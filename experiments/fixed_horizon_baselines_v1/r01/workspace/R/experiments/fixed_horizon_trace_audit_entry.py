"""Command entry for the frozen physical/public trace auditor."""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import itertools
import json
import math
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

WORKSPACE = Path(__file__).resolve().parents[2]
OUT = WORKSPACE.parent
if str(WORKSPACE) not in sys.path:
    sys.path.insert(0, str(WORKSPACE))

from R.experiments import fixed_horizon_trace_audit as core
from R.experiments import fixed_horizon_integrity as integrity
from R.experiments.fixed_horizon_integrity import (
    require_stage_release, sha256_file, verify_frozen_m0, verify_stage_unchanged,
)
from R.experiments.fixed_horizon_runner import audit_loaded_research_modules, load_isolated_runtime
from R.experiments.fixed_horizon_schedule import bank_keys, build_m1_bridge_keys, smoke_keys


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _group_records_by_trajectory(
    records: Iterable[Mapping[str, Any]],
) -> Iterator[tuple[str, list[dict[str, Any]]]]:
    """Retain one contiguous trajectory group and reject split/reappearing keys."""
    current_key: str | None = None
    current: list[dict[str, Any]] = []
    closed: set[str] = set()
    for record in records:
        key = record.get("trajectory_key")
        if not isinstance(key, str) or not key:
            raise core.AuditFailure("trace record has an empty or non-string trajectory key")
        if key != current_key:
            if current_key is not None:
                closed.add(current_key)
                yield current_key, current
            if key in closed:
                raise core.AuditFailure(f"trajectory trace group is split or reappears: {key}")
            current_key = key
            current = []
        current.append(dict(record))
    if current_key is not None:
        yield current_key, current


def _jsonl_gz_records(path: Path) -> Iterator[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise core.AuditFailure(f"invalid JSONL at {path}:{line_number}: {exc}") from exc
            if not isinstance(value, dict):
                raise core.AuditFailure(f"trace row is not an object at {path}:{line_number}")
            yield value


def _jsonl_gz_groups(path: Path) -> Iterator[tuple[str, list[dict[str, Any]]]]:
    yield from _group_records_by_trajectory(_jsonl_gz_records(path))


def _aligned_trajectory_groups(
    raw_rows: Sequence[Mapping[str, Any]],
    public_groups: Iterable[tuple[str, list[dict[str, Any]]]],
    evaluator_groups: Iterable[tuple[str, list[dict[str, Any]]]],
    physical_groups: Iterable[tuple[str, list[dict[str, Any]]]],
) -> Iterator[tuple[str, Mapping[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]]:
    """Join ordered gzip groups to raw rows and fail on key drift or any EOF mismatch."""
    missing = object()
    for index, item in enumerate(itertools.zip_longest(
        raw_rows, public_groups, evaluator_groups, physical_groups, fillvalue=missing
    )):
        raw, public, evaluator, physical = item
        if any(value is missing for value in item):
            raise core.AuditFailure(f"trajectory trace streams have unequal group counts at EOF/index {index}")
        key = str(raw.get("trajectory_key", ""))
        if not key:
            raise core.AuditFailure(f"raw trajectory row {index} has an empty key")
        for label, group in (("public", public), ("evaluator", evaluator), ("physical", physical)):
            if group[0] != key:
                raise core.AuditFailure(
                    f"{label} trace group key/order mismatch at row {index}: {group[0]!r} != {key!r}"
                )
        yield key, raw, public[1], evaluator[1], physical[1]


def _selfhash(record: Mapping[str, Any], field: str = "receipt_sha256") -> str:
    payload = dict(record)
    recorded = str(payload.pop(field, ""))
    actual = hashlib.sha256(_canonical(payload)).hexdigest()
    if not recorded or recorded != actual:
        raise core.AuditFailure("receipt self-hash is stale")
    return recorded


def _capture_stage_entry_snapshot(gate: Mapping[str, Any]) -> dict[str, str]:
    keys = (*integrity.FROZEN_HASH_KEYS, "release_file_sha256")
    missing = [key for key in keys if not isinstance(gate.get(key), str) or not gate.get(key)]
    if missing:
        raise core.AuditFailure(f"stage release payload lacks entry snapshot hashes: {missing}")
    return {key: str(gate[key]) for key in keys}


def _check_runner_process_fields(runner: Mapping[str, Any], scope: str) -> None:
    core.same(runner.get("runner_process_exit_code"), 0,
              path=f"{scope}.actual_runner_process_exit_code")
    elapsed = runner.get("runner_process_seconds")
    core.fail_if(
        isinstance(elapsed, (int, float)) and not isinstance(elapsed, bool)
        and math.isfinite(float(elapsed)) and float(elapsed) >= 0,
        f"{scope} actual runner process elapsed is missing or invalid",
    )
    if scope in ("bridge", "smoke", "m1"):
        core.same(runner.get("whole_M1_process_scope"), True,
                  path=f"{scope}.whole_M1_process_scope")


def _expected(scope: str) -> set[str]:
    if scope == "bridge":
        return {item.removeprefix("BRIDGE|") for item in build_m1_bridge_keys()}
    if scope == "smoke":
        return smoke_keys()
    if scope == "bank":
        return bank_keys()
    raise ValueError(scope)


def _source_bridge_match(raw_by_key: Mapping[str, Mapping[str, str]], steps_by_key: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[str, Any]:
    legacy_root = Path(r"D:\DSAI\containter_new\R\output")
    roots = {
        "PLAN-051": legacy_root / "r_five_versions_comparison_v1/r01/bank_01",
        "PLAN-052": legacy_root / "r_seven_solvers_five_r_v1/r01/m2_bank_01",
    }
    old_rows: dict[str, dict[str, str]] = {}
    for protocol, root in roots.items():
        path = root / "raw_trajectories.csv"
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            for row in csv.DictReader(stream):
                key = row.get("trajectory_key")
                if key in raw_by_key:
                    if key in old_rows:
                        raise core.AuditFailure(f"historical bridge key is duplicated: {key}")
                    old_rows[key] = row
    if set(old_rows) != set(raw_by_key):
        raise core.AuditFailure("bridge does not map to the exact historical key set")

    old_steps: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for protocol in ("PLAN-051", "PLAN-052"):
        path = roots[protocol] / "public_decision_trace.jsonl.gz"
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            for line in stream:
                if not line.strip():
                    continue
                item = json.loads(line)
                if item.get("trajectory_key") in raw_by_key:
                    old_steps[item["trajectory_key"]].append(item)
    if set(old_steps) != set(raw_by_key):
        raise core.AuditFailure("historical bridge public trace keys differ from the registered 24")

    for key, new in raw_by_key.items():
        old = old_rows[key]
        source_protocol = "PLAN-051" if new["algorithm"] in ("EM", "GP", "ERI", "EG") else "PLAN-052"
        identity_field = "algorithm" if source_protocol == "PLAN-051" else "solver"
        for field in (
            "trajectory_key", "layout_fingerprint", "condition_id", "method",
            "R_raw", "N", "decision_count", "forecast_seed_evaluator_only", "F_hash", "complete",
        ):
            core.same(new.get(field), old.get(field), path=f"bridge.{key}.{field}")
        core.same(old.get("algorithm", old.get("solver")), new["algorithm"],
                  path=f"bridge.{key}.historical_solver_identity")
        old_trace = sorted(old_steps[key], key=lambda item: int(item["decision_index"]))
        new_trace = sorted(steps_by_key[key], key=lambda item: int(item["decision_index"]))
        core.same(len(new_trace), len(old_trace), path=f"bridge.{key}.trace_count")
        for index, (actual, expected) in enumerate(zip(new_trace, old_trace, strict=True)):
            for field in (
                "decision_index", identity_field, "method", "public_frame_before", "own_action",
                "actor_trace", "public_frame_after", "terminal_after_action",
            ):
                core.same(actual.get(field), expected.get(field), path=f"bridge.{key}.step[{index}].{field}")
    return {
        "historical_trace_keys": len(old_steps),
        "public_frame_action_and_actor_trace_exact": True,
        "R_raw_N_decision_count_seed_F_hash_exact": True,
        "timing_fields_ignored": True,
        "source_protocol_by_solver": {a: "PLAN-051" if a in ("EM", "GP", "ERI", "EG") else "PLAN-052" for a in core.ALGORITHMS},
        "native20_EG_reconstruction": "frozen build_eg20_plan replay from every full-H20 group-start public frame; not an independent EG implementation",
    }


def run_audit(scope: str, *, output_path: Path) -> dict[str, Any]:
    started = time.monotonic()
    if scope not in ("bridge", "smoke", "bank"):
        raise ValueError("scope must be bridge, smoke, or bank")
    stage = "M2_BANK" if scope == "bank" else "M1_BRIDGE_SMOKE"
    gate = require_stage_release(stage)
    frozen = verify_frozen_m0(require_root_review=True)
    start_snapshot = _capture_stage_entry_snapshot(gate)
    if scope == "bridge":
        run_dir = OUT / "bridge"
        expected_count = 24
        receipt_path = run_dir / "M1_BRIDGE_RECEIPT.json"
        receipt_status = "PASS_M1_BRIDGE"
        methods = core.BRIDGE_METHODS
    elif scope == "smoke":
        run_dir = OUT / "smoke"
        expected_count = 96
        receipt_path = run_dir / "M1_SMOKE_RECEIPT.json"
        receipt_status = "PASS_M1_SMOKE"
        methods = tuple(f"FIXED_H{h}" for h in core.NEW_HORIZONS)
    else:
        run_dir = OUT / "bank"
        expected_count = 10080
        receipt_path = run_dir / "M2_BANK_RECEIPT.json"
        receipt_status = "PASS_M2_COMPLETE_BANK"
        methods = tuple(f"FIXED_H{h}" for h in core.NEW_HORIZONS)

    config = core.read_json(core.CONFIG_PATH)
    layouts_doc = core.read_json(core.LAYOUT_PATH)
    layouts = {str(row["label_invariant_fingerprint"]): row for row in layouts_doc}
    conditions = core.condition_registry()
    raw_path = run_dir / "raw_trajectories.csv"
    public_path = run_dir / "public_decision_trace.jsonl.gz"
    evaluator_path = run_dir / "private/evaluator_sidecar.jsonl.gz"
    physical_path = run_dir / "private/physical_steps.jsonl.gz"
    for path in (raw_path, public_path, evaluator_path, physical_path, receipt_path):
        core.fail_if(path.is_file(), f"required {scope} evidence is missing: {path}")

    runner = core.read_json(receipt_path)
    runner_selfhash = _selfhash(runner)
    core.same(runner.get("status"), receipt_status, path=f"{scope}.runner_status")
    _check_runner_process_fields(runner, scope)
    core.same(runner.get("trajectory_count"), expected_count, path=f"{scope}.runner_count")
    core.same(runner.get("unique_trajectory_keys"), expected_count, path=f"{scope}.runner_unique_count")
    core.same(runner.get("source_binding_sha256"), frozen["source_binding_sha256"], path=f"{scope}.runner_binding")
    core.same(runner.get("authorized_by_release_sha256"), gate["release_file_sha256"], path=f"{scope}.runner_release")
    for path in (raw_path, public_path, evaluator_path, physical_path):
        name = path.relative_to(run_dir).as_posix()
        core.same(sha256_file(path), runner.get("files_sha256", {}).get(name), path=f"{scope}.file_hash.{name}")

    with raw_path.open("r", encoding="utf-8-sig", newline="") as stream:
        raw_rows = list(csv.DictReader(stream))
    core.fail_if(len(raw_rows) == expected_count, f"{scope} raw count {len(raw_rows)} != {expected_count}")
    raw_by_key: dict[str, dict[str, str]] = {}
    for row in raw_rows:
        key = row.get("trajectory_key", "")
        core.fail_if(key and key not in raw_by_key, f"duplicate/empty trajectory key {key!r}")
        raw_by_key[key] = row
        core.fail_if(row.get("layout_fingerprint") in layouts, f"{key}: unknown layout fingerprint")
        core.fail_if(row.get("condition_id") in conditions, f"{key}: unknown condition")
        core.fail_if(row.get("algorithm") in core.ALGORITHMS and row.get("method") in methods, f"{key}: unregistered method/solver")
        for field, expected in conditions[row["condition_id"]].items():
            actual: Any = row.get(field)
            if field == "p": actual = None if actual in (None, "") else float(actual)
            elif field == "window_size": actual = None if actual in (None, "") else int(actual)
            elif field == "repeat": actual = int(actual)
            core.same(actual, expected, path=f"{key}.{field}")
        core.same(row.get("geometry"), layouts[row["layout_fingerprint"]]["parameter_group"], path=f"{key}.geometry")
        core.same(row.get("layout_id"), layouts[row["layout_fingerprint"]]["episode_id"], path=f"{key}.layout_id")
    core.same(set(raw_by_key), _expected(scope), path=f"{scope}.exact_registered_key_set")

    model_doc = core.read_json(core.ORDER_MODEL_PATH)
    ridge_doc = core.read_json(core.RIDGE_PATH)
    ridge_coefficients = [float(value) for value in ridge_doc["coefficients"]]
    order_rates = [[float(value) for value in row] for row in model_doc["main"]["coefficients"]]
    _protocol, runtime, gp_rule, _objects, imported_at_entry = load_isolated_runtime()
    native_eg_audit = {
        "group_count": 0, "action_count": 0, "cached_move_count": 0,
        "fixed_h_group_count": 0, "fixed_h_action_count": 0,
        "fixed_h_cached_move_count": 0,
    }
    total_steps = 0
    public_trace_records = 0
    evaluator_trace_records = 0
    physical_step_records = 0
    forecast_binding_records = 0
    bridge_steps_by_key: dict[str, list[dict[str, Any]]] = {}
    for key, row, key_steps_unsorted, evaluator_group, physical_group in _aligned_trajectory_groups(
        raw_rows,
        _jsonl_gz_groups(public_path),
        _jsonl_gz_groups(evaluator_path),
        _jsonl_gz_groups(physical_path),
    ):
        key_steps = sorted(key_steps_unsorted, key=lambda item: int(item["decision_index"]))
        public_trace_records += len(key_steps)
        evaluator_trace_records += len(evaluator_group)
        physical_step_records += len(physical_group)
        start_records = [item for item in evaluator_group if item.get("record_type") == "trajectory_start"]
        bindings_for_key = [item for item in evaluator_group if item.get("record_type") == "forecast_binding"]
        core.same(len(start_records), 1, path=f"{key}.trajectory_start_count")
        core.same(len(bindings_for_key), 1, path=f"{key}.forecast_binding_count")
        forecast_binding_records += len(bindings_for_key)
        core.same(int(row["decision_count"]), len(key_steps), path=f"{key}.decision_count")
        core.same(len(physical_group), len(key_steps), path=f"{key}.physical_step_count")
        if scope == "bridge":
            bridge_steps_by_key[key] = key_steps
        total_steps += core.replay_trajectory(
            row, start_records[0], key_steps, evaluator_group + physical_group,
            ridge_coefficients, order_rates, int(layouts[row["layout_fingerprint"]]["max_tiers"]),
            native_eg_audit, runtime, gp_rule,
        )

    bridge_compare = _source_bridge_match(raw_by_key, bridge_steps_by_key) if scope == "bridge" else None
    smoke_reuse: dict[str, Any] | None = None
    if scope == "bank":
        smoke_receipt_path = OUT / "smoke/M1_SMOKE_RECEIPT.json"
        smoke_receipt = core.read_json(smoke_receipt_path)
        _selfhash(smoke_receipt)
        with (OUT / "smoke/raw_trajectories.csv").open("r", encoding="utf-8-sig", newline="") as stream:
            smoke_rows = {row["trajectory_key"]: row for row in csv.DictReader(stream)}
        core.same(len(smoke_rows), 96, path="bank.smoke_exact_key_count")
        for key, row in smoke_rows.items():
            core.same(raw_by_key.get(key), row, path=f"bank.exact_smoke_row.{key}")
        core.same(runner.get("smoke_source_receipt_sha256"), smoke_receipt["receipt_sha256"], path="bank.smoke_receipt_binding")
        smoke_reuse = {"row_count": 96, "raw_rows_identical": True, "input_receipt_sha256": smoke_receipt["receipt_sha256"]}

    report: dict[str, Any] = {
        "protocol_id": "ridge_fixed_horizon_baselines_v1", "run_id": "r01", "scope": scope,
        "status": "PASS_M2_TRACE_AUDIT" if scope == "bank" else "PASS",
        "performance_scoring": False,
        "trajectory_count": len(raw_by_key), "unique_trajectory_keys": len(raw_by_key),
        "step_count": total_steps, "public_trace_records": public_trace_records,
        "evaluator_trace_records": evaluator_trace_records,
        "physical_step_records": physical_step_records, "forecast_binding_records": forecast_binding_records,
        "runner_receipt_sha256": runner_selfhash, "runner_receipt_file_sha256": sha256_file(receipt_path),
        "root_release_sha256": gate["release_file_sha256"],
        "m0_source_binding_sha256": frozen["source_binding_sha256"],
        "native20_EG_reconstructed_group_count": native_eg_audit["group_count"],
        "native20_EG_reconstructed_action_count": native_eg_audit["action_count"],
        "native20_EG_cached_move_count": native_eg_audit["cached_move_count"],
        "fixed_h_EG_reconstructed_group_count": native_eg_audit["fixed_h_group_count"],
        "fixed_h_EG_reconstructed_action_count": native_eg_audit["fixed_h_action_count"],
        "fixed_h_EG_cached_move_count": native_eg_audit["fixed_h_cached_move_count"],
        "bridge_comparison": bridge_compare, "smoke_reuse": smoke_reuse,
        "checks": {
            "registered_key_set_and_condition_namespace": True,
            "three_trace_streams_grouped_and_joined_by_raw_trajectory_order": True,
            "three_trace_streams_equal_eof_and_no_split_groups": True,
            "physical_transitions_and_retrieval_cascades_replayed": True,
            "public_forecast_history_q_and_crane_reconstructed": True,
            "legal_action_and_private_trace_exclusion": True,
            "fixed_H_identity_and_requested_H_checked": scope != "bridge",
            "fixed_H_frozen_proposer_replay_per_step": scope != "bridge",
            "independent_H_prefix_formula_checked": scope != "bridge",
            "full_EG_group_commit_actions_checked": True,
            "fixed_H_EG_plan_replayed_from_projected_public_frame": scope != "bridge",
            "bridge_exact_historical_public_frames_and_actions": scope == "bridge",
            "M1_smoke_rows_exactly_reused_in_M2": scope == "bank",
            "performance_comparison_computed": False,
        },
        "elapsed_seconds": round(time.monotonic() - started, 6),
        "audit_script_sha256": sha256_file(Path(__file__).resolve()),
        "source_basis_script_sha256": sha256_file(core.SCRIPT),
        "research_module_files_at_audit_entry": imported_at_entry,
        "research_module_files_at_audit_exit": audit_loaded_research_modules(),
        "source_unchanged_exit_check": verify_stage_unchanged(stage, start_snapshot),
    }
    report["receipt_sha256"] = hashlib.sha256(_canonical(report)).hexdigest()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, sort_keys=True, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return report


def combine_m1_audits(output_path: Path) -> dict[str, Any]:
    started = time.monotonic()
    gate = require_stage_release("M1_BRIDGE_SMOKE")
    start_snapshot = _capture_stage_entry_snapshot(gate)
    components = {}
    stage_receipt = core.read_json(OUT / "audit/M1_BRIDGE_SMOKE_RECEIPT.json")
    _selfhash(stage_receipt)
    core.same(stage_receipt.get("status"), "PASS_M1_BRIDGE_AND_SMOKE", path="m1.stage_runner_status")
    _check_runner_process_fields(stage_receipt, "m1")
    for scope, path in (
        ("bridge", OUT / "audit/M1_BRIDGE_TRACE_AUDIT.json"),
        ("smoke", OUT / "audit/M1_SMOKE_TRACE_AUDIT.json"),
    ):
        if not path.is_file():
            raise core.AuditFailure(f"M1 component trace audit is missing: {path}")
        receipt = core.read_json(path)
        _selfhash(receipt)
        core.same(receipt.get("status"), "PASS", path=f"m1.{scope}.audit_status")
        core.same(receipt.get("scope"), scope, path=f"m1.{scope}.scope")
        runner_path = OUT / scope / ("M1_BRIDGE_RECEIPT.json" if scope == "bridge" else "M1_SMOKE_RECEIPT.json")
        runner = core.read_json(runner_path)
        _selfhash(runner)
        _check_runner_process_fields(runner, scope)
        core.same(runner.get("runner_process_exit_code"), stage_receipt.get("runner_process_exit_code"),
                  path=f"m1.{scope}.runner_exit_matches_whole_M1")
        core.same(runner.get("runner_process_seconds"), stage_receipt.get("runner_process_seconds"),
                  path=f"m1.{scope}.runner_elapsed_matches_whole_M1")
        core.same(stage_receipt.get(f"{scope}_receipt_sha256"), runner.get("receipt_sha256"),
                  path=f"m1.{scope}.combined_runner_receipt_hash")
        core.same(receipt.get("runner_receipt_sha256"), runner.get("receipt_sha256"),
                  path=f"m1.{scope}.audit_runner_receipt_hash")
        core.same(receipt.get("runner_receipt_file_sha256"), sha256_file(runner_path),
                  path=f"m1.{scope}.audit_runner_receipt_file_hash")
        components[scope] = {
            "path": path.relative_to(OUT).as_posix(),
            "file_sha256": sha256_file(path),
            "receipt_sha256": receipt["receipt_sha256"],
            "trajectory_count": receipt.get("trajectory_count"),
            "step_count": receipt.get("step_count"),
        }
    bridge, smoke = components["bridge"], components["smoke"]
    report = {
        "protocol_id": "ridge_fixed_horizon_baselines_v1", "run_id": "r01",
        "scope": "M1_BRIDGE_AND_SMOKE", "status": "PASS_M1_TRACE_AUDIT",
        "performance_scoring": False, "trajectory_count": 120,
        "step_count": int(bridge.get("step_count") or 0) + int(smoke.get("step_count") or 0),
        "runner_receipt_file_sha256": sha256_file(OUT / "audit/M1_BRIDGE_SMOKE_RECEIPT.json"),
        "runner_receipt_sha256": stage_receipt["receipt_sha256"],
        "bridge_receipt_sha256": bridge["receipt_sha256"],
        "smoke_receipt_sha256": smoke["receipt_sha256"],
        "components": components, "elapsed_seconds": round(time.monotonic() - started, 6),
        "source_unchanged_exit_check": verify_stage_unchanged("M1_BRIDGE_SMOKE", start_snapshot),
    }
    report["receipt_sha256"] = hashlib.sha256(_canonical(report)).hexdigest()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, sort_keys=True, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope", choices=("bridge", "smoke", "m1", "bank"), required=True)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    default_paths = {
        "bridge": OUT / "audit/M1_BRIDGE_TRACE_AUDIT.json",
        "smoke": OUT / "audit/M1_SMOKE_TRACE_AUDIT.json",
        "m1": OUT / "audit/M1_TRACE_AUDIT_RECEIPT.json",
        "bank": OUT / "audit/M2_TRACE_AUDIT_RECEIPT.json",
    }
    path = args.output or default_paths[args.scope]
    try:
        result = combine_m1_audits(path) if args.scope == "m1" else run_audit(args.scope, output_path=path)
    except BaseException as exc:
        failure = {"protocol_id": "ridge_fixed_horizon_baselines_v1", "scope": args.scope,
                   "status": "FAIL", "exception": f"{type(exc).__name__}: {exc}"}
        failure["receipt_sha256"] = hashlib.sha256(_canonical(failure)).hexdigest()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(failure, sort_keys=True, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(json.dumps(failure, sort_keys=True))
        return 1
    print(json.dumps({"status": result["status"], "scope": args.scope,
                      "trajectory_count": result["trajectory_count"], "step_count": result["step_count"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
