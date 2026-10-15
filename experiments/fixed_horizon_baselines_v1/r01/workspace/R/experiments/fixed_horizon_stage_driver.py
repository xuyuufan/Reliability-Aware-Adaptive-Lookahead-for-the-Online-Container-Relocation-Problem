"""ROOT-release-gated M1 bridge/smoke and M2 fixed-H bank orchestrator."""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import pathlib
import sys
import time
from typing import Any, Mapping

HERE = pathlib.Path(__file__).resolve()
WORKSPACE = HERE.parents[2]
OUT = WORKSPACE.parent
if str(WORKSPACE) not in sys.path:
    sys.path.insert(0, str(WORKSPACE))

from R.experiments.fixed_horizon_integrity import (
    canonical_sha, read_json, require_stage_release, sha256_file,
    verify_stage_unchanged,
)
from R.experiments.fixed_horizon_runner import (
    _check_stage_call, audit_loaded_research_modules, load_isolated_runtime,
    open_stage_context, run_one_fixed,
)
from R.experiments.fixed_horizon_schedule import (
    ALGORITHMS, HORIZONS, bank_keys, build_m1_bridge_keys, parse_fixed_bank_key, remaining_bank_keys,
    registered_schedule, smoke_keys,
)

RAW_FIELDS = (
    "trajectory_key", "layout_id", "layout_fingerprint", "geometry", "condition_id",
    "family", "strength", "p", "window_size", "repeat", "algorithm", "method",
    "R_raw", "N", "R_over_N", "forecast_seed_evaluator_only", "F_hash", "complete",
    "decision_count", "wall_seconds", "actor_time_seconds", "eg_cached_moves",
)


def _safe(value: Any) -> Any:
    if hasattr(value, "__dataclass_fields__"):
        from dataclasses import asdict
        return _safe(asdict(value))
    if isinstance(value, Mapping):
        return {str(k): _safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(v) for v in value]
    if hasattr(value, "tolist"):
        return _safe(value.tolist())
    if hasattr(value, "item"):
        return _safe(value.item())
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"cannot serialize {type(value).__name__}")


def _json_line(stream: Any, value: Mapping[str, Any]) -> None:
    stream.write(json.dumps(_safe(value), sort_keys=True, ensure_ascii=False,
                            separators=(",", ":"), allow_nan=False) + "\n")


def _write_receipt(path: pathlib.Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    record = _safe(dict(payload))
    record["receipt_sha256"] = canonical_sha(record)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(record, stream, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    return record


def _files_sha256(run_dir: pathlib.Path) -> dict[str, str]:
    names = (
        "raw_trajectories.csv", "public_decision_trace.jsonl.gz",
        "private/evaluator_sidecar.jsonl.gz", "private/physical_steps.jsonl.gz",
    )
    out = {}
    for name in names:
        path = run_dir / name
        if not path.is_file():
            raise RuntimeError(f"required completed stage artifact is missing: {path}")
        out[name] = sha256_file(path)
    return out


def _read_csv(path: pathlib.Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def _append_csv(path: pathlib.Path, row: Mapping[str, Any], *, header: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "w" if header else "a"
    with path.open(mode, encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=RAW_FIELDS, extrasaction="ignore", lineterminator="\n")
        if header:
            writer.writeheader()
        else:
            writer.writerow({name: row.get(name) for name in RAW_FIELDS})
        stream.flush()


def _trace_sinks(run_dir: pathlib.Path):
    public_path = run_dir / "public_decision_trace.jsonl.gz"
    evaluator_path = run_dir / "private/evaluator_sidecar.jsonl.gz"
    physical_path = run_dir / "private/physical_steps.jsonl.gz"
    for path in (public_path, evaluator_path, physical_path):
        path.parent.mkdir(parents=True, exist_ok=True)

    def execute(callable_trajectory):
        public_mode = "at" if public_path.exists() else "wt"
        evaluator_mode = "at" if evaluator_path.exists() else "wt"
        physical_mode = "at" if physical_path.exists() else "wt"
        with gzip.open(public_path, public_mode, encoding="utf-8") as public_stream, \
             gzip.open(evaluator_path, evaluator_mode, encoding="utf-8") as evaluator_stream, \
             gzip.open(physical_path, physical_mode, encoding="utf-8") as physical_stream:
            def public_sink(item):
                _json_line(public_stream, item)

            def private_sink(item):
                target = physical_stream if item.get("record_type") == "physical_step" else evaluator_stream
                _json_line(target, item)

            return callable_trajectory(public_sink, private_sink)
    return execute


def _assert_budget(context: Mapping[str, Any]) -> None:
    elapsed = time.monotonic() - float(context["started_monotonic"])
    if (elapsed >= float(context["science_cap_seconds"])
            or float(context["previous_science_seconds"]) + elapsed >= 7200.0):
        raise RuntimeError("cumulative stage science budget exhausted")


def _m1(stage_context: dict[str, Any], verified: Mapping[str, Any]) -> dict[str, Any]:
    protocol, runtime, gp_rule, objects, _imports = load_isolated_runtime()
    comparison = objects["comparison"]
    five_runner = objects["five_runner"]
    # Load frozen models through the byte-identical PLAN-051 adapter. This also
    # checks the pre-registered Ridge and R-Order model contracts.
    _old_protocol, runtime, gp_rule, models = comparison._load_runtime()
    layouts_raw, conditions = registered_schedule()
    episodes = [(row, runtime["OnlineCRPEpisode"].from_record(row)) for row in layouts_raw]
    first_layouts = {
        geometry: next(pair for pair in episodes if pair[0].get("parameter_group") == geometry)
        for geometry in ("G1", "G2")
    }
    clean = next(item for item in conditions if item["family"] == "CLEAN")
    delayed = next(item for item in conditions
                   if item["family"] == "delayed_positions" and item["p"] == 0.3 and item["repeat"] == 0)
    selected_conditions = (clean, delayed)
    bridge_dir, smoke_dir = OUT / "bridge", OUT / "smoke"
    bridge_dir.mkdir(parents=True, exist_ok=False)
    smoke_dir.mkdir(parents=True, exist_ok=False)
    _append_csv(bridge_dir / "raw_trajectories.csv", {}, header=True)
    _append_csv(smoke_dir / "raw_trajectories.csv", {}, header=True)
    bridge_exec, smoke_exec = _trace_sinks(bridge_dir), _trace_sinks(smoke_dir)
    forecasts: dict[str, tuple[tuple[int, ...], int | None]] = {}
    bridge_rows: list[dict[str, Any]] = []
    smoke_rows: list[dict[str, Any]] = []

    for condition in selected_conditions:
        layout_row, episode = first_layouts["G1"]
        fp = str(layout_row["label_invariant_fingerprint"])
        cache_key = f"{fp}|{condition['condition_id']}"
        forecasts.setdefault(cache_key, comparison._forecast_for(episode, fp, condition, runtime))
        forecast, seed = forecasts[cache_key]
        for algorithm in ALGORITHMS:
            for method in ("R_UTILITY_RIDGE", "NATIVE20"):
                _assert_budget(stage_context)
                raw_key = comparison._trajectory_key(layout_row, condition, algorithm, method)
                whitelist_key = "BRIDGE|" + raw_key
                _check_stage_call(stage_context, whitelist_key)

                def do_bridge(public_sink, private_sink):
                    private_sink({
                        "record_type": "forecast_binding", "trajectory_key": raw_key,
                        "layout_fingerprint": fp, "condition_evaluator_only": condition,
                        "forecast_seed_evaluator_only": seed,
                        "forecast_sha256_evaluator_only": comparison._public_hash(forecast),
                    })
                    if algorithm in ("EM", "GP", "ERI", "EG"):
                        return comparison._run_trajectory(
                            episode=episode, row=layout_row, condition=condition,
                            algorithm=algorithm, method=method, forecast=forecast, seed=seed,
                            runtime=runtime, gp_rule=gp_rule, models=models,
                            trajectory_key=raw_key, public_sink=public_sink, private_sink=private_sink,
                        )
                    return five_runner._run_one(
                        episode=episode, row=layout_row, condition=condition,
                        solver=algorithm, method=method, forecast=forecast, seed=seed,
                        runtime=runtime, models=models, trajectory_key=raw_key,
                        public_sink=public_sink, private_sink=private_sink,
                        comparison=comparison,
                    )

                row = bridge_exec(do_bridge)
                if float(row["wall_seconds"]) > 30.0:
                    raise RuntimeError(f"trajectory hard cap exceeded: {raw_key}")
                stage_context["used_keys"].add(whitelist_key)
                _append_csv(bridge_dir / "raw_trajectories.csv", row)
                bridge_rows.append(row)

    for geometry in ("G1", "G2"):
        layout_row, episode = first_layouts[geometry]
        fp = str(layout_row["label_invariant_fingerprint"])
        for condition in selected_conditions:
            cache_key = f"{fp}|{condition['condition_id']}"
            if cache_key not in forecasts:
                forecasts[cache_key] = comparison._forecast_for(episode, fp, condition, runtime)
            forecast, seed = forecasts[cache_key]
            fhash = comparison._public_hash(forecast)
            for algorithm in ALGORITHMS:
                for horizon in HORIZONS:
                    _assert_budget(stage_context)
                    method = f"FIXED_H{horizon}"
                    key = comparison._trajectory_key(layout_row, condition, algorithm, method)

                    def do_smoke(public_sink, private_sink):
                        return run_one_fixed(
                            stage_context=stage_context, episode=episode, row=layout_row,
                            condition=condition, algorithm=algorithm, horizon=horizon,
                            forecast=forecast, seed=seed, runtime=runtime, gp_rule=gp_rule,
                            trajectory_key=key, public_sink=public_sink, private_sink=private_sink,
                        )

                    row = smoke_exec(do_smoke)
                    if row["F_hash"] != fhash:
                        raise RuntimeError(f"fixed forecast drift in smoke trajectory {key}")
                    _append_csv(smoke_dir / "raw_trajectories.csv", row)
                    smoke_rows.append(row)

    if len(bridge_rows) != 24 or len(smoke_rows) != 96 or len(stage_context["used_keys"]) != 120:
        raise RuntimeError("M1 completed trajectory/key counts differ from 24 bridge + 96 smoke")
    if {"BRIDGE|" + row["trajectory_key"] for row in bridge_rows} | {row["trajectory_key"] for row in smoke_rows} != stage_context["allowed_keys"]:
        raise RuntimeError("M1 executed keys differ from the exact ROOT release whitelist")
    elapsed = time.monotonic() - float(stage_context["started_monotonic"])
    if elapsed >= float(stage_context["science_cap_seconds"]):
        raise RuntimeError("M1 science cap exhausted before receipts were committed")

    common = {
        "protocol_id": "ridge_fixed_horizon_baselines_v1", "run_id": "r01",
        "source_binding_sha256": verified["source_binding_sha256"],
        "authorized_by_release_sha256": verified["release_file_sha256"],
        "research_module_files_at_stage_exit": audit_loaded_research_modules(),
    }
    bridge_receipt = _write_receipt(bridge_dir / "M1_BRIDGE_RECEIPT.json", {
        **common, "scope": "bridge", "status": "PASS_M1_BRIDGE", "trajectory_count": 24,
        "unique_trajectory_keys": len({row["trajectory_key"] for row in bridge_rows}),
        "files_sha256": _files_sha256(bridge_dir),
    })
    smoke_receipt = _write_receipt(smoke_dir / "M1_SMOKE_RECEIPT.json", {
        **common, "scope": "smoke", "status": "PASS_M1_SMOKE", "trajectory_count": 96,
        "unique_trajectory_keys": len({row["trajectory_key"] for row in smoke_rows}),
        "files_sha256": _files_sha256(smoke_dir),
    })
    combined_path = OUT / "audit/M1_BRIDGE_SMOKE_RECEIPT.json"
    combined = _write_receipt(combined_path, {
        **common, "scope": "registered_bridge_24_plus_smoke_96",
        "status": "PASS_M1_BRIDGE_AND_SMOKE", "trajectory_count": 120,
        "bridge_count": 24, "fixed_h_smoke_count": 96,
        "science_wall_seconds": elapsed,
        "bridge_receipt_sha256": bridge_receipt["receipt_sha256"],
        "smoke_receipt_sha256": smoke_receipt["receipt_sha256"],
        "runner_process_exit_code": None, "runner_process_seconds": None,
        "files_sha256": {
            "bridge/" + name: value for name, value in bridge_receipt["files_sha256"].items()
        } | {
            "smoke/" + name: value for name, value in smoke_receipt["files_sha256"].items()
        },
    })
    frozen_after = verify_stage_unchanged("M1_BRIDGE_SMOKE", stage_context["snapshot"])
    return {"status": combined["status"], "receipt_path": str(combined_path),
            "trajectory_count": 120, "science_wall_seconds": elapsed,
            "frozen_exit_check": frozen_after}


def _m2(stage_context: dict[str, Any], verified: Mapping[str, Any]) -> dict[str, Any]:
    protocol, runtime, gp_rule, objects, _imports = load_isolated_runtime()
    comparison = objects["comparison"]
    _old_protocol, runtime, gp_rule, _models = comparison._load_runtime()
    layouts_raw, conditions = registered_schedule()
    episodes = [(row, runtime["OnlineCRPEpisode"].from_record(row)) for row in layouts_raw]
    smoke_path = OUT / "smoke/raw_trajectories.csv"
    smoke_rows = _read_csv(smoke_path)
    smoke_receipt = read_json(OUT / "smoke/M1_SMOKE_RECEIPT.json")
    previous_receipt = read_json(OUT / "audit/M1_BRIDGE_SMOKE_RECEIPT.json")
    if len(smoke_rows) != 96 or len({row["trajectory_key"] for row in smoke_rows}) != 96:
        raise RuntimeError("M2 requires exactly 96 unique completed smoke rows")
    if previous_receipt.get("runner_process_exit_code") != 0:
        raise RuntimeError("M2 requires an M1 stage driver receipt with actual process exit 0")
    if stage_context["previous_science_seconds"] < float(previous_receipt["science_wall_seconds"]):
        raise RuntimeError("M2 cumulative ledger omits prior M1 and/or failed M2 process elapsed")

    bank_dir = OUT / "bank"
    bank_dir.mkdir(parents=True, exist_ok=False)
    raw_path = bank_dir / "raw_trajectories.csv"
    _append_csv(raw_path, {}, header=True)
    for row in smoke_rows:
        _append_csv(raw_path, row)
    for source, target in (
        (OUT / "smoke/public_decision_trace.jsonl.gz", bank_dir / "public_decision_trace.jsonl.gz"),
        (OUT / "smoke/private/evaluator_sidecar.jsonl.gz", bank_dir / "private/evaluator_sidecar.jsonl.gz"),
        (OUT / "smoke/private/physical_steps.jsonl.gz", bank_dir / "private/physical_steps.jsonl.gz"),
    ):
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())
    bank_exec = _trace_sinks(bank_dir)
    smoke_key_set = {row["trajectory_key"] for row in smoke_rows}
    if smoke_key_set != smoke_keys():
        raise RuntimeError("M1 smoke rows do not exactly match the registered 96-key set")

    layouts = {str(row["label_invariant_fingerprint"]): (row, runtime["OnlineCRPEpisode"].from_record(row))
               for row in layouts_raw}
    conditions_by_id = {str(row["condition_id"]): row for row in conditions}
    forecast_cache: dict[str, tuple[tuple[int, ...], int | None]] = {}
    new_rows: list[dict[str, Any]] = []
    for key in sorted(remaining_bank_keys()):
        _assert_budget(stage_context)
        fp, condition_id, algorithm, method = parse_fixed_bank_key(key)
        layout_row, episode = layouts[fp]
        condition = conditions_by_id[condition_id]
        horizon = int(method.removeprefix("FIXED_H"))
        cache_key = f"{fp}|{condition_id}"
        if cache_key not in forecast_cache:
            forecast_cache[cache_key] = comparison._forecast_for(episode, fp, condition, runtime)
        forecast, seed = forecast_cache[cache_key]
        fhash = comparison._public_hash(forecast)

        def do_bank(public_sink, private_sink):
            return run_one_fixed(
                stage_context=stage_context, episode=episode, row=layout_row,
                condition=condition, algorithm=algorithm, horizon=horizon,
                forecast=forecast, seed=seed, runtime=runtime, gp_rule=gp_rule,
                trajectory_key=key, public_sink=public_sink, private_sink=private_sink,
            )

        row = bank_exec(do_bank)
        if row["F_hash"] != fhash:
            raise RuntimeError(f"fixed forecast drift in bank trajectory {key}")
        _append_csv(raw_path, row)
        new_rows.append(row)
    if len(new_rows) != 9984 or len(stage_context["used_keys"]) != 9984:
        raise RuntimeError("M2 completed trajectory/key counts differ from the exact 9,984 whitelist")

    all_rows = _read_csv(raw_path)
    if len(all_rows) != 10080 or {row["trajectory_key"] for row in all_rows} != bank_keys():
        raise RuntimeError("M2 merged bank rows differ from 96 exact smoke + 9,984 new keys")
    for row in smoke_rows:
        match = next(item for item in all_rows if item["trajectory_key"] == row["trajectory_key"])
        if match != row:
            raise RuntimeError(f"M1 smoke raw row changed during exact M2 reuse: {row['trajectory_key']}")
    elapsed = time.monotonic() - float(stage_context["started_monotonic"])
    if elapsed >= float(stage_context["science_cap_seconds"]):
        raise RuntimeError("M2 science cap exhausted before receipt commit")
    receipt_path = bank_dir / "M2_BANK_RECEIPT.json"
    receipt = _write_receipt(receipt_path, {
        "protocol_id": "ridge_fixed_horizon_baselines_v1", "run_id": "r01",
        "scope": "registered_fixed_h_bank_remaining_after_exact_smoke_reuse",
        "status": "PASS_M2_COMPLETE_BANK", "trajectory_count": 10080,
        "unique_trajectory_keys": 10080, "new_trajectory_count": 9984,
        "reused_smoke_row_count": 96, "source_binding_sha256": verified["source_binding_sha256"],
        "authorized_by_release_sha256": verified["release_file_sha256"],
        "science_wall_seconds": stage_context["previous_science_seconds"] + elapsed,
        "stage_wall_seconds": elapsed,
        "smoke_source_receipt_sha256": smoke_receipt["receipt_sha256"],
        "m1_combined_receipt_sha256": previous_receipt["receipt_sha256"],
        "files_sha256": _files_sha256(bank_dir),
        "research_module_files_at_stage_exit": audit_loaded_research_modules(),
        "runner_process_exit_code": None, "runner_process_seconds": None,
    })
    frozen_after = verify_stage_unchanged("M2_BANK", stage_context["snapshot"])
    return {"status": receipt["status"], "receipt_path": str(receipt_path),
            "trajectory_count": 10080, "science_wall_seconds": receipt["science_wall_seconds"],
            "frozen_exit_check": frozen_after}


def run(stage: str, previous_science_seconds: float = 0.0) -> dict[str, Any]:
    stage_name = "M1_BRIDGE_SMOKE" if stage == "M1" else "M2_BANK"
    previous = float(previous_science_seconds)
    if stage == "M2":
        m1 = read_json(OUT / "audit/M1_BRIDGE_SMOKE_RECEIPT.json")
        previous += float(m1["science_wall_seconds"])
    context, verified, snapshot = open_stage_context(stage_name, previous_science_seconds=previous)
    context["snapshot"] = snapshot
    result = _m1(context, verified) if stage == "M1" else _m2(context, verified)
    if len(context["used_keys"]) != context["allowed_count"]:
        raise RuntimeError("stage completed without consuming its entire exact release whitelist")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("M1", "M2"), required=True)
    parser.add_argument("--previous-science-seconds", type=float, default=0.0)
    args = parser.parse_args()
    try:
        result = run(args.stage, args.previous_science_seconds)
    except BaseException as exc:
        print(json.dumps({"status": "FAIL", "stage": args.stage,
                          "exception": f"{type(exc).__name__}: {exc}"}, sort_keys=True))
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
