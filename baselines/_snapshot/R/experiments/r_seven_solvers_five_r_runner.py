"""PLAN-052 M1 preparation and release-gated seven-solver runner.

The preparation command performs only hash, registry, and synthetic-contract
checks. Smoke and bank execution require separate ROOT release files. Forecast
construction and provider synchronization are delegated unchanged to PLAN-051.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import importlib
import io
import json
import math
from pathlib import Path
import re
import subprocess
import sys
import time
import traceback
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


WORKSPACE = Path(__file__).resolve().parents[2]
if str(WORKSPACE) not in sys.path:
    sys.path.insert(0, str(WORKSPACE))
R_ROOT = WORKSPACE / "R"
OUTPUT_ROOT = R_ROOT / "output/r_seven_solvers_five_r_v1/r01"
PREPARATION_ROOT = OUTPUT_ROOT / "preparation/online/r04"
CONFIG_PATH = R_ROOT / "configs/r_seven_solvers_five_r_v1.json"
PLAN_PATH = R_ROOT / "docs/planning/PLAN_052_NEW_SOLVERS_ONLINE_FIVE_R.md"
LAYOUT_PATH = R_ROOT / "output/r_frozen_cross_solver_transfer_v1/r01/data/select20.json"
OLD_OUTPUT_ROOT = R_ROOT / "output/r_five_versions_comparison_v1/r01"
OLD_BINDING_PATH = OLD_OUTPUT_ROOT / "m0/prepare/M0_SOURCE_BINDINGS.json"
OLD_RAW_PATH = OLD_OUTPUT_ROOT / "bank_01/raw_trajectories.csv"
OLD_ACCEPTANCE_PATH = OLD_OUTPUT_ROOT / "FINAL_ACCEPTANCE.json"
OLD_AUDIT_PATH = OLD_OUTPUT_ROOT / "analysis/bank_raw_physics_public_audit_r04.json"
OLD_BANK_RECEIPT_PATH = OLD_OUTPUT_ROOT / "bank_01/M1_BANK_RECEIPT.json"
ROOT_M0_RELEASE_PATH = OUTPUT_ROOT / "ROOT_M1_PREPARATION_RELEASE.json"
RIDGE_PATH = R_ROOT / "output/universal_r_utility_v1/r02/M1_FIT_01/RIDGE_MODEL.json"
ORDER_MODELS_PATH = R_ROOT / "output/r_order_v1/r01/M1_MODELS.json"
RIDGE_SHA256 = "07bf606c2a7ab1ac1bffa7d39449c34fc719ad8e454eba1243761af44fd7effe"
ORDER_SHA256 = "7b769ad8407ea881fe30a3d2bf130604da3cbbebce585776e5302c02e28991e1"
ROOT_M0_RELEASE_SHA256 = "5d3819b572d801bb71bd4c4d0443908761f4214296b4eeaefcea742300f07d5d"
NEW_SOLVERS = ("RIL", "Chain", "Minmax")
NEW_POLICIES = (
    "OLD_FROZEN_R", "R_SOURCE_FULL", "R_UTILITY_RIDGE", "R_QUAD_H",
    "R_ORDER_MAIN_TAU0_FULL", "NATIVE20",
)
SMOKE_POLICIES = ("NATIVE20", "R_UTILITY_RIDGE", "R_ORDER_MAIN_TAU0_FULL")
SCIENCE_CAP_SECONDS = 7200
TRAJECTORY_CAP_SECONDS = 30
ANALYSIS_CAP_SECONDS = 1800
SMOKE_STAGE_FILES = (
    "raw_trajectories.csv",
    "public_decision_trace.jsonl.gz",
    "private/evaluator_sidecar.jsonl.gz",
    "private/physical_steps.jsonl.gz",
)


class Plan052Stop(RuntimeError):
    """A binding, public-information, physical, release, or budget gate failed."""


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if hasattr(value, "__dataclass_fields__"):
        from dataclasses import asdict
        return _json_safe(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("nonfinite value cannot be serialized")
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"unsupported JSON value {type(value).__name__}")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _verify_self_hash(record: Mapping[str, Any], field: str, label: str) -> None:
    payload = dict(record)
    recorded = str(payload.pop(field, ""))
    if not recorded or recorded != _sha256_bytes(_canonical(payload)):
        raise Plan052Stop(f"{label} self-hash is invalid")


def _stage_file_hashes(stage_dir: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for relative in SMOKE_STAGE_FILES:
        path = stage_dir / relative
        if not path.is_file():
            raise Plan052Stop(f"stage artifact is missing: {relative}")
        hashes[relative] = _sha256_file(path)
    return hashes


def _verify_smoke_artifacts(
    smoke_dir: Path, smoke_receipt: Mapping[str, Any]
) -> tuple[dict[str, bytes], list[dict[str, str]]]:
    _verify_self_hash(smoke_receipt, "receipt_sha256", "M1 smoke receipt")
    if smoke_receipt.get("status") != "PASS" or smoke_receipt.get("scope") != "smoke":
        raise Plan052Stop("bank reuse requires a PASS smoke receipt")
    recorded_hashes = smoke_receipt.get("files_sha256")
    if not isinstance(recorded_hashes, Mapping) or set(recorded_hashes) != set(SMOKE_STAGE_FILES):
        raise Plan052Stop("smoke receipt does not bind all four stage files")
    contents: dict[str, bytes] = {}
    for relative in SMOKE_STAGE_FILES:
        path = smoke_dir / relative
        try:
            payload = path.read_bytes()
        except OSError as exc:
            raise Plan052Stop(f"smoke artifact is unavailable: {relative}") from exc
        if _sha256_bytes(payload) != recorded_hashes.get(relative):
            raise Plan052Stop(f"smoke artifact hash differs from its receipt: {relative}")
        contents[relative] = payload
    csv_text = contents["raw_trajectories.csv"].decode("utf-8-sig")
    rows = list(csv.DictReader(io.StringIO(csv_text, newline="")))
    keys = [row.get("trajectory_key", "") for row in rows]
    if (len(rows) != 9 or len(set(keys)) != 9 or any(not key for key in keys)
            or any(row.get("complete", "").lower() != "true" for row in rows)):
        raise Plan052Stop("smoke CSV is not exactly nine unique complete rows")
    expected_pairs = {(solver, policy) for solver in NEW_SOLVERS for policy in SMOKE_POLICIES}
    if {(row.get("algorithm"), row.get("method")) for row in rows} != expected_pairs:
        raise Plan052Stop("smoke CSV does not contain the registered solver/policy pairs")
    if (smoke_receipt.get("trajectory_count") != 9
            or smoke_receipt.get("unique_trajectory_keys") != 9):
        raise Plan052Stop("smoke receipt row counts do not bind the complete smoke CSV")
    return contents, rows


def _write_new_json(path: Path, value: Any) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(_json_safe(value), sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
    return _sha256_bytes(payload)


def _relative(path: Path) -> str:
    return path.relative_to(WORKSPACE).as_posix()


def _old_runner() -> Any:
    return importlib.import_module("R.experiments.r_five_versions_comparison")


def _verify_m0_release() -> dict[str, Any]:
    if not ROOT_M0_RELEASE_PATH.is_file():
        raise Plan052Stop("ROOT M1 preparation release is missing")
    release = _read_json(ROOT_M0_RELEASE_PATH)
    if _sha256_file(ROOT_M0_RELEASE_PATH) != ROOT_M0_RELEASE_SHA256:
        raise Plan052Stop("ROOT M1 preparation release hash changed")
    if release.get("authorized") is not True or release.get("scope") != "online_preparation_and_synthetic_contract_checks_only":
        raise Plan052Stop("ROOT M1 release does not authorize preparation-only scope")
    receipt_path = OUTPUT_ROOT / "preparation/offline/M0_OFFLINE_RECEIPT.json"
    if release.get("m0_receipt_sha256") != _sha256_file(receipt_path):
        raise Plan052Stop("ROOT M1 release does not bind the current offline receipt")
    for relative, expected in release.get("offline_source_sha256", {}).items():
        path = WORKSPACE / relative
        if not path.is_file() or _sha256_file(path) != expected:
            raise Plan052Stop(f"ROOT M1 release-bound M0 source changed: {relative}")
    return release


def _verify_old_binding() -> tuple[dict[str, Any], dict[str, str]]:
    binding = _read_json(OLD_BINDING_PATH)
    raw_hash = str(binding.get("binding_sha256", ""))
    payload = dict(binding)
    payload.pop("binding_sha256", None)
    if raw_hash != _sha256_bytes(_canonical(payload)):
        raise Plan052Stop("PLAN-051 binding self-hash is invalid")
    source_hashes = binding.get("source_sha256", {})
    if len(source_hashes) != 38:
        raise Plan052Stop(f"PLAN-051 M0 source binding count {len(source_hashes)} != 38")
    stale = []
    for relative, expected in sorted(source_hashes.items()):
        path = WORKSPACE / relative
        if not path.is_file() or _sha256_file(path) != expected:
            stale.append(relative)
    if stale:
        raise Plan052Stop(f"PLAN-051 bound sources changed or vanished: {stale[:5]}")
    return binding, dict(source_hashes)


def _verify_old_bank(old_binding: Mapping[str, Any]) -> dict[str, Any]:
    raw = OLD_RAW_PATH
    acceptance = _read_json(OLD_ACCEPTANCE_PATH)
    audit = _read_json(OLD_AUDIT_PATH)
    bank_receipt = _read_json(OLD_BANK_RECEIPT_PATH)
    _verify_self_hash(bank_receipt, "receipt_sha256", "PLAN-051 bank receipt")
    if acceptance.get("status") != "ROOT_ACCEPTED_COMPLETE_COMPARISON" or acceptance.get("next_science") != "STOP":
        raise Plan052Stop("PLAN-051 final acceptance is missing or not the reviewed STOP")
    if acceptance.get("unique_overall_winner_supported") is not False:
        raise Plan052Stop("PLAN-051 winner boundary changed")
    if audit.get("status") != "PASS" or audit.get("scope") != "bank" or audit.get("trajectory_count") != 10080:
        raise Plan052Stop("PLAN-051 r04 raw physics/public audit is not accepted")
    if bank_receipt.get("status") != "PASS" or bank_receipt.get("trajectory_count") != 10080:
        raise Plan052Stop("PLAN-051 bank receipt is not a complete PASS")
    raw_sha = _sha256_file(raw)
    audit_raw_hash = audit.get("input_sha256", {}).get(_relative(raw).replace("/", "\\"))
    if audit_raw_hash is None:
        audit_raw_hash = audit.get("input_sha256", {}).get(_relative(raw))
    if raw_sha != audit_raw_hash:
        raise Plan052Stop("PLAN-051 raw CSV hash differs from r04 audit")
    with raw.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    keys = [row["trajectory_key"] for row in rows]
    if len(rows) != 10080 or len(set(keys)) != 10080 or any(row["complete"].lower() != "true" for row in rows):
        raise Plan052Stop("PLAN-051 raw bank is incomplete or has duplicate keys")
    expected_algorithms = {"EM", "GP", "ERI", "EG"}
    expected_policies = set(NEW_POLICIES)
    if {row["algorithm"] for row in rows} != expected_algorithms or {row["method"] for row in rows} != expected_policies:
        raise Plan052Stop("PLAN-051 raw bank solver/policy registry differs from M1 reuse contract")
    counts = old_binding.get("counts", {})
    if counts.get("bank_trajectories") != 10080 or counts.get("conditions_per_layout") != 21:
        raise Plan052Stop("PLAN-051 M0 counts do not bind the registered 10,080-row bank")
    return {
        "raw_path": _relative(raw), "raw_sha256": raw_sha, "trajectory_count": len(rows),
        "unique_trajectory_keys": len(set(keys)),
        "source_binding_sha256": old_binding["binding_sha256"],
        "acceptance_sha256": _sha256_file(OLD_ACCEPTANCE_PATH),
        "audit_sha256": _sha256_file(OLD_AUDIT_PATH),
        "bank_receipt_sha256": _sha256_file(OLD_BANK_RECEIPT_PATH),
    }


def _new_source_paths() -> tuple[str, ...]:
    return (
        "R/docs/planning/PLAN_052_NEW_SOLVERS_ONLINE_FIVE_R.md",
        "R/configs/r_seven_solvers_five_r_v1.json",
        "R/experiments/r_seven_solvers_five_r_runner.py",
        "R/experiments/r_five_versions_comparison.py",
        "R/experiments/r_order_protocol.py",
        "R/scrp/plan052_public.py",
        "R/scrp/plan052_policies.py",
        "R/scrp/r_five_versions_actor.py",
        "R/scrp/cross_solver_reliability_r.py",
        "R/scrp/r_order_public.py",
        "R/scrp/r_order_actor.py",
        "R/scrp/universal_r_utility.py",
        "R/scrp/history_reliability_r.py",
        "R/scrp/r_horizon_robustness.py",
        "R/offline_brp_common.py",
        "RIL/ril_offline.py", "RIL/ril_public.py",
        "Chain/chain_offline.py", "Chain/chain_public.py",
        "Minmax/minmax_offline.py", "Minmax/minmax_public.py",
        "Minmax/vendor/heuristic.cpp", "Minmax/COPYING", "Minmax/SOURCE_PROVENANCE.md",
        "R/tests/test_plan052_offline.py", "R/tests/test_plan052_online.py",
        "EM/scrp/noise_aware_forecast.py",
        "EM/experiments/noise_aware_reference_p0.py",
        "参考论文/RIL.pdf", "参考论文/chain.pdf", "参考论文/minmax.pdf",
    )


def _verify_matrix_contract(config: Mapping[str, Any], old_runner: Any) -> dict[str, Any]:
    conditions = old_runner.condition_specs()
    if len(conditions) != 21 or len({row["condition_id"] for row in conditions}) != 21:
        raise Plan052Stop("new protocol did not reuse PLAN-051's 21 unique conditions")
    layouts = _read_json(LAYOUT_PATH)
    if len(layouts) != 20:
        raise Plan052Stop("PLAN-044 selected layout source must contain exactly 20 rows")
    smoke_condition = next((row for row in conditions if row["family"] == "delayed_positions" and row["p"] == 0.3 and row["repeat"] == 0), None)
    if smoke_condition is None:
        raise Plan052Stop("registered smoke condition is not in PLAN-051's condition namespace")
    g1 = [row for row in layouts if row.get("parameter_group") == "G1"]
    if not g1:
        raise Plan052Stop("registered G1:first smoke layout is missing")
    new_count = len(layouts) * len(conditions) * len(NEW_SOLVERS) * len(NEW_POLICIES)
    merged_count = new_count + 10080
    smoke_count = len(NEW_SOLVERS) * len(SMOKE_POLICIES)
    if (new_count, merged_count, smoke_count) != (7560, 17640, 9):
        raise Plan052Stop("PLAN-052 registered matrix counts are inconsistent")
    counts = config.get("counts", {})
    if counts.get("new_trajectories") != new_count or counts.get("merged_trajectories") != merged_count or counts.get("smoke_trajectories") != smoke_count:
        raise Plan052Stop("PLAN-052 config matrix counts differ from derived registry")
    return {
        "layout_count": len(layouts), "condition_count": len(conditions),
        "new_solver_count": len(NEW_SOLVERS), "policy_count": len(NEW_POLICIES),
        "new_trajectory_count": new_count, "old_reused_trajectory_count": 10080,
        "merged_trajectory_count": merged_count, "smoke_trajectory_count": smoke_count,
        "smoke_layout_fingerprint": g1[0]["label_invariant_fingerprint"],
        "smoke_condition_id": smoke_condition["condition_id"],
    }


def build_m1_bindings() -> dict[str, Any]:
    _verify_m0_release()
    old_binding, old_sources = _verify_old_binding()
    old_bank = _verify_old_bank(old_binding)
    old_runner = _old_runner()
    config = _read_json(CONFIG_PATH)
    matrix = _verify_matrix_contract(config, old_runner)
    release_sha = _sha256_file(ROOT_M0_RELEASE_PATH)
    if release_sha != ROOT_M0_RELEASE_SHA256:
        raise Plan052Stop("ROOT preparation release hash changed")
    offline_root = OUTPUT_ROOT / "preparation/offline"
    offline_receipt = _read_json(offline_root / "M0_OFFLINE_RECEIPT.json")
    offline_review_path = OUTPUT_ROOT / "ROOT_M0_OFFLINE_REVIEW.md"
    offline_review = offline_review_path.read_text(encoding="utf-8") if offline_review_path.is_file() else ""
    if not str(offline_receipt.get("status", "")).startswith("M0_COMPLETE") or offline_receipt.get("science_execution", {}).get("science_seconds") != 0:
        raise Plan052Stop("M0 offline correctness receipt is not a non-science PASS")
    if "Decision: PASS / RELEASE_ONLINE_PREPARATION_ONLY" not in offline_review:
        raise Plan052Stop("ROOT M0 offline review PASS record is missing")
    if _sha256_file(RIDGE_PATH) != RIDGE_SHA256 or _sha256_file(ORDER_MODELS_PATH) != ORDER_SHA256:
        raise Plan052Stop("frozen shared model assets differ from PLAN-051 accepted hashes")
    source_hashes = {}
    for relative in _new_source_paths():
        path = WORKSPACE / relative
        if not path.is_file():
            raise Plan052Stop(f"M1 source asset is missing: {relative}")
        source_hashes[relative] = _sha256_file(path)
    binding = {
        "protocol_id": "r_seven_solvers_five_r_v1", "run_id": "r01",
        "stage": "M1_PREPARATION", "scope": "public_online_adapter_and_runner_contracts",
        "plan": _relative(PLAN_PATH), "config": _relative(CONFIG_PATH),
        "config_sha256": _sha256_file(CONFIG_PATH),
        "new_source_sha256": source_hashes,
        "old_plan051_source_sha256": old_sources,
        "old_plan051_source_binding_sha256": old_binding["binding_sha256"],
        "old_plan051_source_binding_file_sha256": _sha256_file(OLD_BINDING_PATH),
        "root_m0_release_sha256": release_sha,
        "offline_m0_receipt_sha256": _sha256_file(offline_root / "M0_OFFLINE_RECEIPT.json"),
        "root_m0_offline_review_sha256": _sha256_file(offline_review_path),
        "models": {
            "R_UTILITY_RIDGE": {"path": _relative(RIDGE_PATH), "sha256": RIDGE_SHA256, "feature_width": 74, "fit_in_this_plan": False},
            "R_ORDER_MAIN_TAU0_FULL": {"path": _relative(ORDER_MODELS_PATH), "sha256": ORDER_SHA256, "shape": [20, 5], "fit_in_this_plan": False},
        },
        "layout_source": {"path": _relative(LAYOUT_PATH), "sha256": _sha256_file(LAYOUT_PATH), "count": matrix["layout_count"]},
        "old_bank_reuse": old_bank,
        "matrix": matrix,
        "caps_seconds": {"science_cumulative": SCIENCE_CAP_SECONDS, "per_trajectory": TRAJECTORY_CAP_SECONDS, "analysis": ANALYSIS_CAP_SECONDS},
        "execution_status": {"smoke_trajectories": 0, "bank_trajectories": 0, "model_fit": False, "science_seconds": 0},
    }
    binding["binding_sha256"] = _sha256_bytes(_canonical(binding))
    return binding


def prepare_m1() -> dict[str, Any]:
    _verify_m0_release()
    test_command = [sys.executable, "-B", "-m", "pytest", "-q", "R/tests/test_plan052_online.py"]
    test_run = subprocess.run(
        test_command, cwd=WORKSPACE, capture_output=True, text=True, check=False
    )
    test_output = (test_run.stdout or "") + (test_run.stderr or "")
    match = re.search(r"(\d+) passed(?:,| in )", test_output)
    if test_run.returncode != 0 or match is None:
        raise Plan052Stop(f"synthetic M1 contract tests failed: {test_output[-2000:]}")
    binding = build_m1_bindings()
    PREPARATION_ROOT.mkdir(parents=True, exist_ok=False)
    binding_path = PREPARATION_ROOT / "M1_SOURCE_BINDINGS.json"
    _write_new_json(binding_path, binding)
    output_path = PREPARATION_ROOT / "M1_SYNTHETIC_TEST_STDOUT.txt"
    output_path.write_text(test_output, encoding="utf-8")
    test_receipt = {
        "protocol_id": "r_seven_solvers_five_r_v1", "run_id": "r01",
        "stage": "M1_SYNTHETIC_TESTS", "status": "PASS",
        "binding_sha256": binding["binding_sha256"],
        "command": test_command, "exit_code": test_run.returncode,
        "passed": int(match.group(1)), "failed": 0,
        "stdout_path": _relative(output_path),
        "stdout_sha256": _sha256_file(output_path),
        "real_trajectory_count": 0, "smoke_trajectory_count": 0,
        "bank_trajectory_count": 0, "fit_or_refit": False, "science_seconds": 0,
    }
    test_receipt["receipt_sha256"] = _sha256_bytes(_canonical(test_receipt))
    test_receipt_path = PREPARATION_ROOT / "M1_SYNTHETIC_TEST_RECEIPT.json"
    _write_new_json(test_receipt_path, test_receipt)
    receipt = {
        "protocol_id": "r_seven_solvers_five_r_v1", "run_id": "r01",
        "stage": "M1_PREPARATION", "status": "PASS_WAITING_ROOT_REVIEW",
        "binding_path": _relative(binding_path), "binding_sha256": binding["binding_sha256"],
        "synthetic_test_receipt_path": _relative(test_receipt_path),
        "synthetic_test_receipt_sha256": test_receipt["receipt_sha256"],
        "synthetic_test_passed": test_receipt["passed"],
        "new_trajectory_count": 7560, "merged_trajectory_count": 17640,
        "smoke_trajectory_count": 9,
        "synthetic_test_command": "py -3.12 -B -m pytest -q R/tests/test_plan052_online.py",
        "synthetic_tests_only": True,
        "actual_smoke_trajectories": 0, "actual_bank_trajectories": 0,
        "model_fit": False, "science_wall_seconds": 0,
        "release_required_for_smoke": True, "next_release": "ROOT M1 smoke review release",
    }
    receipt["receipt_sha256"] = _sha256_bytes(_canonical(receipt))
    receipt_path = PREPARATION_ROOT / "M1_PREPARATION_RECEIPT.json"
    _write_new_json(receipt_path, receipt)
    return receipt


def _trace_contains_private(value: Any) -> bool:
    forbidden = {"true_retrieval_sequence", "forecast_seed_evaluator_only", "condition_evaluator_only", "physical_stacks", "private_trace", "forecast_future", "provider"}
    if isinstance(value, Mapping):
        return any(str(key) in forbidden or _trace_contains_private(item) for key, item in value.items())
    if isinstance(value, (tuple, list)):
        return any(_trace_contains_private(item) for item in value)
    return False


def _verify_run_release(scope: str) -> tuple[dict[str, Any], dict[str, Any] | None]:
    if scope not in ("smoke", "bank"):
        raise ValueError("scope must be smoke or bank")
    binding_path = PREPARATION_ROOT / "M1_SOURCE_BINDINGS.json"
    receipt_path = PREPARATION_ROOT / "M1_PREPARATION_RECEIPT.json"
    if not binding_path.is_file() or not receipt_path.is_file():
        raise Plan052Stop("reviewable M1 preparation binding and receipt are required")
    binding, receipt = _read_json(binding_path), _read_json(receipt_path)
    _verify_self_hash(receipt, "receipt_sha256", "M1 preparation receipt")
    current = build_m1_bindings()
    if binding.get("binding_sha256") != current.get("binding_sha256"):
        raise Plan052Stop("M1 source/model/old-bank binding changed since preparation")
    if receipt.get("status") != "PASS_WAITING_ROOT_REVIEW" or receipt.get("binding_sha256") != binding.get("binding_sha256"):
        raise Plan052Stop("M1 preparation receipt is stale or not reviewable")
    test_path = PREPARATION_ROOT / "M1_SYNTHETIC_TEST_RECEIPT.json"
    test_receipt = _read_json(test_path)
    _verify_self_hash(test_receipt, "receipt_sha256", "M1 synthetic test receipt")
    if test_receipt.get("status") != "PASS" or test_receipt.get("binding_sha256") != binding.get("binding_sha256"):
        raise Plan052Stop("M1 synthetic contract tests are not bound PASS")
    release_path = OUTPUT_ROOT / ("ROOT_M1_SMOKE_RELEASE.json" if scope == "smoke" else "ROOT_M1_BANK_RELEASE.json")
    if not release_path.is_file():
        raise Plan052Stop(f"ROOT {scope} release is required before output creation")
    release = _read_json(release_path)
    expected = {
        "protocol_id": "r_seven_solvers_five_r_v1", "run_id": "r01",
        "scope": scope, "authorized": True,
        "m1_binding_sha256": binding["binding_sha256"],
        "m1_preparation_receipt_sha256": receipt["receipt_sha256"],
        "m1_synthetic_test_receipt_sha256": test_receipt["receipt_sha256"],
    }
    for key, value in expected.items():
        if release.get(key) != value:
            raise Plan052Stop(f"ROOT {scope} release binding differs at {key}")
    smoke_receipt = None
    if scope == "bank":
        path = OUTPUT_ROOT / "m1_smoke_01/M1_SMOKE_RECEIPT.json"
        if not path.is_file():
            raise Plan052Stop("bank release requires a completed smoke receipt")
        smoke_receipt = _read_json(path)
        _verify_self_hash(smoke_receipt, "receipt_sha256", "M1 smoke receipt")
        if release.get("smoke_receipt_sha256") != smoke_receipt.get("receipt_sha256"):
            raise Plan052Stop("bank release is not bound to the accepted smoke receipt")
        _verify_smoke_artifacts(path.parent, smoke_receipt)
    return release, smoke_receipt


def _selection(layouts: Sequence[tuple[dict[str, Any], Any]], conditions: Sequence[dict[str, Any]], scope: str) -> list[tuple[dict[str, Any], Any, dict[str, Any], str, str]]:
    if scope == "smoke":
        layout_row, episode = next(item for item in layouts if item[0].get("parameter_group") == "G1")
        condition = next(row for row in conditions if row["family"] == "delayed_positions" and row["p"] == 0.3 and row["repeat"] == 0)
        return [(layout_row, episode, condition, solver, policy) for solver in NEW_SOLVERS for policy in SMOKE_POLICIES]
    return [
        (layout_row, episode, condition, solver, policy)
        for layout_row, episode in layouts
        for condition in conditions
        for solver in NEW_SOLVERS
        for policy in NEW_POLICIES
    ]


def _trajectory_key(row: Mapping[str, Any], condition: Mapping[str, Any], solver: str, policy: str) -> str:
    return "|".join((str(row["label_invariant_fingerprint"]), str(condition["condition_id"]), solver, policy))


def _csv_write(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise Plan052Stop("refusing to write empty trajectory CSV")
    with path.open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), extrasaction="raise")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(_json_safe(value), ensure_ascii=False) if isinstance(value, (dict, list, tuple)) else value for key, value in row.items()})


def _jsonl_gz_append(stream: Any, row: Mapping[str, Any]) -> None:
    stream.write(json.dumps(_json_safe(row), ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")


def _run_one(*, episode: Any, row: Mapping[str, Any], condition: Mapping[str, Any], solver: str, method: str,
             forecast: Sequence[int], seed: int | None, runtime: Mapping[str, Any], models: Mapping[str, Any],
             trajectory_key: str, public_sink: Any, private_sink: Any, comparison: Any) -> dict[str, Any]:
    from R.scrp.plan052_policies import Plan052Policy
    from R.scrp.r_five_versions_actor import public_frame_record, require_public_frame
    from R.scrp.r_order_actor import HistoryBoundController
    from Chain.chain_public import propose as propose_chain
    from Minmax.minmax_public import propose as propose_minmax

    if solver not in ("Chain", "Minmax"):
        raise Plan052Stop("isolated bridge adapter only registers Chain and Minmax")
    proposer = {"Chain": propose_chain, "Minmax": propose_minmax}[solver]

    class _PolicySlot:
        def __init__(self) -> None:
            self.slot = Plan052Policy(solver, proposer, ridge_scorer=models["ridge"], rates=models["rates"])

        def select(self, frame: object) -> tuple[int, dict[str, Any]]:
            return self.slot.select(frame, method)

    policy = _PolicySlot()
    controller = HistoryBoundController.fresh(policy)
    started = time.monotonic()
    environment, provider = comparison._make_context(episode, forecast, seed, runtime)
    fingerprint = str(row["label_invariant_fingerprint"])
    fhash = comparison._public_hash(forecast)
    public_before = runtime["public_frame"](episode, environment, provider)
    decisions = 0
    actor_time = 0.0
    max_steps = int(episode.num_containers) * max(1, int(episode.max_tiers) - 1)
    private_sink({
        "record_type": "trajectory_start", "trajectory_key": trajectory_key,
        "layout_fingerprint": fingerprint,
        "initial_stacks_private": episode.initial_stacks,
        "true_retrieval_sequence_private": episode.retrieval_sequence,
        "fixed_forecast_private": tuple(int(value) for value in forecast),
        "forecast_seed_evaluator_only": seed, "forecast_sha256_evaluator_only": fhash,
        "condition_evaluator_only": dict(condition),
    })
    while not environment.state.terminated:
        if time.monotonic() - started > TRAJECTORY_CAP_SECONDS:
            raise Plan052Stop(f"trajectory exceeded {TRAJECTORY_CAP_SECONDS}s: {trajectory_key}")
        require_public_frame(public_before)
        if not public_before.legal_destinations:
            raise Plan052Stop("nonterminal public frame has no legal relocation")
        physical_before = runtime["private_physical_stacks"](environment)
        actor_started = time.perf_counter()
        action, trace, decorated = controller.select(public_before)
        actor_time += time.perf_counter() - actor_started
        require_public_frame(decorated)
        if int(action) not in public_before.legal_destinations or _trace_contains_private(trace):
            raise Plan052Stop("actor returned an illegal action or private trace field")
        before_retrieval = int(environment.state.retrieval_count)
        before_relocations = int(environment.relocation_count)
        environment.step(int(action))
        after_retrieval = int(environment.state.retrieval_count)
        runtime["sync_retrieval_events_with_audit"](
            provider, episode.retrieval_sequence, before_retrieval, after_retrieval
        )
        if int(environment.relocation_count) != before_relocations + 1:
            raise Plan052Stop("relocation count failed to increment exactly once")
        next_frame = None if environment.state.terminated else runtime["public_frame"](episode, environment, provider)
        public_sink({
            "trajectory_key": trajectory_key, "decision_index": decisions,
            "solver": solver, "method": method,
            "public_frame_before": public_frame_record(decorated),
            "own_action": int(action), "actor_trace": trace,
            "public_frame_after": None if next_frame is None else public_frame_record(next_frame),
            "terminal_after_action": bool(environment.state.terminated),
        })
        private_sink({
            "record_type": "physical_step", "trajectory_key": trajectory_key,
            "decision_index": decisions,
            "physical_stacks_before_private": physical_before,
            "physical_stacks_after_private": runtime["private_physical_stacks"](environment),
            "retrieval_count_before": before_retrieval, "retrieval_count_after": after_retrieval,
            "relocation_count_before": before_relocations,
            "relocation_count_after": int(environment.relocation_count),
            "provider_event_count_private": int(provider.event_count),
            "provider_history_hits_private": provider.history.hits,
        })
        decisions += 1
        if decisions > max_steps:
            raise Plan052Stop("trajectory exceeded registered relocation bound")
        if next_frame is not None:
            public_before = next_frame
    if int(environment.state.retrieval_count) != int(episode.num_containers):
        raise Plan052Stop("trajectory terminated before all retrievals")
    elapsed = time.monotonic() - started
    if elapsed > TRAJECTORY_CAP_SECONDS:
        raise Plan052Stop("trajectory exceeded 30-second cap")
    return {
        "trajectory_key": trajectory_key, "layout_id": str(row.get("episode_id", fingerprint)),
        "layout_fingerprint": fingerprint, "geometry": str(episode.parameter_group),
        "condition_id": str(condition["condition_id"]), "family": str(condition["family"]),
        "strength": str(condition["strength"]), "p": condition.get("p"),
        "window_size": condition.get("window_size"), "repeat": int(condition["repeat"]),
        "algorithm": solver, "method": method,
        "R_raw": int(environment.relocation_count), "N": int(episode.num_containers),
        "R_over_N": float(environment.relocation_count) / float(episode.num_containers),
        "forecast_seed_evaluator_only": seed, "F_hash": fhash, "complete": True,
        "decision_count": decisions, "wall_seconds": elapsed, "actor_time_seconds": actor_time,
        "eg_cached_moves": 0,
    }


def _write_failure_receipt(
    *, scope: str, output_dir: Path, raw_rows: Sequence[Mapping[str, Any]],
    release_sha256: str, stage_started: float, previous_wall: float,
    exc: BaseException, traceback_text: str,
) -> dict[str, Any]:
    partial_path: Path | None = None
    partial_sha256: str | None = None
    partial_write_error: str | None = None
    if raw_rows:
        candidate = output_dir / "partial_raw_trajectories.csv"
        try:
            _csv_write(candidate, raw_rows)
            partial_path = candidate
            partial_sha256 = _sha256_file(candidate)
        except BaseException as save_exc:
            partial_write_error = f"{type(save_exc).__name__}: {save_exc}"
    preserved_hashes: dict[str, str] = {}
    for relative in SMOKE_STAGE_FILES[1:]:
        path = output_dir / relative
        if path.is_file():
            preserved_hashes[relative] = _sha256_file(path)
    elapsed = time.monotonic() - stage_started
    failure = {
        "protocol_id": "r_seven_solvers_five_r_v1", "run_id": "r01",
        "scope": scope, "status": "STOP", "error_type": type(exc).__name__,
        "error": str(exc), "traceback": traceback_text,
        "completed_rows": len(raw_rows), "stage_wall_seconds": elapsed,
        "science_wall_seconds": previous_wall + elapsed,
        "release_sha256": release_sha256,
        "partial_raw_trajectory_csv": None if partial_path is None else _relative(partial_path),
        "partial_raw_trajectory_csv_sha256": partial_sha256,
        "partial_csv_write_error": partial_write_error,
        "preserved_trace_sidecar_sha256": preserved_hashes,
        "partial_scoring_performed": False, "use_for_ranking": False,
    }
    failure["receipt_sha256"] = _sha256_bytes(_canonical(failure))
    _write_new_json(output_dir / f"{scope.upper()}_FAILURE_RECEIPT.json", failure)
    return failure


def run_stage(scope: str) -> dict[str, Any]:
    release, smoke_receipt = _verify_run_release(scope)
    previous_wall = 0.0 if smoke_receipt is None else float(smoke_receipt["science_wall_seconds"])
    release_path = OUTPUT_ROOT / ("ROOT_M1_SMOKE_RELEASE.json" if scope == "smoke" else "ROOT_M1_BANK_RELEASE.json")
    release_sha256 = _sha256_file(release_path)
    stage_started = time.monotonic()
    output_dir = OUTPUT_ROOT / ("m1_smoke_01" if scope == "smoke" else "m2_bank_01")
    output_dir.mkdir(parents=True, exist_ok=False)
    raw_rows: list[dict[str, Any]] = []
    smoke_by_key: dict[str, dict[str, str]] = {}
    streams: list[Any] = []
    try:
        comparison = _old_runner()
        protocol, runtime, _gp_rule, models = comparison._load_runtime()
        smoke_artifacts: dict[str, bytes] = {}
        if scope == "bank":
            smoke_dir = OUTPUT_ROOT / "m1_smoke_01"
            smoke_artifacts, smoke_rows = _verify_smoke_artifacts(smoke_dir, smoke_receipt)
            smoke_by_key = {row["trajectory_key"]: row for row in smoke_rows}
            raw_rows.extend(smoke_rows)
            (output_dir / "private").mkdir()
            for relative, payload in smoke_artifacts.items():
                if relative == "raw_trajectories.csv":
                    continue
                destination = output_dir / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(payload)
        else:
            (output_dir / "private").mkdir()
        mode = "at" if scope == "bank" else "wt"
        public_stream = gzip.open(output_dir / "public_decision_trace.jsonl.gz", mode, encoding="utf-8")
        streams.append(public_stream)
        evaluator_stream = gzip.open(output_dir / "private/evaluator_sidecar.jsonl.gz", mode, encoding="utf-8")
        streams.append(evaluator_stream)
        physical_stream = gzip.open(output_dir / "private/physical_steps.jsonl.gz", mode, encoding="utf-8")
        streams.append(physical_stream)
        layouts = _read_json(LAYOUT_PATH)
        episodes = [(row, runtime["OnlineCRPEpisode"].from_record(row)) for row in layouts]
        conditions = comparison.condition_specs()
        selected = _selection(episodes, conditions, scope)
        expected = 9 if scope == "smoke" else 7560
        if len(selected) != expected:
            raise Plan052Stop(f"{scope} selection count {len(selected)} != {expected}")
        forecast_cache: dict[str, tuple[tuple[int, ...], int | None]] = {}
        for layout_row, episode, condition, solver, method in selected:
            key = _trajectory_key(layout_row, condition, solver, method)
            forecast_key = f"{layout_row['label_invariant_fingerprint']}|{condition['condition_id']}"
            if forecast_key not in forecast_cache:
                forecast_cache[forecast_key] = comparison._forecast_for(
                    episode, str(layout_row["label_invariant_fingerprint"]), condition, runtime
                )
            forecast, seed = forecast_cache[forecast_key]
            if scope == "bank" and key in smoke_by_key:
                reused = smoke_by_key[key]
                expected_seed = "" if seed is None else str(seed)
                if (reused.get("F_hash") != comparison._public_hash(forecast)
                        or reused.get("forecast_seed_evaluator_only", "") != expected_seed
                        or reused.get("layout_fingerprint") != str(layout_row["label_invariant_fingerprint"])
                        or reused.get("condition_id") != str(condition["condition_id"])
                        or reused.get("algorithm") != solver or reused.get("method") != method):
                    raise Plan052Stop(f"smoke row does not exactly match bank forecast/key binding: {key}")
                continue
            if previous_wall + time.monotonic() - stage_started > SCIENCE_CAP_SECONDS:
                raise Plan052Stop("cumulative 7,200-second science cap reached")
            _jsonl_gz_append(evaluator_stream, {
                "record_type": "forecast_binding", "trajectory_key": key,
                "layout_fingerprint": layout_row["label_invariant_fingerprint"],
                "condition_evaluator_only": condition, "forecast_seed_evaluator_only": seed,
                "forecast_sha256_evaluator_only": comparison._public_hash(forecast),
            })
            row = _run_one(
                episode=episode, row=layout_row, condition=condition, solver=solver, method=method,
                forecast=forecast, seed=seed, runtime=runtime, models=models,
                trajectory_key=key,
                public_sink=lambda item: _jsonl_gz_append(public_stream, item),
                private_sink=lambda item: _jsonl_gz_append(evaluator_stream if item.get("record_type") == "trajectory_start" else physical_stream, item),
                comparison=comparison,
            )
            if row["F_hash"] != comparison._public_hash(forecast):
                raise Plan052Stop("trajectory forecast differs from the fixed old namespace forecast")
            raw_rows.append(row)
        for stream in streams:
            stream.close()
        streams.clear()
        keys = [row["trajectory_key"] for row in raw_rows]
        if len(keys) != expected or len(set(keys)) != expected:
            raise Plan052Stop("completed trajectory bank is missing or duplicates registered keys")
        raw_path = output_dir / "raw_trajectories.csv"
        _csv_write(raw_path, raw_rows)
        elapsed = time.monotonic() - stage_started
        total_wall = previous_wall + elapsed
        if total_wall > SCIENCE_CAP_SECONDS:
            raise Plan052Stop("cumulative 7,200-second science cap exceeded")
        receipt = {
            "protocol_id": "r_seven_solvers_five_r_v1", "run_id": "r01",
            "scope": scope, "status": "PASS", "release_sha256": release_sha256,
            "m1_binding_sha256": release["m1_binding_sha256"],
            "trajectory_count": len(raw_rows), "unique_trajectory_keys": len(set(keys)),
            "stage_wall_seconds": elapsed, "science_wall_seconds": total_wall,
            "raw_trajectory_csv": _relative(raw_path),
            "public_trace": _relative(output_dir / "public_decision_trace.jsonl.gz"),
            "private_sidecars": [_relative(output_dir / "private/evaluator_sidecar.jsonl.gz"), _relative(output_dir / "private/physical_steps.jsonl.gz")],
            "files_sha256": _stage_file_hashes(output_dir),
        }
        receipt["receipt_sha256"] = _sha256_bytes(_canonical(receipt))
        path = output_dir / ("M1_SMOKE_RECEIPT.json" if scope == "smoke" else "M2_BANK_RECEIPT.json")
        _write_new_json(path, receipt)
        return receipt
    except BaseException as exc:
        for stream in streams:
            try:
                stream.close()
            except BaseException:
                pass
        _write_failure_receipt(
            scope=scope, output_dir=output_dir, raw_rows=raw_rows,
            release_sha256=release_sha256, stage_started=stage_started,
            previous_wall=previous_wall, exc=exc,
            traceback_text=traceback.format_exc(),
        )
        raise


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="PLAN-052 M1 preparation and ROOT-release-gated runner")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("prepare-m1")
    for scope in ("smoke", "bank"):
        sub.add_parser(f"run-{scope}")
    args = parser.parse_args(argv)
    try:
        if args.command == "prepare-m1":
            print(json.dumps(prepare_m1(), indent=2, ensure_ascii=False))
        else:
            print(json.dumps(run_stage(args.command.removeprefix("run-")), indent=2, ensure_ascii=False))
        return 0
    except BaseException as exc:
        print(f"STOP: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
