"""PowerShell-facing, receipt-producing wrapper for released fixed-H stages.

This module is preparation only until a matching ROOT release exists. It captures
the real Python child exit code and elapsed time and charges every attempt to its
registered cumulative science, audit, or analysis budget.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable

WORKSPACE = Path(__file__).resolve().parents[2]
OUT = WORKSPACE.parent
PREP = OUT / "preparation"
LOGS = PREP / "logs"
STAGE_DRIVER = WORKSPACE / "R/experiments/fixed_horizon_stage_driver.py"
AUDIT_DRIVER = WORKSPACE / "R/experiments/fixed_horizon_trace_audit_entry.py"
ANALYSIS_DRIVER = WORKSPACE / "R/experiments/fixed_horizon_analysis.py"
if str(WORKSPACE) not in sys.path:
    sys.path.insert(0, str(WORKSPACE))


def _canonical_sha(value: Any) -> str:
    data = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def _write_json(path: Path, record: dict[str, Any], field: str = "receipt_sha256") -> dict[str, Any]:
    value = dict(record)
    value.pop(field, None)
    value[field] = _canonical_sha(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return value


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _attempt(kind: str) -> int:
    prefix = f"{kind}_attempt_"
    numbers = []
    for path in LOGS.glob(prefix + "*.process_receipt.json"):
        try:
            numbers.append(int(path.name[len(prefix):-len(".process_receipt.json")]))
        except ValueError:
            continue
    return max(numbers, default=0) + 1


def _prior_elapsed(budget_class: str, stage: str) -> float:
    total = 0.0
    for path in LOGS.glob("*_attempt_*.process_receipt.json"):
        try:
            item = _read_json(path)
        except (OSError, ValueError):
            continue
        if item.get("budget_class") == budget_class and item.get("kind") == stage:
            total += float(item.get("elapsed_seconds", 0.0))
    return total


def remaining_budget_seconds(
    release_cap_seconds: float, local_prior_seconds: float,
    cumulative_prior_seconds: float, hard_cap_seconds: float,
) -> float:
    """Apply stage-local ROOT cap and protocol-wide cumulative cap once each."""
    values = (release_cap_seconds, local_prior_seconds, cumulative_prior_seconds, hard_cap_seconds)
    if any(not math.isfinite(float(value)) or float(value) < 0 for value in values):
        raise ValueError("budget caps and elapsed times must be finite and nonnegative")
    return min(float(release_cap_seconds) - float(local_prior_seconds),
               float(hard_cap_seconds) - float(cumulative_prior_seconds))


def _run_one(command: list[str], *, cwd: Path, timeout: float, stdout_path: Path, stderr_path: Path) -> dict[str, Any]:
    started = time.perf_counter()
    timed_out = False
    try:
        completed = subprocess.run(command, cwd=cwd, capture_output=True, text=True,
                                   encoding="utf-8", errors="replace", timeout=max(0.01, timeout),
                                   check=False)
        code = int(completed.returncode)
        stdout, stderr = completed.stdout, completed.stderr
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        code = 124
        stdout = exc.stdout.decode("utf-8", "replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        stderr = exc.stderr.decode("utf-8", "replace") if isinstance(exc.stderr, bytes) else (exc.stderr or "")
        stderr += f"\nTIMEOUT after {timeout:.6f} seconds\n"
    elapsed = time.perf_counter() - started
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    stdout_path.write_text(stdout, encoding="utf-8", errors="replace")
    stderr_path.write_text(stderr, encoding="utf-8", errors="replace")
    return {
        "command": command, "actual_exit_code": code, "timed_out": timed_out,
        "elapsed_seconds": elapsed, "stdout_sha256": _sha(stdout_path),
        "stderr_sha256": _sha(stderr_path), "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
    }


def _finalize_science_receipt(stage: str, *, code: int, elapsed: float, prior: float) -> None:
    if stage == "M1":
        path = OUT / "audit/M1_BRIDGE_SMOKE_RECEIPT.json"
        if not path.is_file():
            return
        combined = _read_json(path)
        component_paths = {
            "bridge": OUT / "bridge/M1_BRIDGE_RECEIPT.json",
            "smoke": OUT / "smoke/M1_SMOKE_RECEIPT.json",
        }
        component_hashes: dict[str, str] = {}
        missing: list[str] = []
        for scope, component_path in component_paths.items():
            if not component_path.is_file():
                missing.append(scope)
                continue
            component = _read_json(component_path)
            _verify_receipt_selfhash(component, f"M1 {scope} runner receipt")
            if code != 0:
                component["status"] = f"FAIL_{stage}_PROCESS"
            component["runner_process_exit_code"] = code
            component["runner_process_seconds"] = elapsed
            component["runner_process_timed_out"] = code == 124
            component["whole_M1_process_scope"] = True
            component["stage_process_receipt_written_by"] = str(Path(__file__).resolve())
            updated_component = _write_json(component_path, component)
            component_hashes[scope] = updated_component["receipt_sha256"]

        if code != 0:
            combined["status"] = f"FAIL_{stage}_PROCESS"
        elif missing:
            combined["status"] = "FAIL_M1_COMPONENT_RECEIPT_MISSING"
        combined["runner_process_exit_code"] = code
        combined["runner_process_seconds"] = elapsed
        combined["runner_process_timed_out"] = code == 124
        combined["whole_M1_process_scope"] = True
        combined["science_wall_seconds"] = prior + elapsed
        combined["stage_process_receipt_written_by"] = str(Path(__file__).resolve())
        for scope, receipt_hash in component_hashes.items():
            combined[f"{scope}_receipt_sha256"] = receipt_hash
        if missing:
            combined["missing_component_receipts"] = missing
        _write_json(path, combined)
        return

    path = OUT / "bank/M2_BANK_RECEIPT.json"
    if not path.is_file():
        return
    receipt = _read_json(path)
    if code != 0:
        receipt["status"] = f"FAIL_{stage}_PROCESS"
    receipt["runner_process_exit_code"] = code
    receipt["runner_process_seconds"] = elapsed
    receipt["runner_process_timed_out"] = code == 124
    receipt["science_wall_seconds"] = prior + elapsed
    receipt["stage_process_receipt_written_by"] = str(Path(__file__).resolve())
    _write_json(path, receipt)


def _verify_receipt_selfhash(record: dict[str, Any], label: str) -> None:
    recorded = str(record.get("receipt_sha256", ""))
    actual = _canonical_sha({key: value for key, value in record.items() if key != "receipt_sha256"})
    if not recorded or recorded != actual:
        raise RuntimeError(f"{label} canonical self-hash mismatch before finalization")


def _finalize_audit_receipt(stage: str, *, code: int, elapsed: float, prior: float) -> None:
    path = OUT / ("audit/M1_TRACE_AUDIT_RECEIPT.json" if stage == "M1_AUDIT" else "audit/M2_TRACE_AUDIT_RECEIPT.json")
    if not path.is_file():
        return
    receipt = _read_json(path)
    if code != 0:
        receipt["status"] = f"FAIL_{stage}_PROCESS"
    receipt["audit_process_exit_code"] = code
    receipt["audit_process_seconds"] = elapsed
    receipt["audit_process_timed_out"] = code == 124
    receipt["audit_cumulative_seconds"] = prior + elapsed
    _write_json(path, receipt)


def run_kind(kind: str) -> tuple[int, dict[str, Any]]:
    LOGS.mkdir(parents=True, exist_ok=True)
    attempt = _attempt(kind)
    stage = "M1" if kind.startswith("M1") else "M2" if kind.startswith("M2") else "M3"
    if kind in ("M1", "M2"):
        from R.experiments.fixed_horizon_integrity import require_stage_release
        from R.experiments.fixed_horizon_runner import sha256_file
        release_stage = "M1_BRIDGE_SMOKE" if kind == "M1" else "M2_BANK"
        release = require_stage_release(release_stage)
        prior_same = _prior_elapsed("science", kind)
        prior_m1 = 0.0
        if kind == "M2":
            m1 = _read_json(OUT / "audit/M1_BRIDGE_SMOKE_RECEIPT.json")
            if m1.get("runner_process_exit_code") != 0:
                raise RuntimeError("M2 requires actual M1 runner process exit code 0")
            prior_m1 = float(m1["science_wall_seconds"])
        previous = prior_same
        total_prior = prior_m1 + prior_same
        cap = float(release.get("science_budget_cap_seconds", 0.0)) if kind == "M1" else float(release.get("remaining_science_budget_seconds", 0.0))
        remaining = remaining_budget_seconds(cap, prior_same, total_prior, 7200.0)
        if remaining <= 0:
            raise RuntimeError("cumulative 7,200-second science budget exhausted before process launch")
        command = [sys.executable, "-B", str(STAGE_DRIVER), "--stage", kind,
                   "--previous-science-seconds", f"{previous:.9f}"]
        budget_class, timeout, prior_for_receipt = "science", remaining, total_prior
    elif kind in ("M1_AUDIT", "M2_AUDIT"):
        from R.experiments.fixed_horizon_integrity import require_stage_release
        release_stage = "M1_BRIDGE_SMOKE" if kind == "M1_AUDIT" else "M2_BANK"
        release = require_stage_release(release_stage)
        prior_same = _prior_elapsed("audit", kind)
        m1_audit_elapsed = 0.0
        if kind == "M2_AUDIT":
            m1a = _read_json(OUT / "audit/M1_TRACE_AUDIT_RECEIPT.json")
            if m1a.get("audit_process_exit_code") != 0:
                raise RuntimeError("M2 audit requires actual M1 audit process exit code 0")
            m1_audit_elapsed = float(m1a.get("audit_cumulative_seconds", 0.0))
        cap = 7200.0
        if kind == "M2_AUDIT":
            cap = float(release["trace_audit_remaining_budget_seconds"])
        cumulative_prior = m1_audit_elapsed + prior_same
        remaining = remaining_budget_seconds(cap, prior_same, cumulative_prior, 7200.0)
        if remaining <= 0:
            raise RuntimeError("cumulative 7,200-second trace audit budget exhausted before process launch")
        scopes = ("bridge", "smoke", "m1") if kind == "M1_AUDIT" else ("bank",)
        command = [sys.executable, "-B", str(AUDIT_DRIVER), "--scope", scopes[0]]
        budget_class, timeout, prior_for_receipt = "audit", remaining, m1_audit_elapsed + prior_same
    elif kind == "M3_ANALYSIS":
        from R.experiments.fixed_horizon_integrity import require_stage_release
        release = require_stage_release("M3_ANALYSIS")
        prior_same = _prior_elapsed("analysis", kind)
        cap = int(release.get("analysis_budget_cap_seconds", 0))
        if cap != 1800:
            raise RuntimeError("M3 release analysis cap must equal the registered 1,800 seconds")
        remaining = remaining_budget_seconds(cap, prior_same, prior_same, 1800.0)
        if remaining <= 0:
            raise RuntimeError("M3 analysis budget exhausted before process launch")
        command = [sys.executable, "-B", str(ANALYSIS_DRIVER)]
        budget_class, timeout, prior_for_receipt = "analysis", remaining, prior_same
    else:
        raise ValueError(f"unregistered stage kind: {kind}")

    stdout_path = LOGS / f"{kind}_attempt_{attempt:02d}.stdout.txt"
    stderr_path = LOGS / f"{kind}_attempt_{attempt:02d}.stderr.txt"
    process_results = []
    started = time.perf_counter()
    if kind == "M1_AUDIT":
        for scope in ("bridge", "smoke", "m1"):
            child = [sys.executable, "-B", str(AUDIT_DRIVER), "--scope", scope]
            child_timeout = timeout - (time.perf_counter() - started)
            if child_timeout <= 0:
                code = 124
                timed_out = True
                break
            result = _run_one(child, cwd=WORKSPACE, timeout=child_timeout,
                              stdout_path=LOGS / f"{kind}_attempt_{attempt:02d}_{scope}.stdout.txt",
                              stderr_path=LOGS / f"{kind}_attempt_{attempt:02d}_{scope}.stderr.txt")
            process_results.append(result)
            if result["actual_exit_code"] != 0:
                break
        if process_results:
            code = process_results[-1]["actual_exit_code"]
        else:
            stdout_path.write_text("", encoding="utf-8")
            stderr_path.write_text("M1 audit budget exhausted before the first audit command.\n", encoding="utf-8")
        elapsed = time.perf_counter() - started
        stdout_path.write_text("\n".join(Path(x["stdout_path"]).read_text(encoding="utf-8") for x in process_results), encoding="utf-8")
        stderr_path.write_text("\n".join(Path(x["stderr_path"]).read_text(encoding="utf-8") for x in process_results), encoding="utf-8")
        timed_out = any(x["timed_out"] for x in process_results)
        aggregate = {"command": [x["command"] for x in process_results], "actual_exit_code": code,
                     "timed_out": timed_out, "elapsed_seconds": elapsed,
                     "stdout_sha256": _sha(stdout_path), "stderr_sha256": _sha(stderr_path),
                     "stdout_path": str(stdout_path), "stderr_path": str(stderr_path),
                     "component_processes": process_results}
    else:
        result = _run_one(command, cwd=WORKSPACE, timeout=timeout,
                          stdout_path=stdout_path, stderr_path=stderr_path)
        process_results.append(result)
        code, elapsed, timed_out = result["actual_exit_code"], result["elapsed_seconds"], result["timed_out"]
        aggregate = result

    if kind in ("M1", "M2"):
        _finalize_science_receipt(kind, code=code, elapsed=elapsed, prior=prior_for_receipt)
    elif kind in ("M1_AUDIT", "M2_AUDIT"):
        _finalize_audit_receipt(kind, code=code, elapsed=elapsed, prior=prior_for_receipt)
    elif kind == "M3_ANALYSIS":
        report_path = OUT / "analysis/M3_ANALYSIS_RECEIPT.json"
        report = {
            "protocol_id": "ridge_fixed_horizon_baselines_v1", "run_id": "r01",
            "scope": "registered_complete_15120_row_development_analysis",
            "status": "PASS_M3_ANALYSIS_COMPLETE_PENDING_ROOT_REVIEW" if code == 0 else "FAIL_M3_ANALYSIS_PROCESS",
            "analysis_process_exit_code": code, "analysis_process_seconds": elapsed,
            "analysis_cumulative_seconds": prior_for_receipt + elapsed,
            "analysis_process_timed_out": timed_out,
            "release_file_sha256": release.get("release_file_sha256"),
            "merged_raw_sha256": _sha(OUT / "analysis/MERGED_RAW_TRAJECTORIES.csv"),
            "analyzer_sha256": _sha(ANALYSIS_DRIVER),
            "process_receipt_stdout_sha256": aggregate["stdout_sha256"],
            "process_receipt_stderr_sha256": aggregate["stderr_sha256"],
            "process_receipt_path": str(LOGS / f"{kind}_attempt_{attempt:02d}.process_receipt.json"),
        }
        _write_json(report_path, report)

    receipt = {
        "protocol_id": "ridge_fixed_horizon_baselines_v1", "run_id": "r01",
        "kind": kind, "stage": stage, "budget_class": budget_class,
        "attempt": attempt, "budget_prior_seconds": prior_for_receipt,
        "budget_cap_seconds": cap, "budget_remaining_before_seconds": timeout,
        "actual_exit_code": code, "timed_out": timed_out, "elapsed_seconds": elapsed,
        "command_result": aggregate, "component_results": process_results,
        "release_file_sha256": release.get("release_file_sha256"),
        "stage_receipt_path": str(
            OUT / ("audit/M1_BRIDGE_SMOKE_RECEIPT.json" if kind == "M1" else "bank/M2_BANK_RECEIPT.json")
        ) if kind in ("M1", "M2") else None,
        "status": "PASS" if code == 0 else "FAIL",
    }
    receipt_path = LOGS / f"{kind}_attempt_{attempt:02d}.process_receipt.json"
    _write_json(receipt_path, receipt)
    result = {"status": receipt["status"], "kind": kind, "actual_exit_code": code,
              "elapsed_seconds": elapsed, "process_receipt_path": str(receipt_path)}
    print(json.dumps(result, sort_keys=True))
    return code, receipt


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=("M1", "M1_AUDIT", "M2", "M2_AUDIT", "M3_ANALYSIS"), required=True)
    args = parser.parse_args(argv)
    try:
        code, _receipt = run_kind(args.kind)
    except BaseException as exc:
        print(json.dumps({"status": "FAIL_CLOSED", "kind": args.kind,
                          "exception": f"{type(exc).__name__}: {exc}"}, sort_keys=True), file=sys.stderr)
        return 2
    return code


if __name__ == "__main__":
    raise SystemExit(main())
