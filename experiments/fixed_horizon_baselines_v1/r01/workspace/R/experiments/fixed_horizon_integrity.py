"""Hash-bound M0 and ROOT stage release checks for the fixed-H run."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping

WORKSPACE = Path(__file__).resolve().parents[2]
OUT = WORKSPACE.parent
PREP = OUT / "preparation"
PACKAGE = OUT.parents[2]
if str(WORKSPACE) not in sys.path:
    sys.path.insert(0, str(WORKSPACE))
PLAN = PACKAGE / "docs/planning/PLAN_FIXED_HORIZON_BASELINES.md"
AUTH = OUT / "ROOT_USER_AUTHORIZATION.json"
BINDINGS = PREP / "SOURCE_BINDINGS.json"
REUSE = PREP / "REUSE_RECEIPT.json"
SYNTHETIC = PREP / "M0_SYNTHETIC_TEST_RECEIPT.json"
M0_RECEIPT = PREP / "M0_PREPARATION_RECEIPT.json"
ROOT_REVIEW = OUT / "ROOT_M0_REVIEW.json"
RELEASE_FILES = {
    "M1_BRIDGE_SMOKE": OUT / "ROOT_M1_BRIDGE_SMOKE_RELEASE.json",
    "M2_BANK": OUT / "ROOT_M2_BANK_RELEASE.json",
    "M3_ANALYSIS": OUT / "ROOT_M3_ANALYSIS_RELEASE.json",
}
FROZEN_HASH_KEYS = (
    "plan_sha256",
    "authorization_file_sha256",
    "source_bindings_file_sha256",
    "source_binding_sha256",
    "reuse_receipt_file_sha256",
    "synthetic_receipt_file_sha256",
    "m0_preparation_receipt_file_sha256",
    "root_m0_review_file_sha256",
    "historical_reuse_only_binding_sha256",
)


class IntegrityError(RuntimeError):
    """A frozen file, receipt, or review gate is absent or stale."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha(value: object) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise IntegrityError(f"required JSON is absent: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise IntegrityError(f"expected JSON object: {path}")
    return value


def verify_self_hash(record: Mapping[str, Any], field: str, label: str) -> str:
    payload = dict(record)
    recorded = str(payload.pop(field, ""))
    actual = canonical_sha(payload)
    if not recorded or recorded != actual:
        raise IntegrityError(f"{label} canonical self-hash mismatch")
    return recorded


def _check_entry(entry: Mapping[str, Any], *, label: str, root: Path | None = None) -> None:
    path = Path(str(entry.get("path", "")))
    expected = str(entry.get("sha256", ""))
    if not path.is_absolute() and root is not None:
        path = root / path
    if not expected or not path.is_file():
        raise IntegrityError(f"{label} is absent or unbound: {path}")
    actual = sha256_file(path)
    if actual != expected:
        raise IntegrityError(f"{label} SHA256 drift: {path}; expected {expected}, got {actual}")


def verify_frozen_m0(*, require_root_review: bool = False) -> dict[str, Any]:
    """Rehash registered inputs once at stage entry/exit; never per trajectory."""
    binding = read_json(BINDINGS)
    binding_hash = verify_self_hash(binding, "binding_sha256", "SOURCE_BINDINGS")
    if binding.get("protocol_id") != "ridge_fixed_horizon_baselines_v1" or binding.get("run_id") != "r01":
        raise IntegrityError("SOURCE_BINDINGS protocol/run identity mismatch")

    plan = binding.get("plan", {})
    plan_path = Path(str(plan.get("path", "")))
    if not plan_path.is_absolute():
        plan_path = PACKAGE / plan_path
    if not plan_path.is_file() or sha256_file(plan_path) != plan.get("sha256"):
        raise IntegrityError("current registered plan differs from the final M0 binding")
    if plan_path.resolve() != PLAN.resolve():
        raise IntegrityError("SOURCE_BINDINGS points to a different plan file")

    authorization = binding.get("authorization", {})
    auth_path = Path(str(authorization.get("path", "")))
    if not auth_path.is_absolute():
        auth_path = OUT / auth_path
    if auth_path.resolve() != AUTH.resolve() or not auth_path.is_file():
        raise IntegrityError("ROOT_USER_AUTHORIZATION path is missing or unexpected")
    if sha256_file(auth_path) != authorization.get("sha256"):
        raise IntegrityError("ROOT_USER_AUTHORIZATION file hash drift")
    auth = read_json(auth_path)
    if auth.get("plan_sha256") != plan.get("sha256") or auth.get("initial_internal_release") != "M0 preparation and synthetic checks only":
        raise IntegrityError("authorization no longer binds the registered plan/M0-only scope")

    for entry in binding.get("original_sources", []):
        _check_entry(entry, label="frozen original research source")
    for entry in binding.get("runtime_copies", []):
        _check_entry(entry, label="isolated runtime copy")
    for entry in binding.get("local_sources", []):
        _check_entry(entry, label="new protocol source", root=OUT)
    for entry in binding.get("frozen_inputs", []):
        _check_entry(entry, label="frozen input or receipt")

    manifest = binding.get("package_manifest", {})
    _check_entry(manifest, label="immutable package manifest")
    targets = manifest.get("targets", [])
    if len(targets) != 87:
        raise IntegrityError(f"frozen package manifest target count changed: {len(targets)}")
    for entry in targets:
        _check_entry(entry, label="immutable package target")

    reuse_binding = read_json(PREP / "SOURCE_BINDINGS_REUSE_ONLY.json")
    reuse_binding_hash = verify_self_hash(reuse_binding, "binding_sha256", "historical reuse-only SOURCE_BINDINGS")
    if reuse_binding_hash != binding.get("historical_reuse_only_binding_sha256"):
        raise IntegrityError("historical reuse-only binding was replaced")
    reuse_only_receipt = read_json(PREP / "REUSE_RECEIPT_REUSE_ONLY.json")
    reuse_only_hash = verify_self_hash(reuse_only_receipt, "receipt_sha256", "historical reuse-only receipt")
    if reuse_only_hash != binding.get("historical_reuse_only_receipt_sha256"):
        raise IntegrityError("historical reuse-only receipt was replaced")

    reuse = read_json(REUSE)
    verify_self_hash(reuse, "receipt_sha256", "REUSE_RECEIPT")
    if reuse.get("status") != "PASS_M0_REUSE_ELIGIBLE_AWAITING_ROOT_REVIEW":
        raise IntegrityError("reuse receipt status is not the registered M0 pass state")
    if reuse.get("source_binding_sha256") != binding_hash or reuse.get("reused_rows") != 5040 or reuse.get("unique_reuse_slots") != 5040:
        raise IntegrityError("reuse receipt does not bind the final 5,040-row reuse audit")
    reuse_map = PREP / "REUSE_MAP.csv"
    if sha256_file(reuse_map) != reuse.get("reuse_map_sha256"):
        raise IntegrityError("REUSE_MAP.csv differs from its receipt")

    synthetic = read_json(SYNTHETIC)
    verify_self_hash(synthetic, "receipt_sha256", "M0 synthetic test receipt")
    if synthetic.get("status") != "PASS" or synthetic.get("source_binding_sha256") != binding_hash:
        raise IntegrityError("synthetic contract receipt is absent, failed, or stale")
    if synthetic.get("real_trajectory_count") != 0 or synthetic.get("fit_or_tuning") is not False:
        raise IntegrityError("M0 synthetic receipt records prohibited science or fitting")

    m0 = read_json(M0_RECEIPT)
    verify_self_hash(m0, "receipt_sha256", "M0 preparation receipt")
    if (m0.get("status") != "PASS_M0_READY_FOR_ROOT_REVIEW"
            or m0.get("source_binding_sha256") != binding_hash
            or m0.get("source_bindings_file_sha256") != sha256_file(BINDINGS)
            or m0.get("reuse_receipt_file_sha256") != sha256_file(REUSE)
            or m0.get("synthetic_receipt_file_sha256") != sha256_file(SYNTHETIC)):
        raise IntegrityError("M0 preparation receipt does not bind the current artifacts")

    review: dict[str, Any] | None = None
    if require_root_review:
        review = read_json(ROOT_REVIEW)
        verify_self_hash(review, "review_sha256", "ROOT M0 review")
        if (review.get("status") != "PASS_M0_READY_FOR_M1_RELEASE_REVIEW"
                or review.get("plan_sha256") != plan.get("sha256")
                or review.get("source_binding_sha256") != binding_hash
                or review.get("m0_preparation_receipt_file_sha256") != sha256_file(M0_RECEIPT)):
            raise IntegrityError("ROOT M0 review is not a PASS on the current frozen M0 artifacts")

    return {
        "plan_sha256": str(plan["sha256"]),
        "authorization_file_sha256": sha256_file(AUTH),
        "source_bindings_file_sha256": sha256_file(BINDINGS),
        "source_binding_sha256": binding_hash,
        "reuse_receipt_file_sha256": sha256_file(REUSE),
        "synthetic_receipt_file_sha256": sha256_file(SYNTHETIC),
        "m0_preparation_receipt_file_sha256": sha256_file(M0_RECEIPT),
        "root_m0_review_file_sha256": sha256_file(ROOT_REVIEW) if review is not None else None,
        "historical_reuse_only_binding_sha256": reuse_binding_hash,
    }


def verify_release_record(stage: str, release: Mapping[str, Any], expected: Mapping[str, str]) -> dict[str, Any]:
    if stage not in RELEASE_FILES:
        raise IntegrityError(f"unregistered stage: {stage}")
    if (release.get("protocol_id") != "ridge_fixed_horizon_baselines_v1"
            or release.get("run_id") != "r01" or release.get("stage") != stage
            or release.get("status") != "ROOT_RELEASED" or release.get("issued_by") != "ROOT"
            or not str(release.get("release_id", "")).strip()):
        raise IntegrityError("ROOT release protocol/run/stage/issuer/status identity mismatch")
    for key, value in expected.items():
        if release.get(key) != value:
            raise IntegrityError(f"ROOT release is missing or stale for {key}")
    return {"stage": stage, "status": "ROOT_RELEASED", "bound_fields": sorted(expected)}


def _verify_receipt_file(path: Path, *, expected_sha256: str, expected_status: str, label: str) -> dict[str, Any]:
    actual_file_hash = sha256_file(path) if path.is_file() else "MISSING"
    if actual_file_hash != expected_sha256:
        raise IntegrityError(f"{label} file hash differs from the ROOT release")
    receipt = read_json(path)
    receipt_self_hash = verify_self_hash(receipt, "receipt_sha256", label)
    if receipt.get("status") != expected_status:
        raise IntegrityError(f"{label} status is not {expected_status}")
    return {"path": str(path), "file_sha256": actual_file_hash, "receipt_sha256": receipt_self_hash, "receipt": receipt}


def require_stage_release(stage: str) -> dict[str, Any]:
    """Check release plus all real prerequisite files before stage work begins."""
    frozen = verify_frozen_m0(require_root_review=True)
    release_path = RELEASE_FILES.get(stage)
    if release_path is None or not release_path.is_file():
        raise IntegrityError(f"required ROOT stage release is absent: {release_path}")
    release = read_json(release_path)
    verified = {**frozen, **verify_release_record(stage, release, frozen)}
    stage_root = OUT

    if stage == "M1_BRIDGE_SMOKE":
        scope = release.get("scope")
        if scope != "registered_bridge_24_plus_smoke_96":
            raise IntegrityError("M1 release scope differs from the registered 24+96 matrix")
        if release.get("authorized_trajectories") != 120 or release.get("science_budget_cap_seconds") != 7200:
            raise IntegrityError("M1 release count or science cap differs from registration")
        from R.experiments.fixed_horizon_schedule import m1_key_digest
        if release.get("authorized_key_whitelist_sha256") != m1_key_digest():
            raise IntegrityError("M1 release does not bind the exact bridge/smoke whitelist")
    elif stage == "M2_BANK":
        if release.get("scope") != "registered_fixed_h_bank_remaining_after_exact_smoke_reuse":
            raise IntegrityError("M2 release scope differs from the registered bank")
        if release.get("authorized_trajectories") != 9984:
            raise IntegrityError("M2 release must authorize exactly the 9,984 remaining bank keys")
        remaining = release.get("remaining_science_budget_seconds")
        if isinstance(remaining, bool) or not isinstance(remaining, (int, float)) or not 0 < float(remaining) <= 7200:
            raise IntegrityError("M2 release remaining cumulative science budget is outside (0,7200]")
        from R.experiments.fixed_horizon_schedule import remaining_bank_keys, whitelist_digest
        if release.get("authorized_key_whitelist_sha256") != whitelist_digest(remaining_bank_keys()):
            raise IntegrityError("M2 release does not bind the exact remaining-bank key whitelist")
        m1 = _verify_receipt_file(
            stage_root / "audit/M1_BRIDGE_SMOKE_RECEIPT.json",
            expected_sha256=str(release.get("m1_bridge_smoke_receipt_file_sha256", "")),
            expected_status="PASS_M1_BRIDGE_AND_SMOKE",
            label="M1 bridge/smoke receipt",
        )
        m1_audit = _verify_receipt_file(
            stage_root / "audit/M1_TRACE_AUDIT_RECEIPT.json",
            expected_sha256=str(release.get("m1_audit_receipt_file_sha256", "")),
            expected_status="PASS_M1_TRACE_AUDIT",
            label="M1 trace audit receipt",
        )
        for key, item in (("m1_bridge_smoke_receipt_sha256", m1), ("m1_audit_receipt_sha256", m1_audit)):
            if release.get(key) != item["receipt_sha256"]:
                raise IntegrityError(f"M2 ROOT release canonical receipt hash mismatch: {key}")
        m1_science = m1["receipt"].get("science_wall_seconds")
        if isinstance(m1_science, bool) or not isinstance(m1_science, (int, float)) or not 0 <= float(m1_science) < 7200:
            raise IntegrityError("M1 receipt science wall time is outside the registered cumulative budget")
        if m1["receipt"].get("runner_process_exit_code") != 0:
            raise IntegrityError("M1 bridge/smoke did not record actual process exit code 0")
        if m1_audit["receipt"].get("audit_process_exit_code") != 0:
            raise IntegrityError("M1 trace audit did not record actual process exit code 0")
        m1_audit_elapsed = m1_audit["receipt"].get("audit_cumulative_seconds")
        if (isinstance(m1_audit_elapsed, bool) or not isinstance(m1_audit_elapsed, (int, float))
                or not 0 <= float(m1_audit_elapsed) <= 7200):
            raise IntegrityError("M1 trace audit time is outside the registered cumulative budget")
        audit_remaining = release.get("trace_audit_remaining_budget_seconds")
        if (isinstance(audit_remaining, bool) or not isinstance(audit_remaining, (int, float))
                or not 0 < float(audit_remaining) <= 7200.0 - float(m1_audit_elapsed)):
            raise IntegrityError("M2 release audit cap exceeds the remaining cumulative 7,200-second budget")
        remaining = float(release["remaining_science_budget_seconds"])
        if remaining > 7200.0 - float(m1_science) + 1e-9:
            raise IntegrityError("M2 release cap exceeds the remaining cumulative 7,200-second science budget")
    elif stage == "M3_ANALYSIS":
        if release.get("scope") != "registered_complete_15120_row_development_analysis":
            raise IntegrityError("M3 release scope differs from the registered complete 15,120-row analysis")
        if release.get("analysis_budget_cap_seconds") != 1800:
            raise IntegrityError("M3 release analysis cap must equal the registered 1,800 seconds")
        m2 = _verify_receipt_file(
            stage_root / "bank/M2_BANK_RECEIPT.json",
            expected_sha256=str(release.get("m2_bank_receipt_file_sha256", "")),
            expected_status="PASS_M2_COMPLETE_BANK",
            label="M2 bank receipt",
        )
        m2_audit = _verify_receipt_file(
            stage_root / "audit/M2_TRACE_AUDIT_RECEIPT.json",
            expected_sha256=str(release.get("m2_audit_receipt_file_sha256", "")),
            expected_status="PASS_M2_TRACE_AUDIT",
            label="M2 trace audit receipt",
        )
        if release.get("m2_bank_receipt_sha256") != m2["receipt_sha256"] or release.get("m2_audit_receipt_sha256") != m2_audit["receipt_sha256"]:
            raise IntegrityError("M3 ROOT release canonical upstream receipt hashes do not match")
        if m2["receipt"].get("runner_process_exit_code") != 0:
            raise IntegrityError("M2 bank did not record actual process exit code 0")
        if m2_audit["receipt"].get("audit_process_exit_code") != 0:
            raise IntegrityError("M2 trace audit did not record actual process exit code 0")
        m2_science = m2["receipt"].get("science_wall_seconds")
        m2_audit_elapsed = m2_audit["receipt"].get("audit_cumulative_seconds")
        if isinstance(m2_science, bool) or not isinstance(m2_science, (int, float)) or not 0 <= float(m2_science) <= 7200:
            raise IntegrityError("M2 cumulative science time exceeds 7,200 seconds")
        if (isinstance(m2_audit_elapsed, bool) or not isinstance(m2_audit_elapsed, (int, float))
                or not 0 <= float(m2_audit_elapsed) <= 7200):
            raise IntegrityError("M2 cumulative trace audit time exceeds 7,200 seconds")
        merged = stage_root / "analysis/MERGED_RAW_TRAJECTORIES.csv"
        if not merged.is_file() or sha256_file(merged) != release.get("merged_raw_sha256"):
            raise IntegrityError("M3 ROOT release does not bind the current complete merged raw CSV")
    verified.update({
        "release_path": str(release_path),
        "release_file_sha256": sha256_file(release_path),
        "root_review_file_sha256": frozen["root_m0_review_file_sha256"],
        "release_scope": release.get("scope"),
        "authorized_trajectories": release.get("authorized_trajectories"),
        "science_budget_cap_seconds": release.get("science_budget_cap_seconds"),
        "remaining_science_budget_seconds": release.get("remaining_science_budget_seconds"),
        "analysis_budget_cap_seconds": release.get("analysis_budget_cap_seconds"),
        "trace_audit_remaining_budget_seconds": release.get("trace_audit_remaining_budget_seconds"),
        "authorized_key_whitelist_sha256": release.get("authorized_key_whitelist_sha256"),
    })
    return verified


def verify_stage_unchanged(stage: str, start_snapshot: Mapping[str, str]) -> dict[str, Any]:
    missing = [key for key in (*FROZEN_HASH_KEYS, "release_file_sha256") if not start_snapshot.get(key)]
    if missing:
        raise IntegrityError(f"stage entry snapshot is incomplete: {missing}")
    current = verify_frozen_m0(require_root_review=True)
    changed = [key for key in FROZEN_HASH_KEYS if current.get(key) != start_snapshot.get(key)]
    if changed:
        raise IntegrityError(f"M0/review source drift during stage {stage}: {changed}")
    # Recheck the exact ROOT release and real upstream files at stage exit as well.
    checked = require_stage_release(stage)
    if checked.get("release_file_sha256") != start_snapshot.get("release_file_sha256"):
        raise IntegrityError(f"ROOT {stage} release file changed while the stage was running")
    return {"stage": stage, "m0_hashes_unchanged": True, "release_file_sha256": checked["release_file_sha256"]}

