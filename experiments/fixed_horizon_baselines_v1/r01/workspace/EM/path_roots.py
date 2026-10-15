"""Path mapping for the separated EM, R, and H20 workspace roots."""

from __future__ import annotations

from pathlib import Path, PurePosixPath, PureWindowsPath


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
EM_ROOT = WORKSPACE_ROOT / "EM"
R_ROOT = WORKSPACE_ROOT / "R"
BASE_ROOT = WORKSPACE_ROOT / "Mijn_RL"

_EM_PREFIXES = (
    "output/noise_aware_lookahead_v1/",
    "configs/noise_aware_lookahead_v1.json",
    "scrp/noise_aware_",
    "scrp/tiny_bayes_reference.py",
    "experiments/noise_aware_",
    "tests/test_noise_aware_",
    "tests/test_tiny_bayes_reference.py",
    "docs/planning/PLAN_036_",
    "docs/planning/PLAN_037_",
    "docs/planning/reviews/REVIEW_036_",
    "docs/planning/reviews/REVIEW_037_",
)
_R_PREFIXES = (
    "output/history_reliability_r_v1/",
    "configs/history_reliability_r_",
    "scrp/history_reliability_r.py",
    "experiments/history_reliability_r",
    "tests/test_history_reliability_r",
    "docs/planning/PLAN_038_",
    "docs/planning/PLAN_039_",
    "docs/planning/RESEARCH_GOAL.md",
    "docs/planning/RESEARCH_STATUS.md",
    "docs/planning/PLAN_INDEX.md",
    "docs/planning/deferred/IDEA_",
    "docs/planning/reviews/REVIEW_038_",
)
_BASE_PREFIXES = (
    "checkpoints/",
    "data/online_crp/",
    "scrp/online_crp.py",
    "scrp/vectorized_online_crp.py",
    "scrp/scale_diverse_policy.py",
    "scrp/attention_data.py",
    "scrp/attention_evaluation.py",
    "scrp/h20_a1_policy.py",
)


def resolve_project_path(relative: str | Path, *, owner: str, write: bool = False) -> Path:
    """Resolve a legacy project-relative path within its explicit owning root.

    Old config strings remain byte-identical. Only the registered EM frozen
    output and H20 shared inputs cross roots; writes must remain in the owner.
    """
    owner_roots = {"EM": EM_ROOT, "R": R_ROOT}
    if owner not in owner_roots:
        raise ValueError(f"unknown project root owner: {owner}")

    raw = str(relative)
    windows_path = PureWindowsPath(raw)
    posix_path = PurePosixPath(raw.replace("\\", "/"))
    if windows_path.is_absolute() or windows_path.drive or posix_path.is_absolute():
        raise ValueError(f"absolute project paths are forbidden: {raw}")
    if any(part in ("..", "") for part in posix_path.parts):
        raise ValueError(f"path traversal is forbidden: {raw}")

    normalized = posix_path.as_posix()
    if normalized.startswith(_EM_PREFIXES):
        target_root = EM_ROOT
    elif normalized.startswith(_R_PREFIXES):
        if owner != "R":
            raise ValueError(f"{owner} is not allowed to read R-owned paths: {raw}")
        target_root = R_ROOT
    elif normalized.startswith(_BASE_PREFIXES):
        target_root = BASE_ROOT
    else:
        target_root = owner_roots[owner]

    if write and target_root.resolve() != owner_roots[owner].resolve():
        raise ValueError(f"write path is outside the {owner} project root: {raw}")

    candidate = (target_root / Path(*posix_path.parts)).resolve()
    try:
        candidate.relative_to(target_root.resolve())
    except ValueError as exc:
        raise ValueError(f"resolved path escapes its allowed root: {raw}") from exc
    return candidate
