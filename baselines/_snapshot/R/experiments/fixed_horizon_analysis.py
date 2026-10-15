"""Fail-closed analysis for the registered Ridge-R fixed-horizon bank.

The self-test uses only an in-memory synthetic matrix.  Formal analysis is
permitted only after the ROOT M3 release, M2 bank/audit receipts, and merged
raw hash pass the frozen integrity gate.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import io
import json
import math
import os
import sys
import tempfile
import time
import unittest
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


SCRIPT = Path(__file__).resolve()
WORKSPACE = SCRIPT.parents[2]
if str(WORKSPACE) not in sys.path:
    sys.path.insert(0, str(WORKSPACE))
OUT = WORKSPACE.parent
REPOSITORY = WORKSPACE.parents[1]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))
EXPECTED_INPUT = REPOSITORY / "data/reference_trajectories.csv"
LAYOUT_PATH = REPOSITORY / "data/layouts/select20.json"
TEST_PATH = WORKSPACE / "R" / "tests" / "test_fixed_horizon_analysis.py"

SOLVERS = ("EM", "GP", "ERI", "EG", "Chain", "Minmax")
HORIZONS = (0, 2, 5, 10, 20)
FIXED_HORIZONS = (0, 2, 5, 10)
POLICIES = ("RIDGE", "FIXED_H0", "FIXED_H2", "FIXED_H5", "FIXED_H10", "FIXED_H20")
RIDGE_INDEX = POLICIES.index("RIDGE")
POLICY_FOR_H = {h: POLICIES.index(f"FIXED_H{h}") for h in HORIZONS}
FAMILIES = ("subset_shuffle", "time_window_shuffle")
GEOMETRIES = ("G1", "G2")
EXPECTED_LAYOUTS = 20
EXPECTED_CONDITIONS = 15
EXPECTED_ROWS = EXPECTED_LAYOUTS * EXPECTED_CONDITIONS * len(SOLVERS) * len(POLICIES)
BOOTSTRAP_REPLICATES = 100_000
BOOTSTRAP_SEED = 202610155301
BOOTSTRAP_RNG = "NumPy Generator(PCG64)"
BOOTSTRAP_LAYOUT_ORDER = "G1 then G2; within geometry lexicographic layout_fingerprint"
BOOTSTRAP_RNG_CALL_ORDER = "for each consecutive batch: rng.integers for G1's 10 draws, then rng.integers for G2's 10 draws; the same index matrix is reused for every scope, solver, policy, metric, and paired contrast"
BOOTSTRAP_QUANTILE_METHOD = "linear"
FAMILY_ALPHA = 0.05
MAIN_COMPARISONS = 35
SIMULTANEOUS_TAIL = FAMILY_ALPHA / (2 * MAIN_COMPARISONS)
ANALYSIS_BUDGET_SECONDS = 1800
BOOTSTRAP_BATCH_SIZE = 1000

RAW_COLUMNS = (
    "trajectory_key", "layout_fingerprint", "geometry", "condition_id", "family",
    "strength", "p", "window_size", "repeat", "algorithm", "method", "R_raw",
    "N", "R_over_N", "forecast_seed_evaluator_only", "F_hash", "complete",
    "decision_count", "wall_seconds", "actor_time_seconds", "eg_cached_moves",
    "source_origin", "source_stage",
)

class AnalysisFailure(ValueError):
    """Input, pairing, inference, release, or output contract failed."""


def _build_conditions() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for family in ("subset_shuffle",):
        for p in (0.1, 0.3, 0.5):
            for repeat in (0, 1):
                rows.append({
                    "family": family, "p": p, "window_size": None, "repeat": repeat,
                    "strength": f"p={p:.1f}",
                    "condition_id": f"{family}|p={p:.1f}|r={repeat}",
                })
    for width in (2, 5, 10, 20):
        for repeat in (0, 1):
            rows.append({
                "family": "time_window_shuffle", "p": None, "window_size": width,
                "repeat": repeat, "strength": f"B={width}",
                "condition_id": f"time_window_shuffle|B={width}|r={repeat}",
            })
    rows.append({
        "family": "CLEAN", "p": None, "window_size": None, "repeat": 0,
        "strength": "CLEAN", "condition_id": "CLEAN",
    })
    return rows


CONDITIONS: tuple[dict[str, Any], ...] = tuple(_build_conditions())
CONDITION_BY_ID = {str(row["condition_id"]): row for row in CONDITIONS}
CONDITION_INDEX = {str(row["condition_id"]): i for i, row in enumerate(CONDITIONS)}


@dataclass(frozen=True)
class ValidatedBank:
    layouts: tuple[str, ...]
    geometry_by_layout: Mapping[str, str]
    n_by_layout: Mapping[str, int]
    raw: np.ndarray  # layout x condition x solver x policy
    rn: np.ndarray
    forecast_pair_count: int
    source_origin_counts: Mapping[str, int]
    source_stage_counts: Mapping[str, int]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _finite_float(value: Any, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise AnalysisFailure(f"{label} is not numeric: {value!r}") from exc
    if not math.isfinite(number):
        raise AnalysisFailure(f"{label} must be finite")
    return number


def _strict_int(value: Any, label: str) -> int:
    number = _finite_float(value, label)
    if not number.is_integer():
        raise AnalysisFailure(f"{label} must be an integer: {value!r}")
    return int(number)


def _truthy_complete(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def load_layout_registry(path: Path = LAYOUT_PATH) -> dict[str, tuple[str, int]]:
    """Return frozen fingerprint -> (geometry, N) mapping from select20."""
    try:
        rows = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AnalysisFailure(f"cannot read frozen select20 layout registry: {path}: {exc}") from exc
    if not isinstance(rows, list) or len(rows) != EXPECTED_LAYOUTS:
        raise AnalysisFailure("frozen select20 must contain exactly 20 layouts")
    registry: dict[str, tuple[str, int]] = {}
    for i, row in enumerate(rows):
        fp = str(row.get("label_invariant_fingerprint", "")).strip()
        geometry = str(row.get("parameter_group", "")).strip()
        n = sum(len(stack) for stack in row.get("initial_stacks", []))
        if not fp or geometry not in GEOMETRIES or n <= 0:
            raise AnalysisFailure(f"select20 row {i} has an invalid fingerprint/geometry/N")
        if fp in registry:
            raise AnalysisFailure(f"select20 duplicates layout fingerprint {fp}")
        registry[fp] = (geometry, int(n))
    counts = {g: sum(value[0] == g for value in registry.values()) for g in GEOMETRIES}
    if counts != {"G1": 10, "G2": 10}:
        raise AnalysisFailure(f"select20 geometry split must be 10/10, found {counts}")
    return registry


def _check_condition_fields(row: Mapping[str, Any], label: str, spec: Mapping[str, Any]) -> None:
    if str(row.get("family", "")).strip() != str(spec["family"]):
        raise AnalysisFailure(f"{label}: family disagrees with {spec['condition_id']}")
    if str(row.get("strength", "")).strip() != str(spec["strength"]):
        raise AnalysisFailure(f"{label}: strength disagrees with {spec['condition_id']}")
    if _strict_int(row.get("repeat"), f"{label} repeat") != int(spec["repeat"]):
        raise AnalysisFailure(f"{label}: repeat disagrees with {spec['condition_id']}")
    p_value = row.get("p")
    if spec["p"] is None:
        if p_value is not None and str(p_value).strip():
            raise AnalysisFailure(f"{label}: p must be empty for {spec['condition_id']}")
    elif not math.isclose(_finite_float(p_value, f"{label} p"), float(spec["p"]), rel_tol=0, abs_tol=1e-12):
        raise AnalysisFailure(f"{label}: p disagrees with {spec['condition_id']}")
    width = row.get("window_size")
    if spec["window_size"] is None:
        if width is not None and str(width).strip():
            raise AnalysisFailure(f"{label}: window_size must be empty for {spec['condition_id']}")
    elif _strict_int(width, f"{label} window_size") != int(spec["window_size"]):
        raise AnalysisFailure(f"{label}: window_size disagrees with {spec['condition_id']}")


def validate_rows(
    rows: Sequence[Mapping[str, Any]],
    expected_layout_registry: Mapping[str, tuple[str, int]] | None = None,
) -> ValidatedBank:
    """Validate exact 15,120-key matrix and common frozen forecast per pair."""
    if len(CONDITIONS) != EXPECTED_CONDITIONS or len(CONDITION_BY_ID) != EXPECTED_CONDITIONS:
        raise AnalysisFailure("internal condition registry is not the registered 21-condition bank")
    if len(rows) != EXPECTED_ROWS:
        raise AnalysisFailure(f"expected {EXPECTED_ROWS} merged rows, found {len(rows)}")

    seen: dict[tuple[str, str, str, str], Mapping[str, Any]] = {}
    trajectory_keys: set[str] = set()
    geometry_by_layout: dict[str, str] = {}
    n_by_layout: dict[str, int] = {}
    forecast_by_pair: dict[tuple[str, str], tuple[str, str]] = {}
    source_origin_counts: dict[str, int] = {}
    source_stage_counts: dict[str, int] = {}
    for number, row in enumerate(rows, start=2):
        label = f"merged CSV row {number}"
        missing = [name for name in RAW_COLUMNS if name not in row]
        if missing:
            raise AnalysisFailure(f"{label}: missing raw columns {missing}")
        fp = str(row.get("layout_fingerprint", "")).strip()
        cid = str(row.get("condition_id", "")).strip()
        solver = str(row.get("algorithm", "")).strip()
        method = str(row.get("method", "")).strip()
        trajectory_key = str(row.get("trajectory_key", "")).strip()
        if not fp or cid not in CONDITION_BY_ID or solver not in SOLVERS or method not in POLICIES:
            raise AnalysisFailure(f"{label}: key fields are outside the registered matrix")
        expected_key = f"{fp}|{cid}|{solver}|{method}"
        if trajectory_key != expected_key:
            raise AnalysisFailure(f"{label}: trajectory_key does not match registered fields")
        if trajectory_key in trajectory_keys:
            raise AnalysisFailure(f"{label}: duplicate trajectory_key {trajectory_key}")
        trajectory_keys.add(trajectory_key)
        key = (fp, cid, solver, method)
        if key in seen:
            raise AnalysisFailure(f"{label}: duplicate layout-condition-solver-policy key {key}")
        seen[key] = row

        geometry = str(row.get("geometry", "")).strip()
        if geometry not in GEOMETRIES:
            raise AnalysisFailure(f"{label}: unregistered geometry {geometry!r}")
        if fp in geometry_by_layout and geometry_by_layout[fp] != geometry:
            raise AnalysisFailure(f"{label}: geometry changes within layout {fp}")
        geometry_by_layout[fp] = geometry
        n = _strict_int(row.get("N"), f"{label} N")
        if n <= 0:
            raise AnalysisFailure(f"{label}: N must be positive")
        if fp in n_by_layout and n_by_layout[fp] != n:
            raise AnalysisFailure(f"{label}: N changes within layout {fp}")
        n_by_layout[fp] = n
        if expected_layout_registry is not None:
            if fp not in expected_layout_registry:
                raise AnalysisFailure(f"{label}: fingerprint is absent from frozen select20")
            exp_geometry, exp_n = expected_layout_registry[fp]
            if (geometry, n) != (exp_geometry, exp_n):
                raise AnalysisFailure(f"{label}: geometry or N differs from frozen select20")

        _check_condition_fields(row, label, CONDITION_BY_ID[cid])
        if not _truthy_complete(row.get("complete")):
            raise AnalysisFailure(f"{label}: trajectory is incomplete")
        relocations = _finite_float(row.get("R_raw"), f"{label} R_raw")
        if relocations < 0 or not relocations.is_integer():
            raise AnalysisFailure(f"{label}: R_raw must be a nonnegative integer")
        rn = _finite_float(row.get("R_over_N"), f"{label} R_over_N")
        if rn < 0 or not math.isclose(rn, relocations / n, rel_tol=1e-8, abs_tol=1e-10):
            raise AnalysisFailure(f"{label}: R_over_N disagrees with R_raw/N")

        seed = str(row.get("forecast_seed_evaluator_only", "")).strip()
        f_hash = str(row.get("F_hash", "")).strip()
        if not f_hash:
            raise AnalysisFailure(f"{label}: F_hash is empty")
        pair_key = (fp, cid)
        pair = (seed, f_hash)
        if pair_key in forecast_by_pair and forecast_by_pair[pair_key] != pair:
            raise AnalysisFailure(f"{label}: frozen forecast seed/F_hash differs within {pair_key}")
        forecast_by_pair[pair_key] = pair

        origin = str(row.get("source_origin", "")).strip()
        stage = str(row.get("source_stage", "")).strip()
        if not origin or not stage:
            raise AnalysisFailure(f"{label}: source_origin/source_stage must be recorded")
        source_origin_counts[origin] = source_origin_counts.get(origin, 0) + 1
        source_stage_counts[stage] = source_stage_counts.get(stage, 0) + 1

    layouts = tuple(sorted(geometry_by_layout, key=lambda fp: (geometry_by_layout[fp], fp)))
    if len(layouts) != EXPECTED_LAYOUTS:
        raise AnalysisFailure(f"expected {EXPECTED_LAYOUTS} layouts, found {len(layouts)}")
    geometry_counts = {g: sum(geometry_by_layout[fp] == g for fp in layouts) for g in GEOMETRIES}
    if geometry_counts != {"G1": 10, "G2": 10}:
        raise AnalysisFailure(f"layout geometry split must be 10/10, found {geometry_counts}")
    if expected_layout_registry is not None and set(layouts) != set(expected_layout_registry):
        raise AnalysisFailure("merged fingerprint set differs from frozen select20")

    expected_keys = {
        (fp, str(spec["condition_id"]), solver, method)
        for fp in layouts for spec in CONDITIONS for solver in SOLVERS for method in POLICIES
    }
    if set(seen) != expected_keys:
        missing = expected_keys - set(seen)
        extra = set(seen) - expected_keys
        raise AnalysisFailure(f"full key matrix mismatch: missing={len(missing)}, extra={len(extra)}")
    if len(forecast_by_pair) != EXPECTED_LAYOUTS * EXPECTED_CONDITIONS:
        raise AnalysisFailure("not all 420 layout-condition pairs have a common forecast binding")

    raw = np.empty((EXPECTED_LAYOUTS, EXPECTED_CONDITIONS, len(SOLVERS), len(POLICIES)), dtype=np.float64)
    rn_values = np.empty_like(raw)
    for li, fp in enumerate(layouts):
        for ci, spec in enumerate(CONDITIONS):
            cid = str(spec["condition_id"])
            for si, solver in enumerate(SOLVERS):
                for pi, method in enumerate(POLICIES):
                    row = seen[(fp, cid, solver, method)]
                    value = float(row["R_raw"])
                    raw[li, ci, si, pi] = value
                    rn_values[li, ci, si, pi] = value / n_by_layout[fp]
    return ValidatedBank(
        layouts=layouts,
        geometry_by_layout=geometry_by_layout,
        n_by_layout=n_by_layout,
        raw=raw,
        rn=rn_values,
        forecast_pair_count=len(forecast_by_pair),
        source_origin_counts=source_origin_counts,
        source_stage_counts=source_stage_counts,
    )


def _family_layout_values(values: np.ndarray, family: str) -> np.ndarray:
    """Repeat -> strength -> family mean, retaining layout/solver/policy axes."""
    strength_groups: dict[str, list[int]] = {}
    for ci, spec in enumerate(CONDITIONS):
        if spec["family"] == family:
            strength_groups.setdefault(str(spec["strength"]), []).append(ci)
    if not strength_groups:
        raise AnalysisFailure(f"unknown/empty condition family {family}")
    strength_means = [values[:, idxs, :, :].mean(axis=1) for _, idxs in sorted(strength_groups.items())]
    return np.stack(strength_means, axis=1).mean(axis=1)


def scope_layout_values(values: np.ndarray) -> dict[str, np.ndarray]:
    """Create all registered macro, family, CLEAN, and condition scopes."""
    if values.shape != (EXPECTED_LAYOUTS, EXPECTED_CONDITIONS, len(SOLVERS), len(POLICIES)):
        raise AnalysisFailure(f"unexpected statistic matrix shape {values.shape}")
    result: dict[str, np.ndarray] = {}
    family_values: dict[str, np.ndarray] = {}
    for family in FAMILIES:
        family_values[family] = _family_layout_values(values, family)
        result[f"family:{family}"] = family_values[family]
    result["noisy"] = np.stack([family_values[family] for family in FAMILIES], axis=1).mean(axis=1)
    clean_index = CONDITION_INDEX["CLEAN"]
    result["clean"] = values[:, clean_index, :, :]
    for ci, spec in enumerate(CONDITIONS):
        result[f"condition:{spec['condition_id']}"] = values[:, ci, :, :]
    return result


def geometry_mean(layout_values: np.ndarray, geometry_by_layout: Mapping[str, str], layouts: Sequence[str]) -> np.ndarray:
    pieces = []
    for geometry in GEOMETRIES:
        idx = [i for i, fp in enumerate(layouts) if geometry_by_layout[fp] == geometry]
        if len(idx) != 10:
            raise AnalysisFailure(f"bootstrap/point estimate requires ten {geometry} layouts")
        pieces.append(layout_values[idx].mean(axis=0))
    return (pieces[0] + pieces[1]) / 2.0


def make_bootstrap_indices(
    layouts: Sequence[str], geometry_by_layout: Mapping[str, str], *,
    replicates: int = BOOTSTRAP_REPLICATES, seed: int = BOOTSTRAP_SEED,
    batch_size: int = BOOTSTRAP_BATCH_SIZE,
) -> np.ndarray:
    """One common PCG64 draw matrix: 10 with replacement per geometry."""
    if replicates <= 0 or batch_size <= 0:
        raise AnalysisFailure("bootstrap replicates and batch size must be positive")
    groups = {g: np.asarray([i for i, fp in enumerate(layouts) if geometry_by_layout[fp] == g], dtype=np.int16)
              for g in GEOMETRIES}
    if any(len(groups[g]) != 10 for g in GEOMETRIES):
        raise AnalysisFailure("registered bootstrap requires exactly ten layouts in each geometry")
    rng = np.random.Generator(np.random.PCG64(seed))
    indices = np.empty((replicates, EXPECTED_LAYOUTS), dtype=np.int16)
    for start in range(0, replicates, batch_size):
        stop = min(start + batch_size, replicates)
        n = stop - start
        indices[start:stop, :10] = groups["G1"][rng.integers(0, 10, size=(n, 10))]
        indices[start:stop, 10:] = groups["G2"][rng.integers(0, 10, size=(n, 10))]
    return indices


def bootstrap_scope_pair(
    raw_layout_values: np.ndarray,
    rn_layout_values: np.ndarray,
    indices: np.ndarray,
    *, batch_size: int = BOOTSTRAP_BATCH_SIZE,
    deadline: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Bootstrap paired policy means in chunks, avoiding a replicate x bank tensor."""
    reps = len(indices)
    raw_draws = np.empty((reps, len(SOLVERS), len(POLICIES)), dtype=np.float64)
    rn_draws = np.empty_like(raw_draws)
    for start in range(0, reps, batch_size):
        if deadline is not None and time.perf_counter() > deadline:
            raise AnalysisFailure("M3 analysis exceeded the registered 1,800 second cap")
        stop = min(start + batch_size, reps)
        draw_indices = indices[start:stop]
        raw_draws[start:stop] = raw_layout_values[draw_indices].mean(axis=1)
        rn_draws[start:stop] = rn_layout_values[draw_indices].mean(axis=1)
    return raw_draws, rn_draws


def _quantiles(values: np.ndarray, probs: tuple[float, float]) -> tuple[float | None, float | None]:
    arr = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(arr)):
        return None, None
    lo, hi = np.quantile(arr, probs, method=BOOTSTRAP_QUANTILE_METHOD)
    return float(lo), float(hi)


def _safe_ratio_samples(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray | None:
    if np.any(denominator <= 0) or not np.all(np.isfinite(denominator)):
        return None
    return 100.0 * numerator / denominator


def _ci_fields(values: np.ndarray, probs: tuple[float, float]) -> tuple[float | None, float | None]:
    return _quantiles(values, probs)


def primary_comparisons(
    point: np.ndarray,
    boot: np.ndarray,
    *, replicates: int | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return 30 solver contrasts and five equal-six contrasts (35 family)."""
    if point.shape != (len(SOLVERS), len(POLICIES)) or boot.ndim != 3 or boot.shape[1:] != point.shape:
        raise AnalysisFailure("primary comparison arrays have an invalid shape")
    reps = int(len(boot) if replicates is None else replicates)
    primary_rows: list[dict[str, Any]] = []
    overall_rows: list[dict[str, Any]] = []
    sim_probs = (SIMULTANEOUS_TAIL, 1.0 - SIMULTANEOUS_TAIL)
    per_solver_draws: dict[int, np.ndarray | None] = {}
    for h in HORIZONS:
        pi = POLICY_FOR_H[h]
        per_solver_draws[h] = None
        valid_draws: list[np.ndarray] = []
        point_rates: list[float] = []
        abs_draws: list[np.ndarray] = []
        point_abs: list[float] = []
        all_valid = True
        for si, solver in enumerate(SOLVERS):
            fixed = float(point[si, pi])
            ridge = float(point[si, RIDGE_INDEX])
            abs_diff = fixed - ridge
            status = "PASS"
            rate = None if fixed <= 0 else 100.0 * abs_diff / fixed
            if fixed <= 0:
                status = "UNDEFINED_ZERO_FIXED_DENOMINATOR"
            diff_draw = boot[:, si, pi] - boot[:, si, RIDGE_INDEX]
            rate_draw = _safe_ratio_samples(diff_draw, boot[:, si, pi])
            if rate_draw is None:
                status = "UNDEFINED_ZERO_BOOTSTRAP_DENOMINATOR"
            if rate is None or rate_draw is None:
                all_valid = False
            else:
                point_rates.append(rate)
                valid_draws.append(rate_draw)
            abs_draws.append(diff_draw)
            point_abs.append(abs_diff)
            abs_lo, abs_hi = _ci_fields(diff_draw, (0.025, 0.975))
            if rate is None or rate_draw is None:
                ci_lo = ci_hi = sim_lo = sim_hi = None
            else:
                ci_lo, ci_hi = _ci_fields(rate_draw, (0.025, 0.975))
                sim_lo, sim_hi = _ci_fields(rate_draw, sim_probs)
            primary_rows.append({
                "algorithm": solver, "horizon": h, "fixed_method": f"FIXED_H{h}",
                "fixed_mean_R_raw": fixed, "ridge_mean_R_raw": ridge,
                "absolute_difference_fixed_minus_ridge": abs_diff,
                "improvement_percent": rate,
                "ci95_low_percent": ci_lo, "ci95_high_percent": ci_hi,
                "simultaneous_low_percent": sim_lo, "simultaneous_high_percent": sim_hi,
                "absolute_ci95_low_moves": abs_lo, "absolute_ci95_high_moves": abs_hi,
                "familywise_alpha": FAMILY_ALPHA, "comparison_family_size": MAIN_COMPARISONS,
                "per_comparison_two_sided_alpha": FAMILY_ALPHA / MAIN_COMPARISONS,
                "bootstrap_replicates": reps, "bootstrap_seed": BOOTSTRAP_SEED,
                "higher_is_better": True, "status": status,
            })
        if all_valid:
            per_solver_draws[h] = np.stack(valid_draws, axis=1)
            rate_point = float(np.mean(point_rates))
            rate_draws_equal6 = per_solver_draws[h].mean(axis=1)
            abs_equal6 = float(np.mean(point_abs))
            abs_draws_equal6 = np.stack(abs_draws, axis=1).mean(axis=1)
            ci_lo, ci_hi = _ci_fields(rate_draws_equal6, (0.025, 0.975))
            sim_lo, sim_hi = _ci_fields(rate_draws_equal6, sim_probs)
            abs_lo, abs_hi = _ci_fields(abs_draws_equal6, (0.025, 0.975))
            status = "PASS"
        else:
            rate_point = ci_lo = ci_hi = sim_lo = sim_hi = abs_lo = abs_hi = None
            abs_equal6 = None
            status = "UNDEFINED_ZERO_DENOMINATOR_IN_AT_LEAST_ONE_SOLVER"
        overall_rows.append({
            "horizon": h, "fixed_method": f"FIXED_H{h}",
            "equal6_improvement_percent": rate_point,
            "ci95_low_percent": ci_lo, "ci95_high_percent": ci_hi,
            "simultaneous_low_percent": sim_lo, "simultaneous_high_percent": sim_hi,
            "mean_solver_absolute_difference_fixed_minus_ridge": abs_equal6,
            "absolute_ci95_low_moves": abs_lo, "absolute_ci95_high_moves": abs_hi,
            "algorithm_count": len(SOLVERS), "algorithm_weight": 1.0 / len(SOLVERS),
            "familywise_alpha": FAMILY_ALPHA, "comparison_family_size": MAIN_COMPARISONS,
            "per_comparison_two_sided_alpha": FAMILY_ALPHA / MAIN_COMPARISONS,
            "bootstrap_replicates": reps, "bootstrap_seed": BOOTSTRAP_SEED,
            "higher_is_better": True, "status": status,
        })
    if len(primary_rows) + len(overall_rows) != MAIN_COMPARISONS:
        raise AnalysisFailure("internal main comparison family is not exactly 35")
    return primary_rows, overall_rows


def _descriptive_rows(
    scope: str, raw_point: np.ndarray, rn_point: np.ndarray,
    raw_boot: np.ndarray, rn_boot: np.ndarray, *, replicates: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for metric, point, boot in (("R_raw", raw_point, raw_boot), ("RN", rn_point, rn_boot)):
        for si, solver in enumerate(SOLVERS):
            ridge = float(point[si, RIDGE_INDEX])
            for pi, method in enumerate(POLICIES):
                estimate = float(point[si, pi])
                mean_lo, mean_hi = _ci_fields(boot[:, si, pi], (0.025, 0.975))
                if pi == RIDGE_INDEX:
                    improvement = 0.0
                    delta_samples = np.zeros(replicates, dtype=np.float64)
                    rate_samples = np.zeros(replicates, dtype=np.float64)
                    status = "REFERENCE"
                else:
                    delta_samples = boot[:, si, pi] - boot[:, si, RIDGE_INDEX]
                    denom = boot[:, si, pi]
                    rate_samples = _safe_ratio_samples(delta_samples, denom)
                    if estimate <= 0:
                        improvement = None
                        status = "UNDEFINED_ZERO_POLICY_DENOMINATOR"
                    else:
                        improvement = 100.0 * (estimate - ridge) / estimate
                        status = "DESCRIPTIVE"
                    if rate_samples is None:
                        status = "UNDEFINED_ZERO_BOOTSTRAP_DENOMINATOR"
                delta_lo, delta_hi = _ci_fields(delta_samples, (0.025, 0.975))
                if rate_samples is None:
                    rate_lo = rate_hi = None
                else:
                    rate_lo, rate_hi = _ci_fields(rate_samples, (0.025, 0.975))
                rows.append({
                    "metric": metric, "scope": scope, "algorithm": solver, "method": method,
                    "estimate": estimate, "ci95_low": mean_lo, "ci95_high": mean_hi,
                    "ridge_estimate": ridge,
                    "absolute_difference_policy_minus_ridge": estimate - ridge,
                    "absolute_difference_ci95_low": delta_lo,
                    "absolute_difference_ci95_high": delta_hi,
                    "ridge_improvement_percent": improvement,
                    "ridge_improvement_ci95_low_percent": rate_lo,
                    "ridge_improvement_ci95_high_percent": rate_hi,
                    "bootstrap_replicates": replicates, "bootstrap_seed": BOOTSTRAP_SEED,
                    "interval_role": "descriptive_ordinary_95_percentile",
                    "higher_improvement_favors_ridge": True,
                    "direction": "lower_is_better", "status": status,
                })
    return rows


def _fixed_h_vs_h20_rows(
    raw_point: np.ndarray, rn_point: np.ndarray, raw_boot: np.ndarray, rn_boot: np.ndarray,
    *, replicates: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    h20_index = POLICY_FOR_H[20]
    for metric, point, boot in (("R_raw", raw_point, raw_boot), ("RN", rn_point, rn_boot)):
        solver_rates: dict[int, list[np.ndarray]] = {h: [] for h in FIXED_HORIZONS}
        solver_point_rates: dict[int, list[float]] = {h: [] for h in FIXED_HORIZONS}
        solver_abs: dict[int, list[np.ndarray]] = {h: [] for h in FIXED_HORIZONS}
        solver_point_abs: dict[int, list[float]] = {h: [] for h in FIXED_HORIZONS}
        all_valid: dict[int, bool] = {h: True for h in FIXED_HORIZONS}
        for h in FIXED_HORIZONS:
            pi = POLICY_FOR_H[h]
            for si, solver in enumerate(SOLVERS):
                fixed = float(point[si, pi])
                native = float(point[si, h20_index])
                delta = native - fixed
                samples_delta = boot[:, si, h20_index] - boot[:, si, pi]
                samples = _safe_ratio_samples(samples_delta, boot[:, si, h20_index])
                status = "DESCRIPTIVE"
                rate = None if native <= 0 else 100.0 * delta / native
                if rate is None or samples is None:
                    all_valid[h] = False
                    status = "UNDEFINED_ZERO_H20_DENOMINATOR"
                else:
                    solver_rates[h].append(samples)
                    solver_point_rates[h].append(rate)
                solver_abs[h].append(samples_delta)
                solver_point_abs[h].append(delta)
                rate_lo, rate_hi = (None, None) if rate is None or samples is None else _ci_fields(samples, (0.025, 0.975))
                abs_lo, abs_hi = _ci_fields(samples_delta, (0.025, 0.975))
                rows.append({
                    "metric": metric, "scope": "noisy", "algorithm": solver, "horizon": h,
                    "fixed_method": f"FIXED_H{h}", "baseline_method": "FIXED_H20",
                    "fixed_mean": fixed, "h20_mean": native,
                    "absolute_difference_h20_minus_fixed": delta,
                    "fixed_improvement_percent_vs_h20": rate,
                    "ci95_low_percent": rate_lo, "ci95_high_percent": rate_hi,
                    "absolute_ci95_low": abs_lo, "absolute_ci95_high": abs_hi,
                    "bootstrap_replicates": replicates, "bootstrap_seed": BOOTSTRAP_SEED,
                    "interval_role": "descriptive_ordinary_95_percentile",
                    "status": status,
                })
            if all_valid[h]:
                equal6_rate_samples = np.stack(solver_rates[h], axis=1).mean(axis=1)
                equal6_point_rate = float(np.mean(solver_point_rates[h]))
                equal6_abs_samples = np.stack(solver_abs[h], axis=1).mean(axis=1)
                equal6_abs_point = float(np.mean(solver_point_abs[h]))
                rate_lo, rate_hi = _ci_fields(equal6_rate_samples, (0.025, 0.975))
                abs_lo, abs_hi = _ci_fields(equal6_abs_samples, (0.025, 0.975))
                status = "DESCRIPTIVE_EQUAL6"
            else:
                equal6_point_rate = rate_lo = rate_hi = equal6_abs_point = abs_lo = abs_hi = None
                status = "UNDEFINED_ZERO_H20_DENOMINATOR_IN_AT_LEAST_ONE_SOLVER"
            rows.append({
                "metric": metric, "scope": "noisy", "algorithm": "EQUAL6", "horizon": h,
                "fixed_method": f"FIXED_H{h}", "baseline_method": "FIXED_H20",
                "fixed_mean": None, "h20_mean": None,
                "absolute_difference_h20_minus_fixed": equal6_abs_point,
                "fixed_improvement_percent_vs_h20": equal6_point_rate,
                "ci95_low_percent": rate_lo, "ci95_high_percent": rate_hi,
                "absolute_ci95_low": abs_lo, "absolute_ci95_high": abs_hi,
                "bootstrap_replicates": replicates, "bootstrap_seed": BOOTSTRAP_SEED,
                "interval_role": "descriptive_ordinary_95_percentile",
                "status": status,
            })
    return rows


def _best_fixed_rows(
    raw_point: np.ndarray, raw_boot: np.ndarray, primary_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    primary_by_key = {(str(r["algorithm"]), int(r["horizon"])): r for r in primary_rows}
    rows: list[dict[str, Any]] = []
    package_fixed: list[float] = []
    package_ridge: list[float] = []
    package_rate: list[float] = []
    for si, solver in enumerate(SOLVERS):
        best_h = min(HORIZONS, key=lambda h: (float(raw_point[si, POLICY_FOR_H[h]]), h))
        fixed_mean = float(raw_point[si, POLICY_FOR_H[best_h]])
        ridge_mean = float(raw_point[si, RIDGE_INDEX])
        primary = primary_by_key[(solver, best_h)]
        rate = primary["improvement_percent"]
        package_fixed.append(fixed_mean)
        package_ridge.append(ridge_mean)
        if rate is not None:
            package_rate.append(float(rate))
        rows.append({
            "algorithm": solver, "development_best_horizon": best_h,
            "development_best_fixed_mean_R_raw": fixed_mean, "ridge_mean_R_raw": ridge_mean,
            "absolute_difference_fixed_minus_ridge": fixed_mean - ridge_mean,
            "ridge_improvement_percent": rate,
            "simultaneous_low_percent": primary["simultaneous_low_percent"],
            "simultaneous_high_percent": primary["simultaneous_high_percent"],
            "selection_role": "descriptive_selected_on_same_development_bank",
            "inference_note": "35-comparison simultaneous interval is inherited from the declared H comparison; same-bank H selection is not independent confirmation",
            "status": primary["status"],
        })
    if len(package_rate) == len(SOLVERS):
        rows.append({
            "algorithm": "EQUAL6_SOLVER_SPECIFIC_ENVELOPE", "development_best_horizon": "solver_specific",
            "development_best_fixed_mean_R_raw": float(np.mean(package_fixed)),
            "ridge_mean_R_raw": float(np.mean(package_ridge)),
            "absolute_difference_fixed_minus_ridge": float(np.mean(package_fixed) - np.mean(package_ridge)),
            "ridge_improvement_percent": float(np.mean(package_rate)),
            "simultaneous_low_percent": None, "simultaneous_high_percent": None,
            "selection_role": "descriptive_five_fixed_h_development_envelope",
            "inference_note": "solver-specific selected means; not a deployable common-H baseline and no selection-adjusted interval",
            "status": "DESCRIPTIVE_ONLY",
        })
    return rows


def _layout_aggregate_rows(
    bank: ValidatedBank,
    raw_scopes: Mapping[str, np.ndarray],
    rn_scopes: Mapping[str, np.ndarray],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    labels: list[tuple[str, str, str, str, str]] = [("noisy", "noisy", "NOISY_EQUAL_WEIGHTED", "", "")]
    for family in FAMILIES:
        labels.append((f"family:{family}", "family", family, family, ""))
    labels.append(("clean", "CLEAN", "CLEAN", "CLEAN", ""))
    for spec in CONDITIONS:
        labels.append((f"condition:{spec['condition_id']}", "condition", str(spec["condition_id"]), str(spec["family"]), str(spec["strength"])))
    for scope_key, scope_type, scope, family, strength in labels:
        raw = raw_scopes[scope_key]
        rn = rn_scopes[scope_key]
        for li, fp in enumerate(bank.layouts):
            for si, solver in enumerate(SOLVERS):
                for pi, policy in enumerate(POLICIES):
                    rows.append({
                        "scope_type": scope_type, "scope": scope, "family": family,
                        "strength": strength, "layout_fingerprint": fp,
                        "geometry": bank.geometry_by_layout[fp], "N": bank.n_by_layout[fp],
                        "algorithm": solver, "method": policy,
                        "R_raw_mean": float(raw[li, si, pi]), "RN_mean": float(rn[li, si, pi]),
                        "repeat_strength_geometry_weighting": "repeat_mean_then_strength_equal_then_family_equal_then_geometry_equal",
                    })
    return rows


def _read_csv(path: Path) -> list[dict[str, str]]:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames is None or len(reader.fieldnames) != len(set(reader.fieldnames)):
                raise AnalysisFailure(f"merged CSV header is missing or duplicated: {path}")
            missing = sorted(set(RAW_COLUMNS) - set(reader.fieldnames))
            if missing:
                raise AnalysisFailure(f"merged CSV lacks required columns: {missing}")
            rows = [dict(row) for row in reader]
    except OSError as exc:
        raise AnalysisFailure(f"cannot read merged raw CSV: {path}: {exc}") from exc
    if any(None in row or any(value is None for value in row.values()) for row in rows):
        raise AnalysisFailure("merged CSV has malformed row width")
    return rows


def _csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(fields), extrasaction="raise")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key)) for key in fields})


def _fixed_curve_rows(point: np.ndarray, boot: np.ndarray, replicates: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for si, solver in enumerate(SOLVERS):
        for h in HORIZONS:
            pi = POLICY_FOR_H[h]
            lo, hi = _ci_fields(boot[:, si, pi], (0.025, 0.975))
            rows.append({
                "algorithm": solver, "method": f"FIXED_H{h}", "horizon": h,
                "mean_R_raw": float(point[si, pi]), "ci95_low_R_raw": lo, "ci95_high_R_raw": hi,
                "direction": "lower_is_better", "interval_role": "descriptive_ordinary_95_percentile",
                "bootstrap_replicates": replicates,
            })
        lo, hi = _ci_fields(boot[:, si, RIDGE_INDEX], (0.025, 0.975))
        ridge = float(point[si, RIDGE_INDEX])
        for h in HORIZONS:
            rows.append({
                "algorithm": solver, "method": "RIDGE", "horizon": h,
                "mean_R_raw": ridge, "ci95_low_R_raw": lo, "ci95_high_R_raw": hi,
                "direction": "lower_is_better", "interval_role": "descriptive_ordinary_95_percentile",
                "bootstrap_replicates": replicates,
            })
    return rows


def _conclusion_for_solver(rows: Sequence[Mapping[str, Any]], solver: str) -> str:
    subset = [r for r in rows if r["algorithm"] == solver]
    positive = {int(r["horizon"]) for r in subset if r["status"] == "PASS" and r["simultaneous_low_percent"] is not None and float(r["simultaneous_low_percent"]) > 0}
    negative = {int(r["horizon"]) for r in subset if r["status"] == "PASS" and r["simultaneous_high_percent"] is not None and float(r["simultaneous_high_percent"]) < 0}
    if len(positive) == len(HORIZONS):
        return "simultaneous evidence favors Ridge over all five registered fixed horizons"
    if positive and negative:
        return (
            "supports Ridge over H " + ",".join(str(h) for h in sorted(positive))
            + "; degradation evidence versus H " + ",".join(str(h) for h in sorted(negative))
        )
    if 20 in positive and not any(h in positive for h in FIXED_HORIZONS):
        return "supports Ridge over H20; evidence does not establish superiority over short fixed horizons"
    if positive:
        return "supports Ridge only over horizons " + ",".join(str(h) for h in sorted(positive))
    if negative:
        return "degradation evidence versus horizons " + ",".join(str(h) for h in sorted(negative))
    return "comparisons remain unresolved by the simultaneous intervals"


def _equal6_conclusion(rows: Sequence[Mapping[str, Any]]) -> str:
    positive = {int(row["horizon"]) for row in rows
                if row["status"] == "PASS" and row["simultaneous_low_percent"] is not None
                and float(row["simultaneous_low_percent"]) > 0}
    negative = {int(row["horizon"]) for row in rows
                if row["status"] == "PASS" and row["simultaneous_high_percent"] is not None
                and float(row["simultaneous_high_percent"]) < 0}
    if len(positive) == len(HORIZONS):
        return "the equal-six macro score supports Ridge over all five registered fixed horizons"
    if positive and negative:
        return "equal-six evidence favors Ridge over H " + ",".join(str(h) for h in sorted(positive)) + " and favors fixed H over Ridge at H " + ",".join(str(h) for h in sorted(negative))
    if positive:
        return "equal-six evidence favors Ridge only over H " + ",".join(str(h) for h in sorted(positive))
    if negative:
        return "equal-six degradation evidence versus H " + ",".join(str(h) for h in sorted(negative))
    return "equal-six comparisons remain unresolved by the simultaneous intervals"


def _format_num(value: Any, digits: int = 3) -> str:
    if value is None:
        return "undefined"
    return f"{float(value):.{digits}f}"


def _build_final_report(primary, overall, best_rows, bank, *, input_sha256, analyzer_sha256):
    from reproduction.reporting import build_report
    clean_point = geometry_mean(scope_layout_values(bank.raw)["clean"], bank.geometry_by_layout, bank.layouts)
    return build_report(primary, overall, best_rows, bank, clean_point=clean_point,
                        solvers=SOLVERS, policies=POLICIES, input_sha256=input_sha256,
                        analyzer_sha256=analyzer_sha256)


def _render_figures(
    out_dir: Path, curve_rows: Sequence[Mapping[str, Any]], primary_rows: Sequence[Mapping[str, Any]],
    family_rows: Sequence[Mapping[str, Any]], clean_rows: Sequence[Mapping[str, Any]],
) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    created: list[str] = []

    def save(fig: Any, stem: str) -> None:
        fig.savefig(out_dir / f"{stem}.png", dpi=220, bbox_inches="tight")
        fig.savefig(out_dir / f"{stem}.pdf", bbox_inches="tight")
        plt.close(fig)
        created.extend([f"{stem}.png", f"{stem}.pdf"])

    def spans(y: Sequence[float], low: Sequence[float], high: Sequence[float]) -> list[list[float]]:
        lower: list[float] = []
        upper: list[float] = []
        for center, lo, hi in zip(y, low, high):
            if not (np.isfinite(center) and np.isfinite(lo) and np.isfinite(hi)):
                lower.append(0.0)
                upper.append(0.0)
            else:
                lower.append(max(0.0, float(center - lo)))
                upper.append(max(0.0, float(hi - center)))
        return [lower, upper]

    horizons = list(HORIZONS)
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), sharex=True)
    for ax, solver in zip(axes.flat, SOLVERS):
        rows = [r for r in curve_rows if r["algorithm"] == solver]
        fixed = [next(r for r in rows if r["method"] == f"FIXED_H{h}") for h in horizons]
        curve_y = [float(r["mean_R_raw"]) for r in fixed]
        curve_lo = [np.nan if r["ci95_low_R_raw"] is None else float(r["ci95_low_R_raw"]) for r in fixed]
        curve_hi = [np.nan if r["ci95_high_R_raw"] is None else float(r["ci95_high_R_raw"]) for r in fixed]
        ax.errorbar(horizons, curve_y, yerr=spans(curve_y, curve_lo, curve_hi),
                    marker="o", capsize=3, linewidth=1.7, label="Fixed H")
        ridge = next(r for r in rows if r["method"] == "RIDGE")
        ax.axhline(ridge["mean_R_raw"], color="black", linestyle="--", linewidth=1.4, label="Ridge")
        ax.set_title(solver)
        ax.set_xticks(horizons)
        ax.set_xlabel("Numeric forecast horizon H")
        ax.set_ylabel("Mean R_raw (relocations; lower is better)")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    fig.suptitle("Fixed-horizon relocation counts with Ridge reference; noisy macro", y=1.01)
    fig.tight_layout()
    save(fig, "FIXED_H_RRAW_BY_SOLVER")

    fig, axes = plt.subplots(2, 3, figsize=(15, 9), sharex=True)
    for ax, solver in zip(axes.flat, SOLVERS):
        rows = sorted((r for r in primary_rows if r["algorithm"] == solver), key=lambda r: int(r["horizon"]))
        x = [int(r["horizon"]) for r in rows]
        y = [np.nan if r["improvement_percent"] is None else float(r["improvement_percent"]) for r in rows]
        lo = [np.nan if r["simultaneous_low_percent"] is None else float(r["simultaneous_low_percent"]) for r in rows]
        hi = [np.nan if r["simultaneous_high_percent"] is None else float(r["simultaneous_high_percent"]) for r in rows]
        ax.axhline(0.0, color="black", linewidth=1)
        ax.errorbar(x, y, yerr=spans(y, lo, hi),
                    marker="o", capsize=3)
        ax.set_title(solver)
        ax.set_xticks(HORIZONS)
        ax.set_xlabel("Numeric fixed horizon H")
        ax.set_ylabel("Ridge improvement (%)")
        ax.grid(alpha=0.25)
    fig.suptitle("Ridge vs fixed H; Bonferroni simultaneous 95% intervals (35 comparisons)", y=1.01)
    fig.tight_layout()
    save(fig, "RIDGE_VS_FIXED_H_SIMULTANEOUS")

    families = list(FAMILIES)
    fig, axes = plt.subplots(1, len(FAMILIES), figsize=(6 * len(FAMILIES), 5), sharey=True)
    for ax, family in zip(axes.flat, families):
        for solver in SOLVERS:
            subset = [r for r in family_rows if r["family"] == family and r["algorithm"] == solver
                      and r["metric"] == "R_raw" and r["method"] != "RIDGE"]
            subset = sorted(subset, key=lambda r: int(r["method"].replace("FIXED_H", "")))
            x = [int(r["method"].replace("FIXED_H", "")) for r in subset]
            y = [np.nan if r["ridge_improvement_percent"] is None else float(r["ridge_improvement_percent"]) for r in subset]
            lo = [np.nan if r["ridge_improvement_ci95_low_percent"] is None else float(r["ridge_improvement_ci95_low_percent"]) for r in subset]
            hi = [np.nan if r["ridge_improvement_ci95_high_percent"] is None else float(r["ridge_improvement_ci95_high_percent"]) for r in subset]
            ax.errorbar(x, y, yerr=spans(y, lo, hi),
                        marker="o", linewidth=1, capsize=2, label=solver)
        ax.axhline(0.0, color="black", linewidth=1)
        ax.set_title(family)
        ax.set_xticks(HORIZONS)
        ax.set_xlabel("Numeric fixed horizon H")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Ridge improvement (%)")
    axes[-1].legend(fontsize=8, loc="best")
    fig.suptitle("Ridge vs fixed H by noise family; descriptive ordinary 95% intervals", y=1.02)
    fig.tight_layout()
    save(fig, "FAMILY_RESULTS_FACETS")

    fig, axes = plt.subplots(2, 3, figsize=(15, 9), sharex=True)
    for ax, solver in zip(axes.flat, SOLVERS):
        subset = [r for r in clean_rows if r["algorithm"] == solver and r["metric"] == "R_raw" and r["method"] != "RIDGE"]
        subset = sorted(subset, key=lambda r: int(r["method"].replace("FIXED_H", "")))
        x = [int(r["method"].replace("FIXED_H", "")) for r in subset]
        y = [np.nan if r["ridge_improvement_percent"] is None else float(r["ridge_improvement_percent"]) for r in subset]
        lo = [np.nan if r["ridge_improvement_ci95_low_percent"] is None else float(r["ridge_improvement_ci95_low_percent"]) for r in subset]
        hi = [np.nan if r["ridge_improvement_ci95_high_percent"] is None else float(r["ridge_improvement_ci95_high_percent"]) for r in subset]
        ax.errorbar(x, y, yerr=spans(y, lo, hi),
                    marker="o", capsize=3)
        ax.axhline(0.0, color="black", linewidth=1)
        ax.set_title(solver)
        ax.set_xticks(HORIZONS)
        ax.set_xlabel("Numeric fixed horizon H")
        ax.set_ylabel("Ridge improvement (%)")
        ax.grid(alpha=0.25)
    fig.suptitle("CLEAN Ridge vs fixed H; descriptive ordinary 95% intervals", y=1.01)
    fig.tight_layout()
    save(fig, "CLEAN_RESULTS")
    return created


def _write_analysis_outputs(
    out_dir: Path, bank: ValidatedBank,
    raw_scopes: Mapping[str, np.ndarray], rn_scopes: Mapping[str, np.ndarray],
    scope_boots: Mapping[str, tuple[np.ndarray, np.ndarray]],
    primary_rows: Sequence[Mapping[str, Any]], overall_rows: Sequence[Mapping[str, Any]],
    fixed_h20_rows: Sequence[Mapping[str, Any]], best_rows: Sequence[Mapping[str, Any]],
    curve_rows: Sequence[Mapping[str, Any]], *, input_sha256: str, analyzer_sha256: str,
    replicates: int,
) -> list[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    layout_rows = _layout_aggregate_rows(bank, raw_scopes, rn_scopes)
    family_rows: list[dict[str, Any]] = []
    condition_rows: list[dict[str, Any]] = []
    clean_rows: list[dict[str, Any]] = []
    for family in FAMILIES:
        scope = f"family:{family}"
        raw_point = geometry_mean(raw_scopes[scope], bank.geometry_by_layout, bank.layouts)
        rn_point = geometry_mean(rn_scopes[scope], bank.geometry_by_layout, bank.layouts)
        raw_boot, rn_boot = scope_boots[scope]
        rows = _descriptive_rows(f"family:{family}", raw_point, rn_point, raw_boot, rn_boot, replicates=replicates)
        for row in rows:
            row["family"] = family
            row["descriptive"] = True
            row["condition_id"] = ""
        family_rows.extend(rows)
    scope = "clean"
    raw_point = geometry_mean(raw_scopes[scope], bank.geometry_by_layout, bank.layouts)
    rn_point = geometry_mean(rn_scopes[scope], bank.geometry_by_layout, bank.layouts)
    raw_boot, rn_boot = scope_boots[scope]
    clean_rows = _descriptive_rows("CLEAN", raw_point, rn_point, raw_boot, rn_boot, replicates=replicates)
    for row in clean_rows:
        row["family"] = "CLEAN"
        row["condition_id"] = "CLEAN"
        row["strength"] = "CLEAN"
        row["repeat"] = 0
        row["descriptive"] = True
    for spec in CONDITIONS:
        scope = f"condition:{spec['condition_id']}"
        raw_point = geometry_mean(raw_scopes[scope], bank.geometry_by_layout, bank.layouts)
        rn_point = geometry_mean(rn_scopes[scope], bank.geometry_by_layout, bank.layouts)
        raw_boot, rn_boot = scope_boots[scope]
        rows = _descriptive_rows(str(spec["condition_id"]), raw_point, rn_point, raw_boot, rn_boot, replicates=replicates)
        for row in rows:
            row.update({
                "condition_id": spec["condition_id"], "family": spec["family"],
                "strength": spec["strength"], "repeat": spec["repeat"], "descriptive": True,
            })
        condition_rows.extend(rows)

    report = _build_final_report(
        primary_rows, overall_rows, best_rows, bank, input_sha256=input_sha256,
        analyzer_sha256=analyzer_sha256,
    )
    curve_fields = ("algorithm", "method", "horizon", "mean_R_raw", "ci95_low_R_raw", "ci95_high_R_raw", "direction", "interval_role", "bootstrap_replicates")
    primary_fields = tuple(primary_rows[0].keys())
    overall_fields = tuple(overall_rows[0].keys())
    fixed_h20_fields = tuple(fixed_h20_rows[0].keys())
    best_fields = tuple(best_rows[0].keys())
    descriptive_fields = (
        "metric", "scope", "family", "condition_id", "strength", "repeat", "algorithm", "method",
        "estimate", "ci95_low", "ci95_high", "ridge_estimate", "absolute_difference_policy_minus_ridge",
        "absolute_difference_ci95_low", "absolute_difference_ci95_high", "ridge_improvement_percent",
        "ridge_improvement_ci95_low_percent", "ridge_improvement_ci95_high_percent", "bootstrap_replicates",
        "bootstrap_seed", "interval_role", "higher_improvement_favors_ridge", "direction", "descriptive", "status",
    )
    layout_fields = tuple(layout_rows[0].keys())
    outputs: dict[str, tuple[Sequence[Mapping[str, Any]], Sequence[str]]] = {
        "LAYOUT_AGGREGATES.csv": (layout_rows, layout_fields),
        "RIDGE_VS_FIXED_H_MACRO.csv": (primary_rows, primary_fields),
        "OVERALL_EQUAL6_RESULTS.csv": (overall_rows, overall_fields),
        "FIXED_H_VS_H20.csv": (fixed_h20_rows, fixed_h20_fields),
        "FAMILY_RESULTS.csv": (family_rows, descriptive_fields),
        "CONDITION_RESULTS.csv": (condition_rows, descriptive_fields),
        "CLEAN_RESULTS.csv": (clean_rows, descriptive_fields),
        "DEVELOPMENT_BEST_FIXED_H.csv": (best_rows, best_fields),
        "PLOT_DATA_FIXED_H_RRAW.csv": (curve_rows, curve_fields),
        "PLOT_DATA_RIDGE_VS_FIXED_H.csv": (primary_rows, primary_fields),
        "PLOT_DATA_FAMILIES.csv": (family_rows, descriptive_fields),
        "PLOT_DATA_CLEAN.csv": (clean_rows, descriptive_fields),
    }
    for name, (rows, fields) in outputs.items():
        _write_csv(out_dir / name, rows, fields)
    (out_dir / "FINAL_REPORT.md").write_text(report, encoding="utf-8")
    chart_names = _render_figures(out_dir, curve_rows, primary_rows, family_rows, clean_rows)
    return [*outputs.keys(), "FINAL_REPORT.md", *chart_names]


def _check_budget(start: float, cap_seconds: float) -> None:
    if time.perf_counter() - start > cap_seconds:
        raise AnalysisFailure(f"M3 analysis exceeded its registered {cap_seconds:g} second cap")


def _validate_requested_analysis_cap(cap_seconds: float) -> float:
    if isinstance(cap_seconds, bool) or not isinstance(cap_seconds, (int, float)):
        raise AnalysisFailure("M3 analysis cap must be a finite positive number of seconds")
    cap = float(cap_seconds)
    if not math.isfinite(cap) or cap <= 0:
        raise AnalysisFailure("M3 analysis cap must be a finite positive number of seconds")
    if cap > ANALYSIS_BUDGET_SECONDS:
        raise AnalysisFailure(
            f"M3 analysis cap {cap:g}s exceeds the registered {ANALYSIS_BUDGET_SECONDS}s maximum"
        )
    return cap


def _capture_m3_entry_snapshot(integrity: Any) -> tuple[Mapping[str, Any], dict[str, str]]:
    """Capture only hash-bound M0 fields and the M3 release file hash.

    The release payload also contains descriptive metadata (scope, budget, and
    authorization counts).  Passing that full object to the verifier would
    blur the verifier's frozen-hash contract, so formal M3 keeps a narrow,
    explicit snapshot instead.
    """
    stage = "M3_ANALYSIS"
    payload = integrity.require_stage_release(stage)
    if not isinstance(payload, Mapping):
        raise AnalysisFailure("M3 integrity gate returned a non-mapping release payload")
    frozen_keys = tuple(integrity.FROZEN_HASH_KEYS)
    if len(set(frozen_keys)) != len(frozen_keys):
        raise AnalysisFailure("M3 integrity gate exposes duplicate frozen hash keys")
    snapshot_keys = (*frozen_keys, "release_file_sha256")
    missing = [key for key in snapshot_keys if not isinstance(payload.get(key), str) or not payload.get(key)]
    if missing:
        raise AnalysisFailure(f"M3 release payload lacks frozen snapshot hashes: {missing}")
    return payload, {key: str(payload[key]) for key in snapshot_keys}


def _verify_m3_snapshot_unchanged(integrity: Any, snapshot: Mapping[str, str]) -> Mapping[str, Any]:
    stage = "M3_ANALYSIS"
    result = integrity.verify_stage_unchanged(stage, snapshot)
    if not isinstance(result, Mapping):
        raise AnalysisFailure("M3 integrity exit gate returned a non-mapping result")
    if result.get("stage") != stage or result.get("m0_hashes_unchanged") is not True:
        raise AnalysisFailure("M3 integrity exit gate did not confirm unchanged frozen sources")
    if result.get("release_file_sha256") != snapshot.get("release_file_sha256"):
        raise AnalysisFailure("M3 integrity exit gate returned a different ROOT release hash")
    return result


def _validate_release_analysis_cap(payload: Mapping[str, Any], requested_cap: float) -> float:
    release_cap = payload.get("analysis_budget_cap_seconds")
    if isinstance(release_cap, bool) or not isinstance(release_cap, (int, float)):
        raise AnalysisFailure("M3 ROOT release lacks a numeric analysis_budget_cap_seconds")
    release_cap_value = float(release_cap)
    if not math.isfinite(release_cap_value) or release_cap_value != ANALYSIS_BUDGET_SECONDS:
        raise AnalysisFailure(
            "M3 ROOT analysis budget must exactly match the registered "
            f"{ANALYSIS_BUDGET_SECONDS}s cap"
        )
    if requested_cap > release_cap_value:
        raise AnalysisFailure(
            f"requested M3 cap {requested_cap:g}s exceeds the ROOT release cap {release_cap_value:g}s"
        )
    return release_cap_value


def run_formal_analysis(
    input_path: Path = EXPECTED_INPUT,
    output_dir: Path | None = None,
    *, replicates: int = BOOTSTRAP_REPLICATES,
    cap_seconds: float = ANALYSIS_BUDGET_SECONDS,
) -> dict[str, Any]:
    """Run only after M3 release; check frozen sources and merged hash at entry/exit."""
    if replicates != BOOTSTRAP_REPLICATES:
        raise AnalysisFailure("formal M3 requires exactly 100,000 registered bootstrap replicates")
    cap_seconds = _validate_requested_analysis_cap(cap_seconds)
    start = time.perf_counter()
    deadline = start + cap_seconds
    input_path = input_path.resolve()
    target_dir = (output_dir or EXPECTED_INPUT.parent).resolve()
    if input_path != EXPECTED_INPUT.resolve():
        raise AnalysisFailure(f"formal analysis input must be the released merged bank: {EXPECTED_INPUT}")
    if target_dir != EXPECTED_INPUT.parent.resolve():
        raise AnalysisFailure(f"formal analysis output directory must be: {EXPECTED_INPUT.parent}")
    analyzer_sha_entry = _sha256_file(SCRIPT)

    from R.experiments import fixed_horizon_integrity as integrity
    entry_payload, entry_snapshot = _capture_m3_entry_snapshot(integrity)
    registered_cap = _validate_release_analysis_cap(entry_payload, cap_seconds)
    input_sha_entry = _sha256_file(input_path)
    rows = _read_csv(input_path)
    registry = load_layout_registry()
    bank = validate_rows(rows, registry)
    raw_scopes = scope_layout_values(bank.raw)
    rn_scopes = scope_layout_values(bank.rn)
    indices = make_bootstrap_indices(
        bank.layouts, bank.geometry_by_layout, replicates=replicates,
        seed=BOOTSTRAP_SEED, batch_size=BOOTSTRAP_BATCH_SIZE,
    )

    scope_boots: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for scope_name in raw_scopes:
        _check_budget(start, cap_seconds)
        scope_boots[scope_name] = bootstrap_scope_pair(
            raw_scopes[scope_name], rn_scopes[scope_name], indices,
            batch_size=BOOTSTRAP_BATCH_SIZE, deadline=deadline,
        )
    noisy_raw_point = geometry_mean(raw_scopes["noisy"], bank.geometry_by_layout, bank.layouts)
    noisy_rn_point = geometry_mean(rn_scopes["noisy"], bank.geometry_by_layout, bank.layouts)
    noisy_raw_boot, noisy_rn_boot = scope_boots["noisy"]
    primary_rows, overall_rows = primary_comparisons(noisy_raw_point, noisy_raw_boot, replicates=replicates)
    fixed_h20_rows = _fixed_h_vs_h20_rows(noisy_raw_point, noisy_rn_point, noisy_raw_boot, noisy_rn_boot, replicates=replicates)
    best_rows = _best_fixed_rows(noisy_raw_point, noisy_raw_boot, primary_rows)
    curve_rows = _fixed_curve_rows(noisy_raw_point, noisy_raw_boot, replicates)

    input_sha_exit_before = _sha256_file(input_path)
    analyzer_sha_exit_before = _sha256_file(SCRIPT)
    if input_sha_exit_before != input_sha_entry or analyzer_sha_exit_before != analyzer_sha_entry:
        raise AnalysisFailure("merged raw or analyzer source changed during M3 analysis")

    _check_budget(start, cap_seconds)
    with tempfile.TemporaryDirectory(prefix=".fixed_horizon_analysis_", dir=str(target_dir)) as temp_name:
        temp_dir = Path(temp_name)
        staged_names = _write_analysis_outputs(
            temp_dir, bank, raw_scopes, rn_scopes, scope_boots, primary_rows, overall_rows,
            fixed_h20_rows, best_rows, curve_rows, input_sha256=input_sha_entry,
            analyzer_sha256=analyzer_sha_entry, replicates=replicates,
        )
        _check_budget(start, cap_seconds)
        unchanged = _verify_m3_snapshot_unchanged(integrity, entry_snapshot)
        if _sha256_file(input_path) != input_sha_entry or _sha256_file(SCRIPT) != analyzer_sha_entry:
            raise AnalysisFailure("merged raw or analyzer source changed before M3 output commit")
        for name in staged_names:
            destination = target_dir / name
            if destination.exists():
                raise AnalysisFailure(f"refusing to overwrite existing M3 artifact: {destination}")
            os.replace(temp_dir / name, destination)
        _check_budget(start, cap_seconds)
        # Exit gate rechecks M0 bindings, ROOT M3 release, M2 bank/audit receipts,
        # and the full merged CSV hash after all analysis outputs have been written.
        exit_snapshot = _verify_m3_snapshot_unchanged(integrity, entry_snapshot)
        if _sha256_file(input_path) != input_sha_entry or _sha256_file(SCRIPT) != analyzer_sha_entry:
            raise AnalysisFailure("merged raw or analyzer source changed at M3 exit")
    return {
        "status": "PASS_M3_ANALYSIS_COMPLETE_PENDING_ROOT_REVIEW",
        "input_sha256": input_sha_entry, "analyzer_sha256": analyzer_sha_entry,
        "row_count": len(rows), "layout_count": len(bank.layouts),
        "forecast_pair_count": bank.forecast_pair_count,
        "bootstrap_replicates": replicates, "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_rng": BOOTSTRAP_RNG, "bootstrap_batch_size": BOOTSTRAP_BATCH_SIZE,
        "bootstrap_layout_order": BOOTSTRAP_LAYOUT_ORDER,
        "bootstrap_rng_call_order": BOOTSTRAP_RNG_CALL_ORDER,
        "bootstrap_quantile_method": BOOTSTRAP_QUANTILE_METHOD,
        "main_comparison_count": MAIN_COMPARISONS, "outputs": staged_names,
        "registered_analysis_budget_cap_seconds": registered_cap,
        "effective_run_cap_seconds": cap_seconds,
        "entry_snapshot": entry_snapshot, "exit_snapshot": exit_snapshot,
        "run_elapsed_seconds": time.perf_counter() - start,
    }


def build_synthetic_layout_registry() -> dict[str, tuple[str, int]]:
    return {
        f"SYN-G{1 if i < 10 else 2}-{i:02d}": ("G1" if i < 10 else "G2", 100 if i < 10 else 200)
        for i in range(EXPECTED_LAYOUTS)
    }


def build_synthetic_rows() -> list[dict[str, str]]:
    """A fully synthetic arithmetic matrix; it reads no workspace data."""
    registry = build_synthetic_layout_registry()
    family_strength_means = {
        "subset_shuffle": {"p=0.1": 10, "p=0.3": 30, "p=0.5": 50},
        "time_window_shuffle": {"B=2": 210, "B=5": 230, "B=10": 250, "B=20": 270},
    }
    method_delta = {"RIDGE": -20, "FIXED_H0": 0, "FIXED_H2": 6, "FIXED_H5": 10, "FIXED_H10": 14, "FIXED_H20": 18}
    rows: list[dict[str, str]] = []
    for li, (fp, (geometry, n)) in enumerate(registry.items()):
        base = (100 if geometry == "G1" else 200) + li
        for ci, spec in enumerate(CONDITIONS):
            cid = str(spec["condition_id"])
            if spec["family"] == "CLEAN":
                condition_value = 400
            else:
                condition_value = family_strength_means[str(spec["family"])][str(spec["strength"])]
                condition_value += int(spec["repeat"]) * (8 if spec["family"] == "time_window_shuffle" else 4)
            for si, solver in enumerate(SOLVERS):
                for method in POLICIES:
                    value = base + condition_value + 3 * si + method_delta[method]
                    f_hash = f"SYN-F-{li}-{ci}"
                    seed = "" if spec["family"] == "CLEAN" else f"SYN-SEED-{li}-{ci}"
                    rows.append({
                        "trajectory_key": f"{fp}|{cid}|{solver}|{method}",
                        "layout_fingerprint": fp, "geometry": geometry,
                        "condition_id": cid, "family": str(spec["family"]),
                        "strength": str(spec["strength"]),
                        "p": "" if spec["p"] is None else str(spec["p"]),
                        "window_size": "" if spec["window_size"] is None else str(spec["window_size"]),
                        "repeat": str(spec["repeat"]), "algorithm": solver, "method": method,
                        "R_raw": str(value), "N": str(n), "R_over_N": str(value / n),
                        "forecast_seed_evaluator_only": seed, "F_hash": f_hash,
                        "complete": "true", "decision_count": "1", "wall_seconds": "0.0",
                        "actor_time_seconds": "0.0", "eg_cached_moves": "0",
                        "source_origin": "synthetic", "source_stage": "M0_SELF_TEST",
                    })
    return rows


def _run_self_test(receipt_out: Path | None = None) -> tuple[int, dict[str, Any], str]:
    start = time.perf_counter()
    if not TEST_PATH.is_file():
        raise AnalysisFailure(f"independent test file is missing: {TEST_PATH}")
    spec = importlib.util.spec_from_file_location("test_fixed_horizon_analysis", TEST_PATH)
    if spec is None or spec.loader is None:
        raise AnalysisFailure("cannot load independent synthetic analysis tests")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    suite = unittest.defaultTestLoader.loadTestsFromModule(module)
    stream = io.StringIO()
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    elapsed = time.perf_counter() - start
    test_count = int(result.testsRun)
    failed = len(result.failures) + len(result.errors)
    passed = test_count - failed
    report = {
        "status": "PASS" if failed == 0 and test_count > 0 else "FAIL",
        "tests_passed": passed,
        "test_count": test_count,
        "failed": failed,
        "synthetic_rows_checked": EXPECTED_ROWS if failed == 0 else 0,
        "real_data_read": False,
        "real_trajectory_count": 0,
        "fit_or_tuning": False,
        "elapsed_seconds": elapsed,
        "command": " ".join(sys.argv),
        "actual_exit_code": 0 if failed == 0 and test_count > 0 else 1,
    }
    if receipt_out is not None:
        receipt_out.parent.mkdir(parents=True, exist_ok=True)
        receipt_out.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return report["actual_exit_code"], report, stream.getvalue()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true", help="run only independent synthetic checks")
    parser.add_argument("--receipt-out", type=Path, help="write self-test details JSON (synthetic-only mode)")
    parser.add_argument("--input", type=Path, default=EXPECTED_INPUT, help="released complete merged raw CSV")
    parser.add_argument("--output-dir", type=Path, default=EXPECTED_INPUT.parent, help="M3 analysis output directory")
    parser.add_argument("--bootstrap-replicates", type=int, default=BOOTSTRAP_REPLICATES)
    parser.add_argument("--cap-seconds", type=float, default=ANALYSIS_BUDGET_SECONDS,
                        help="M3 wall-time cap; the default and maximum are the registered 1800 seconds")
    args = parser.parse_args(argv)
    if args.self_test:
        exit_code, report, detail = _run_self_test(args.receipt_out)
        if detail:
            sys.stdout.write(detail)
        sys.stdout.write(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
        return exit_code
    try:
        result = run_formal_analysis(
            args.input, args.output_dir, replicates=args.bootstrap_replicates,
            cap_seconds=args.cap_seconds,
        )
    except Exception as exc:
        sys.stderr.write(f"FAIL_CLOSED: {type(exc).__name__}: {exc}\n")
        return 2
    sys.stdout.write(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
