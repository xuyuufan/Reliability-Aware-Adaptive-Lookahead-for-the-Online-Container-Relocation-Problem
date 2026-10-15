"""Evaluator-only pair labels and the frozen PLAN-049 Plackett-Luce fit."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence

import numpy as np

from R.scrp.r_order_public import (
    PlackettLuceRates,
    PublicHistorySummary,
    UNKNOWN_RANK,
)


L2 = 1e-3
MAX_ITER = 500
GTOL = 1e-6
FTOL = 1e-12
_RATE_FEATURE_INDEX = {"intercept": 0, "q": 1, "displacement": 2, "coverage": 3, "retrieval_fraction": 4}


@dataclass(frozen=True)
class PairwiseLabel:
    """One eligible public pair; private identities never leave label creation."""

    rank_first: int
    rank_second: int
    summary: PublicHistorySummary
    outcome_first_precedes_truth: int
    behavior_algorithm: str
    geometry: str
    layout_fingerprint: str
    regime: str
    strength: str
    repeat: int
    capture_id: str
    condition_id: str
    weight: float = 0.0


@dataclass(frozen=True)
class PairwiseFit:
    model: PlackettLuceRates
    objective: float
    gradient_inf_norm: float
    iterations: int
    converged: bool
    row_count: int
    weight_sum: float
    active_feature_indices: tuple[int, ...]
    scipy_message: str


@dataclass(frozen=True)
class QOnlyFit:
    intercept: float
    q_slope: float
    objective: float
    gradient_inf_norm: float
    iterations: int
    converged: bool
    row_count: int
    weight_sum: float
    scipy_message: str


def _future_rank(public_rank: int) -> int | None:
    if public_rank == 1:
        return None
    if 2 <= public_rank <= 21:
        return public_rank - 1
    if public_rank == UNKNOWN_RANK:
        return UNKNOWN_RANK
    raise ValueError("public rank outside current/future/sentinel range")


def extract_pairwise_labels(
    public_frame: object,
    private_physical_stacks: Sequence[Sequence[int]],
    true_retrieval_sequence: Sequence[int],
    *,
    behavior_algorithm: str,
    geometry: str,
    layout_fingerprint: str,
    regime: str,
    strength: str,
    repeat: int,
    capture_id: str,
    condition_id: str,
) -> tuple[PairwiseLabel, ...]:
    """Create training-only pairs from public ranks and evaluator truth.

    The evaluator uses private physical IDs solely to attach the true order
    label to public-rank pairs. The returned rows contain no ID, family, raw
    layout, seed, or private forecast field accepted by the fitted model.
    """
    if behavior_algorithm not in ("EM", "GP", "ERI", "EG"):
        raise ValueError("unknown behavior algorithm")
    observation = getattr(public_frame, "observation")
    ranks = np.asarray(observation.rank_tensor, dtype=np.int32)
    occupied = np.asarray(observation.occupied_mask, dtype=np.bool_)
    if len(private_physical_stacks) != ranks.shape[0]:
        raise ValueError("private physical stack count differs from public frame")
    if any(len(private_physical_stacks[index]) != int(occupied[index].sum()) for index in range(ranks.shape[0])):
        raise ValueError("private physical stack heights differ from public occupancy")
    truth = tuple(int(value) for value in true_retrieval_sequence)
    if len(truth) != len(set(truth)):
        raise ValueError("true retrieval sequence must be unique")
    truth_position = {value: index for index, value in enumerate(truth)}

    objects: list[tuple[int, int]] = []
    for stack in range(ranks.shape[0]):
        height = int(occupied[stack].sum())
        for tier in range(height):
            public_rank = int(ranks[stack, tier])
            if public_rank == 1:
                continue
            if public_rank != UNKNOWN_RANK and not 2 <= public_rank <= 21:
                raise ValueError("occupied public ranks must be 1..21 or 22")
            token = int(private_physical_stacks[stack][tier])
            if token not in truth_position:
                raise ValueError("private live token is missing from the true retrieval sequence")
            objects.append((public_rank, truth_position[token]))

    rows: list[PairwiseLabel] = []
    for first_index in range(len(objects)):
        rank_left, truth_left = objects[first_index]
        future_left = _future_rank(rank_left)
        for second_index in range(first_index + 1, len(objects)):
            rank_right, truth_right = objects[second_index]
            future_right = _future_rank(rank_right)
            # Current target and unknown-unknown pairs have no learned relation.
            if future_left is None or future_right is None:
                continue
            if future_left == UNKNOWN_RANK and future_right == UNKNOWN_RANK:
                continue
            if future_left < future_right:
                rank_first, rank_second = future_left, future_right
                truth_first, truth_second = truth_left, truth_right
            else:
                rank_first, rank_second = future_right, future_left
                truth_first, truth_second = truth_right, truth_left
            if rank_first == rank_second:
                raise ValueError("eligible public pair unexpectedly has equal normalized rank")
            rows.append(
                PairwiseLabel(
                    rank_first=rank_first,
                    rank_second=rank_second,
                    summary=getattr(observation, "r_order_summary"),
                    outcome_first_precedes_truth=int(truth_first < truth_second),
                    behavior_algorithm=behavior_algorithm,
                    geometry=geometry,
                    layout_fingerprint=layout_fingerprint,
                    regime=regime,
                    strength=strength,
                    repeat=int(repeat),
                    capture_id=capture_id,
                    condition_id=condition_id,
                )
            )
    return tuple(rows)


def normalized_hierarchical_weights(rows: Sequence[PairwiseLabel]) -> tuple[PairwiseLabel, ...]:
    """Apply equal geometry/layout/regime/strength/repeat/capture/pair weights."""
    if not rows:
        raise ValueError("cannot weight an empty pair table")
    groups: dict[tuple[Any, ...], list[int]] = {}
    fields = (
        "geometry", "layout_fingerprint", "regime", "strength", "repeat",
        "condition_id", "capture_id",
    )
    for index, row in enumerate(rows):
        if row.geometry not in ("G1", "G2") or row.regime not in ("DP", "TWS", "CLEAN"):
            raise ValueError("pair metadata is outside the frozen aggregation strata")
        if row.rank_first == row.rank_second or row.outcome_first_precedes_truth not in (0, 1):
            raise ValueError("invalid pair ranks or binary truth outcome")
        key = tuple(getattr(row, field) for field in fields)
        groups.setdefault(key, []).append(index)

    # Build the hierarchy from deepest complete keys. Each supported child gets
    # equal weight; within a capture, each eligible unordered pair is equal.
    capture_keys = list(groups)
    by_condition: dict[tuple[Any, ...], list[tuple[Any, ...]]] = {}
    by_repeat: dict[tuple[Any, ...], list[tuple[Any, ...]]] = {}
    by_strength: dict[tuple[Any, ...], list[tuple[Any, ...]]] = {}
    by_regime: dict[tuple[Any, ...], list[tuple[Any, ...]]] = {}
    by_layout: dict[tuple[Any, ...], list[tuple[Any, ...]]] = {}
    by_geometry: dict[tuple[Any, ...], list[tuple[Any, ...]]] = {}
    for key in capture_keys:
        geometry, layout, regime, strength, repeat, condition, capture = key
        by_condition.setdefault((geometry, layout, regime, strength, repeat, condition), []).append(key)
        by_repeat.setdefault((geometry, layout, regime, strength, repeat), []).append((geometry, layout, regime, strength, repeat, condition))
        by_strength.setdefault((geometry, layout, regime, strength), []).append((geometry, layout, regime, strength, repeat))
        by_regime.setdefault((geometry, layout, regime), []).append((geometry, layout, regime, strength))
        by_layout.setdefault((geometry, layout), []).append((geometry, layout, regime))
        by_geometry.setdefault((geometry,), []).append((geometry, layout))

    def child_weights(
        children: Mapping[tuple[Any, ...], Sequence[tuple[Any, ...]]],
        parent: tuple[Any, ...],
        parent_weight: float,
    ) -> dict[tuple[Any, ...], float]:
        unique = sorted(set(children[parent]), key=repr)
        return {child: parent_weight / len(unique) for child in unique}

    geom_values = sorted({row.geometry for row in rows})
    weight_by_capture: dict[tuple[Any, ...], float] = {}
    for geometry in geom_values:
        layout_weights = child_weights(by_geometry, (geometry,), 1.0 / len(geom_values))
        for layout_key, layout_weight in layout_weights.items():
            regime_weights = child_weights(by_layout, layout_key, layout_weight)
            for regime_key, regime_weight in regime_weights.items():
                strength_weights = child_weights(by_regime, regime_key, regime_weight)
                for strength_key, strength_weight in strength_weights.items():
                    repeat_weights = child_weights(by_strength, strength_key, strength_weight)
                    for repeat_key, repeat_weight in repeat_weights.items():
                        condition_weights = child_weights(by_repeat, repeat_key, repeat_weight)
                        for condition_key, condition_weight in condition_weights.items():
                            captures = sorted(set(by_condition[condition_key]), key=repr)
                            capture_weight = condition_weight / len(captures)
                            for capture_key in captures:
                                pair_indices = groups[capture_key]
                                weight_by_capture[capture_key] = capture_weight / len(pair_indices)

    # Assign every original pair its capture-level mass divided by pair count.
    result: list[PairwiseLabel] = []
    for row in rows:
        key = tuple(getattr(row, field) for field in fields)
        result.append(PairwiseLabel(**{**row.__dict__, "weight": weight_by_capture[key]}))
    total = sum(row.weight for row in result)
    if not math.isfinite(total) or not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(f"hierarchical pair weights do not sum to one ({total})")
    return tuple(result)


def _active_indices(no_history: bool) -> tuple[int, ...]:
    return (0, 4) if no_history else (0, 1, 2, 3, 4)


def _model_logits(rows: Sequence[PairwiseLabel], flat: np.ndarray, active: tuple[int, ...]) -> np.ndarray:
    full = np.zeros((20, 5), dtype=np.float64)
    full[:, active] = flat.reshape(20, len(active))
    output = np.empty(len(rows), dtype=np.float64)
    for index, row in enumerate(rows):
        x = np.asarray((1.0, row.summary.q, row.summary.displacement, row.summary.coverage, row.summary.retrieval_fraction), dtype=np.float64)
        first_log = 0.0 if row.rank_first == UNKNOWN_RANK else float(full[row.rank_first - 1] @ x)
        second_log = 0.0 if row.rank_second == UNKNOWN_RANK else float(full[row.rank_second - 1] @ x)
        output[index] = first_log - second_log
    return output


def _prepare_fit_arrays(
    rows: Sequence[PairwiseLabel], active: tuple[int, ...]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build compact arrays once; optimizer iterations do no Python row loops."""
    weighted = normalized_hierarchical_weights(rows)
    first = np.fromiter((row.rank_first for row in weighted), dtype=np.int16, count=len(weighted))
    second = np.fromiter((row.rank_second for row in weighted), dtype=np.int16, count=len(weighted))
    y = np.fromiter((row.outcome_first_precedes_truth for row in weighted), dtype=np.float64, count=len(weighted))
    sample_weight = np.fromiter((row.weight for row in weighted), dtype=np.float64, count=len(weighted))
    features = np.asarray(
        [
            (1.0, row.summary.q, row.summary.displacement, row.summary.coverage, row.summary.retrieval_fraction)
            for row in weighted
        ],
        dtype=np.float64,
    )
    if features.shape != (len(weighted), 5) or not np.isfinite(features).all():
        raise ValueError("pair feature matrix must be finite [n,5]")
    return first, second, y, sample_weight, features


def _logits_from_arrays(
    first: np.ndarray,
    second: np.ndarray,
    features: np.ndarray,
    flat: np.ndarray,
    active: tuple[int, ...],
) -> np.ndarray:
    coefficients = np.zeros((20, 5), dtype=np.float64)
    coefficients[:, active] = flat.reshape(20, len(active))
    output = np.zeros(len(first), dtype=np.float64)
    known_first = first != UNKNOWN_RANK
    known_second = second != UNKNOWN_RANK
    output[known_first] += np.einsum(
        "ij,ij->i", coefficients[first[known_first] - 1], features[known_first], optimize=False
    )
    output[known_second] -= np.einsum(
        "ij,ij->i", coefficients[second[known_second] - 1], features[known_second], optimize=False
    )
    return output


def _pl_objective_arrays(
    theta: np.ndarray,
    first: np.ndarray,
    second: np.ndarray,
    y: np.ndarray,
    sample_weight: np.ndarray,
    features: np.ndarray,
    active: tuple[int, ...],
) -> tuple[float, np.ndarray]:
    """Compute the exact frozen loss and gradient without an n-by-100 matrix."""
    logits = _logits_from_arrays(first, second, features, theta, active)
    probabilities = 1.0 / (1.0 + np.exp(-np.clip(logits, -700.0, 700.0)))
    loss = np.logaddexp(0.0, logits) - y * logits
    value = float(np.dot(sample_weight, loss) + 0.5 * L2 * np.dot(theta, theta))
    errors = sample_weight * (probabilities - y)
    gradient_full = np.zeros((20, 5), dtype=np.float64)
    known_first = first != UNKNOWN_RANK
    known_second = second != UNKNOWN_RANK
    np.add.at(
        gradient_full,
        first[known_first] - 1,
        errors[known_first, None] * features[known_first],
    )
    np.add.at(
        gradient_full,
        second[known_second] - 1,
        -errors[known_second, None] * features[known_second],
    )
    gradient = gradient_full[:, active].reshape(-1) + L2 * theta
    return value, gradient


def fit_plackett_luce(
    rows: Sequence[PairwiseLabel], *, no_history: bool = False, model_id: str = "r_order"
) -> PairwiseFit:
    """Fit fixed weighted pairwise logistic rates once with L-BFGS-B."""
    if not rows:
        raise ValueError("cannot fit Plackett-Luce rates without eligible pairs")
    from scipy.optimize import minimize

    active = _active_indices(no_history)
    ranks = np.arange(1, 21, dtype=np.float64)
    initial_full = np.zeros((20, 5), dtype=np.float64)
    initial_full[:, 0] = np.log(21.0 / ranks)
    initial = initial_full[:, active].reshape(-1)
    first, second, y, sample_weight, features = _prepare_fit_arrays(rows, active)

    def objective(theta: np.ndarray) -> tuple[float, np.ndarray]:
        return _pl_objective_arrays(
            theta, first, second, y, sample_weight, features, active
        )

    result = minimize(
        objective,
        initial,
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": MAX_ITER, "gtol": GTOL, "ftol": FTOL},
    )
    value, gradient = objective(np.asarray(result.x, dtype=np.float64))
    coefficients = np.zeros((20, 5), dtype=np.float64)
    coefficients[:, active] = np.asarray(result.x, dtype=np.float64).reshape(20, len(active))
    gradient_inf = float(np.max(np.abs(gradient)))
    converged = bool(result.success and np.isfinite(value) and np.isfinite(coefficients).all() and gradient_inf <= GTOL)
    model = PlackettLuceRates(coefficients, active, model_id)
    return PairwiseFit(
        model=model,
        objective=float(value),
        gradient_inf_norm=gradient_inf,
        iterations=int(getattr(result, "nit", 0)),
        converged=converged,
        row_count=len(rows),
        weight_sum=1.0,
        active_feature_indices=active,
        scipy_message=str(result.message),
    )


def fit_q_only(rows: Sequence[PairwiseLabel]) -> QOnlyFit:
    """Fit the registered independent two-parameter q-only pair correctness model."""
    if not rows:
        raise ValueError("cannot fit q-only comparator without eligible pairs")
    from scipy.optimize import minimize

    weighted = normalized_hierarchical_weights(rows)
    y = np.asarray([row.outcome_first_precedes_truth for row in weighted], dtype=np.float64)
    weights = np.asarray([row.weight for row in weighted], dtype=np.float64)
    q = np.asarray([row.summary.q - 0.5 for row in weighted], dtype=np.float64)
    x = np.column_stack((np.ones_like(q), q))

    def objective(theta: np.ndarray) -> tuple[float, np.ndarray]:
        logits = x @ theta
        probabilities = 1.0 / (1.0 + np.exp(-np.clip(logits, -700.0, 700.0)))
        loss = np.logaddexp(0.0, logits) - y * logits
        value = float(np.dot(weights, loss) + 0.5 * L2 * np.dot(theta, theta))
        gradient = x.T @ (weights * (probabilities - y)) + L2 * theta
        return value, gradient

    result = minimize(
        objective,
        np.zeros(2, dtype=np.float64),
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": MAX_ITER, "gtol": GTOL, "ftol": FTOL},
    )
    value, gradient = objective(np.asarray(result.x, dtype=np.float64))
    gradient_inf = float(np.max(np.abs(gradient)))
    theta = np.asarray(result.x, dtype=np.float64)
    converged = bool(result.success and np.isfinite(value) and np.isfinite(theta).all() and gradient_inf <= GTOL)
    return QOnlyFit(
        intercept=float(theta[0]),
        q_slope=float(theta[1]),
        objective=float(value),
        gradient_inf_norm=gradient_inf,
        iterations=int(getattr(result, "nit", 0)),
        converged=converged,
        row_count=len(weighted),
        weight_sum=float(sum(row.weight for row in weighted)),
        scipy_message=str(result.message),
    )


def predict_pair_probability(model: PlackettLuceRates, row: PairwiseLabel) -> float:
    first = model.log_rate(row.rank_first, row.summary)
    second = model.log_rate(row.rank_second, row.summary)
    logit = first - second
    if logit >= 0:
        return float(1.0 / (1.0 + math.exp(-logit)))
    exp_logit = math.exp(logit)
    return float(exp_logit / (1.0 + exp_logit))


def predict_q_only_probability(model: QOnlyFit, row: PairwiseLabel) -> float:
    logit = model.intercept + model.q_slope * (row.summary.q - 0.5)
    if logit >= 0:
        return float(1.0 / (1.0 + math.exp(-logit)))
    exp_logit = math.exp(logit)
    return float(exp_logit / (1.0 + exp_logit))


def weighted_brier(
    rows: Sequence[PairwiseLabel], probability_fn: Any, *, filter_behavior: str | None = None
) -> float:
    selected = [row for row in rows if filter_behavior is None or row.behavior_algorithm == filter_behavior]
    if not selected:
        raise ValueError("Brier evaluation has no supported rows")
    weighted = normalized_hierarchical_weights(selected)
    score = sum(
        row.weight * (float(probability_fn(row)) - row.outcome_first_precedes_truth) ** 2
        for row in weighted
    )
    if not math.isfinite(score):
        raise ValueError("Brier score is nonfinite")
    return float(score)


def relation_calibration_tables(rows: Sequence[PairwiseLabel], probability_fn: Any) -> dict[str, Any]:
    """Return weight-preserving calibration groups and ten-bin reliability rows.

    The global pair weights are exactly those used by ``weighted_brier``. Each
    partition row reports its share of that global mass, so group Briers can be
    recombined into the same pooled Brier without introducing a new weighting
    rule. This diagnostic does not fit or alter a model.
    """
    weighted = normalized_hierarchical_weights(rows)
    scored: list[dict[str, Any]] = []
    for row in weighted:
        probability = float(probability_fn(row))
        if not math.isfinite(probability) or probability < 0.0 or probability > 1.0:
            raise ValueError("calibration probability must be finite and within [0, 1]")
        if row.rank_first == UNKNOWN_RANK or row.rank_second == UNKNOWN_RANK:
            rank_distance = "known-tail"
        else:
            distance = abs(int(row.rank_second) - int(row.rank_first))
            rank_distance = (
                "1-2" if distance <= 2 else
                "3-5" if distance <= 5 else
                "6-10" if distance <= 10 else
                "11-19"
            )
        censor_history = any(float(value) == 1.0 for value in row.summary.records)
        fraction = float(row.summary.retrieval_fraction)
        retrieval_phase = "early" if fraction < (1.0 / 3.0) else "mid" if fraction < (2.0 / 3.0) else "late"
        scored.append({
            "rank_first": int(row.rank_first),
            "rank_second": int(row.rank_second),
            "outcome_first_precedes_truth": int(row.outcome_first_precedes_truth),
            "probability": probability,
            "weight": float(row.weight),
            "geometry": row.geometry,
            "layout_fingerprint": row.layout_fingerprint,
            "regime": row.regime,
            "strength": row.strength,
            "repeat": int(row.repeat),
            "condition_id": row.condition_id,
            "capture_id": row.capture_id,
            "q": float(row.summary.q),
            "displacement": float(row.summary.displacement),
            "coverage": float(row.summary.coverage),
            "retrieval_fraction": fraction,
            "history_records": [float(value) for value in row.summary.records],
            "noise_family": row.regime,
            "rank_distance": rank_distance,
            "censor_history": censor_history,
            "retrieval_phase": retrieval_phase,
            "probability_bin": min(9, int(probability * 10.0)),
        })

    def aggregate(group_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        mass = float(sum(float(row["weight"]) for row in group_rows))
        count = len(group_rows)
        if not count or mass <= 0.0:
            return {
                "row_count": 0, "layout_count": 0, "capture_count": 0, "weight_mass": 0.0,
                "weighted_brier": None, "mean_prediction": None,
                "observed_rate": None, "calibration_gap": None,
            }
        brier = sum(float(row["weight"]) * (float(row["probability"]) - int(row["outcome_first_precedes_truth"])) ** 2
                    for row in group_rows) / mass
        mean_prediction = sum(float(row["weight"]) * float(row["probability"]) for row in group_rows) / mass
        observed_rate = sum(float(row["weight"]) * int(row["outcome_first_precedes_truth"])
                            for row in group_rows) / mass
        captures = {
            (row["geometry"], row["layout_fingerprint"], row["condition_id"], row["capture_id"])
            for row in group_rows
        }
        layouts = {(row["geometry"], row["layout_fingerprint"]) for row in group_rows}
        return {
            "row_count": count, "layout_count": len(layouts), "capture_count": len(captures), "weight_mass": mass,
            "weighted_brier": float(brier), "mean_prediction": float(mean_prediction),
            "observed_rate": float(observed_rate),
            "calibration_gap": float(mean_prediction - observed_rate),
        }

    pooled = aggregate(scored)
    dimensions = (
        ("noise_family", ("DP", "TWS", "CLEAN")),
        ("rank_distance", ("1-2", "3-5", "6-10", "11-19", "known-tail")),
        ("censor_history", (False, True)),
        ("retrieval_phase", ("early", "mid", "late")),
    )
    group_metrics: list[dict[str, Any]] = []
    bin_metrics: list[dict[str, Any]] = []

    def add_partition(dimension: str, category: Any, selected: Sequence[Mapping[str, Any]]) -> None:
        group_metrics.append({"dimension": dimension, "category": category, **aggregate(selected)})
        for bin_index in range(10):
            in_bin = [row for row in selected if int(row["probability_bin"]) == bin_index]
            bin_metrics.append({
                "dimension": dimension, "category": category,
                "probability_bin": bin_index,
                "bin_lower": bin_index / 10.0, "bin_upper": (bin_index + 1) / 10.0,
                **aggregate(in_bin),
            })

    add_partition("overall", "all", scored)
    for dimension, categories in dimensions:
        for category in categories:
            add_partition(dimension, category, [row for row in scored if row[dimension] == category])
    return {
        "overall": pooled,
        "scored_rows": scored,
        "group_rows": group_metrics,
        "probability_bin_rows": bin_metrics,
        "weight_definition": "normalized_hierarchical_weights over the complete supplied pair table; each subgroup uses original weight mass",
        "probability_bin_definition": "ten equal-width bins [0,.1),...,[.9,1], with p=1 in bin 9",
    }


__all__ = (
    "FTOL", "GTOL", "L2", "MAX_ITER", "PairwiseFit", "PairwiseLabel", "QOnlyFit",
    "extract_pairwise_labels", "fit_plackett_luce", "fit_q_only",
    "normalized_hierarchical_weights", "predict_pair_probability",
    "predict_q_only_probability", "relation_calibration_tables", "weighted_brier",
)
