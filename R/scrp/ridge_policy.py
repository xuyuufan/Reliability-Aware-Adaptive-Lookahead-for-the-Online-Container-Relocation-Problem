"""Ridge-only actor adapter for six non-RIL solver proposals.

Adapted from the PLAN-051 FiveRPolicy Ridge branch, PLAN-052 Plan052Policy
R_UTILITY_RIDGE branch, and shared R-EG sequence commit logic. This module
contains no alternative R policy or RIL implementation. A caller supplies its
existing solver proposal callback; this is not a bank runner.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from R.scrp.ridge_public_features import HORIZONS
from R.scrp.universal_r_utility import (
    CandidateOrigin, CandidateSequence, NativeCandidatePool, ScoredSelection,
    _current_target_blocker_count, _validate_sequence, select_strictly_below_h20,
)

RIDGE_ALGORITHMS = ("EM", "GP", "ERI", "EG", "Chain", "Minmax")
_CRANE_TRAVEL_TIME = 1.2
_PICK_DROP_TIME = 30.0


def require_public_frame(frame: object) -> None:
    """Fail closed on missing crane scalars or actor-visible private fields."""
    required = (
        "observation", "legal_destinations", "max_tiers", "retrieval_count",
        "relocation_count", "crane_position", "crane_time",
    )
    missing = [name for name in required if not hasattr(frame, name)]
    if missing:
        raise ValueError(f"public frame is missing registered fields: {missing}")
    forbidden = (
        "environment", "provider", "forecast", "retrieval_sequence", "seed",
        "family", "p", "window_size", "layout_fingerprint", "private_trace",
    )
    present = [name for name in forbidden if hasattr(frame, name)]
    if present:
        raise ValueError(f"actor frame contains evaluator/private fields: {present}")
    obs = frame.observation
    observation_fields = set(getattr(obs, "__dataclass_fields__", {}))
    if observation_fields:
        allowed_observation_fields = {
            "rank_tensor", "occupied_mask", "legal_destination_mask",
            "history_hits", "q_prior", "q_t", "r_order_summary",
        }
        unexpected = sorted(observation_fields - allowed_observation_fields)
        if unexpected:
            raise ValueError(f"actor observation contains unregistered fields: {unexpected}")
    nested_private = [name for name in forbidden if hasattr(obs, name)]
    if nested_private:
        raise ValueError(f"actor observation contains evaluator/private fields: {nested_private}")
    ranks = np.asarray(obs.rank_tensor, dtype=np.int32)
    occupied = np.asarray(obs.occupied_mask, dtype=np.bool_)
    legal_mask = np.asarray(obs.legal_destination_mask, dtype=np.bool_)
    legal = tuple(int(value) for value in frame.legal_destinations)
    if ranks.ndim != 2 or ranks.shape != occupied.shape or ranks.shape[0] != legal_mask.shape[0]:
        raise ValueError("public rank, occupancy, and legal-mask shapes differ")
    if ranks.shape[1] != int(frame.max_tiers):
        raise ValueError("public rank width differs from max_tiers")
    if tuple(int(value) for value in np.flatnonzero(legal_mask)) != legal:
        raise ValueError("public legal destinations differ from their mask")
    if np.any(ranks[~occupied] != 0) or np.any(occupied & ((ranks < 1) | (ranks > 22))):
        raise ValueError("public occupancy/rank sentinel contract failed")
    if np.count_nonzero(occupied & (ranks == 1)) != 1:
        raise ValueError("public frame must contain exactly one current target")
    position = frame.crane_position
    if position is not None and (isinstance(position, bool) or int(position) != position or int(position) < 0):
        raise ValueError("crane_position must be None or a nonnegative public stack position")
    crane_time = float(frame.crane_time)
    if not math.isfinite(crane_time) or crane_time < 0:
        raise ValueError("crane_time must be finite and nonnegative")


@dataclass(frozen=True)
class RidgeScorer:
    """Frozen 74-coefficient float64 linear score."""

    coefficients: tuple[float, ...]

    def __call__(self, features: Sequence[float]) -> float:
        vector = np.asarray(features, dtype=np.float64)
        weights = np.asarray(self.coefficients, dtype=np.float64)
        if vector.shape != (74,) or weights.shape != (74,) or not np.isfinite(vector).all() or not np.isfinite(weights).all():
            raise ValueError("frozen PLAN-047 Ridge requires 74 finite features and coefficients")
        return float(np.dot(weights, vector))


def load_frozen_ridge(path: str | Path) -> RidgeScorer:
    """Load the frozen model; this function never fits or calibrates."""
    record = json.loads(Path(path).read_text(encoding="utf-8"))
    if (
        record.get("model_kind") != "weighted_float64_ridge"
        or record.get("intercept") is not False
        or record.get("standardized") is not False
        or float(record.get("ridge_lambda", -1)) != 1.0
        or len(record.get("coefficients", ())) != 74
    ):
        raise ValueError("PLAN-047 frozen Ridge artifact differs from its M1 contract")
    return RidgeScorer(tuple(float(value) for value in record["coefficients"]))


def select_ridge_sequence(
    public_frame: object,
    algorithm: str,
    proposals: Mapping[int, Sequence[int]],
    scorer: Callable[[np.ndarray], float],
) -> tuple[ScoredSelection, dict[str, Any]]:
    """Apply strict-below-H20 Ridge selection to five native proposals."""
    if algorithm not in RIDGE_ALGORITHMS:
        raise ValueError(f"algorithm must be one of {RIDGE_ALGORITHMS}")
    if set(proposals) != set(HORIZONS) or len(proposals) != len(HORIZONS):
        raise ValueError(f"{algorithm} must provide exactly one proposal for each horizon {HORIZONS}")
    by_sequence: dict[tuple[int, ...], list[CandidateOrigin]] = {}
    for horizon in HORIZONS:
        sequence = _validate_sequence(public_frame, proposals[horizon])
        expected_length = _current_target_blocker_count(public_frame) if algorithm == "EG" else 1
        if len(sequence) != expected_length:
            expected = "complete current blocker group" if algorithm == "EG" else "one destination"
            raise ValueError(f"{algorithm} proposal must contain {expected}")
        origin = CandidateOrigin(algorithm, int(horizon), f"{algorithm}:h{horizon}")
        by_sequence.setdefault(sequence, []).append(origin)
    anchor = tuple(int(value) for value in proposals[20])
    candidates = tuple(
        CandidateSequence(
            destinations=sequence,
            origins=tuple(sorted(origins, key=lambda row: (row.horizon, row.proposal_id))),
            is_h20_anchor=(sequence == anchor),
        )
        for sequence, origins in sorted(by_sequence.items())
    )
    if sum(bool(item.is_h20_anchor) for item in candidates) != 1:
        raise AssertionError("R-Utility pool lost its explicit H20 sequence anchor")
    pool = NativeCandidatePool(algorithm, candidates, anchor)
    selection = select_strictly_below_h20(public_frame, pool, scorer)
    score_by_sequence = dict(selection.scored_sequences)
    trace = {
        "policy": "R_UTILITY_RIDGE",
        "solver": algorithm,
        "selected_sequence": list(selection.sequence),
        "anchor_sequence": list(selection.anchor_sequence),
        "score": float(selection.score),
        "anchor_score": float(selection.anchor_score),
        "used_anchor": bool(selection.used_anchor),
        "candidates": [
            {
                "sequence": list(item.destinations),
                "origins": [
                    {"algorithm": row.algorithm, "h": int(row.horizon), "proposal_id": row.proposal_id}
                    for row in item.origins
                ],
                "is_h20_anchor": bool(item.is_h20_anchor),
                "score": float(score_by_sequence[item.destinations]),
            }
            for item in pool.candidates
        ],
    }
    return selection, trace


@dataclass(frozen=True)
class _PublicSnapshot:
    ranks: np.ndarray
    occupied: np.ndarray
    legal_mask: np.ndarray
    legal_destinations: tuple[int, ...]
    max_tiers: int
    retrieval_count: int
    relocation_count: int
    history_hits: tuple[bool, ...]
    q_prior: float
    q_t: float
    crane_position: int | None
    crane_time: float


def _snapshot(frame: Any) -> _PublicSnapshot:
    obs = frame.observation
    return _PublicSnapshot(
        ranks=np.asarray(obs.rank_tensor, dtype=np.int32).copy(),
        occupied=np.asarray(obs.occupied_mask, dtype=np.bool_).copy(),
        legal_mask=np.asarray(obs.legal_destination_mask, dtype=np.bool_).copy(),
        legal_destinations=tuple(int(value) for value in frame.legal_destinations),
        max_tiers=int(frame.max_tiers),
        retrieval_count=int(frame.retrieval_count),
        relocation_count=int(frame.relocation_count),
        history_hits=tuple(bool(value) for value in obs.history_hits),
        q_prior=float(obs.q_prior),
        q_t=float(obs.q_t),
        crane_position=None if frame.crane_position is None else int(frame.crane_position),
        crane_time=float(frame.crane_time),
    )


def _after_move(
    frame: Any,
    destination: int,
    *,
    crane_travel_time: float,
    pick_drop_time: float,
) -> _PublicSnapshot:
    before = _snapshot(frame)
    target_locations = np.argwhere(before.occupied & (before.ranks == 1))
    if target_locations.shape != (1, 2):
        raise ValueError("R-EG cache requires exactly one public current target")
    source = int(target_locations[0, 0])
    heights = before.occupied.sum(axis=1).astype(int).tolist()
    if destination not in before.legal_destinations:
        raise ValueError("R-EG plan emitted a destination outside the current legal mask")
    src_height = heights[source]
    dst_height = heights[destination]
    ranks = before.ranks.copy()
    occupied = before.occupied.copy()
    ranks[source, src_height - 1] = 0
    occupied[source, src_height - 1] = False
    ranks[destination, dst_height] = int(frame.observation.rank_tensor[source, src_height - 1])
    occupied[destination, dst_height] = True
    heights[source] -= 1
    heights[destination] += 1
    next_legal_mask = np.asarray(
        [stack != source and heights[stack] < before.max_tiers for stack in range(len(heights))],
        dtype=np.bool_,
    )
    source_position = source + 1
    destination_position = int(destination) + 1
    crane_time = before.crane_time
    if before.crane_position is not None:
        crane_time += abs(before.crane_position - source_position) * float(crane_travel_time)
    crane_time += abs(source_position - destination_position) * float(crane_travel_time) + float(pick_drop_time)
    return _PublicSnapshot(
        ranks=ranks,
        occupied=occupied,
        legal_mask=next_legal_mask,
        legal_destinations=tuple(int(value) for value in np.flatnonzero(next_legal_mask)),
        max_tiers=before.max_tiers,
        retrieval_count=before.retrieval_count,
        relocation_count=before.relocation_count + 1,
        history_hits=before.history_hits,
        q_prior=before.q_prior,
        q_t=before.q_t,
        crane_position=destination_position,
        crane_time=crane_time,
    )


class EGPlanMismatchError(RuntimeError):
    """The public frame changed before a frozen Ridge-EG group was committed."""


def _assert_snapshot(frame: Any, expected: _PublicSnapshot) -> None:
    actual = _snapshot(frame)
    if not (
        np.array_equal(actual.ranks, expected.ranks)
        and np.array_equal(actual.occupied, expected.occupied)
        and np.array_equal(actual.legal_mask, expected.legal_mask)
        and actual.legal_destinations == expected.legal_destinations
        and actual.max_tiers == expected.max_tiers
        and actual.retrieval_count == expected.retrieval_count
        and actual.relocation_count == expected.relocation_count
        and actual.history_hits == expected.history_hits
        and math.isclose(actual.q_prior, expected.q_prior, rel_tol=0.0, abs_tol=1e-15)
        and math.isclose(actual.q_t, expected.q_t, rel_tol=0.0, abs_tol=1e-15)
        and actual.crane_position == expected.crane_position
        and math.isclose(actual.crane_time, expected.crane_time, rel_tol=0.0, abs_tol=1e-12)
    ):
        raise EGPlanMismatchError("full H20 public frame changed during a committed R-EG group")


class _SequenceCommitter:
    """Commit the selected complete EG blocker group against the full public frame."""

    def __init__(
        self,
        algorithm: str,
        selector: Callable[[object], tuple[tuple[int, ...], dict[str, Any]]],
    ) -> None:
        self.algorithm = algorithm
        self.selector = selector
        self.sequence: tuple[int, ...] | None = None
        self.next_index = 0
        self.expected_full: _PublicSnapshot | None = None
        self._cached_moves = 0

    @property
    def cached_moves(self) -> int:
        return self._cached_moves

    def reset(self) -> None:
        self.sequence = None
        self.next_index = 0
        self.expected_full = None
        self._cached_moves = 0

    def select(self, frame: object) -> tuple[int, dict[str, Any]]:
        require_public_frame(frame)
        if self.algorithm == "EG" and self.sequence is not None:
            if self.expected_full is None:
                raise RuntimeError("EG sequence cache lost its expected public frame")
            _assert_snapshot(frame, self.expected_full)
            action = int(self.sequence[self.next_index])
            self._cached_moves += 1
            self.next_index += 1
            trace = {
                "group_start": False,
                "cached_move": True,
                "committed_sequence": list(self.sequence),
                "sequence_index": self.next_index,
            }
            if self.next_index < len(self.sequence):
                self.expected_full = _after_move(
                    frame, action,
                    crane_travel_time=_CRANE_TRAVEL_TIME,
                    pick_drop_time=_PICK_DROP_TIME,
                )
            else:
                self.sequence, self.expected_full, self.next_index = None, None, 0
            return action, trace

        sequence, trace = self.selector(frame)
        sequence = _validate_sequence(frame, sequence)
        required_length = _current_target_blocker_count(frame) if self.algorithm == "EG" else 1
        if len(sequence) != required_length:
            expected = "complete current blocker group" if self.algorithm == "EG" else "one destination"
            raise ValueError(f"{self.algorithm} Ridge policy must commit {expected}")
        if self.algorithm != "EG":
            return sequence[0], {**trace, "group_start": True, "cached_move": False}
        self.sequence = sequence
        self.next_index = 1
        if len(sequence) > 1:
            self.expected_full = _after_move(
                frame, sequence[0],
                crane_travel_time=_CRANE_TRAVEL_TIME,
                pick_drop_time=_PICK_DROP_TIME,
            )
        else:
            self.sequence, self.expected_full, self.next_index = None, None, 0
        return sequence[0], {
            **trace,
            "group_start": True,
            "cached_move": False,
            "committed_sequence": list(sequence),
        }


class RidgeUtilityPolicy:
    """Ridge selector and commit adapter over the existing solver proposer.

    proposer(public_frame, horizon) returns one destination for EM, GP, ERI,
    Chain, and Minmax, or the full current EG blocker sequence for EG.
    """

    def __init__(
        self,
        algorithm: str,
        proposer: Callable[[object, int], Sequence[int]],
        ridge_scorer: Callable[[np.ndarray], float],
    ) -> None:
        if algorithm not in RIDGE_ALGORITHMS:
            raise ValueError(f"algorithm must be one of {RIDGE_ALGORITHMS}")
        self.algorithm = algorithm
        self.proposer = proposer
        self.ridge_scorer = ridge_scorer
        self._committer = _SequenceCommitter(algorithm, self._select_sequence)

    @property
    def cached_moves(self) -> int:
        return self._committer.cached_moves

    def reset(self) -> None:
        self._committer.reset()

    def select(self, public_frame: object) -> tuple[int, dict[str, Any]]:
        return self._committer.select(public_frame)

    def _select_sequence(self, public_frame: object) -> tuple[tuple[int, ...], dict[str, Any]]:
        require_public_frame(public_frame)
        proposals = {
            int(horizon): tuple(int(value) for value in self.proposer(public_frame, int(horizon)))
            for horizon in HORIZONS
        }
        selection, trace = select_ridge_sequence(
            public_frame, self.algorithm, proposals, self.ridge_scorer
        )
        return tuple(selection.sequence), trace


__all__ = [
    "RIDGE_ALGORITHMS", "RidgeScorer", "RidgeUtilityPolicy",
    "load_frozen_ridge", "require_public_frame", "select_ridge_sequence",
]


