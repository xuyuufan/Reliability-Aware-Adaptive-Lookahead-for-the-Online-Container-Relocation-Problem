"""PLAN-041 R selectors that retain every EM horizon as a candidate source.

The original R selector remains unchanged and deduplicates by destination.
These development-only selectors differ only in candidate construction: one
candidate is retained for each fixed horizon, including repeated destinations.
"""

from __future__ import annotations

import math
from typing import Literal

from scrp.history_reliability_r import (
    FROZEN_R_COEFFICIENTS,
    _m2_public_state_statistics,
    _m2_select_scored_candidates,
)
from scrp.noise_aware_em import ActionCandidate, SelectorDecision
from scrp.noise_aware_forecast import HORIZONS, PublicObservation


ProvenanceMethod = Literal["R_SOURCE_FULL", "R_SOURCE_CONST"]
PROVENANCE_METHODS: tuple[ProvenanceMethod, ...] = (
    "R_SOURCE_FULL",
    "R_SOURCE_CONST",
)


def full_source_candidates(
    observation: PublicObservation,
    *,
    max_tiers: int,
    reliability_error: float,
) -> tuple[ActionCandidate, ...]:
    """Build one public EM candidate per requested horizon without deduping.

    Repeated destinations are retained with their distinct ``requested_h``
    and R features. They remain alternative provenance records for one
    physical destination; the caller executes only the selected destination.
    """
    error = float(reliability_error)
    if not math.isfinite(error) or not 0.0 <= error <= 1.0:
        raise ValueError("reliability_error must be a finite probability complement")
    statistics = _m2_public_state_statistics(observation, max_tiers=max_tiers)
    candidates: list[ActionCandidate] = []
    for requested_h in HORIZONS:
        destination = statistics.destination_at_h(requested_h)
        h_scaled = requested_h / 20.0
        candidates.append(
            ActionCandidate(
                destination=int(destination),
                requested_h=int(requested_h),
                features=(
                    1.0,
                    h_scaled,
                    error,
                    h_scaled * error,
                ),
            )
        )
    return tuple(candidates)


def select_r_source_action(
    observation: PublicObservation,
    *,
    method: ProvenanceMethod,
    max_tiers: int,
) -> SelectorDecision:
    """Score all horizon-source candidates with the frozen R4 rule."""
    if method not in PROVENANCE_METHODS:
        raise ValueError(f"unsupported provenance method {method!r}")
    reliability = (
        float(observation.q_t)
        if method == "R_SOURCE_FULL"
        else float(observation.q_prior)
    )
    candidates = full_source_candidates(
        observation,
        max_tiers=max_tiers,
        reliability_error=1.0 - reliability,
    )
    return _m2_select_scored_candidates(
        candidates,
        coefficients=FROZEN_R_COEFFICIENTS,
        variant=method,
    )


__all__ = (
    "PROVENANCE_METHODS",
    "ProvenanceMethod",
    "full_source_candidates",
    "select_r_source_action",
)
