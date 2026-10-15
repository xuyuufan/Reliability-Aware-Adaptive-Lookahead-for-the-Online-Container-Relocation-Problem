"""PLAN-048 q-only requested-horizon proxy and single-call EM actor."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable, Literal, Sequence

import numpy as np

from scrp.history_reliability_r import FROZEN_R_COEFFICIENTS


HORIZONS = (0, 2, 5, 10, 20)
Q_PRIOR = 5183 / 9920
Rule = Literal["linear", "quad"]


@dataclass(frozen=True)
class HorizonAction:
    action: int
    requested_h: int
    effective_h: int
    proxy_score: float
    proxy_scores: tuple[tuple[int, float], ...]


def _checked_q(q_t: float) -> float:
    if isinstance(q_t, (bool, np.bool_)):
        raise ValueError("q_t must be a probability, not a boolean")
    value = float(q_t)
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError("q_t must be finite and in [0, 1]")
    return value


def _checked_rule(rule: str) -> Rule:
    if rule not in ("linear", "quad"):
        raise ValueError("rule must be 'linear' or 'quad'")
    return rule  # type: ignore[return-value]


def proxy_score(h: int, q_t: float, rule: Rule = "quad") -> float:
    """Frozen R-coefficient proxy score; it is not a relocation-delta estimate."""
    q = _checked_q(q_t)
    variant = _checked_rule(rule)
    if isinstance(h, (bool, np.bool_)) or int(h) != h or int(h) not in HORIZONS:
        raise ValueError(f"h must be in {HORIZONS}")
    horizon = int(h)
    if horizon == 0:
        return 0.0
    beta0, beta1, beta2, beta3 = FROZEN_R_COEFFICIENTS
    x = horizon / 20.0
    error = 1.0 - q
    score = beta0 + beta1 * x + beta2 * error + beta3 * error * x
    if variant == "quad":
        score += beta3 * error * x * x
    return float(score)


def all_proxy_scores(q_t: float, rule: Rule = "quad") -> tuple[tuple[int, float], ...]:
    q = _checked_q(q_t)
    variant = _checked_rule(rule)
    return tuple((h, proxy_score(h, q, variant)) for h in HORIZONS)


def choose_r_horizon(q_t: float, rule: Rule = "quad") -> tuple[int, float]:
    """Choose across the full requested grid before any action deduplication.

    The h=0 anchor has score 0. A nonzero horizon is selected only for a
    strictly negative proxy; ties go to the smaller requested horizon.
    """
    scores = all_proxy_scores(q_t, rule)
    best_h, best_score = min(scores[1:], key=lambda row: (row[1], row[0]))
    if best_score < 0.0:
        return int(best_h), float(best_score)
    return 0, 0.0


def pairwise_switch_q(h_left: int, h_right: int, rule: Rule = "quad") -> float | None:
    """Analytic q root for s(h_left,q)=s(h_right,q), with s=A+(1-q)C."""
    variant = _checked_rule(rule)
    for horizon in (h_left, h_right):
        if isinstance(horizon, (bool, np.bool_)) or int(horizon) != horizon or int(horizon) not in HORIZONS:
            raise ValueError(f"horizons must be in {HORIZONS}")

    def components(horizon: int) -> tuple[float, float]:
        if horizon == 0:
            return 0.0, 0.0
        beta0, beta1, beta2, beta3 = FROZEN_R_COEFFICIENTS
        x = horizon / 20.0
        a = beta0 + beta1 * x
        c = beta2 + beta3 * x
        if variant == "quad":
            c += beta3 * x * x
        return a, c

    a0, c0 = components(int(h_left))
    a1, c1 = components(int(h_right))
    delta_c = c0 - c1
    if delta_c == 0.0:
        return None
    return float(1.0 + (a0 - a1) / delta_c)


def project_public_observation(observation: object, horizon: int):
    """Evaluator-side adapter creating one selected nested public prefix.

    Call this before the action module.  The adaptive actor function below
    accepts only the returned prefix, never the source H20 observation.
    """
    if isinstance(horizon, (bool, np.bool_)) or int(horizon) != horizon or int(horizon) < 0:
        raise ValueError("effective horizon must be a non-negative integer")
    h = int(horizon)
    from scrp.noise_aware_forecast import PublicObservation

    ranks = np.asarray(getattr(observation, "rank_tensor"), dtype=np.int32)
    occupied = np.asarray(getattr(observation, "occupied_mask"), dtype=np.bool_)
    legal = np.asarray(getattr(observation, "legal_destination_mask"), dtype=np.bool_)
    if ranks.ndim != 2 or ranks.shape != occupied.shape:
        raise ValueError("rank tensor and occupied mask must be equal 2-D shapes")
    if ranks.shape[0] != legal.shape[0]:
        raise ValueError("legal mask does not match public stacks")
    if np.count_nonzero(occupied & (ranks == 1)) != 1:
        raise ValueError("public frame must contain exactly one current target")
    if np.any(ranks[~occupied] != 0):
        raise ValueError("empty padding must remain rank 0")
    if np.any(occupied & ((ranks < 1) | (ranks > 22))):
        raise ValueError("occupied rank is outside the public rank/sentinel range")
    prefix = ranks.copy()
    prefix[occupied & (prefix > h + 1)] = h + 2
    prefix[~occupied] = 0
    projected = PublicObservation(
        rank_tensor=prefix,
        occupied_mask=occupied.copy(),
        legal_destination_mask=legal.copy(),
        history_hits=tuple(getattr(observation, "history_hits")),
        q_prior=float(getattr(observation, "q_prior")),
        q_t=_checked_q(float(getattr(observation, "q_t"))),
    )
    if not np.array_equal(projected.occupied_mask, occupied) or not np.array_equal(
        projected.legal_destination_mask, legal
    ):
        raise AssertionError("horizon projection changed occupancy/legal masks")
    if np.count_nonzero(projected.occupied_mask & (projected.rank_tensor == 1)) != 1:
        raise AssertionError("horizon projection changed the current target")
    return projected


def select_adaptive_em_action(
    observation: object,
    *,
    requested_h: int,
    effective_h: int,
    max_tiers: int,
    rule: Rule,
    em_select: Callable[..., int] | None = None,
) -> HorizonAction:
    """Make one EM call from an already projected public view.

    Horizon selection and projection happen in the evaluator adapter. This
    actor boundary accepts only q-bearing selected-view public data and
    geometry; it never receives the source H20 frame, provider, environment,
    evaluator, truth, or IDs.
    """
    if max_tiers <= 0:
        raise ValueError("max_tiers must be positive")
    q_t = _checked_q(float(getattr(observation, "q_t")))
    scores = all_proxy_scores(q_t, rule)
    selected_h, selected_score = choose_r_horizon(q_t, rule)
    if isinstance(requested_h, (bool, np.bool_)) or int(requested_h) != requested_h:
        raise ValueError("requested_h must be an integer")
    if int(requested_h) != selected_h:
        raise ValueError("requested_h does not match the frozen full-grid choice")
    if isinstance(effective_h, (bool, np.bool_)) or int(effective_h) != effective_h:
        raise ValueError("effective_h must be an integer")
    occupied = np.asarray(getattr(observation, "occupied_mask"), dtype=np.bool_)
    ranks = np.asarray(getattr(observation, "rank_tensor"), dtype=np.int32)
    effective_h = int(effective_h)
    if effective_h < 0 or effective_h > int(requested_h):
        raise ValueError("effective_h must be between zero and requested_h")
    if ranks.shape != occupied.shape or np.any(ranks[~occupied] != 0):
        raise ValueError("selected public view has invalid rank or occupancy shape")
    if np.any(occupied & ((ranks < 1) | (ranks > effective_h + 2))):
        raise ValueError("action module received ranks outside the selected horizon")
    if np.count_nonzero(occupied & (ranks == 1)) != 1:
        raise ValueError("selected public view must contain exactly one current target")
    if em_select is None:
        from scrp.noise_aware_em import select_em_destination

        em_select = select_em_destination
    action = int(em_select(observation, max_tiers=max_tiers, h=effective_h))
    legal_mask = np.asarray(getattr(observation, "legal_destination_mask"), dtype=np.bool_)
    legal = tuple(int(value) for value in np.flatnonzero(legal_mask))
    if action not in legal:
        raise ValueError("single-call EM selector returned an illegal action")
    return HorizonAction(
        action=action,
        requested_h=int(requested_h),
        effective_h=int(effective_h),
        proxy_score=float(selected_score),
        proxy_scores=scores,
    )


__all__ = [
    "HORIZONS",
    "Q_PRIOR",
    "HorizonAction",
    "all_proxy_scores",
    "choose_r_horizon",
    "pairwise_switch_q",
    "project_public_observation",
    "proxy_score",
    "select_adaptive_em_action",
]
