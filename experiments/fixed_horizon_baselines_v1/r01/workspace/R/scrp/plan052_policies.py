"""Five frozen R policy adapters for the three PLAN-052 one-step proposers.

Old PLAN-051 registries are deliberately not extended or patched. Proposal
origin metadata is kept in traces and never enters either shared model feature
builder.
"""

from __future__ import annotations

from typing import Any, Callable, Mapping

import numpy as np

from R.scrp.plan052_public import HORIZONS, NEW_POLICIES, NEW_SOLVERS


_FROZEN_R_COEFFICIENTS = (
    -0.10860203113738334,
    -0.7425766885981918,
    0.20742547985980178,
    1.1399906546560534,
)


class Plan052Policy:
    """A stateless six-slot policy for one new solver; every call returns one move."""

    def __init__(
        self,
        solver: str,
        proposer: Callable[[object, int], tuple[int, dict[str, Any]]],
        *,
        ridge_scorer: Callable[[np.ndarray], float] | None = None,
        rates: object | None = None,
    ) -> None:
        if solver not in NEW_SOLVERS:
            raise ValueError(f"unknown PLAN-052 solver {solver!r}")
        self.solver = solver
        self.proposer = proposer
        self.ridge_scorer = ridge_scorer
        self.rates = rates

    def select(self, public_frame: object, method: str) -> tuple[int, dict[str, Any]]:
        if method not in NEW_POLICIES:
            raise ValueError(f"unknown PLAN-052 policy {method!r}")
        from R.scrp.r_five_versions_actor import require_public_frame

        require_public_frame(public_frame)
        if method == "NATIVE20":
            action, raw_trace = self.proposer(public_frame, 20)
            return self._legal(public_frame, action), {
                "solver": self.solver,
                "policy": method,
                "requested_h": 20,
                "selected_h": 20,
                "proposals": [raw_trace],
                **raw_trace,
            }

        if method == "R_QUAD_H":
            from R.scrp.r_horizon_robustness import all_proxy_scores, choose_r_horizon

            q = float(public_frame.observation.q_t)
            selected_h, score = choose_r_horizon(q, rule="quad")
            action, raw_trace = self.proposer(public_frame, selected_h)
            return self._legal(public_frame, action), {
                "solver": self.solver,
                "policy": method,
                "q_used": q,
                "selected_h": int(selected_h),
                "proxy_score": float(score),
                "all_proxy_scores": [[int(h), float(value)] for h, value in all_proxy_scores(q, "quad")],
                "proposals": [raw_trace],
            }

        proposals = {
            int(h): self.proposer(public_frame, int(h))
            for h in HORIZONS
        }
        destinations = {h: int(row[0]) for h, row in proposals.items()}
        proposal_trace = [
            {"h": h, "destination": destinations[h], "trace": proposals[h][1]}
            for h in HORIZONS
        ]

        if method in ("OLD_FROZEN_R", "R_SOURCE_FULL"):
            from R.scrp.history_reliability_r import _m2_select_scored_candidates
            from scrp.noise_aware_em import ActionCandidate

            if method == "OLD_FROZEN_R":
                # First-action deduplication keeps the lowest requested h.
                source_h: dict[int, int] = {}
                for h in HORIZONS:
                    source_h.setdefault(destinations[h], h)
                candidate_rows = tuple(
                    ActionCandidate(
                        destination=destination,
                        requested_h=h,
                        features=self._old_r_features(public_frame, h),
                    )
                    for destination, h in sorted(source_h.items(), key=lambda item: item[1])
                )
            else:
                # R_SOURCE_FULL retains every source horizon, including duplicate actions.
                candidate_rows = tuple(
                    ActionCandidate(
                        destination=destinations[h],
                        requested_h=h,
                        features=self._old_r_features(public_frame, h),
                    )
                    for h in HORIZONS
                )
            decision = _m2_select_scored_candidates(
                candidate_rows,
                coefficients=_FROZEN_R_COEFFICIENTS,
                variant=method,
            )
            return self._legal(public_frame, int(decision.destination)), {
                "solver": self.solver,
                "policy": method,
                "selected_h": int(decision.requested_h),
                "delta": float(decision.predicted_delta),
                "source_candidates": [
                    {"h": h, "destination": destinations[h]}
                    for h in HORIZONS
                ],
                "scored_candidates": [
                    {
                        "h": int(candidate.requested_h),
                        "destination": int(candidate.destination),
                        "features": [float(value) for value in candidate.features],
                    }
                    for candidate in candidate_rows
                ],
                "proposals": proposal_trace,
            }

        if method == "R_UTILITY_RIDGE":
            if self.ridge_scorer is None:
                raise ValueError("the frozen PLAN-047 Ridge scorer is required")
            from R.scrp.universal_r_utility import score_candidate

            origins: dict[int, list[int]] = {}
            for h in HORIZONS:
                origins.setdefault(destinations[h], []).append(h)
            scores = {
                destination: score_candidate(
                    public_frame, (destination,), self.ridge_scorer
                )
                for destination in origins
            }
            anchor = destinations[20]
            anchor_score = scores[anchor]
            alternatives = [
                (score, destination)
                for destination, score in scores.items()
                if destination != anchor and score < anchor_score
            ]
            if alternatives:
                selected_score, action = min(alternatives)
                used_anchor = False
            else:
                selected_score, action, used_anchor = anchor_score, anchor, True
            selected_h = min(origins[int(action)])
            return self._legal(public_frame, int(action)), {
                "solver": self.solver,
                "policy": method,
                "selected_h_source_min": int(selected_h),
                "anchor_destination": int(anchor),
                "selected_destination": int(action),
                "score": float(selected_score),
                "anchor_score": float(anchor_score),
                "used_anchor": bool(used_anchor),
                "candidate_pool": [
                    {
                        "destination": int(destination),
                        "source_horizons": sorted(int(h) for h in horizons),
                        "score": float(scores[destination]),
                        "is_h20_anchor": destination == anchor,
                    }
                    for destination, horizons in sorted(origins.items())
                ],
                "proposals": proposal_trace,
            }

        if method == "R_ORDER_MAIN_TAU0_FULL":
            if self.rates is None:
                raise ValueError("the frozen R-Order main model is required")
            from R.scrp.r_order_public import _after_move, expected_blocking_units

            origins: dict[int, list[str]] = {}
            for h in HORIZONS:
                origins.setdefault(destinations[h], []).append(f"native_h{h}")
            risk_scores: list[tuple[float, int]] = []
            for destination in tuple(int(value) for value in public_frame.legal_destinations):
                after = _after_move(public_frame, int(destination))
                risk_scores.append((
                    1.0 + float(expected_blocking_units(after, self.rates)),
                    int(destination),
                ))
            risk_score, risk_destination = min(risk_scores, key=lambda item: (item[0], item[1]))
            origins.setdefault(risk_destination, []).append("risk")
            # Score only the registered pool: five native first actions plus
            # the one risk-proposal first action, deduplicated by destination.
            score_by_destination = {destination: score for score, destination in risk_scores}
            scores = {destination: score_by_destination[destination] for destination in origins}
            # The risk action is one legal relocation scored by the shared
            # 20x5 model; no solver label enters the feature builder.
            anchor = destinations[20]
            anchor_score = scores[anchor]
            alternatives = [
                (score, destination)
                for destination, score in scores.items()
                if destination != anchor and score < anchor_score
            ]
            if alternatives:
                selected_score, action = min(alternatives)
                used_anchor = False
            else:
                selected_score, action, used_anchor = anchor_score, anchor, True
            return self._legal(public_frame, int(action)), {
                "solver": self.solver,
                "policy": method,
                "tau": 0.0,
                "selected_destination": int(action),
                "anchor_destination": int(anchor),
                "score": float(selected_score),
                "anchor_score": float(anchor_score),
                "used_anchor": bool(used_anchor),
                "risk_proposal": int(risk_destination),
                "risk_score": float(risk_score),
                "candidate_pool": [
                    {
                        "destination": int(destination),
                        "origins": sorted(origins[destination]),
                        "score": float(scores[destination]),
                        "is_h20_anchor": destination == anchor,
                    }
                    for destination in sorted(origins)
                ],
                "proposals": proposal_trace,
            }
        raise AssertionError("unreachable PLAN-052 policy dispatch")

    @staticmethod
    def _old_r_features(frame: object, h: int) -> tuple[float, float, float, float]:
        q = float(frame.observation.q_t)
        scaled = float(h) / 20.0
        error = 1.0 - q
        return 1.0, scaled, error, scaled * error

    @staticmethod
    def _legal(frame: object, action: int) -> int:
        if int(action) not in tuple(int(value) for value in frame.legal_destinations):
            raise ValueError("PLAN-052 policy selected an illegal public destination")
        return int(action)


def load_frozen_ridge(path: str) -> Callable[[np.ndarray], float]:
    """Load only the frozen shared Ridge artifact; never fit or calibrate."""
    import json
    from pathlib import Path
    from R.scrp.r_five_versions_actor import RidgeScorer

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


def load_frozen_order_rates(path: str) -> object:
    """Load the shared frozen main 20x5 R-Order model, with no solver key."""
    import json
    from pathlib import Path
    from R.scrp.r_order_public import PlackettLuceRates

    record = json.loads(Path(path).read_text(encoding="utf-8"))
    main = record.get("main", {})
    if (
        main.get("model_id") != "r_order_main"
        or main.get("converged") is not True
        or np.asarray(main.get("coefficients", ())).shape != (20, 5)
    ):
        raise ValueError("PLAN-049 frozen shared R-Order main model is missing or malformed")
    return PlackettLuceRates(
        main["coefficients"],
        tuple(int(value) for value in main["active_feature_indices"]),
        main["model_id"],
    )
