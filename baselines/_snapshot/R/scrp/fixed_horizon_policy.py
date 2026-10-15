"""Fixed-H wrappers around the frozen solver proposers for this registered run."""
from __future__ import annotations
from typing import Any, Callable, Mapping

ALGORITHMS = ("EM", "GP", "ERI", "EG", "Chain", "Minmax")
NEW_HORIZONS = (0, 2, 5, 10)
_ACTOR_RUNTIME_KEYS = ("truncate_public_frame", "eri_eg", "em_select", "eri_select")


class FixedHorizonPolicy:
    """One proposer call at the registered H; EG retains its native group cache."""
    def __init__(self, algorithm: str, horizon: int, runtime: Mapping[str, Any],
                 gp_rule: Mapping[str, Any], *, proposer: Callable | None = None) -> None:
        if algorithm not in ALGORITHMS:
            raise ValueError(f"unregistered solver: {algorithm!r}")
        if isinstance(horizon, bool) or int(horizon) != horizon or int(horizon) not in NEW_HORIZONS:
            raise ValueError(f"new fixed H must be one of {NEW_HORIZONS}")
        self.algorithm = algorithm
        self.horizon = int(horizon)
        self.proposal_calls = 0
        self._actor = None
        self._proposer = None
        if algorithm in ("EM", "GP", "ERI", "EG"):
            from R.scrp.r_order_actor import FixedHorizonActor
            actor_runtime = {key: runtime[key] for key in _ACTOR_RUNTIME_KEYS}
            self._actor = FixedHorizonActor(algorithm, self.horizon, actor_runtime, gp_rule)
        else:
            if proposer is None:
                if algorithm == "Chain":
                    from Chain.chain_public import propose as proposer
                else:
                    from Minmax.minmax_public import propose as proposer
            self._proposer = proposer

    @property
    def is_group_start(self) -> bool:
        return True if self._actor is None else bool(self._actor.is_group_start)

    @property
    def cached_moves(self) -> int:
        return 0 if self._actor is None else int(self._actor.cached_moves)

    def reset(self) -> None:
        if self._actor is not None:
            self._actor.reset()
        self.proposal_calls = 0

    def select(self, public_frame: object) -> tuple[int, dict[str, Any]]:
        from R.scrp.r_five_versions_actor import require_public_frame
        require_public_frame(public_frame)
        if self._actor is not None:
            action, native_trace = self._actor.select(public_frame)
            if self.algorithm != "EG":
                self.proposal_calls += 1
            elif bool(native_trace.get("group_start")):
                self.proposal_calls += 1
        else:
            action, native_trace = self._proposer(public_frame, self.horizon)
            self.proposal_calls += 1
        action = int(action)
        if action not in tuple(int(value) for value in public_frame.legal_destinations):
            raise ValueError(f"{self.algorithm} H{self.horizon} returned an illegal action")
        if int(native_trace.get("requested_h", -1)) != self.horizon:
            raise ValueError(f"{self.algorithm} did not use exactly registered H={self.horizon}")
        trace = dict(native_trace)
        trace.update({
            "solver": self.algorithm,
            "policy": f"FIXED_H{self.horizon}",
            "requested_h": self.horizon,
            "fixed_h_only": True,
        })
        return action, trace

