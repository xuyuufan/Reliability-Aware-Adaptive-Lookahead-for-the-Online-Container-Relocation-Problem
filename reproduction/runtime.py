"""Load the included runtime and expose only the registered policy slots."""
from __future__ import annotations

from functools import lru_cache
import importlib
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = ROOT / "baselines/_snapshot"
REFERENCE = ROOT / "data/reference_trajectories.csv"
ALGORITHMS = ("EM", "GP", "ERI", "EG", "Chain", "Minmax")
METHODS = ("RIDGE", "FIXED_H0", "FIXED_H2", "FIXED_H5", "FIXED_H10", "FIXED_H20")


@lru_cache(maxsize=1)
def load_runtime():
    # Original sources use these package names. Reject an already loaded export
    # rather than silently mixing the compact root adapter with the snapshot.
    loaded = sys.modules.get("R")
    if loaded and any(not Path(p).resolve().is_relative_to(SNAPSHOT) for p in loaded.__path__):
        raise RuntimeError("R was imported from outside the baseline snapshot; use a fresh Python process")
    sys.path.insert(0, str(SNAPSHOT))
    import hashlib
    import json
    comparison = importlib.import_module("R.experiments.r_five_versions_comparison")
    protocol = importlib.import_module("R.experiments.r_order_protocol")
    runtime = protocol._runtime_imports()
    gp_record = json.loads((ROOT / "models/GP_RULE.json").read_text(encoding="utf-8"))
    from GP.expression import Node
    from GP.simulator import select_destination
    tree = Node.from_prefix(tuple(gp_record["selected_best_prefix"]))
    expression = tree.expression()
    if (expression != gp_record["selected_best_expression"]
            or hashlib.sha256(expression.encode()).hexdigest() != gp_record["selected_expression_sha256"]):
        raise ValueError("Frozen GP prefix/expression identity mismatch")
    def selector(state, legal_destinations, *, max_tiers):
        return int(select_destination(tree, state, legal_destinations, max_tiers=max_tiers))
    gp = {"selector": selector, "tree": tree}
    ridge_record = json.loads((ROOT / "models/RIDGE_MODEL.json").read_text(encoding="utf-8"))
    if (ridge_record["model_kind"] != "weighted_float64_ridge"
            or ridge_record["intercept"] is not False or ridge_record["standardized"] is not False
            or float(ridge_record["ridge_lambda"]) != 1.0 or len(ridge_record["coefficients"]) != 74):
        raise ValueError("Frozen Ridge model contract changed")
    from R.scrp.r_five_versions_actor import RidgeScorer
    models = {"ridge": RidgeScorer(tuple(float(v) for v in ridge_record["coefficients"]))}
    audit_imports()
    return {"protocol": protocol, "runtime": runtime, "gp": gp,
            "models": models, "comparison": comparison}


def audit_imports():
    """The runtime must not read research modules from a local source workspace."""
    files = {}
    for name, module in tuple(sys.modules.items()):
        if name.split(".")[0] not in {"R", "EM", "GP", "Chain", "Minmax", "scrp", "experiments"}:
            continue
        path = getattr(module, "__file__", None)
        if path:
            resolved = Path(path).resolve()
            if not resolved.is_relative_to(SNAPSHOT):
                raise RuntimeError(f"Research import escaped the included snapshot: {name} -> {resolved}")
            files[name] = resolved.relative_to(ROOT).as_posix()
    return files


class _OneStepPolicy:
    def __init__(self, proposer, horizon=None, ridge=None, algorithm=None):
        self.proposer, self.horizon, self.ridge = proposer, horizon, ridge
        if ridge is not None:
            from R.scrp.plan052_policies import Plan052Policy
            self.policy = Plan052Policy(algorithm, proposer, ridge_scorer=ridge)

    @property
    def cached_moves(self):
        return 0

    def select(self, frame):
        if self.ridge is not None:
            return self.policy.select(frame, "R_UTILITY_RIDGE")
        return self.proposer(frame, self.horizon)


class _LevelingPolicy:
    cached_moves = 0

    def select(self, frame):
        from R.scrp.r_five_versions_actor import require_public_frame
        require_public_frame(frame)
        heights = frame.observation.occupied_mask.sum(axis=1)
        destination = min(frame.legal_destinations, key=lambda s: (int(heights[s]), s))
        return int(destination), {"policy": "LEVELING", "requested_h": 0}


def make_policy(algorithm, method, context):
    if algorithm == "Leveling":
        if method != "FIXED_H0":
            raise ValueError("Leveling is forecast-independent and exposes only FIXED_H0")
        return _LevelingPolicy()
    if algorithm not in ALGORITHMS or method not in METHODS:
        raise ValueError("Algorithm or method is outside the six-solver frozen experiment")
    actor_runtime = context["comparison"]._actor_runtime(context["runtime"])
    if algorithm in ("Chain", "Minmax"):
        module = importlib.import_module(f"{algorithm}.{algorithm.lower()}_public")
        return _OneStepPolicy(module.propose,
                              horizon=None if method == "RIDGE" else int(method.removeprefix("FIXED_H")),
                              ridge=context["models"]["ridge"] if method == "RIDGE" else None,
                              algorithm=algorithm)
    if method == "RIDGE":
        from R.scrp.r_five_versions_actor import FiveRPolicy
        return FiveRPolicy("R_UTILITY_RIDGE", algorithm, actor_runtime, context["gp"],
                           ridge_scorer=context["models"]["ridge"])
    from R.scrp.r_order_actor import FixedHorizonActor
    return FixedHorizonActor(algorithm, int(method.removeprefix("FIXED_H")),
                             actor_runtime, context["gp"])
