"""Release-gated fixed-H trajectory calls for ridge_fixed_horizon_baselines_v1."""
from __future__ import annotations
import argparse, json, pathlib, sys, time
from typing import Any, Callable, Mapping, Sequence

HERE = pathlib.Path(__file__).resolve()
WORKSPACE = HERE.parents[2]
OUT = WORKSPACE.parent
PREP = OUT / "preparation"
CONFIG_PATH = WORKSPACE / "R/configs/r_fixed_horizon_baselines_v1.json"
POLICY_NAMES = {0:"FIXED_H0", 2:"FIXED_H2", 5:"FIXED_H5", 10:"FIXED_H10"}
RELEASE_FILES = {"M1_BRIDGE_SMOKE", "M2_BANK", "M3_ANALYSIS"}
PACKAGE_ROOT = OUT.parents[2]
ORIGINAL_ROOT = pathlib.Path(r"D:\DSAI\containter_new")
RESEARCH_DIRS = ("R", "GP", "Chain", "Minmax", "EM", "Mijn_RL", "ERI_EG", "Ridge_R_analysis", "Leveling", "experiments")
if str(WORKSPACE) not in sys.path:
    sys.path.insert(0, str(WORKSPACE))

from R.experiments.fixed_horizon_integrity import (
    IntegrityError as StageReleaseError, require_stage_release,
    sha256_file, verify_stage_unchanged,
)
from R.experiments import fixed_horizon_integrity as integrity
from R.experiments.fixed_horizon_schedule import build_m1_bridge_keys, remaining_bank_keys, smoke_keys

_STAGE_CONTEXT_MARKER = object()


def _capture_stage_entry_snapshot(verified: Mapping[str, Any]) -> dict[str, str]:
    keys = (*integrity.FROZEN_HASH_KEYS, "release_file_sha256")
    missing = [key for key in keys if not isinstance(verified.get(key), str) or not verified.get(key)]
    if missing:
        raise StageReleaseError(f"stage release payload lacks entry snapshot hashes: {missing}")
    return {key: str(verified[key]) for key in keys}


def open_stage_context(stage: str, *, previous_science_seconds: float = 0.0) -> tuple[dict[str, Any], dict[str, Any], dict[str, str]]:
    """Create the only trajectory-call context after the full release gate passes."""
    verified = require_stage_release(stage)
    if stage == "M1_BRIDGE_SMOKE":
        allowed, count = build_m1_bridge_keys() | smoke_keys(), 120
        cap = min(float(verified["science_budget_cap_seconds"]), 7200.0 - float(previous_science_seconds))
    elif stage == "M2_BANK":
        allowed, count = remaining_bank_keys(), 9984
        cap = min(float(verified["remaining_science_budget_seconds"]), 7200.0 - float(previous_science_seconds))
    else:
        raise StageReleaseError("trajectory context is available only for M1/M2")
    if cap <= 0 or len(allowed) != count:
        raise StageReleaseError("stage whitelist or cumulative science budget is empty/invalid")
    context = {
        "stage": stage, "allowed_keys": frozenset(allowed), "allowed_count": count,
        "science_cap_seconds": cap, "previous_science_seconds": float(previous_science_seconds),
        "started_monotonic": time.monotonic(), "used_keys": set(), "marker": _STAGE_CONTEXT_MARKER,
    }
    snapshot = _capture_stage_entry_snapshot(verified)
    return context, verified, snapshot


def _check_stage_call(context: Mapping[str, Any], trajectory_key: str) -> None:
    if context.get("marker") is not _STAGE_CONTEXT_MARKER:
        raise StageReleaseError("trajectory call did not receive a validated stage context")
    if trajectory_key not in context["allowed_keys"]:
        raise StageReleaseError("trajectory key is outside the exact released stage whitelist")
    if trajectory_key in context["used_keys"]:
        raise StageReleaseError("duplicate trajectory key in released stage")
    if len(context["used_keys"]) >= context["allowed_count"]:
        raise StageReleaseError("released trajectory count cap has been reached")
    elapsed = time.monotonic() - context["started_monotonic"]
    if elapsed >= context["science_cap_seconds"] or context["previous_science_seconds"] + elapsed >= 7200:
        raise StageReleaseError("cumulative 7,200-second science budget exhausted")


def audit_loaded_research_modules() -> dict[str, str]:
    """Reject research modules loaded from the read-only source or bundle root."""
    allowed = WORKSPACE.resolve()
    watched_roots = [
        root.resolve() / name
        for root in (PACKAGE_ROOT, ORIGINAL_ROOT)
        for name in RESEARCH_DIRS
        if (root / name).exists()
    ]
    loaded: dict[str, str] = {}
    escaped: list[str] = []
    for name, module in sorted(sys.modules.items()):
        file_name = getattr(module, "__file__", None)
        if not file_name:
            continue
        path = pathlib.Path(file_name).resolve()
        if not any(path == root or root in path.parents for root in watched_roots):
            continue
        if path != allowed and allowed not in path.parents:
            escaped.append(f"{name} -> {path}")
        else:
            loaded[name] = str(path)
    if escaped:
        raise RuntimeError("research import escaped isolated r01 workspace: " + "; ".join(escaped[:12]))
    return loaded

def load_isolated_runtime() -> tuple[Any,dict[str,Any],dict[str,Any],dict[str,Any],dict[str,str]]:
    """Load only the byte-identical r01 copy; fail if research imports escape it."""
    if not WORKSPACE.is_dir(): raise FileNotFoundError(WORKSPACE)
    protocol_path=WORKSPACE/"R/experiments/r_order_protocol.py"
    if not protocol_path.is_file(): raise FileNotFoundError(protocol_path)
    protocol_dir=str(WORKSPACE)
    if protocol_dir not in sys.path: sys.path.insert(0,protocol_dir)
    import importlib
    protocol=importlib.import_module("R.experiments.r_order_protocol")
    if pathlib.Path(protocol.__file__).resolve()!=protocol_path.resolve():
        raise RuntimeError(f"protocol import escaped isolated copy: {protocol.__file__}")
    runtime=protocol._runtime_imports()
    gp_rule,gp_identity=protocol._load_gp_rule(runtime)
    comparison=importlib.import_module("R.experiments.r_five_versions_comparison")
    five_runner=importlib.import_module("R.experiments.r_seven_solvers_five_r_runner")
    from R.scrp.fixed_horizon_policy import FixedHorizonPolicy
    importlib.import_module("Chain.chain_public")
    importlib.import_module("Minmax.minmax_public")
    imported=audit_loaded_research_modules()
    return protocol,runtime,gp_rule,{"identity":gp_identity,"comparison":comparison,"five_runner":five_runner,"FixedHorizonPolicy":FixedHorizonPolicy},imported

def run_one_fixed(*, stage_context: Mapping[str, Any], episode: Any, row: Mapping[str,Any], condition: Mapping[str,Any],
                  algorithm: str, horizon: int, forecast: Sequence[int], seed: int|None,
                  runtime: Mapping[str,Any], gp_rule: Mapping[str,Any],
                  trajectory_key: str, public_sink: Callable, private_sink: Callable) -> dict[str,Any]:
    """Run one whitelisted fixed-H key inside its M1 or M2 released stage."""
    _check_stage_call(stage_context, trajectory_key)
    expected_stage = "M1_BRIDGE_SMOKE" if trajectory_key.startswith("BRIDGE|") or trajectory_key in smoke_keys() else "M2_BANK"
    if stage_context["stage"] != expected_stage:
        raise StageReleaseError(f"trajectory key belongs to {expected_stage}, not {stage_context['stage']}")
    if horizon not in POLICY_NAMES: raise ValueError(f"unregistered new horizon: {horizon}")
    import R.scrp.fixed_horizon_policy as policy_module
    from R.scrp.r_order_actor import HistoryBoundController
    comparison=__import__("R.experiments.r_five_versions_comparison",fromlist=["*"])
    policy=policy_module.FixedHorizonPolicy(algorithm,horizon,runtime,gp_rule)
    controller=HistoryBoundController.fresh(policy)
    started=time.monotonic()
    environment,provider=comparison._make_context(episode,forecast,seed,runtime)
    fingerprint=str(row["label_invariant_fingerprint"])
    fhash=comparison._public_hash(forecast)
    frame=runtime["public_frame"](episode,environment,provider)
    decisions=0; actor_time=0.0
    max_steps=int(episode.num_containers)*max(1,int(episode.max_tiers)-1)
    private_sink({"record_type":"trajectory_start","trajectory_key":trajectory_key,
        "layout_fingerprint":fingerprint,"initial_stacks_private":episode.initial_stacks,
        "true_retrieval_sequence_private":episode.retrieval_sequence,
        "fixed_forecast_private":tuple(int(v) for v in forecast),
        "forecast_seed_evaluator_only":seed,"forecast_sha256_evaluator_only":fhash,
        "condition_evaluator_only":dict(condition)})
    private_sink({"record_type":"forecast_binding","trajectory_key":trajectory_key,
        "layout_fingerprint":fingerprint,"forecast_seed_evaluator_only":seed,
        "forecast_sha256_evaluator_only":fhash,"condition_evaluator_only":dict(condition)})
    while not environment.state.terminated:
        elapsed_stage = time.monotonic() - stage_context["started_monotonic"]
        if (elapsed_stage >= stage_context["science_cap_seconds"]
                or stage_context["previous_science_seconds"] + elapsed_stage >= 7200):
            raise StageReleaseError("cumulative 7,200-second science budget exhausted")
        if time.monotonic()-started>30: raise RuntimeError(f"trajectory cap 30s exceeded: {trajectory_key}")
        if not frame.legal_destinations: raise RuntimeError("nonterminal frame has no legal move")
        physical_before=runtime["private_physical_stacks"](environment)
        tick=time.perf_counter(); action,trace,decorated=controller.select(frame); actor_time+=time.perf_counter()-tick
        if int(action) not in frame.legal_destinations: raise RuntimeError("policy selected illegal action")
        before_retrieval=int(environment.state.retrieval_count); before_relocations=int(environment.relocation_count)
        environment.step(int(action)); after_retrieval=int(environment.state.retrieval_count)
        runtime["sync_retrieval_events_with_audit"](provider,episode.retrieval_sequence,before_retrieval,after_retrieval)
        if int(environment.relocation_count)!=before_relocations+1: raise RuntimeError("relocation conservation failed")
        next_frame=None if environment.state.terminated else runtime["public_frame"](episode,environment,provider)
        public_sink({"trajectory_key":trajectory_key,"decision_index":decisions,"algorithm":algorithm,
            "method":POLICY_NAMES[horizon],"public_frame_before":comparison._frame_record(decorated),
            "own_action":int(action),"actor_trace":trace,
            "public_frame_after":None if next_frame is None else comparison._frame_record(next_frame),
            "terminal_after_action":bool(environment.state.terminated)})
        private_sink({"record_type":"physical_step","trajectory_key":trajectory_key,"decision_index":decisions,
            "physical_stacks_before_private":physical_before,
            "physical_stacks_after_private":runtime["private_physical_stacks"](environment),
            "retrieval_count_before":before_retrieval,"retrieval_count_after":after_retrieval,
            "relocation_count_before":before_relocations,"relocation_count_after":int(environment.relocation_count),
            "provider_event_count_private":int(provider.event_count),"provider_history_hits_private":provider.history.hits})
        decisions+=1
        if decisions>max_steps: raise RuntimeError("registered relocation bound exceeded")
        if next_frame is not None: frame=next_frame
    if int(environment.state.retrieval_count)!=int(episode.num_containers): raise RuntimeError("trajectory incomplete")
    elapsed=time.monotonic()-started
    if elapsed>30: raise RuntimeError("trajectory cap 30s exceeded")
    stage_context["used_keys"].add(trajectory_key)
    return {"trajectory_key":trajectory_key,"layout_fingerprint":fingerprint,
        "layout_id":str(row.get("episode_id",fingerprint)),
        "geometry":str(episode.parameter_group),"condition_id":str(condition["condition_id"]),
        "family":str(condition["family"]),"strength":str(condition["strength"]),
        "p":condition.get("p"),"window_size":condition.get("window_size"),"repeat":int(condition["repeat"]),
        "algorithm":algorithm,"method":POLICY_NAMES[horizon],"R_raw":int(environment.relocation_count),
        "N":int(episode.num_containers),"R_over_N":float(environment.relocation_count)/int(episode.num_containers),
        "forecast_seed_evaluator_only":seed,"F_hash":fhash,"complete":True,"decision_count":decisions,
        "wall_seconds":elapsed,"actor_time_seconds":actor_time,"eg_cached_moves":int(policy.cached_moves)}

def main(argv: Sequence[str]|None=None) -> int:
    parser=argparse.ArgumentParser()
    parser.add_argument("--preflight",action="store_true",help="load and verify the isolated dependency closure; never runs a trajectory")
    parser.add_argument("--check-release",choices=tuple(RELEASE_FILES))
    args=parser.parse_args(argv)
    if args.check_release:
        result=require_stage_release(args.check_release)
        print(json.dumps(result,sort_keys=True)); return 0
    if args.preflight:
        _protocol,_runtime,_gp,_objects,imported=load_isolated_runtime()
        print(json.dumps({"status":"PASS_ISOLATED_PREFLIGHT","loaded_research_modules":len(imported),"module_files":imported,"trajectories_run":0},sort_keys=True))
        return 0
    parser.error("select --preflight or --check-release; trajectory execution is callable only after a stage release")
    return 2

if __name__=="__main__": raise SystemExit(main())
