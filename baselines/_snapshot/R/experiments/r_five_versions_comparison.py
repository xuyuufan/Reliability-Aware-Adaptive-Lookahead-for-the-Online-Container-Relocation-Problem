"""PLAN-051 M0 and release-gated five-R comparison runner.

The runner reuses PLAN-049's owner-first runtime, public-frame construction,
noise-aware environment/provider, and retrieval-audit synchronization. It does
not fit or calibrate any model. M1 commands require ROOT-authored, hash-bound
release files and preserve public actions separately from evaluator-private
truth and physical-state audit streams.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import replace
import gzip
import hashlib
import importlib
import json
import math
from pathlib import Path
import shutil
import sys
import time
import traceback
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


WORKSPACE = Path(__file__).resolve().parents[2]
if str(WORKSPACE) not in sys.path:
    sys.path.insert(0, str(WORKSPACE))
R_ROOT = WORKSPACE / "R"
CONFIG_PATH = R_ROOT / "configs/r_five_versions_comparison_v1.json"
OUTPUT_ROOT = R_ROOT / "output/r_five_versions_comparison_v1/r01"
PLAN_PATH = R_ROOT / "docs/planning/PLAN_051_FIVE_R_FOUR_SOLVERS_COMPARISON.md"
LAYOUT_PATH = R_ROOT / "output/r_frozen_cross_solver_transfer_v1/r01/data/select20.json"
RIDGE_PATH = R_ROOT / "output/universal_r_utility_v1/r02/M1_FIT_01/RIDGE_MODEL.json"
ORDER_MODELS_PATH = R_ROOT / "output/r_order_v1/r01/M1_MODELS.json"
POLICIES = (
    "OLD_FROZEN_R", "R_SOURCE_FULL", "R_UTILITY_RIDGE", "R_QUAD_H",
    "R_ORDER_MAIN_TAU0_FULL", "NATIVE20",
)
ALGORITHMS = ("EM", "GP", "ERI", "EG")
SCIENCE_CAP_SECONDS = 7200
TRAJECTORY_CAP_SECONDS = 30


class ComparisonStop(RuntimeError):
    pass


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if hasattr(value, "__dataclass_fields__"):
        from dataclasses import asdict
        return _json_safe(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("nonfinite value cannot be written to JSON")
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"unsupported JSON value {type(value).__name__}")


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_new_json(path: Path, value: Any) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(_json_safe(value), sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
    return _sha256_bytes(payload)


def _load_runtime() -> tuple[Any, dict[str, Any], dict[str, Any], dict[str, Any]]:
    protocol = importlib.import_module("R.experiments.r_order_protocol")
    runtime = protocol._runtime_imports()
    gp_rule, gp_identity = protocol._load_gp_rule(runtime)
    ridge_record = _read_json(RIDGE_PATH)
    if (ridge_record.get("model_kind") != "weighted_float64_ridge"
            or ridge_record.get("intercept") is not False
            or ridge_record.get("standardized") is not False
            or float(ridge_record.get("ridge_lambda", -1)) != 1.0
            or len(ridge_record.get("coefficients", ())) != 74):
        raise ComparisonStop("PLAN-047 Ridge artifact violates its frozen M1 contract")
    order_record = _read_json(ORDER_MODELS_PATH)
    main = order_record.get("main", {})
    if (main.get("model_id") != "r_order_main" or main.get("converged") is not True
            or len(main.get("coefficients", ())) != 20
            or any(len(row) != 5 for row in main["coefficients"])):
        raise ComparisonStop("PLAN-049 frozen main model is missing or malformed")
    from R.scrp.r_five_versions_actor import RidgeScorer
    from R.scrp.r_order_public import PlackettLuceRates
    models = {
        "ridge": RidgeScorer(tuple(float(v) for v in ridge_record["coefficients"])),
        "rates": PlackettLuceRates(
            main["coefficients"], tuple(main["active_feature_indices"]), main["model_id"]
        ),
        "ridge_record": ridge_record,
        "order_record": order_record,
    }
    return protocol, runtime, gp_rule, {"identity": gp_identity, **models}


def _actor_runtime(runtime: Mapping[str, Any]) -> dict[str, Any]:
    """Give policy objects only public proposal and selector capabilities."""
    keys = (
        "truncate_public_frame", "eri_eg", "r_eri_eg", "select_r_action",
        "frozen_actor_call", "em_select", "eri_select",
    )
    return {key: runtime[key] for key in keys}


def _source_paths(protocol: Any) -> tuple[str, ...]:
    return tuple(dict.fromkeys((
        "R/docs/planning/PLAN_051_FIVE_R_FOUR_SOLVERS_COMPARISON.md",
        "R/configs/r_five_versions_comparison_v1.json",
        "R/experiments/r_five_versions_comparison.py",
        "R/experiments/r_order_protocol.py",
        "R/scrp/r_five_versions_actor.py",
        "R/output/r_frozen_cross_solver_transfer_v1/r01/data/select20.json",
        "R/scrp/history_reliability_r.py",
        "R/scrp/history_reliability_r_provenance.py",
        "R/scrp/cross_solver_reliability_r.py",
        "R/scrp/r_eri_eg_public.py",
        "R/scrp/eri_eg_public.py",
        "R/scrp/r_order_public.py",
        "R/scrp/r_order_actor.py",
        "R/scrp/r_order_training.py",
        "R/scrp/universal_r_utility.py",
        "R/scrp/universal_r_utility_m1.py",
        "R/scrp/r_horizon_robustness.py",
        "R/scrp/h20_r_bank.py",
        "R/scrp/h20_r_c2.py",
        "R/scrp/h20_r_data.py",
        "R/experiments/history_reliability_r_protocol.py",
        "R/experiments/time_window_shuffle_protocol.py",
        "R/experiments/time_window_shuffle_acceptance.py",
        "R/configs/r_noise_diagnostics_v1.json",
        "R/output/universal_r_utility_v1/r02/M1_FIT_01/RIDGE_MODEL.json",
        "R/output/universal_r_utility_v1/r02/RECOVERY_AUDIT/negative_result_delivery_01/M1_FINAL_NEGATIVE_RESULT.md",
        "R/output/r_order_v1/r01/M1_MODELS.json",
        "EM/scrp/noise_aware_forecast.py",
        "EM/scrp/noise_aware_em.py",
        "EM/experiments/noise_aware_reference_p0.py",
        "EM/experiments/noise_aware_lookahead.py",
        "Mijn_RL/scrp/online_crp.py",
        "Mijn_RL/scrp/attention_data.py",
        *protocol.GP_ACCEPTED_ARTIFACT_PATHS,
    )))


def _build_bindings() -> dict[str, Any]:
    protocol, runtime, gp_rule, models = _load_runtime()
    config = _read_json(CONFIG_PATH)
    layouts_raw = _read_json(LAYOUT_PATH)
    if len(layouts_raw) != 20:
        raise ComparisonStop(f"PLAN-044 layout source has {len(layouts_raw)} rows, expected 20")
    layouts = [runtime["OnlineCRPEpisode"].from_record(row) for row in layouts_raw]
    seen: set[str] = set()
    geometry_counts = {"G1": 0, "G2": 0}
    fingerprints: list[str] = []
    for episode, row in zip(layouts, layouts_raw, strict=True):
        fingerprint = str(runtime["label_invariant_fingerprint"](episode))
        if fingerprint != row.get("label_invariant_fingerprint") or fingerprint in seen:
            raise ComparisonStop("PLAN-044 layout fingerprint is missing, changed, or duplicated")
        seen.add(fingerprint)
        geometry = str(episode.parameter_group)
        if geometry not in geometry_counts:
            raise ComparisonStop(f"unregistered geometry in select20: {geometry}")
        geometry_counts[geometry] += 1
        fingerprints.append(fingerprint)
    if geometry_counts != {"G1": 10, "G2": 10}:
        raise ComparisonStop(f"PLAN-044 geometry counts differ: {geometry_counts}")
    source_hashes: dict[str, str] = {}
    for relative in _source_paths(protocol):
        path = WORKSPACE / relative
        if not path.is_file():
            raise ComparisonStop(f"registered source asset is absent: {relative}")
        source_hashes[relative] = _sha256_file(path)
    ridge_record = models["ridge_record"]
    main = models["order_record"]["main"]
    binding = {
        "protocol_id": "r_five_versions_comparison_v1",
        "run_id": "r01",
        "plan_status_at_binding": "REGISTERED / M0_PREPARATION",
        "config": config,
        "source_sha256": source_hashes,
        "layout": {
            "source": LAYOUT_PATH.relative_to(WORKSPACE).as_posix(),
            "source_sha256": source_hashes[LAYOUT_PATH.relative_to(WORKSPACE).as_posix()],
            "count": len(layouts),
            "geometry_counts": geometry_counts,
            "fingerprints": fingerprints,
        },
        "models": {
            "R_UTILITY_RIDGE": {
                "path": RIDGE_PATH.relative_to(WORKSPACE).as_posix(),
                "sha256": source_hashes[RIDGE_PATH.relative_to(WORKSPACE).as_posix()],
                "model_kind": ridge_record["model_kind"],
                "coefficient_count": len(ridge_record["coefficients"]),
                "ridge_lambda": ridge_record["ridge_lambda"],
                "intercept": ridge_record["intercept"],
                "standardized": ridge_record["standardized"],
                "fixed_reason": "pre-result PLAN-047 Ridge; historical capture exceeded fixed MLP",
            },
            "R_ORDER_MAIN_TAU0_FULL": {
                "path": ORDER_MODELS_PATH.relative_to(WORKSPACE).as_posix(),
                "sha256": source_hashes[ORDER_MODELS_PATH.relative_to(WORKSPACE).as_posix()],
                "model_id": main["model_id"],
                "converged": main["converged"],
                "active_feature_indices": main["active_feature_indices"],
                "tau": 0.0,
                "calibrated_tau_available": False,
                "includes_risk_proposal": True,
            },
            "GP": models["identity"],
        },
        "counts": {
            "conditions_per_layout": len(condition_specs()),
            "candidate_instances_per_solver": len(POLICIES),
            "algorithms": len(ALGORITHMS),
            "bank_trajectories": len(layouts) * len(condition_specs()) * len(POLICIES) * len(ALGORITHMS),
            "smoke_trajectories": 12,
        },
    }
    if binding["counts"]["conditions_per_layout"] != 21 or binding["counts"]["bank_trajectories"] != 10080:
        raise ComparisonStop("registered condition or trajectory count is inconsistent")
    binding["binding_sha256"] = _sha256_bytes(_canonical(binding))
    return binding


def condition_specs() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for family in ("subset_shuffle", "delayed_positions"):
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


def _forecast_for(episode: Any, fingerprint: str, condition: Mapping[str, Any], runtime: Mapping[str, Any]) -> tuple[tuple[int, ...], int | None]:
    true_future = tuple(int(value) for value in episode.retrieval_sequence[1:])
    family = str(condition["family"])
    repeat = int(condition["repeat"])
    if family == "CLEAN":
        return true_future, None
    if family == "time_window_shuffle":
        seed = int(runtime["semantic_time_window_seed"](
            fingerprint, window_size=int(condition["window_size"]), repeat=repeat
        ))
        forecast = tuple(runtime["make_time_window_forecast"](
            true_future, window_size=int(condition["window_size"]), seed=seed
        ))
    else:
        seed = int(runtime["semantic_forecast_seed"](fingerprint, family, repeat))
        forecast = tuple(runtime["make_evaluator_forecast"](
            true_future, family=family, p_evaluator_only=float(condition["p"]), seed=seed
        ))
    if len(forecast) != len(true_future) or set(forecast) != set(true_future):
        raise ComparisonStop("forecast generator did not return the fixed full permutation")
    return forecast, seed


def _public_hash(forecast: Sequence[int]) -> str:
    return _sha256_bytes(_canonical([int(value) for value in forecast]))


def _make_context(episode: Any, forecast: Sequence[int], seed: int | None, runtime: Mapping[str, Any]) -> tuple[Any, Any]:
    from R.scrp.r_order_public import Q_PRIOR
    provider = runtime["ForecastProvider"](
        episode.retrieval_sequence, p=0.0, seed=0 if seed is None else int(seed),
        q_prior=Q_PRIOR, max_horizon=20, forecast_future=tuple(int(v) for v in forecast),
    )
    environment = runtime["OnlineCRPEnvironment"](episode, lookahead_horizon=0)
    before = 0
    after = int(environment.state.retrieval_count)
    runtime["sync_retrieval_events_with_audit"](provider, episode.retrieval_sequence, before, after)
    if provider.max_horizon != 20 or not math.isclose(provider.history.q_prior, Q_PRIOR, rel_tol=0.0, abs_tol=1e-15):
        raise ComparisonStop("fresh provider differs from the registered horizon or prior")
    return environment, provider


def _frame_record(frame: object) -> dict[str, Any]:
    from R.scrp.r_five_versions_actor import public_frame_record
    return public_frame_record(frame)


def _run_trajectory(
    *, episode: Any, row: Mapping[str, Any], condition: Mapping[str, Any],
    algorithm: str, method: str, forecast: Sequence[int], seed: int | None,
    runtime: Mapping[str, Any], gp_rule: Mapping[str, Any], models: Mapping[str, Any],
    trajectory_key: str, public_sink: Any, private_sink: Any,
) -> dict[str, Any]:
    from R.scrp.r_five_versions_actor import FiveRPolicy
    from R.scrp.r_order_actor import HistoryBoundController
    started = time.monotonic()
    environment, provider = _make_context(episode, forecast, seed, runtime)
    policy = FiveRPolicy(
        method, algorithm, _actor_runtime(runtime), gp_rule,
        ridge_scorer=models["ridge"], rates=models["rates"],
    )
    controller = HistoryBoundController.fresh(policy)
    fingerprint = str(row["label_invariant_fingerprint"])
    fhash = _public_hash(forecast)
    public_before = runtime["public_frame"](episode, environment, provider)
    decisions = 0
    actor_time = 0.0
    max_steps = int(episode.num_containers) * max(1, int(episode.max_tiers) - 1)
    private_sink({
        "record_type": "trajectory_start", "trajectory_key": trajectory_key,
        "layout_fingerprint": fingerprint,
        "initial_stacks_private": episode.initial_stacks,
        "true_retrieval_sequence_private": episode.retrieval_sequence,
        "fixed_forecast_private": tuple(int(v) for v in forecast),
        "forecast_seed_evaluator_only": seed,
        "forecast_sha256_evaluator_only": fhash,
        "condition_evaluator_only": dict(condition),
    })
    while not environment.state.terminated:
        if time.monotonic() - started > TRAJECTORY_CAP_SECONDS:
            raise ComparisonStop(f"trajectory exceeded {TRAJECTORY_CAP_SECONDS}s: {trajectory_key}")
        if not public_before.legal_destinations:
            raise ComparisonStop("nonterminal public frame has no legal relocation")
        physical_before = runtime["private_physical_stacks"](environment)
        actor_started = time.perf_counter()
        action, trace, decorated = controller.select(public_before)
        actor_time += time.perf_counter() - actor_started
        if int(action) not in public_before.legal_destinations:
            raise ComparisonStop("actor returned a destination outside the shared legal set")
        before_retrieval = int(environment.state.retrieval_count)
        before_relocations = int(environment.relocation_count)
        environment.step(int(action))
        after_retrieval = int(environment.state.retrieval_count)
        runtime["sync_retrieval_events_with_audit"](
            provider, episode.retrieval_sequence, before_retrieval, after_retrieval
        )
        if int(environment.relocation_count) != before_relocations + 1:
            raise ComparisonStop("legal trajectory action failed relocation conservation")
        next_frame = None if environment.state.terminated else runtime["public_frame"](episode, environment, provider)
        public_sink({
            "trajectory_key": trajectory_key, "decision_index": decisions,
            "algorithm": algorithm, "method": method,
            "public_frame_before": _frame_record(decorated), "own_action": int(action),
            "actor_trace": trace,
            "public_frame_after": None if next_frame is None else _frame_record(next_frame),
            "terminal_after_action": bool(environment.state.terminated),
        })
        private_sink({
            "record_type": "physical_step", "trajectory_key": trajectory_key,
            "decision_index": decisions,
            "physical_stacks_before_private": physical_before,
            "physical_stacks_after_private": runtime["private_physical_stacks"](environment),
            "retrieval_count_before": before_retrieval,
            "retrieval_count_after": after_retrieval,
            "relocation_count_before": before_relocations,
            "relocation_count_after": int(environment.relocation_count),
            "provider_event_count_private": int(provider.event_count),
            "provider_history_hits_private": provider.history.hits,
        })
        decisions += 1
        if decisions > max_steps:
            raise ComparisonStop("trajectory exceeded the registered physical relocation bound")
        if next_frame is not None:
            public_before = next_frame
    if int(environment.state.retrieval_count) != int(episode.num_containers):
        raise ComparisonStop("trajectory terminated before full retrieval")
    elapsed = time.monotonic() - started
    if elapsed > TRAJECTORY_CAP_SECONDS:
        raise ComparisonStop(f"trajectory exceeded {TRAJECTORY_CAP_SECONDS}s: {trajectory_key}")
    return {
        "trajectory_key": trajectory_key,
        "layout_id": str(row.get("episode_id", fingerprint)),
        "layout_fingerprint": fingerprint,
        "geometry": str(episode.parameter_group),
        "condition_id": str(condition["condition_id"]),
        "family": str(condition["family"]),
        "strength": str(condition["strength"]),
        "p": condition.get("p"), "window_size": condition.get("window_size"),
        "repeat": int(condition["repeat"]),
        "algorithm": algorithm, "method": method,
        "R_raw": int(environment.relocation_count),
        "N": int(episode.num_containers),
        "R_over_N": float(environment.relocation_count) / float(episode.num_containers),
        "forecast_seed_evaluator_only": seed,
        "F_hash": fhash,
        "complete": True, "decision_count": decisions,
        "wall_seconds": elapsed, "actor_time_seconds": actor_time,
        "eg_cached_moves": int(getattr(policy, "cached_moves", 0)),
    }


def _trajectory_key(row: Mapping[str, Any], condition: Mapping[str, Any], algorithm: str, method: str) -> str:
    return "|".join((str(row["label_invariant_fingerprint"]), str(condition["condition_id"]), algorithm, method))


def _load_receipts() -> tuple[dict[str, Any], dict[str, Any]]:
    bindings_path = OUTPUT_ROOT / "m0/prepare/M0_SOURCE_BINDINGS.json"
    prepare_path = OUTPUT_ROOT / "m0/prepare/M0_PREPARE_RECEIPT.json"
    test_path = OUTPUT_ROOT / "m0/tests/M0_TEST_RECEIPT.json"
    bindings, prepare = _read_json(bindings_path), _read_json(prepare_path)
    if prepare.get("status") != "PASS":
        raise ComparisonStop("M0 prepare receipt is not PASS")
    digest = str(bindings.get("binding_sha256", ""))
    if digest != prepare.get("binding_sha256"):
        raise ComparisonStop("M0 prepare receipt binds different source/model assets")
    if not test_path.is_file():
        raise ComparisonStop("M0 test receipt is not present")
    test = _read_json(test_path)
    if test.get("status") != "PASS" or digest != test.get("binding_sha256"):
        raise ComparisonStop("M0 test receipt is not PASS or has a stale source binding")
    return bindings, test


def _load_prepare_receipt() -> tuple[dict[str, Any], dict[str, Any]]:
    binding_path = OUTPUT_ROOT / "m0/prepare/M0_SOURCE_BINDINGS.json"
    prepare_path = OUTPUT_ROOT / "m0/prepare/M0_PREPARE_RECEIPT.json"
    bindings, prepare = _read_json(binding_path), _read_json(prepare_path)
    if prepare.get("status") != "PASS" or bindings.get("binding_sha256") != prepare.get("binding_sha256"):
        raise ComparisonStop("M0 prepare receipt is missing, stale, or not PASS")
    return bindings, prepare


def prepare_m0() -> dict[str, Any]:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    binding = _build_bindings()
    binding_path = OUTPUT_ROOT / "m0/prepare/M0_SOURCE_BINDINGS.json"
    _write_new_json(binding_path, binding)
    receipt = {
        "protocol_id": "r_five_versions_comparison_v1", "run_id": "r01",
        "stage": "M0_PREPARE", "status": "PASS",
        "binding_sha256": binding["binding_sha256"],
        "binding_path": binding_path.relative_to(WORKSPACE).as_posix(),
        "condition_count": 21, "bank_trajectory_count": 10080,
        "smoke_trajectory_count": 12, "performance_scoring_run": False,
        "training_or_calibration_run": False,
    }
    path = OUTPUT_ROOT / "m0/prepare/M0_PREPARE_RECEIPT.json"
    receipt["receipt_sha256"] = _sha256_bytes(_canonical(receipt))
    _write_new_json(path, receipt)
    return receipt


def _make_test_frame(runtime: Mapping[str, Any]) -> Any:
    from scrp.noise_aware_forecast import PublicObservation
    from experiments.noise_aware_reference_p0 import PublicDecisionFrame
    ranks = np.asarray([[1, 2, 3, 0], [4, 0, 0, 0], [5, 0, 0, 0], [6, 0, 0, 0]], dtype=np.int32)
    occupied = ranks != 0
    legal = np.asarray([False, True, True, True], dtype=np.bool_)
    observation = PublicObservation(
        rank_tensor=ranks, occupied_mask=occupied, legal_destination_mask=legal,
        history_hits=(), q_prior=5183 / 9920, q_t=5183 / 9920,
    )
    return PublicDecisionFrame(
        observation=observation, legal_destinations=(1, 2, 3), max_tiers=4,
        crane_position=None, crane_time=0.0, retrieval_count=0, relocation_count=0,
    )


def _test_eg_commitment(runtime: Mapping[str, Any]) -> dict[str, Any]:
    from types import SimpleNamespace
    from R.scrp.r_five_versions_actor import _SequenceCommitter
    from R.scrp.r_eri_eg_public import EGPlanMismatchError
    from R.scrp.r_eri_eg_public import _after_move as expected_after_move
    from R.scrp.r_order_public import PublicHistoryTracker, attach_history_summary
    frame = _make_test_frame(runtime)
    frame = attach_history_summary(frame, PublicHistoryTracker().summary(frame))
    sequence = (1, 2)
    committer = _SequenceCommitter("EG", lambda _frame: (sequence, {"test": True}))
    first, first_trace = committer.select(frame)
    if first != 1 or first_trace.get("committed_sequence") != [1, 2]:
        raise AssertionError("EG first decision did not commit the full selected plan")
    after = expected_after_move(
        frame, 1, crane_travel_time=1.2, pick_drop_time=30.0
    )
    # The legacy R-EG validator owns the complete public physical snapshot,
    # including crane position/time; deliberately perturb one public scalar.
    bad_observation = SimpleNamespace(
        rank_tensor=after.ranks,
        occupied_mask=after.occupied,
        legal_destination_mask=after.legal_mask,
        history_hits=after.history_hits,
        q_prior=after.q_prior, q_t=after.q_t,
    )
    bad_frame = SimpleNamespace(
        observation=bad_observation, legal_destinations=after.legal_destinations,
        max_tiers=after.max_tiers, crane_position=after.crane_position,
        crane_time=after.crane_time + 1.0, retrieval_count=after.retrieval_count,
        relocation_count=after.relocation_count,
    )
    try:
        committer.select(bad_frame)
    except EGPlanMismatchError:
        return {"full_plan_committed": True, "crane_mismatch_fail_stop": True}
    raise AssertionError("EG cache accepted a public crane-time mismatch")


def test_m0() -> dict[str, Any]:
    bindings, _ = _load_prepare_receipt()
    current = _build_bindings()
    if current["binding_sha256"] != bindings["binding_sha256"]:
        raise ComparisonStop("source or frozen model hash changed after M0 preparation")
    protocol, runtime, gp_rule, models = _load_runtime()
    source_rows = _read_json(LAYOUT_PATH)
    row = next(row for row in source_rows if row.get("parameter_group") == "G1")
    episode = runtime["OnlineCRPEpisode"].from_record(row)
    clean = next(condition for condition in condition_specs() if condition["family"] == "CLEAN")
    forecast, seed = _forecast_for(episode, row["label_invariant_fingerprint"], clean, runtime)
    environment, provider = _make_context(episode, forecast, seed, runtime)
    frame = runtime["public_frame"](episode, environment, provider)
    from R.scrp.r_five_versions_actor import (
        FiveRPolicy, public_frame_record, require_public_frame,
        strict_public_frame_from_record,
    )
    from R.scrp.r_order_actor import HistoryBoundController, FrozenRActor
    actor_runtime = _actor_runtime(runtime)
    decorated = HistoryBoundController.fresh(FiveRPolicy(
        "NATIVE20", "EM", actor_runtime, gp_rule, ridge_scorer=models["ridge"], rates=models["rates"]
    )).observe(frame)
    frame_record = public_frame_record(decorated)
    roundtrip = strict_public_frame_from_record(frame_record)
    if roundtrip.crane_position != frame.crane_position or roundtrip.crane_time != frame.crane_time:
        raise AssertionError("strict public frame round trip changed crane fields")
    incomplete_record = dict(frame_record)
    incomplete_record.pop("crane_position")
    try:
        strict_public_frame_from_record(incomplete_record)
    except ValueError:
        pass
    else:
        raise AssertionError("strict public frame restore defaulted a missing crane field")
    env_a, provider_a = _make_context(episode, forecast, seed, runtime)
    env_b, provider_b = _make_context(episode, forecast, seed, runtime)
    if env_a is env_b or provider_a is provider_b or provider_a.history is provider_b.history:
        raise AssertionError("fresh trajectory contexts share mutable environment/provider state")
    test_frame = _make_test_frame(runtime)
    require_public_frame(test_frame)
    # Occupied unknown 22 remains distinct from an empty zero sentinel.
    unknown_ranks = test_frame.observation.rank_tensor.copy()
    unknown_ranks[1, 0] = 22
    from scrp.noise_aware_forecast import PublicObservation
    unknown_obs = PublicObservation(
        rank_tensor=unknown_ranks,
        occupied_mask=test_frame.observation.occupied_mask,
        legal_destination_mask=test_frame.observation.legal_destination_mask,
        history_hits=(), q_prior=5183 / 9920, q_t=5183 / 9920,
    )
    unknown_frame = replace(test_frame, observation=unknown_obs)
    require_public_frame(unknown_frame)
    if int(unknown_frame.observation.rank_tensor[1, 0]) != 22 or int(unknown_frame.observation.rank_tensor[1, 1]) != 0:
        raise AssertionError("unknown occupied rank collapsed into empty rank")
    private_frame = type("PrivateFrame", (), {
        "observation": frame.observation, "legal_destinations": frame.legal_destinations,
        "max_tiers": frame.max_tiers, "crane_position": frame.crane_position,
        "crane_time": frame.crane_time, "retrieval_count": frame.retrieval_count,
        "relocation_count": frame.relocation_count, "forecast": (1, 2, 3),
    })()
    try:
        require_public_frame(private_frame)
    except ValueError:
        pass
    else:
        raise AssertionError("actor frame accepted evaluator forecast metadata")
    actor_calls = 0
    if any(key in actor_runtime for key in (
        "ForecastProvider", "private_physical_stacks", "make_evaluator_forecast",
        "OnlineCRPEnvironment", "public_frame",
    )):
        raise AssertionError("actor received evaluator-only runtime capabilities")
    for algorithm in ALGORITHMS:
        old = FiveRPolicy("OLD_FROZEN_R", algorithm, actor_runtime, gp_rule,
                          ridge_scorer=models["ridge"], rates=models["rates"])
        frozen_reference = FrozenRActor(algorithm, actor_runtime, gp_rule)
        old_action, _ = old.select(decorated)
        reference_action, _ = frozen_reference.select(decorated)
        if int(old_action) != int(reference_action):
            raise AssertionError(f"OLD_FROZEN_R changed delegate behavior for {algorithm}")
        for method in POLICIES:
            policy = FiveRPolicy(
                method, algorithm, actor_runtime, gp_rule,
                ridge_scorer=models["ridge"], rates=models["rates"],
            )
            controller = HistoryBoundController.fresh(policy)
            action, trace, _decorated = controller.select(frame)
            if int(action) not in frame.legal_destinations:
                raise AssertionError(f"{method}/{algorithm} returned illegal action {action}")
            if method == "R_SOURCE_FULL" and [item["h"] for item in trace["source_candidates"]] != [0, 2, 5, 10, 20]:
                raise AssertionError("R_SOURCE_FULL failed to retain every registered source h")
            if method == "R_QUAD_H":
                from R.scrp.r_horizon_robustness import choose_r_horizon
                if int(trace["selected_h"]) != choose_r_horizon(float(frame.observation.q_t), rule="quad")[0]:
                    raise AssertionError("R_QUAD_H differs from the frozen q-to-H mapping")
            if method == "R_ORDER_MAIN_TAU0_FULL" and float(trace["tau"]) != 0.0:
                raise AssertionError("R-Order candidate does not use the fixed registered tau=0")
            actor_calls += 1
    eg = _test_eg_commitment(runtime)
    return {
        "protocol_id": "r_five_versions_comparison_v1", "run_id": "r01",
        "stage": "M0_TEST", "status": "PASS",
        "binding_sha256": bindings["binding_sha256"],
        "checks": {
            "source_and_model_hashes_stable": True,
            "strict_public_crane_roundtrip_and_missing_field_rejection": True,
            "fresh_environment_provider_state": True,
            "forecast_private_frame_rejected": True,
            "actor_runtime_capability_allowlist": True,
            "unknown_22_distinct_from_empty_0": True,
            "registered_condition_keys_unique_21": len(condition_specs()) == len({row["condition_id"] for row in condition_specs()}) == 21,
            "original_R_delegate_regression_four_solvers": True,
            "all_policy_solver_first_action_calls": actor_calls,
            "R_SOURCE_FULL_retains_all_source_h": True,
            "R_QUAD_H_uses_frozen_rule": True,
            "R_ORDER_tau_fixed_zero": True,
            "EG_cache": eg,
        },
        "performance_scoring_run": False,
        "trajectories_run": 0,
        "smoke_authorized_or_run": False,
    }


def run_test_m0() -> dict[str, Any]:
    receipt = test_m0()
    path = OUTPUT_ROOT / "m0/tests/M0_TEST_RECEIPT.json"
    receipt["receipt_sha256"] = _sha256_bytes(_canonical(receipt))
    _write_new_json(path, receipt)
    return receipt


def _verify_release(scope: str) -> tuple[dict[str, Any], dict[str, Any]]:
    bindings, test_receipt = _load_receipts()
    current_binding = _build_bindings()
    if current_binding["binding_sha256"] != bindings.get("binding_sha256"):
        raise ComparisonStop("current source/model hashes differ from reviewed M0 bindings")
    marker_path = OUTPUT_ROOT / ("ROOT_M1_SMOKE_RELEASE.json" if scope == "smoke" else "ROOT_M1_BANK_RELEASE.json")
    if not marker_path.is_file():
        raise ComparisonStop(f"ROOT release marker is required before {scope}: {marker_path.relative_to(WORKSPACE)}")
    marker = _read_json(marker_path)
    expected = {
        "protocol_id": "r_five_versions_comparison_v1", "run_id": "r01",
        "scope": scope, "authorized": True,
        "m0_binding_sha256": bindings["binding_sha256"],
        "m0_test_receipt_sha256": test_receipt["receipt_sha256"],
    }
    for key, value in expected.items():
        if marker.get(key) != value:
            raise ComparisonStop(f"ROOT {scope} release differs at {key}")
    if scope == "bank":
        smoke_receipt_path = OUTPUT_ROOT / "smoke_01/M1_SMOKE_RECEIPT.json"
        smoke_receipt = _read_json(smoke_receipt_path)
        if smoke_receipt.get("status") != "PASS" or marker.get("smoke_receipt_sha256") != smoke_receipt.get("receipt_sha256"):
            raise ComparisonStop("bank release must bind the reviewed PASS smoke receipt")
    else:
        smoke_receipt = {}
    return marker, smoke_receipt


def _csv_write(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    rows = list(rows)
    if not rows:
        raise ComparisonStop("no raw trajectory rows to write")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with path.open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(_json_safe(value), ensure_ascii=False) if isinstance(value, (dict, list, tuple)) else value for key, value in row.items()})


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def _jsonl_gz_append(stream: Any, row: Mapping[str, Any]) -> None:
    stream.write(json.dumps(_json_safe(row), ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")


def _selection(layouts: Sequence[tuple[dict[str, Any], Any]], conditions: Sequence[dict[str, Any]], scope: str) -> list[tuple[dict[str, Any], Any, dict[str, Any], str, str]]:
    output: list[tuple[dict[str, Any], Any, dict[str, Any], str, str]] = []
    if scope == "smoke":
        selected_layout = next(item for item in layouts if item[0].get("parameter_group") == "G1")
        condition = next(row for row in conditions if row["family"] == "delayed_positions" and row["p"] == 0.3 and row["repeat"] == 0)
        for algorithm in ALGORITHMS:
            for method in ("OLD_FROZEN_R", "R_UTILITY_RIDGE", "R_ORDER_MAIN_TAU0_FULL"):
                output.append((selected_layout[0], selected_layout[1], condition, algorithm, method))
        return output
    for row, episode in layouts:
        for condition in conditions:
            for algorithm in ALGORITHMS:
                for method in POLICIES:
                    output.append((row, episode, condition, algorithm, method))
    return output


def run_stage(scope: str) -> dict[str, Any]:
    if scope not in ("smoke", "bank"):
        raise ValueError("scope must be smoke or bank")
    marker, _ = _verify_release(scope)
    stage_started = time.monotonic()
    output_dir = OUTPUT_ROOT / ("smoke_01" if scope == "smoke" else "bank_01")
    output_dir.mkdir(parents=True, exist_ok=False)
    protocol, runtime, gp_rule, models = _load_runtime()
    raw_rows: list[dict[str, Any]] = []
    smoke_keys: set[str] = set()
    smoke_rows_by_key: dict[str, dict[str, str]] = {}
    if scope == "bank":
        smoke_dir = OUTPUT_ROOT / "smoke_01"
        raw_rows = _read_csv(smoke_dir / "raw_trajectories.csv")
        smoke_keys = {row["trajectory_key"] for row in raw_rows}
        if len(smoke_keys) != 12 or any(row["complete"].lower() != "true" for row in raw_rows):
            raise ComparisonStop("PASS smoke must contain exactly 12 unique complete rows")
        smoke_rows_by_key = {row["trajectory_key"]: row for row in raw_rows}
        shutil.copyfile(smoke_dir / "public_decision_trace.jsonl.gz", output_dir / "public_decision_trace.jsonl.gz")
        (output_dir / "private").mkdir()
        for name in ("evaluator_sidecar.jsonl.gz", "physical_steps.jsonl.gz"):
            shutil.copyfile(smoke_dir / "private" / name, output_dir / "private" / name)
    else:
        (output_dir / "private").mkdir()
    if scope == "bank":
        previous_wall = float(_read_json(OUTPUT_ROOT / "smoke_01/M1_SMOKE_RECEIPT.json")["science_wall_seconds"])
        if previous_wall >= SCIENCE_CAP_SECONDS:
            raise ComparisonStop("smoke exhausted the registered cumulative science cap")
    else:
        previous_wall = 0.0
    if scope == "smoke":
        public_stream = gzip.open(output_dir / "public_decision_trace.jsonl.gz", "wt", encoding="utf-8")
        evaluator_stream = gzip.open(output_dir / "private/evaluator_sidecar.jsonl.gz", "wt", encoding="utf-8")
        physical_stream = gzip.open(output_dir / "private/physical_steps.jsonl.gz", "wt", encoding="utf-8")
    else:
        public_stream = gzip.open(output_dir / "public_decision_trace.jsonl.gz", "at", encoding="utf-8")
        evaluator_stream = gzip.open(output_dir / "private/evaluator_sidecar.jsonl.gz", "at", encoding="utf-8")
        physical_stream = gzip.open(output_dir / "private/physical_steps.jsonl.gz", "at", encoding="utf-8")
    try:
        raw_layouts = _read_json(LAYOUT_PATH)
        layouts = [(row, runtime["OnlineCRPEpisode"].from_record(row)) for row in raw_layouts]
        conditions = condition_specs()
        selected = _selection(layouts, conditions, scope)
        expected = 12 if scope == "smoke" else 10080
        if len(selected) != expected:
            raise ComparisonStop(f"{scope} selection count {len(selected)} != {expected}")
        forecast_cache: dict[str, tuple[tuple[int, ...], int | None]] = {}
        for layout_row, episode, condition, algorithm, method in selected:
            key = _trajectory_key(layout_row, condition, algorithm, method)
            forecast_key = f"{layout_row['label_invariant_fingerprint']}|{condition['condition_id']}"
            if forecast_key not in forecast_cache:
                forecast_cache[forecast_key] = _forecast_for(
                    episode, str(layout_row["label_invariant_fingerprint"]), condition, runtime
                )
            forecast, seed = forecast_cache[forecast_key]
            fhash = _public_hash(forecast)
            if scope == "bank" and key in smoke_keys:
                reused = smoke_rows_by_key[key]
                reused_seed = reused.get("forecast_seed_evaluator_only", "")
                expected_seed = "" if seed is None else str(seed)
                if (reused.get("F_hash") != fhash
                        or reused.get("forecast_seed_evaluator_only", "") != expected_seed
                        or reused.get("layout_fingerprint") != str(layout_row["label_invariant_fingerprint"])
                        or reused.get("condition_id") != str(condition["condition_id"])
                        or reused.get("algorithm") != algorithm or reused.get("method") != method):
                    raise ComparisonStop(f"smoke reuse key/hash mismatch: {key}")
                continue
            if previous_wall + time.monotonic() - stage_started > SCIENCE_CAP_SECONDS:
                raise ComparisonStop("cumulative registered 2h science wall cap reached")
            _jsonl_gz_append(evaluator_stream, {
                "record_type": "forecast_binding", "trajectory_key": key,
                "layout_fingerprint": layout_row["label_invariant_fingerprint"],
                "condition_evaluator_only": condition, "forecast_seed_evaluator_only": seed,
                "forecast_sha256_evaluator_only": fhash,
            })
            row = _run_trajectory(
                episode=episode, row=layout_row, condition=condition,
                algorithm=algorithm, method=method, forecast=forecast, seed=seed,
                runtime=runtime, gp_rule=gp_rule, models=models,
                trajectory_key=key,
                public_sink=lambda item: _jsonl_gz_append(public_stream, item),
                private_sink=lambda item: _jsonl_gz_append(evaluator_stream if item.get("record_type") == "trajectory_start" else physical_stream, item),
            )
            if row["F_hash"] != fhash:
                raise ComparisonStop("actor trajectory forecast differs from shared immutable condition F")
            raw_rows.append(row)
        public_stream.close()
        evaluator_stream.close()
        physical_stream.close()
        if len({row["trajectory_key"] for row in raw_rows}) != len(raw_rows):
            raise ComparisonStop("duplicate trajectory key in raw bank")
        if len(raw_rows) != expected:
            raise ComparisonStop(f"completed trajectory row count {len(raw_rows)} != {expected}")
        _csv_write(output_dir / "raw_trajectories.csv", raw_rows)
        elapsed = time.monotonic() - stage_started
        if previous_wall + elapsed > SCIENCE_CAP_SECONDS:
            raise ComparisonStop("cumulative registered 2h science cap exceeded")
        receipt = {
            "protocol_id": "r_five_versions_comparison_v1", "run_id": "r01",
            "scope": scope, "status": "PASS", "authorized_by_release_sha256": _sha256_file(
                OUTPUT_ROOT / ("ROOT_M1_SMOKE_RELEASE.json" if scope == "smoke" else "ROOT_M1_BANK_RELEASE.json")
            ),
            "m0_binding_sha256": marker["m0_binding_sha256"],
            "trajectory_count": len(raw_rows), "unique_trajectory_keys": len({r["trajectory_key"] for r in raw_rows}),
            "science_wall_seconds": previous_wall + elapsed,
            "stage_wall_seconds": elapsed,
            "raw_trajectory_csv": (output_dir / "raw_trajectories.csv").relative_to(WORKSPACE).as_posix(),
            "public_trace": (output_dir / "public_decision_trace.jsonl.gz").relative_to(WORKSPACE).as_posix(),
            "private_sidecars": [
                (output_dir / "private/evaluator_sidecar.jsonl.gz").relative_to(WORKSPACE).as_posix(),
                (output_dir / "private/physical_steps.jsonl.gz").relative_to(WORKSPACE).as_posix(),
            ],
        }
        receipt_path = output_dir / ("M1_SMOKE_RECEIPT.json" if scope == "smoke" else "M1_BANK_RECEIPT.json")
        receipt["receipt_sha256"] = _sha256_bytes(_canonical(receipt))
        _write_new_json(receipt_path, receipt)
        return receipt
    except BaseException as exc:
        for stream in (public_stream, evaluator_stream, physical_stream):
            try:
                stream.close()
            except BaseException:
                pass
        elapsed = time.monotonic() - stage_started
        partial_path = output_dir / "partial_raw_trajectories.csv"
        partial_sha = None
        if raw_rows:
            try:
                _csv_write(partial_path, raw_rows)
                partial_sha = _sha256_file(partial_path)
            except BaseException:
                partial_path = None
        failure = {
            "protocol_id": "r_five_versions_comparison_v1", "run_id": "r01",
            "scope": scope, "status": "STOP", "error_type": type(exc).__name__,
            "error": str(exc), "traceback": traceback.format_exc(),
            "completed_rows": len(raw_rows), "m0_binding_sha256": marker["m0_binding_sha256"],
            "stage_wall_seconds": elapsed,
            "science_wall_seconds": previous_wall + elapsed,
            "partial_raw_csv": None if partial_path is None else partial_path.relative_to(WORKSPACE).as_posix(),
            "partial_raw_csv_sha256": partial_sha,
        }
        try:
            _write_new_json(output_dir / f"{scope.upper()}_FAILURE_RECEIPT.json", failure)
        finally:
            raise


def status() -> dict[str, Any]:
    items = {}
    for relative in (
        "m0/prepare/M0_PREPARE_RECEIPT.json", "m0/tests/M0_TEST_RECEIPT.json",
        "smoke_01/M1_SMOKE_RECEIPT.json", "bank_01/M1_BANK_RECEIPT.json",
    ):
        path = OUTPUT_ROOT / relative
        items[relative] = _read_json(path) if path.is_file() else "not run"
    return {"protocol_id": "r_five_versions_comparison_v1", "run_id": "r01", "receipts": items}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="PLAN-051 M0 and ROOT-released comparison runner")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("prepare-m0")
    sub.add_parser("test-m0")
    sub.add_parser("run-smoke")
    sub.add_parser("run-bank")
    sub.add_parser("status")
    args = parser.parse_args(argv)
    try:
        if args.command == "prepare-m0":
            result = prepare_m0()
        elif args.command == "test-m0":
            result = run_test_m0()
        elif args.command == "run-smoke":
            result = run_stage("smoke")
        elif args.command == "run-bank":
            result = run_stage("bank")
        else:
            result = status()
        print(json.dumps(_json_safe(result), sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False))
        return 0
    except BaseException as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
