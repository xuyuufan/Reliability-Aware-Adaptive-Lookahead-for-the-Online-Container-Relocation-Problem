"""Shared fit-state assignment and counterfactual label contracts for PLAN-045."""

from __future__ import annotations

import copy
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Callable, Mapping, Sequence

import numpy as np


R_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = R_ROOT.parent
EM_ROOT = WORKSPACE / "EM"
BASE_ROOT = WORKSPACE / "Mijn_RL"
M0_DIR = R_ROOT / "output/h20_r_adaptation_v1/M0"
M0_RECEIPT_PATH = M0_DIR / "M0_PREPARATION_RECEIPT.json"
FIT_LAYOUT_PATH = M0_DIR / "fit.json"
CONFIG_PATH = R_ROOT / "configs/h20_r_adaptation_v1.json"
CHECKPOINT_PATH = WORKSPACE / "Mijn_RL/checkpoints/h20.best.pt"
PROTOCOL_ID = "h20_r_adaptation_v1"
RELEASE_SCOPE = ("fit_label_bank_only",)
EXPECTED_CHECKPOINT_SHA256 = "aedd5e744607359318352f699cb4320f85dbbfc2c12b510047e04bd0013ef2d0"
EXPECTED_FIT_LAYOUT_SHA256 = "0cf94bffae49a5e58b955e36cdfc600b3ebaa85e1f4e3884903850aed92f0767"
EXPECTED_M0_RECEIPT_SHA256 = "c6326fa9bcc4a10d9a044465f8dcb9cfe10d3d87df45530924ec50585e358839"
BANK_SOURCE_PATHS = (
    "EM/experiments/noise_aware_lookahead.py",
    "EM/experiments/noise_aware_reference_p0.py",
    "EM/scrp/noise_aware_forecast.py",
    "Mijn_RL/checkpoints/h20.best.pt",
    "Mijn_RL/scrp/attention_data.py",
    "Mijn_RL/scrp/h20_a1_policy.py",
    "Mijn_RL/scrp/online_crp.py",
    "Mijn_RL/scrp/scale_diverse_policy.py",
    "R/configs/h20_r_adaptation_v1.json",
    "R/experiments/h20_r_label_bank.py",
    "R/experiments/h20_r_prepare.py",
    "R/experiments/history_reliability_r_protocol.py",
    "R/scrp/cross_solver_reliability_r.py",
    "R/scrp/h20_r_bank.py",
    "R/scrp/h20_r_data.py",
)


def _install_workspace_import_paths() -> None:
    for root in (WORKSPACE, R_ROOT, EM_ROOT, BASE_ROOT):
        text = str(root.resolve())
        if text not in sys.path:
            sys.path.insert(0, text)
    for package, roots in (
        ("scrp", (R_ROOT, EM_ROOT, BASE_ROOT)),
        ("experiments", (EM_ROOT, R_ROOT, BASE_ROOT)),
    ):
        module = sys.modules.get(package)
        if module is None:
            __import__(package)
            module = sys.modules[package]
        locations = [str((root / package).resolve()) for root in roots]
        module.__path__ = locations
        spec = getattr(module, "__spec__", None)
        if spec is not None and spec.submodule_search_locations is not None:
            spec.submodule_search_locations[:] = locations


_install_workspace_import_paths()

from experiments.history_reliability_r_protocol import (  # noqa: E402
    FAMILIES,
    P_VALUES,
    REPEATS,
    make_evaluator_forecast,
    semantic_forecast_seed,
)
from experiments.noise_aware_lookahead import (  # noqa: E402
    _private_physical_stacks,
    sync_retrieval_events,
)
from experiments.noise_aware_reference_p0 import (  # noqa: E402
    PublicDecisionFrame,
    h20_action,
    load_frozen_h20,
    make_public_frame,
)
from scrp.attention_data import file_sha256, label_invariant_fingerprint  # noqa: E402
from scrp.cross_solver_reliability_r import truncate_public_frame  # noqa: E402
from scrp.h20_r_data import HORIZONS, Q_PRIOR  # noqa: E402
from scrp.noise_aware_forecast import ForecastProvider  # noqa: E402
from scrp.online_crp import OnlineCRPEpisode, OnlineCRPEnvironment  # noqa: E402


@dataclass(frozen=True)
class FitStateAssignment:
    state_id: str
    family: str
    p: float
    repeat: int
    geometry_id: str
    layout_fingerprint: str
    stage_target: float


@dataclass(frozen=True)
class DecisionPosition:
    decision_index: int
    retrieval_count: int
    frame: PublicDecisionFrame


@dataclass(frozen=True)
class CapturedDecision:
    decision_index: int
    retrieval_count: int
    relocation_count: int
    frame: PublicDecisionFrame
    anchor_action: int
    environment: OnlineCRPEnvironment
    provider: ForecastProvider


@dataclass(frozen=True)
class BehaviorCapture:
    selected: CapturedDecision | None
    decisions: int
    behavior_tail_relocations: int | None
    forecast_sha256: str
    evaluator_private_binding: Mapping[str, Any]


@dataclass(frozen=True)
class ContinuationResult:
    remaining_relocations: int
    private_trace: tuple[Mapping[str, Any], ...]


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def current_source_sha256(config_path: Path = CONFIG_PATH) -> dict[str, str]:
    config_path = config_path.resolve()
    try:
        config_relative = config_path.relative_to(WORKSPACE.resolve()).as_posix()
    except ValueError as exc:
        raise ValueError("release-bound config must remain inside the workspace") from exc
    paths = tuple(
        config_relative if relative == "R/configs/h20_r_adaptation_v1.json" else relative
        for relative in BANK_SOURCE_PATHS
    )
    hashes: dict[str, str] = {}
    for relative in paths:
        path = WORKSPACE.joinpath(*relative.split("/"))
        if not path.is_file():
            raise FileNotFoundError(f"release-bound source is missing: {relative}")
        hashes[relative] = file_sha256(path)
    return hashes


def current_m0_binding(config_path: Path = CONFIG_PATH) -> dict[str, Any]:
    if not M0_RECEIPT_PATH.is_file() or not FIT_LAYOUT_PATH.is_file():
        raise FileNotFoundError("accepted M0 fit manifest or receipt is missing")
    receipt = json.loads(M0_RECEIPT_PATH.read_text(encoding="utf-8"))
    fit_sha = file_sha256(FIT_LAYOUT_PATH)
    if (
        receipt.get("protocol_id") != PROTOCOL_ID
        or int(receipt.get("exit_code", -1)) != 0
        or receipt.get("counterfactual_labels_generated") is not False
        or receipt.get("fit_states_sampled") is not False
        or receipt.get("model_loaded") is not False
        or receipt.get("h20_checkpoint_sha256") != EXPECTED_CHECKPOINT_SHA256
        or receipt.get("config_sha256") != file_sha256(config_path.resolve())
    ):
        raise ValueError("M0 receipt is not the accepted no-label preparation receipt")
    fit_record = receipt.get("splits", {}).get("fit", {})
    if (
        fit_record.get("path") != "output/h20_r_adaptation_v1/M0/fit.json"
        or fit_record.get("sha256") != fit_sha
        or fit_sha != EXPECTED_FIT_LAYOUT_SHA256
    ):
        raise ValueError("M0 fit manifest differs from its accepted receipt")
    receipt_sha = file_sha256(M0_RECEIPT_PATH)
    if receipt_sha != EXPECTED_M0_RECEIPT_SHA256:
        raise ValueError("M0 preparation receipt bytes differ from the accepted artifact")
    return {
        "m0_receipt_sha256": receipt_sha,
        "fit_layout_sha256": fit_sha,
        "fit_layout_path": str(FIT_LAYOUT_PATH.resolve()),
    }


def expected_release_contract(config_path: Path = CONFIG_PATH) -> dict[str, Any]:
    """Return the exact root-authored release payload needed to start fit labels."""
    from scrp.h20_r_data import Q_PRIOR as DATA_Q_PRIOR

    if abs(DATA_Q_PRIOR - Q_PRIOR) > 0.0:
        raise AssertionError("public history prior changed between bank and feature modules")
    if file_sha256(CHECKPOINT_PATH) != EXPECTED_CHECKPOINT_SHA256:
        raise ValueError("frozen H20 checkpoint hash differs from PLAN-045")
    binding = current_m0_binding(config_path)
    return {
        "protocol_id": PROTOCOL_ID,
        "stage": "fit_label_bank",
        "status": "RELEASED",
        "scope": list(RELEASE_SCOPE),
        "config_sha256": file_sha256(config_path.resolve()),
        "fit_layout_sha256": binding["fit_layout_sha256"],
        "m0_receipt_sha256": binding["m0_receipt_sha256"],
        "h20_checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
        "source_sha256": current_source_sha256(config_path),
    }


def verify_fit_label_release(
    release_path: Path,
    *,
    config_path: Path = CONFIG_PATH,
) -> dict[str, Any]:
    """Fail closed on missing, stale, broader-than-fit, or malformed release."""
    if not release_path.is_file():
        raise FileNotFoundError(f"fit-label release is missing: {release_path}")
    release = json.loads(release_path.read_text(encoding="utf-8"))
    expected = expected_release_contract(config_path)
    allowed = set(expected) | {"release_id", "review_receipt_path", "review_receipt_sha256"}
    required = set(expected)
    if not required.issubset(release) or set(release) - allowed:
        raise ValueError("fit-label release has missing or unregistered fields")
    if release.get("scope") != list(RELEASE_SCOPE):
        raise ValueError("fit-label release scope must be fit_label_bank_only")
    for key, expected_value in expected.items():
        if release.get(key) != expected_value:
            raise ValueError(f"fit-label release is stale or mismatched at {key}")
    for key in ("review_receipt_path", "review_receipt_sha256"):
        if key in release and not str(release[key]).strip():
            raise ValueError(f"optional release field {key} cannot be empty")
    return {
        "release_sha256": file_sha256(release_path),
        "release": release,
        "expected": expected,
    }


def load_fit_episodes(path: Path = FIT_LAYOUT_PATH) -> tuple[OnlineCRPEpisode, ...]:
    if not path.is_file():
        raise FileNotFoundError(f"fit layout manifest is missing: {path}")
    records = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise ValueError("fit layout manifest must be a JSON list")
    episodes = tuple(OnlineCRPEpisode.from_record(row) for row in records)
    fingerprints = [label_invariant_fingerprint(episode) for episode in episodes]
    if len(fingerprints) != len(set(fingerprints)):
        raise ValueError("fit layout manifest contains duplicate semantic fingerprints")
    counts = {
        geometry: sum(episode.parameter_group == geometry for episode in episodes)
        for geometry in ("G1", "G2")
    }
    if len(episodes) != 80 or counts != {"G1": 40, "G2": 40}:
        raise ValueError(f"fit layout split differs from M0 contract: {counts}")
    return episodes


def build_fit_state_assignments(
    episodes: Sequence[OnlineCRPEpisode],
    *,
    seed: int = 202610114503,
    per_cell: int = 32,
    per_geometry: int = 16,
    stage_buckets: Sequence[float] = (0.15, 0.50, 0.85),
) -> tuple[FitStateAssignment, ...]:
    """Build the registered matrix using layout identity and a fixed PCG64 seed only."""
    stages = tuple(float(value) for value in stage_buckets)
    if not stages or any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in stages):
        raise ValueError("stage buckets must be finite fractions in [0, 1]")
    if len(set(stages)) != len(stages) or per_cell <= 0 or per_geometry <= 0:
        raise ValueError("stage buckets and sample counts must be positive and unique")
    if per_cell != per_geometry * 2:
        raise ValueError("registered matrix requires exactly two balanced geometries")

    by_geometry: dict[str, list[tuple[str, OnlineCRPEpisode]]] = {"G1": [], "G2": []}
    for episode in episodes:
        geometry = str(episode.parameter_group)
        if geometry not in by_geometry:
            raise ValueError(f"unexpected fit geometry {geometry!r}")
        by_geometry[geometry].append((label_invariant_fingerprint(episode), episode))
    for geometry, rows in by_geometry.items():
        rows.sort(key=lambda pair: pair[0])
        if len(rows) < per_geometry:
            raise ValueError(f"insufficient {geometry} layouts for the registered state matrix")
    if len({fingerprint for rows in by_geometry.values() for fingerprint, _ in rows}) != sum(
        len(rows) for rows in by_geometry.values()
    ):
        raise ValueError("fit layout fingerprints must be unique across geometries")

    quota, remainder = divmod(per_cell, len(stages))
    stage_schedule = tuple(
        stage
        for index, stage in enumerate(stages)
        for _ in range(quota + int(index < remainder))
    )
    assignments: list[FitStateAssignment] = []
    cell_index = 0
    for family in FAMILIES:
        for p in P_VALUES:
            for repeat in REPEATS:
                rng = np.random.Generator(np.random.PCG64(int(seed) + cell_index))
                selected: list[tuple[str, str]] = []
                for geometry in ("G1", "G2"):
                    rows = by_geometry[geometry]
                    indices = rng.permutation(len(rows))[:per_geometry]
                    selected.extend((geometry, rows[int(index)][0]) for index in indices)
                stage_order = rng.permutation(np.asarray(stage_schedule, dtype=np.float64))
                selected_order = rng.permutation(len(selected))
                for offset, selected_index in enumerate(selected_order.tolist()):
                    geometry, fingerprint = selected[selected_index]
                    state_id = (
                        f"{family}|p={p:.1f}|repeat={repeat}|state={offset:02d}"
                    )
                    assignments.append(
                        FitStateAssignment(
                            state_id=state_id,
                            family=family,
                            p=float(p),
                            repeat=int(repeat),
                            geometry_id=geometry,
                            layout_fingerprint=fingerprint,
                            stage_target=float(stage_order[offset]),
                        )
                    )
                cell_index += 1
    if len(assignments) != 512:
        if len(episodes) == 80 and per_cell == 32 and per_geometry == 16:
            raise AssertionError("registered 512-state matrix size changed")
    return tuple(assignments)


def assignment_manifest(
    assignments: Sequence[FitStateAssignment], *, seed: int = 202610114503
) -> dict[str, Any]:
    rows = [
        {
            "state_id": row.state_id,
            "family": row.family,
            "p": row.p,
            "repeat": row.repeat,
            "geometry_id": row.geometry_id,
            "layout_fingerprint": row.layout_fingerprint,
            "stage_target": row.stage_target,
        }
        for row in assignments
    ]
    cells: dict[str, dict[str, Any]] = {}
    for family in FAMILIES:
        for p in P_VALUES:
            for repeat in REPEATS:
                cell_rows = [
                    row
                    for row in assignments
                    if row.family == family and row.p == p and row.repeat == repeat
                ]
                key = f"{family}|p={p:.1f}|repeat={repeat}"
                cells[key] = {
                    "count": len(cell_rows),
                    "by_geometry": {
                        geometry: sum(row.geometry_id == geometry for row in cell_rows)
                        for geometry in ("G1", "G2")
                    },
                    "by_stage": {
                        f"{stage:.2f}": sum(row.stage_target == stage for row in cell_rows)
                        for stage in (0.15, 0.50, 0.85)
                    },
                }
    first = rows[: next(iter(cells.values()))["count"]]
    return {
        "protocol_id": PROTOCOL_ID,
        "artifact_role": "prelabel_state_assignment_only",
        "sampling_rule": "PCG64(seed + cell_index); 16 layouts per geometry; balanced stage schedule; no cost or outcome inputs",
        "sampling_seed": int(seed),
        "total_assignments": len(rows),
        "stage_schedule_per_cell": {f"{stage:.2f}": count for stage, count in _stage_counts(first).items()},
        "cells": cells,
        "assignments": rows,
        "scientific_labels_generated": False,
    }


def _stage_counts(rows: Sequence[Mapping[str, Any]]) -> dict[float, int]:
    return {
        stage: sum(float(row["stage_target"]) == stage for row in rows)
        for stage in (0.15, 0.50, 0.85)
    }


def choose_nearest_decision(
    decisions: Sequence[DecisionPosition], *, stage_target: float, original_n: int
) -> DecisionPosition | None:
    if original_n <= 0 or not math.isfinite(stage_target) or not 0.0 <= stage_target <= 1.0:
        raise ValueError("invalid stage target or original episode size")
    if not decisions:
        return None
    for decision in decisions:
        if decision.frame.retrieval_count != decision.retrieval_count:
            raise ValueError("decision retrieval count differs from its public frame")
        if not decision.frame.legal_destinations:
            raise ValueError("only legal relocation decisions may be sampled")
    return min(
        decisions,
        key=lambda item: (
            abs(item.retrieval_count - stage_target * original_n),
            item.decision_index,
        ),
    )


def _forecast_hash(forecast: Sequence[int]) -> str:
    return hashlib.sha256(_canonical_json_bytes([int(value) for value in forecast])).hexdigest()


def _private_signature(
    environment: OnlineCRPEnvironment, provider: ForecastProvider
) -> tuple[Any, ...]:
    return (
        tuple(tuple(int(item) for item in stack) for stack in environment._stacks),
        tuple(sorted((int(key), value) for key, value in environment._locations.items())),
        int(environment._next_index),
        int(environment._relocations),
        environment._crane_position,
        float(environment._crane_time),
        bool(environment._terminated),
        tuple(provider._true_sequence),
        tuple(provider._forecast_future),
        provider._current,
        tuple(sorted(provider._retrieved)),
        tuple(sorted(provider._seen)),
        provider.history.hits,
        float(provider.history.q_prior),
        int(provider.history.window),
        int(provider.event_count),
    )


def _make_context(
    episode: OnlineCRPEpisode, *, family: str, p: float, repeat: int
) -> tuple[OnlineCRPEnvironment, ForecastProvider, tuple[int, ...], dict[str, Any]]:
    fingerprint = label_invariant_fingerprint(episode)
    seed = semantic_forecast_seed(fingerprint, family, repeat)
    forecast = make_evaluator_forecast(
        episode.retrieval_sequence[1:],
        family=family,
        p_evaluator_only=p,
        seed=seed,
    )
    provider = ForecastProvider(
        episode.retrieval_sequence,
        p=p,
        seed=seed,
        q_prior=Q_PRIOR,
        max_horizon=20,
        forecast_future=forecast,
    )
    environment = OnlineCRPEnvironment(episode, lookahead_horizon=0)
    reset_count = int(environment.state.retrieval_count)
    reset_events = _sync_retrieval_events_with_audit(
        provider, episode.retrieval_sequence, 0, reset_count
    )
    reset_record = {
        "retrieval_count_after_reset": reset_count,
        "revealed_targets_private": [int(row["revealed_target_id_private"]) for row in reset_events],
        "reveal_events_private": reset_events,
    }
    return environment, provider, tuple(forecast), reset_record


def _rolling_q(history: Sequence[bool], q_prior: float) -> float:
    window = tuple(bool(value) for value in history[-8:])
    return float((2.0 * q_prior + sum(window)) / (2.0 + len(window)))


def _sync_retrieval_events_with_audit(
    provider: ForecastProvider,
    true_sequence: Sequence[int],
    before_count: int,
    after_count: int,
) -> list[dict[str, Any]]:
    prior_hits = list(provider.history.hits)
    prior_current = provider._current
    prior_retrieved = set(provider._retrieved)
    event_rows: list[dict[str, Any]] = []
    sync_retrieval_events(
        provider,
        tuple(int(value) for value in true_sequence),
        before_count,
        after_count,
        event_rows=event_rows,
        record_q_t_before=True,
    )
    expected_events = max(0, min(after_count, len(true_sequence) - 1) - min(before_count, len(true_sequence) - 1))
    if len(event_rows) != expected_events:
        raise AssertionError("reveal event count differs from retrieval-count advancement")

    hits = prior_hits[-8:]
    current = prior_current
    retrieved = prior_retrieved
    audited: list[dict[str, Any]] = []
    event_offset = 0
    for retrieved_index in range(before_count, after_count):
        next_index = retrieved_index + 1
        if next_index >= len(true_sequence):
            continue
        event = event_rows[event_offset]
        event_offset += 1
        if current is None:
            raise AssertionError("provider ended before a required reveal event")
        excluded = retrieved | {int(current)}
        predicted = next(
            (int(item) for item in provider._forecast_future if int(item) not in excluded),
            None,
        )
        revealed = int(true_sequence[next_index])
        hit = bool(event["hit"])
        q_before = _rolling_q(hits, provider.history.q_prior)
        if abs(q_before - float(event["q_t_before_reveal"])) > 1e-15:
            raise AssertionError("causal history q_t differs before a private reveal")
        if (predicted == revealed) != hit:
            raise AssertionError("private forecast hit differs from the recorded reveal")
        hits.append(hit)
        hits = hits[-8:]
        q_after = _rolling_q(hits, provider.history.q_prior)
        audited.append(
            {
                "event_index": int(event["event_index"]),
                "previous_target_id_private": int(current),
                "predicted_next_target_id_private": predicted,
                "revealed_target_id_private": revealed,
                "hit_private": hit,
                "q_t_before_reveal": q_before,
                "q_t_after_reveal": q_after,
            }
        )
        retrieved.add(int(current))
        current = revealed
    if event_offset != len(event_rows):
        raise AssertionError("not all synchronization events were privately audited")
    if tuple(hits) != provider.history.hits or current != provider._current:
        raise AssertionError("audited reveal state differs from the live causal provider")
    return audited


def _private_capture_record(
    episode: OnlineCRPEpisode,
    environment: OnlineCRPEnvironment,
    provider: ForecastProvider,
) -> dict[str, Any]:
    return {
        "physical_stacks_private": _private_physical_stacks(environment),
        "current_target_id_private": environment.current_target_id,
        "retrieval_count": int(environment.state.retrieval_count),
        "relocation_count": int(environment.relocation_count),
        "crane_position": environment.state.crane_position,
        "crane_time": float(environment.state.crane_time),
        "history_hits_private": list(provider.history.hits),
        "q_prior": float(provider.history.q_prior),
        "q_t": float(provider.history.q_t),
        "history_event_count": int(provider.event_count),
        "provider_current_target_id_private": provider._current,
        "provider_retrieved_target_ids_private": sorted(int(item) for item in provider._retrieved),
        "provider_seen_target_ids_private": sorted(int(item) for item in provider._seen),
        "episode_num_containers": int(episode.num_containers),
    }


def _public_frame(
    episode: OnlineCRPEpisode,
    environment: OnlineCRPEnvironment,
    provider: ForecastProvider,
) -> PublicDecisionFrame:
    if provider.max_horizon != 20:
        raise ValueError("bank actor frames require the fixed max_horizon=20 provider")
    legal = environment.legal_destinations()
    observation = provider.public_observation(
        _private_physical_stacks(environment), legal, max_tiers=episode.max_tiers
    )
    return make_public_frame(
        observation, environment.state, legal, max_tiers=episode.max_tiers
    )


def simulate_h20_behavior(
    episode: OnlineCRPEpisode,
    *,
    family: str,
    p: float,
    repeat: int,
    stage_target: float,
    policy: Any,
    check_budget: Callable[[], None] | None = None,
) -> BehaviorCapture:
    """Capture the nearest legal relocation on one fixed H20@20 behavior path."""
    if family not in FAMILIES or p not in P_VALUES or repeat not in REPEATS:
        raise ValueError("trajectory condition differs from the frozen fit grid")
    if episode.num_containers <= 0:
        raise ValueError("fit trajectory requires a positive original N")
    environment, provider, forecast, reset_record = _make_context(
        episode, family=family, p=p, repeat=repeat
    )
    best: CapturedDecision | None = None
    best_key: tuple[float, int] | None = None
    decisions = 0
    max_steps = episode.num_containers * (episode.max_tiers - 1)
    target_retrieval_count = stage_target * episode.num_containers
    while not environment.state.terminated:
        if check_budget is not None:
            check_budget()
        frame = _public_frame(episode, environment, provider)
        if not frame.legal_destinations:
            raise RuntimeError("nonterminal H20 behavior state has no legal relocation")
        action = h20_action(frame, policy)
        distance = abs(frame.retrieval_count - target_retrieval_count)
        key = (distance, decisions)
        if best_key is None or key < best_key:
            best_key = key
            best = CapturedDecision(
                decision_index=decisions,
                retrieval_count=frame.retrieval_count,
                relocation_count=int(environment.relocation_count),
                frame=frame,
                anchor_action=action,
                environment=copy.deepcopy(environment),
                provider=copy.deepcopy(provider),
            )
        before = int(environment.state.retrieval_count)
        environment.step(action)
        after = int(environment.state.retrieval_count)
        _sync_retrieval_events_with_audit(provider, episode.retrieval_sequence, before, after)
        decisions += 1
        if decisions > max_steps:
            raise RuntimeError("H20 fit behavior exceeded the physical safety bound")
    if not environment.state.terminated:
        raise AssertionError("H20 behavior trajectory did not complete")
    tail = None if best is None else int(environment.relocation_count - best.relocation_count)
    return BehaviorCapture(
        selected=best,
        decisions=decisions,
        behavior_tail_relocations=tail,
        forecast_sha256=_forecast_hash(forecast),
        evaluator_private_binding={
            "initial_stacks_private": episode.initial_stacks,
            "true_retrieval_sequence_private": episode.retrieval_sequence,
            "fixed_forecast_private": forecast,
            "initial_reset": reset_record,
            "capture": None
            if best is None
            else _private_capture_record(episode, best.environment, best.provider),
        },
    )


def _continue_h20_from_action(
    episode: OnlineCRPEpisode,
    environment: OnlineCRPEnvironment,
    provider: ForecastProvider,
    first_action: int,
    policy: Any,
    first_frame: PublicDecisionFrame,
    check_budget: Callable[[], None] | None = None,
) -> ContinuationResult:
    if environment.state.terminated:
        raise ValueError("counterfactual source state must be nonterminal")
    start_relocations = int(environment.relocation_count)
    max_steps = episode.num_containers * (episode.max_tiers - 1)
    trace: list[Mapping[str, Any]] = []

    def advance(action: int, *, phase: str, frame: PublicDecisionFrame) -> None:
        if check_budget is not None:
            check_budget()
        legal = environment.legal_destinations()
        if int(action) not in legal:
            raise ValueError("continuation action is not legal in its public frame")
        before_stacks = _private_physical_stacks(environment)
        target = environment.current_target_id
        source = next(index for index, stack in enumerate(before_stacks) if target in stack)
        blocker = before_stacks[source][-1]
        retrieval_before = int(environment.state.retrieval_count)
        relocation_before = int(environment.relocation_count)
        q_before = float(provider.history.q_t)
        hits_before = provider.history.hits
        crane_before = float(environment.state.crane_time)
        frame_record = public_frame_record(frame)
        environment.step(int(action))
        retrieval_after = int(environment.state.retrieval_count)
        relocation_after = int(environment.relocation_count)
        reveal_events = _sync_retrieval_events_with_audit(
            provider, episode.retrieval_sequence, retrieval_before, retrieval_after
        )
        if relocation_after != relocation_before + 1:
            raise AssertionError("one continuation action must cause exactly one relocation")
        if not reveal_events and provider.history.q_t != q_before:
            raise AssertionError("q_t changed without a new reveal")
        trace.append(
            {
                "phase": phase,
                "step_index": len(trace),
                "forced_first_action": phase == "forced_first_action",
                "current_target_id_private": int(target),
                "source_stack_private": int(source),
                "moving_blocker_id_private": int(blocker),
                "legal_destinations": [int(value) for value in legal],
                "action_destination": int(action),
                "public_frame_sha256": hashlib.sha256(
                    _canonical_json_bytes(frame_record)
                ).hexdigest(),
                "public_frame": frame_record,
                "physical_stacks_before_private": before_stacks,
                "physical_stacks_after_private": _private_physical_stacks(environment),
                "retrieval_count_before": retrieval_before,
                "retrieval_count_after": retrieval_after,
                "automatic_retrieved_target_ids_private": [
                    int(value)
                    for value in episode.retrieval_sequence[retrieval_before:retrieval_after]
                ],
                "relocation_count_before": relocation_before,
                "relocation_count_after": relocation_after,
                "q_t_before_action": q_before,
                "q_t_after_reveals": float(provider.history.q_t),
                "history_hits_before_private": list(hits_before),
                "history_hits_after_private": list(provider.history.hits),
                "reveal_events_private": reveal_events,
                "crane_time_before": crane_before,
                "crane_time_after": float(environment.state.crane_time),
            }
        )

    advance(int(first_action), phase="forced_first_action", frame=first_frame)
    steps = 1
    while not environment.state.terminated:
        if check_budget is not None:
            check_budget()
        frame = _public_frame(episode, environment, provider)
        if not frame.legal_destinations:
            raise RuntimeError("nonterminal H20 continuation state has no legal relocation")
        action = h20_action(frame, policy)
        advance(action, phase="h20_at_20_continuation", frame=frame)
        steps += 1
        if steps > max_steps:
            raise RuntimeError("counterfactual H20 continuation exceeded its safety bound")
    if int(environment.state.retrieval_count) != episode.num_containers:
        raise AssertionError("H20 continuation terminated before all containers were retrieved")
    return ContinuationResult(
        remaining_relocations=int(environment.relocation_count - start_relocations),
        private_trace=tuple(trace),
    )


def raw_h20_proposals(
    frame: PublicDecisionFrame,
    policy: Any,
    *,
    check_budget: Callable[[], None] | None = None,
) -> dict[int, int]:
    if tuple(frame.legal_destinations) != tuple(sorted(set(frame.legal_destinations))):
        raise ValueError("public legal destinations must be unique and sorted")
    proposals: dict[int, int] = {}
    for horizon in HORIZONS:
        if check_budget is not None:
            check_budget()
        proposals[horizon] = h20_action(truncate_public_frame(frame, horizon), policy)
    return proposals


def public_frame_record(frame: PublicDecisionFrame) -> dict[str, Any]:
    observation = frame.observation
    return {
        "rank_tensor": np.asarray(observation.rank_tensor, dtype=np.int32).tolist(),
        "occupied_mask": np.asarray(observation.occupied_mask, dtype=np.bool_).tolist(),
        "legal_destination_mask": np.asarray(
            observation.legal_destination_mask, dtype=np.bool_
        ).tolist(),
        "legal_destinations": list(frame.legal_destinations),
        "max_tiers": int(frame.max_tiers),
        "q_prior": float(observation.q_prior),
        "q_t": float(observation.q_t),
        "history_hits": [bool(hit) for hit in observation.history_hits],
        "retrieval_count": int(frame.retrieval_count),
        "relocation_count": int(frame.relocation_count),
        "crane_position": frame.crane_position,
        "crane_time": float(frame.crane_time),
    }


def counterfactual_label_row(
    episode: OnlineCRPEpisode,
    assignment: FitStateAssignment,
    capture: BehaviorCapture,
    *,
    policy: Any,
    check_budget: Callable[[], None] | None = None,
) -> dict[str, Any]:
    selected = capture.selected
    if selected is None or capture.behavior_tail_relocations is None:
        raise ValueError("cannot label an assignment with no legal relocation decision")
    if label_invariant_fingerprint(episode) != assignment.layout_fingerprint:
        raise ValueError("captured trajectory layout differs from its preassigned fingerprint")
    if selected.frame.retrieval_count != selected.retrieval_count:
        raise AssertionError("captured public frame retrieval count changed")

    source_signature = _private_signature(selected.environment, selected.provider)
    proposals = raw_h20_proposals(selected.frame, policy, check_budget=check_budget)
    anchor = int(proposals[20])
    if anchor != selected.anchor_action:
        raise AssertionError("captured H20 behavior action differs from its H20@20 proposal")

    label_by_destination: dict[int, int] = {}
    trace_by_destination: dict[int, tuple[Mapping[str, Any], ...]] = {}
    rollout_count = 0
    for destination in selected.frame.legal_destinations:
        clone_environment = copy.deepcopy(selected.environment)
        clone_provider = copy.deepcopy(selected.provider)
        outcome = _continue_h20_from_action(
            episode,
            clone_environment,
            clone_provider,
            int(destination),
            policy,
            first_frame=selected.frame,
            check_budget=check_budget,
        )
        label_by_destination[int(destination)] = outcome.remaining_relocations
        trace_by_destination[int(destination)] = outcome.private_trace
        rollout_count += 1
    if source_signature != _private_signature(selected.environment, selected.provider):
        raise AssertionError("counterfactual clones mutated their shared capture source")
    if label_by_destination.get(anchor) != capture.behavior_tail_relocations:
        raise AssertionError("anchor continuation differs from the captured H20 behavior tail")
    if set(label_by_destination) != set(selected.frame.legal_destinations):
        raise AssertionError("bank must label each legal physical destination exactly once")

    anchor_cost = int(label_by_destination[anchor])
    labels = [
        {
            "destination": int(destination),
            "remaining_relocations": int(cost),
            "normalized_remaining_cost": float(cost / episode.num_containers),
            "relative_label": float((cost - anchor_cost) / episode.num_containers),
            "is_anchor": int(destination) == anchor,
        }
        for destination, cost in sorted(label_by_destination.items())
    ]
    if any(row["is_anchor"] and row["relative_label"] != 0.0 for row in labels):
        raise AssertionError("H20@20 anchor relative label must be exactly zero")
    proposal_provenance: dict[int, list[int]] = {}
    for horizon, destination in proposals.items():
        proposal_provenance.setdefault(destination, []).append(horizon)
    return {
        "state_id": assignment.state_id,
        "family": assignment.family,
        "p": assignment.p,
        "repeat": assignment.repeat,
        "geometry_id": assignment.geometry_id,
        "layout_fingerprint": assignment.layout_fingerprint,
        "stage_target": assignment.stage_target,
        "stage_actual": selected.retrieval_count / episode.num_containers,
        "stage_abs_error": abs(
            selected.retrieval_count / episode.num_containers - assignment.stage_target
        ),
        "original_n": episode.num_containers,
        "decision_index": selected.decision_index,
        "retrieval_count": selected.retrieval_count,
        "capture_relocation_count": selected.relocation_count,
        "public_frame": public_frame_record(selected.frame),
        "raw_h20_proposals": {str(h): int(action) for h, action in proposals.items()},
        "proposal_destination_provenance": {
            str(destination): horizons
            for destination, horizons in sorted(proposal_provenance.items())
        },
        "anchor_destination": anchor,
        "anchor_remaining_relocations": anchor_cost,
        "behavior_tail_remaining_relocations": int(capture.behavior_tail_relocations),
        "legal_destination_labels": labels,
        "counterfactual_rollouts": rollout_count,
        "forecast_sha256": capture.forecast_sha256,
        "private_audit": {
            **dict(capture.evaluator_private_binding),
            "candidate_rollouts_private": [
                {
                    "first_action_destination": int(destination),
                    "remaining_relocations": int(label_by_destination[destination]),
                    "full_action_reveal_trace": list(trace_by_destination[destination]),
                }
                for destination in sorted(trace_by_destination)
            ],
            "forecast_sha256": capture.forecast_sha256,
            "audit_role": "evaluator_only; never passed to an actor or model feature builder",
        },
        "label_semantics": "candidate first relocation then frozen public H20@20 continuation; lower relative label is better",
        "actor_input_boundary": "public_frame_only; no raw IDs, future sequence, forecast, p, or evaluator labels",
    }


def build_label_bank(
    assignments: Sequence[FitStateAssignment],
    episodes: Sequence[OnlineCRPEpisode],
    *,
    policy: Any,
    check_budget: Callable[[], None] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Generate fit-only labels; the CLI release gate must run before this call."""
    by_fingerprint = {
        label_invariant_fingerprint(episode): episode for episode in episodes
    }
    if len(by_fingerprint) != len(episodes):
        raise ValueError("fit episodes contain duplicate semantic fingerprints")
    rows: list[dict[str, Any]] = []
    shortages: list[dict[str, Any]] = []
    for assignment in assignments:
        episode = by_fingerprint.get(assignment.layout_fingerprint)
        if episode is None:
            raise ValueError("assignment references a layout outside the M0 fit split")
        row, shortage = label_one_assignment(
            episode, assignment, policy=policy, check_budget=check_budget
        )
        if shortage is not None:
            shortages.append(shortage)
        else:
            assert row is not None
            rows.append(row)
    return rows, shortages


def label_one_assignment(
    episode: OnlineCRPEpisode,
    assignment: FitStateAssignment,
    *,
    policy: Any,
    check_budget: Callable[[], None] | None = None,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    if label_invariant_fingerprint(episode) != assignment.layout_fingerprint:
        raise ValueError("assignment references a different fit layout")
    capture = simulate_h20_behavior(
        episode,
        family=assignment.family,
        p=assignment.p,
        repeat=assignment.repeat,
        stage_target=assignment.stage_target,
        policy=policy,
        check_budget=check_budget,
    )
    if capture.selected is None:
        return None, {
            "state_id": assignment.state_id,
            "layout_fingerprint": assignment.layout_fingerprint,
            "family": assignment.family,
            "p": assignment.p,
            "repeat": assignment.repeat,
            "stage_target": assignment.stage_target,
            "status": "NO_LEGAL_RELOCATION_DECISION",
        }
    return (
        counterfactual_label_row(
            episode,
            assignment,
            capture,
            policy=policy,
            check_budget=check_budget,
        ),
        None,
    )


__all__ = (
    "BANK_SOURCE_PATHS",
    "BehaviorCapture",
    "CapturedDecision",
    "DecisionPosition",
    "FitStateAssignment",
    "assignment_manifest",
    "build_fit_state_assignments",
    "build_label_bank",
    "choose_nearest_decision",
    "counterfactual_label_row",
    "current_m0_binding",
    "current_source_sha256",
    "expected_release_contract",
    "load_fit_episodes",
    "label_one_assignment",
    "public_frame_record",
    "raw_h20_proposals",
    "simulate_h20_behavior",
    "verify_fit_label_release",
)
