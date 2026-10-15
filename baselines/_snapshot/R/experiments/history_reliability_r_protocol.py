"""M3 confirmation protocol preparation; no scoring or model fitting lives here.

This module deliberately reuses only the frozen, read-only PLAN-036 layout and
source-inventory helpers.  It does not import the M2 R selector or any actor.
The guarded layout builder is present for review but remains disabled until a
separate preparation release is recorded in the config.
"""

from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from EM.path_roots import resolve_project_path
from experiments.noise_aware_lookahead import (
    _rank_layout_episode,
    _source_manifest_inventory,
)
from scrp.attention_data import file_sha256, label_invariant_fingerprint
from scrp.online_crp import OnlineCRPEpisode


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/history_reliability_r_confirm_v1.json"
PROTOCOL_ID = "history_reliability_r_v1"
FORECAST_SEED_NAMESPACE = "history-reliability-r-v1"
FAMILIES = ("subset_shuffle", "delayed_positions")
HORIZONS = (0, 2, 5, 10, 20)
P_VALUES = (0.0, 0.1, 0.3, 0.5)
REPEATS = (0, 1)
METHODS = ("EM0", "EM2", "EM5", "EM10", "EM20", "R", "R_CONST", "R_NO_HISTORY")
LAYOUT_SEED = 2026101001
LAYOUT_COUNT = 80
LAYOUTS_PER_GEOMETRY = 40
MAX_OVERLAP_RETRIES = 100
RUN_ID = "r01_m3_confirm80"
GEOMETRY_CONTRACT = (
    ("G1", 6, 5, 24),
    ("G2", 8, 6, 40),
)


def _project_path(relative_path: str) -> Path:
    return resolve_project_path(relative_path, owner="R")


def read_config(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    """Read only the new confirm config and fail closed on any scoring release."""
    path = path.resolve()
    try:
        path.relative_to(ROOT)
    except ValueError as exc:
        raise ValueError("confirm config must remain inside Mijn_RL") from exc
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("protocol_id") != PROTOCOL_ID:
        raise ValueError("unexpected confirm protocol_id")
    scope = config.get("release_scope", {})
    if scope.get("scientific_scoring_released") is not False:
        raise ValueError("M3 preparation cannot release scientific scoring")
    if scope.get("confirm_layout_generation_released") not in (False, True):
        raise ValueError("confirm-layout release state must be explicit")
    if scope.get("method_source_freeze_pending") is not True:
        raise ValueError("M2 final method source is still pending")
    for unreleased in ("formal_evaluation_released", "cost_measurement_released"):
        if scope.get(unreleased) is not False:
            raise ValueError(f"M3 preparation cannot release {unreleased}")
    layout = config["layout_generation"]
    if (
        config.get("run_id") != RUN_ID
        or layout.get("split") != "confirm"
        or int(layout.get("master_seed", -1)) != LAYOUT_SEED
        or int(layout.get("total_layouts", -1)) != LAYOUT_COUNT
        or int(layout.get("per_geometry", -1)) != LAYOUTS_PER_GEOMETRY
        or int(layout.get("max_overlap_retries_per_layout", -1))
        != MAX_OVERLAP_RETRIES
    ):
        raise ValueError("confirm layout seed/count/retry contract differs from REVIEW-038-08")
    geometries = tuple(
        (
            str(item["id"]),
            int(item["num_stacks"]),
            int(item["max_tiers"]),
            int(item["num_containers"]),
        )
        for item in layout["geometries"]
    )
    if geometries != GEOMETRY_CONTRACT:
        raise ValueError("confirm geometries differ from REVIEW-038-08")
    if (
        config["evaluation"].get("layout_count") != LAYOUT_COUNT
        or config["evaluation"].get("expected_scoring_keys") != 10240
    ):
        raise ValueError("confirm evaluation layout/key counts differ from PLAN-038")
    if set(config["legacy_r01_layout_exclusions"]) != {"fit", "select", "screen"}:
        raise ValueError("explicit old r01 exclusion splits must be fit/select/screen")
    if {
        split: int(spec["expected_count"])
        for split, spec in config["legacy_r01_layout_exclusions"].items()
    } != {"fit": 40, "select": 40, "screen": 80}:
        raise ValueError("explicit old r01 fit/select/screen counts differ from REVIEW-038-08")
    if tuple(config["evaluation"]["families"]) != FAMILIES:
        raise ValueError("confirm noise families differ from REVIEW-038-08")
    if tuple(int(value) for value in config["evaluation"]["horizons"]) != HORIZONS:
        raise ValueError("confirm horizon grid differs from the frozen method")
    if tuple(float(value) for value in config["evaluation"]["p_values"]) != P_VALUES:
        raise ValueError("confirm p grid differs from REVIEW-038-08")
    if tuple(int(value) for value in config["evaluation"]["repeats"]) != REPEATS:
        raise ValueError("confirm repeat grid differs from REVIEW-038-08")
    if tuple(config["evaluation"]["methods"]) != METHODS:
        raise ValueError("confirm method list differs from PLAN-038")
    expected = (
        LAYOUT_COUNT
        * len(FAMILIES)
        * len(P_VALUES)
        * len(REPEATS)
        * len(METHODS)
    )
    if expected != int(config["evaluation"]["expected_scoring_keys"]):
        raise ValueError("confirm scoring-key arithmetic is inconsistent")
    return config


def collect_exclusion_inventory(config: Mapping[str, Any]) -> dict[str, Any]:
    """Read old exclusion sources plus explicit r01 fit/select/screen layouts.

    This performs no generation, labels, rollouts, forecasts, or scoring.  The
    established inventory does not scan output/; therefore the old 40+40+80
    r01 layouts are loaded explicitly and fingerprinted into the same union.
    """
    base_fingerprints, base_sources = _source_manifest_inventory()
    n1_binding = config["legacy_source_inventory"]
    n1_report_path = _project_path(str(n1_binding["n1_contract_report"]["path"]))
    if file_sha256(n1_report_path) != str(n1_binding["n1_contract_report"]["sha256"]):
        raise ValueError("legacy N1 source inventory report hash changed")
    n1_report = json.loads(n1_report_path.read_text(encoding="utf-8"))
    historical_sources = n1_report["data"]["source_manifests"]

    def source_key(item: Mapping[str, Any]) -> tuple[str, str]:
        return str(item["category"]), str(Path(item["path"]).resolve())

    current_by_key = {source_key(item): item for item in base_sources}
    historical_by_key = {source_key(item): item for item in historical_sources}
    if len(current_by_key) != int(n1_binding["expected_source_manifest_count"]):
        raise ValueError("legacy source manifest count changed")
    if set(current_by_key) != set(historical_by_key):
        missing = sorted(set(historical_by_key) - set(current_by_key))
        added = sorted(set(current_by_key) - set(historical_by_key))
        raise ValueError(f"legacy source inventory path set changed; missing={missing}, added={added}")
    for key, current in current_by_key.items():
        historical = historical_by_key[key]
        for field in ("sha256", "episode_count", "unique_semantic_fingerprints"):
            if current[field] != historical[field]:
                raise ValueError(f"legacy source inventory changed at {key}: {field}")
    expected_base_count = int(n1_report["data"]["source_fingerprint_union"])
    if len(base_fingerprints) != expected_base_count:
        raise ValueError("legacy source-manifest fingerprint union changed")

    explicit_records: list[dict[str, Any]] = []
    explicit_sets: dict[str, set[str]] = {}
    for split, spec in config["legacy_r01_layout_exclusions"].items():
        source_path = _project_path(str(spec["path"]))
        source_hash = file_sha256(source_path)
        if source_hash != str(spec["sha256"]):
            raise ValueError(f"legacy {split} layout source hash changed")
        raw = json.loads(source_path.read_text(encoding="utf-8"))
        expected_count = int(spec["expected_count"])
        if not isinstance(raw, list) or len(raw) != expected_count:
            raise ValueError(f"legacy {split} layout count changed")
        episodes = [OnlineCRPEpisode.from_record(record) for record in raw]
        fingerprints = [label_invariant_fingerprint(episode) for episode in episodes]
        fp_set = set(fingerprints)
        if len(fp_set) != expected_count:
            raise ValueError(f"legacy {split} layouts contain duplicate semantic fingerprints")
        actual_geometry_counts = {
            geometry_id: sum(episode.parameter_group == geometry_id for episode in episodes)
            for geometry_id in spec["by_geometry"]
        }
        if actual_geometry_counts != {
            str(key): int(value) for key, value in spec["by_geometry"].items()
        }:
            raise ValueError(f"legacy {split} geometry counts changed")
        if any(episode.metadata.get("split") != split for episode in episodes):
            raise ValueError(f"legacy {split} source contains a different split role")
        if fp_set & base_fingerprints:
            raise ValueError(f"legacy {split} layouts overlap the old source-manifest union")
        explicit_sets[split] = fp_set
        explicit_records.append(
            {
                "split": split,
                "path": str(source_path),
                "sha256": source_hash,
                "episode_count": len(episodes),
                "unique_semantic_fingerprints": len(fp_set),
                "by_geometry": actual_geometry_counts,
                "semantic_fingerprints": sorted(fp_set),
            }
        )

    split_names = tuple(explicit_sets)
    for index, left in enumerate(split_names):
        for right in split_names[index + 1 :]:
            if explicit_sets[left] & explicit_sets[right]:
                raise ValueError(f"legacy r01 {left}/{right} layouts overlap")
    explicit_union = set().union(*explicit_sets.values())
    if len(explicit_union) != 160:
        raise ValueError("explicit legacy r01 exclusion must contain 160 layouts")
    all_exclusions = set(base_fingerprints) | explicit_union
    if len(all_exclusions) != int(
        n1_binding["expected_combined_exclusion_fingerprint_count"]
    ):
        raise ValueError("combined old-source exclusion fingerprint count changed")
    return {
        "status": "READ_ONLY_EXCLUSION_INVENTORY_COMPLETE",
        "protocol_id": PROTOCOL_ID,
        "source_manifest_inventory": base_sources,
        "source_manifest_fingerprint_union_count": len(base_fingerprints),
        "explicit_legacy_r01_layouts": explicit_records,
        "explicit_legacy_r01_fingerprint_union_count": len(explicit_union),
        "overlap_between_old_sources_and_explicit_r01": 0,
        "overlap_among_explicit_r01_splits": 0,
        "combined_exclusion_fingerprint_count": len(all_exclusions),
        "combined_exclusion_fingerprints": sorted(all_exclusions),
        "new_confirm_layouts_generated": 0,
        "labels_or_scores_read": False,
    }


def _new_confirm_episode(
    config: Mapping[str, Any],
    *,
    geometry: Mapping[str, Any],
    index: int,
    retry: int,
) -> OnlineCRPEpisode:
    """Adapt the old rank-layout sampler to the new confirm metadata contract."""
    run_id = str(config["run_id"])
    geometry_id = str(geometry["id"])
    episode_id = f"{run_id}-confirm-{geometry_id}-{index:03d}"
    episode = _rank_layout_episode(
        episode_id=episode_id,
        geometry_id=geometry_id,
        split="confirm",
        num_stacks=int(geometry["num_stacks"]),
        max_tiers=int(geometry["max_tiers"]),
        num_containers=int(geometry["num_containers"]),
        master_seed=int(config["layout_generation"]["master_seed"]),
        index=index,
        retry=retry,
    )
    metadata = dict(episode.metadata)
    metadata.update(
        {
            "protocol_id": PROTOCOL_ID,
            "run_id": run_id,
            "split": "confirm",
            "geometry_id": geometry_id,
            "data_role": "confirm_unscored_layout",
        }
    )
    return replace(episode, episode_id=episode_id, metadata=metadata)


def generate_confirm_layouts(
    config: Mapping[str, Any], inventory: Mapping[str, Any]
) -> list[OnlineCRPEpisode]:
    """Build the future 80-layout set only after an explicit preparation release."""
    scope = config.get("release_scope", {})
    if scope.get("confirm_layout_generation_released") is not True:
        raise RuntimeError("confirm-layout generation is not released")
    if scope.get("scientific_scoring_released") is not False:
        raise RuntimeError("this module never releases scientific scoring")
    if inventory.get("status") != "READ_ONLY_EXCLUSION_INVENTORY_COMPLETE":
        raise ValueError("a complete, current exclusion inventory is required")
    excluded = set(inventory["combined_exclusion_fingerprints"])
    generated: list[OnlineCRPEpisode] = []
    seen = set(excluded)
    retry_limit = int(config["layout_generation"]["max_overlap_retries_per_layout"])
    per_geometry = int(config["layout_generation"]["per_geometry"])
    for geometry in config["layout_generation"]["geometries"]:
        geometry_id = str(geometry["id"])
        for index in range(per_geometry):
            for retry in range(retry_limit):
                episode = _new_confirm_episode(
                    config, geometry=geometry, index=index, retry=retry
                )
                fingerprint = label_invariant_fingerprint(episode)
                if fingerprint in seen:
                    continue
                seen.add(fingerprint)
                generated.append(episode)
                break
            else:
                raise RuntimeError(
                    f"confirm-layout overlap retry limit exhausted at {geometry_id}/{index}"
                )
    if len(generated) != int(config["layout_generation"]["total_layouts"]):
        raise RuntimeError("confirm-layout generation count differs from the frozen contract")
    fingerprints = [label_invariant_fingerprint(episode) for episode in generated]
    if len(set(fingerprints)) != len(generated) or set(fingerprints) & excluded:
        raise RuntimeError("confirm-layout semantic exclusion verification failed")
    return generated


def semantic_forecast_seed(
    semantic_fingerprint: str, family: str, repeat: int
) -> int:
    """Family/repeat seed; p, horizon, method, and evaluator mask are excluded."""
    if not semantic_fingerprint or family not in FAMILIES or int(repeat) not in REPEATS:
        raise ValueError("invalid semantic forecast seed key")
    payload = (
        f"{FORECAST_SEED_NAMESPACE}|{semantic_fingerprint}|{family}|{int(repeat)}"
    ).encode("utf-8")
    return int(hashlib.sha256(payload).hexdigest()[:16], 16)


def _validate_full_future(true_future: Sequence[int]) -> tuple[int, ...]:
    sequence = tuple(int(value) for value in true_future)
    if len(set(sequence)) != len(sequence):
        raise ValueError("evaluator future must be a full no-duplicate permutation")
    return sequence


def make_evaluator_forecast(
    true_future: Sequence[int], *, family: str, p_evaluator_only: float, seed: int
) -> tuple[int, ...]:
    """Create one private complete F; never pass true_future or p to an actor."""
    truth = _validate_full_future(true_future)
    p = float(p_evaluator_only)
    if family not in FAMILIES or p not in P_VALUES:
        raise ValueError("forecast family or p differs from the frozen grid")
    rng = np.random.Generator(np.random.PCG64(int(seed)))
    if family == "subset_shuffle":
        selected = np.flatnonzero(rng.random(len(truth)) < p)
        forecast = list(truth)
        if len(selected) > 1:
            values = np.asarray([truth[int(index)] for index in selected], dtype=np.int64)
            shuffled = rng.permutation(values)
            for index, value in zip(selected, shuffled):
                forecast[int(index)] = int(value)
    else:
        delay = int(math.ceil(len(truth) / 4.0))
        mask = rng.random(len(truth)) < p
        order = sorted(
            range(len(truth)),
            key=lambda index: (index + delay * int(mask[index]), index),
        )
        forecast = [truth[index] for index in order]
    result = tuple(forecast)
    if len(result) != len(truth) or set(result) != set(truth):
        raise RuntimeError("evaluator forecast is not a complete legal permutation")
    return result


def evaluator_forecast_diagnostics(
    true_future: Sequence[int], forecast: Sequence[int]
) -> dict[str, float]:
    """Evaluator-only diagnostics; never expose these values to the actor."""
    truth = _validate_full_future(true_future)
    predicted = _validate_full_future(forecast)
    if len(predicted) != len(truth) or set(predicted) != set(truth):
        raise ValueError("diagnostics require a complete legal forecast permutation")
    true_position = {token: index for index, token in enumerate(truth)}
    predicted_true_positions = [true_position[token] for token in predicted]
    length = len(truth)
    pair_count = length * (length - 1) // 2
    inversions = sum(
        predicted_true_positions[left] > predicted_true_positions[right]
        for left in range(length)
        for right in range(left + 1, length)
    )
    inversion_fraction = float(inversions / pair_count) if pair_count else 0.0
    displacement = (
        float(
            np.mean(
                [
                    abs(predicted_position - true_position[token])
                    for predicted_position, token in enumerate(predicted)
                ]
            )
        )
        if length
        else 0.0
    )
    return {
        "mean_absolute_position_displacement": displacement,
        "normalized_inversion_fraction": inversion_fraction,
        "kendall_tau": 1.0 - 2.0 * inversion_fraction,
    }


def evaluator_prefix_membership_errors(
    true_future: Sequence[int], forecast: Sequence[int]
) -> dict[str, float]:
    """Return evaluator-only nested-prefix errors for the frozen H grid."""
    truth = _validate_full_future(true_future)
    predicted = _validate_full_future(forecast)
    if len(predicted) != len(truth) or set(predicted) != set(truth):
        raise ValueError("prefix diagnostics require a complete legal forecast")
    values: dict[str, float] = {}
    for horizon in HORIZONS:
        length = min(horizon, len(truth))
        error = (
            1.0
            - len(set(truth[:length]).intersection(predicted[:length])) / length
            if length
            else 0.0
        )
        values[str(horizon)] = float(error)
    return values
