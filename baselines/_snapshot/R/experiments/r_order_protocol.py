"""PLAN-049 one-shot data preparation and release-gated science runner.

M0 only builds layout manifests/fingerprints and runs synthetic tests. Every
label, fit, bank, cost, confirmation, and report command requires a separate
ROOT-authored, hash-bound phase release.
"""

from __future__ import annotations

from collections import defaultdict
from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import csv
import argparse
import gzip
import hashlib
import importlib
import io
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


WORKSPACE = Path(__file__).resolve().parents[2]
R_ROOT = WORKSPACE / "R"
CONFIG_PATH = R_ROOT / "configs/r_order_v1.json"
PLAN_PATH = R_ROOT / "docs/planning/PLAN_049_R_ORDER.md"
OUTPUT_ROOT = R_ROOT / "output/r_order_v1/r01"
M0_REVISION_ROOT = OUTPUT_ROOT / "m0_review_rev02"
REV01_ARCHIVE_ROOT = R_ROOT / "output/r_order_v1/archive/M0_REVIEW_REV01"
REV01_ARCHIVE_MANIFEST_PATH = REV01_ARCHIVE_ROOT / "ARCHIVE_MANIFEST.json"
PREFLIGHT_AUDIT_PATH = R_ROOT / "output/r_order_v1/r01_preflight_audit.json"
REV02_PREPARE_ATTEMPT_AUDIT_PATH = R_ROOT / "output/r_order_v1/M0_REV02_PREPARE_ATTEMPT_01.json"
REV02_PREPARE_ATTEMPT_02_AUDIT_PATH = R_ROOT / "output/r_order_v1/M0_REV02_PREPARE_ATTEMPT_02.json"
M0_REVISION_SOURCE_BINDINGS_PATH = M0_REVISION_ROOT / "M0_SOURCE_BINDINGS_REV02.json"
M0_REVISION_PREPARE_RECEIPT_PATH = M0_REVISION_ROOT / "M0_PREPARE_RECEIPT_REV02.json"
M0_REVISION_TEST_RECEIPT_PATH = M0_REVISION_ROOT / "M0_TEST_RECEIPT_REV02.json"
M0_REVISION_SUMMARY_PATH = M0_REVISION_ROOT / "M0_COMPACT_SUMMARY_REV02.md"
BASE_SOURCE_BINDINGS_PATH = OUTPUT_ROOT / "M0_SOURCE_BINDINGS.json"
BASE_PREPARE_RECEIPT_PATH = OUTPUT_ROOT / "M0_PREPARE_RECEIPT.json"
BASE_TEST_RECEIPT_PATH = OUTPUT_ROOT / "M0_TEST_RECEIPT.json"
BASE_M0_COMPACT_SUMMARY_PATH = OUTPUT_ROOT / "M0_COMPACT_SUMMARY.md"
SOURCE_BINDINGS_PATH = M0_REVISION_SOURCE_BINDINGS_PATH
PREPARE_RECEIPT_PATH = M0_REVISION_PREPARE_RECEIPT_PATH
TEST_RECEIPT_PATH = M0_REVISION_TEST_RECEIPT_PATH
M0_COMPACT_SUMMARY_PATH = M0_REVISION_SUMMARY_PATH
ROOT_M0_REVIEW_PATH = M0_REVISION_ROOT / "ROOT_M0_REVIEW_REV02.json"
SCIENCE_LEDGER_PATH = OUTPUT_ROOT / "SCIENCE_WALL.jsonl"
ALGORITHMS = ("EM", "GP", "ERI", "EG")
HORIZONS = (0, 2, 5, 10, 20)
REGIMES = ("DP", "TWS", "CLEAN")
SPLITS = ("fit", "cal", "select", "confirm_candidates")
GEOMETRIES = (
    ("G1", 6, 5, 24),
    ("G2", 8, 6, 40),
)
SOURCE_RELATIVE_PATHS = (
    "R/docs/planning/PLAN_049_R_ORDER.md",
    "R/configs/r_order_v1.json",
    "R/scrp/r_order_public.py",
    "R/scrp/r_order_training.py",
    "R/scrp/r_order_actor.py",
    "R/experiments/r_order_protocol.py",
    "R/experiments/r_order_analysis.py",
    "R/tests/test_r_order_public.py",
    "R/tests/test_r_order_training.py",
    "R/tests/test_r_order_analysis.py",
    "R/tests/test_r_order_protocol.py",
    "R/scrp/universal_r_utility.py",
    "R/scrp/universal_r_utility_m1.py",
    "R/scrp/eri_eg_public.py",
    "R/scrp/r_eri_eg_public.py",
    "R/scrp/cross_solver_reliability_r.py",
    "R/scrp/history_reliability_r.py",
    "EM/scrp/noise_aware_forecast.py",
    "EM/scrp/noise_aware_em.py",
    "EM/experiments/noise_aware_lookahead.py",
    "EM/experiments/noise_aware_reference_p0.py",
    "EM/path_roots.py",
    "R/experiments/r_horizon_robustness.py",
    "R/experiments/r_horizon_robustness_analysis.py",
    "R/experiments/time_window_shuffle_protocol.py",
    "R/experiments/history_reliability_r_protocol.py",
    "R/experiments/time_window_shuffle_acceptance.py",
    "R/scrp/h20_r_bank.py",
    "Mijn_RL/scrp/attention_data.py",
    "Mijn_RL/scrp/online_crp.py",
    "R/configs/r_noise_diagnostics_v1.json",
    "GP/expression.py",
    "GP/simulator.py",
    "GP/data.py",
    "GP/online_crp_oracle.py",
    "R/scrp/h20_r_data.py",
)
M0_REVISION_ALLOWED_CHANGED_SOURCES = {
    "R/docs/planning/PLAN_049_R_ORDER.md",
    "R/configs/r_order_v1.json",
    "R/scrp/r_order_training.py",
    "R/experiments/r_order_protocol.py",
    "R/tests/test_r_order_training.py",
    "R/tests/test_r_order_protocol.py",
}
GP_ACCEPTED_ARTIFACT_PATHS = (
    "R/output/r_frozen_cross_solver_transfer_v1/r01/M0_PREPARATION_RECEIPT.json",
    "R/output/r_frozen_cross_solver_transfer_v1/r01/M1_select/GP/M1_RECEIPT.json",
    "GP/results/h20_four_terminal_sets/runs/H20/C/run_25/run_complete.json",
    "GP/results/h20_four_terminal_sets/runs/H20/C/run_25/train_complete.json",
    "GP/data/caserta_train_840_seed20260920.json",
)
KNOWN_EXCLUSION_PATHS = (
    "R/output/universal_r_utility_v1/r01/M0_EXCLUSION_MANIFEST.json",
    "R/output/universal_r_utility_v1/r01/fit.json",
    "R/output/universal_r_utility_v1/r01/validation.json",
    "R/output/r_horizon_robustness_v1/r01/data/EXCLUSION_FINGERPRINTS.json",
    "R/output/r_horizon_robustness_v1/r01/data/development20.json",
    "R/output/r_horizon_robustness_v1/r01/data/confirm40.json",
)
FROZEN_OLD_ROOTS = (
    R_ROOT / "output/universal_r_utility_v1",
    R_ROOT / "output/r_horizon_robustness_v1",
)
FROZEN_OLD_FILES = (
    R_ROOT / "scrp/history_reliability_r.py",
    R_ROOT / "experiments/history_reliability_r_protocol.py",
    R_ROOT / "docs/planning/HANDOFF_PLAN047_20261012.md",
    R_ROOT / "docs/planning/HANDOFF_PLAN048_20261013.md",
)
RELEASE_PATHS = {
    "M1": OUTPUT_ROOT / "ROOT_M1_RELEASE.json",
    "M2": OUTPUT_ROOT / "ROOT_M2_RELEASE.json",
    "M3": OUTPUT_ROOT / "ROOT_M3_RELEASE.json",
    "M4": OUTPUT_ROOT / "ROOT_M4_RELEASE.json",
}
PHASE_CAP_SECONDS = {"M1": 4 * 3600, "M2": 6 * 3600, "M3": 12 * 3600, "M4": 2 * 3600}
TOTAL_CAP_SECONDS = 24 * 3600


class ROrderStop(RuntimeError):
    """A frozen source, information, completeness, budget, or science gate failed."""


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        _json_safe(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(_json_safe(value), sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")


def _json_safe(value: Any) -> Any:
    """Convert internal numpy/dataclass/tuple-key analysis values to strict JSON."""
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, Path):
        return value.as_posix()
    if hasattr(value, "__dataclass_fields__"):
        return _json_safe(asdict(value))
    if isinstance(value, Mapping):
        output: dict[str, Any] = {}
        for key, item in value.items():
            if isinstance(key, tuple):
                key_text = json.dumps(_json_safe(list(key)), ensure_ascii=False, separators=(",", ":"))
            elif isinstance(key, (str, int, float, bool)):
                key_text = str(key)
            else:
                key_text = str(key)
            if key_text in output:
                raise ValueError(f"JSON key collision after canonical serialization: {key_text}")
            output[key_text] = _json_safe(item)
        return output
    if isinstance(value, (list, tuple, set)):
        items = list(value)
        if isinstance(value, set):
            items.sort(key=repr)
        return [_json_safe(item) for item in items]
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("nonfinite float cannot be written to frozen JSON")
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"unsupported JSON value: {type(value).__name__}")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_new(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as destination:
        destination.write(payload)
        destination.flush()
        os.fsync(destination.fileno())


def _write_json_new(path: Path, value: Any) -> None:
    _write_new(path, _json_bytes(value))


def _append_jsonl_checkpoint(path: Path, row: Mapping[str, Any]) -> None:
    """Durably checkpoint one completed trajectory/branch without resume semantics."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("ab") as destination:
        destination.write(json.dumps(_json_safe(row), sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":"), allow_nan=False).encode("utf-8"))
        destination.write(b"\n")
        destination.flush()
        os.fsync(destination.fileno())


def _fit_attempt_safe(value: Any) -> Any:
    """Preserve failed optimizer diagnostics while keeping strict JSON receipts."""
    if isinstance(value, np.generic):
        return _fit_attempt_safe(value.item())
    if isinstance(value, np.ndarray):
        return _fit_attempt_safe(value.tolist())
    if isinstance(value, Mapping):
        return {str(key): _fit_attempt_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_fit_attempt_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return "NaN" if math.isnan(value) else ("Infinity" if value > 0 else "-Infinity")
    return value


class _JsonlCheckpointStream:
    """Single-run append-only gzip stream; fsync at each completed trajectory."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._raw: Any | None = None
        self._gzip: Any | None = None

    def open(self) -> "_JsonlCheckpointStream":
        if self._raw is not None:
            raise RuntimeError("checkpoint stream is already open")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._raw = self.path.open("xb")
        self._gzip = gzip.GzipFile(fileobj=self._raw, mode="wb", compresslevel=6, mtime=0)
        return self

    def append(self, row: Mapping[str, Any]) -> None:
        if self._gzip is None:
            raise RuntimeError("checkpoint stream is not open")
        payload = json.dumps(_json_safe(row), sort_keys=True, ensure_ascii=False,
                             separators=(",", ":"), allow_nan=False).encode("utf-8") + b"\n"
        self._gzip.write(payload)

    def sync(self) -> None:
        if self._gzip is not None and self._raw is not None:
            self._gzip.flush()
            self._raw.flush()
            os.fsync(self._raw.fileno())

    def close(self) -> None:
        if self._gzip is not None:
            self._gzip.close()
            self._gzip = None
        if self._raw is not None:
            self._raw.flush()
            os.fsync(self._raw.fileno())
            self._raw.close()
            self._raw = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


def _read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _workspace_path(relative: str) -> Path:
    path = (WORKSPACE / Path(*str(relative).replace("\\", "/").split("/"))).resolve()
    try:
        path.relative_to(WORKSPACE.resolve())
    except ValueError as exc:
        raise ROrderStop(f"registered path escapes workspace: {relative}") from exc
    return path


def _load_config() -> dict[str, Any]:
    config = _read_json(CONFIG_PATH)
    if config.get("protocol_id") != "r_order_v1" or config.get("run_id") != "r01":
        raise ROrderStop("unexpected protocol_id/run_id")
    if tuple(config.get("geometries", ())) != tuple(
        {"id": key, "num_stacks": stacks, "max_tiers": tiers, "num_containers": count}
        for key, stacks, tiers, count in GEOMETRIES
    ):
        raise ROrderStop("registered geometries changed")
    condition_count = int(config["conditions"]["unique_per_layout"])
    if condition_count != 15 or config["conditions"].get("same_layout_condition_forecast_for_all_algorithms") is not True:
        raise ROrderStop("unique condition count or common-forecast rule changed")
    if config["statistics"].get("primary_metric") != "R_over_N":
        raise ROrderStop("primary metric is not R/N")
    if config["candidate_selector"].get("thresholds_tau") != [0.0, 0.25]:
        raise ROrderStop("tau grid differs from registration")
    if int(config["budgets"]["total_science_wall_seconds"]) != TOTAL_CAP_SECONDS:
        raise ROrderStop("total science budget changed")
    return config


def _runtime_imports() -> dict[str, Any]:
    """Install the established owner-first shared package paths."""
    roots = (R_ROOT, WORKSPACE / "EM", WORKSPACE / "Mijn_RL", WORKSPACE)
    for root in reversed(roots):
        value = str(root.resolve())
        while value in sys.path:
            sys.path.remove(value)
        sys.path.insert(0, value)
    scrp = importlib.import_module("scrp")
    scrp_paths = [str((root / "scrp").resolve()) for root in roots[:3]]
    scrp.__path__ = scrp_paths
    if scrp.__spec__ is not None and scrp.__spec__.submodule_search_locations is not None:
        scrp.__spec__.submodule_search_locations[:] = scrp_paths
    experiments = importlib.import_module("experiments")
    experiment_roots = (WORKSPACE / "EM", R_ROOT, WORKSPACE / "Mijn_RL")
    experiment_paths = [str((root / "experiments").resolve()) for root in experiment_roots]
    experiments.__path__ = experiment_paths
    if experiments.__spec__ is not None and experiments.__spec__.submodule_search_locations is not None:
        experiments.__spec__.submodule_search_locations[:] = experiment_paths

    online = importlib.import_module("scrp.online_crp")
    attention = importlib.import_module("scrp.attention_data")
    em_runner = importlib.import_module("experiments.noise_aware_lookahead")
    old_r_protocol = importlib.import_module("experiments.history_reliability_r_protocol")
    time_window = importlib.import_module("experiments.time_window_shuffle_protocol")
    forecast = importlib.import_module("scrp.noise_aware_forecast")
    em = importlib.import_module("scrp.noise_aware_em")
    history_r = importlib.import_module("scrp.history_reliability_r")
    eri_eg = importlib.import_module("scrp.eri_eg_public")
    r_eri_eg = importlib.import_module("scrp.r_eri_eg_public")
    cross = importlib.import_module("scrp.cross_solver_reliability_r")
    reference = importlib.import_module("experiments.noise_aware_reference_p0")
    bank = importlib.import_module("R.scrp.h20_r_bank")
    return {
        "OnlineCRPEpisode": online.OnlineCRPEpisode,
        "OnlineCRPEnvironment": online.OnlineCRPEnvironment,
        "episode_to_record": attention.episode_to_record,
        "label_invariant_fingerprint": attention.label_invariant_fingerprint,
        "file_sha256": attention.file_sha256,
        "rank_layout_episode": em_runner._rank_layout_episode,
        "source_manifest_inventory": em_runner._source_manifest_inventory,
        "private_physical_stacks": em_runner._private_physical_stacks,
        "ForecastProvider": forecast.ForecastProvider,
        "make_evaluator_forecast": old_r_protocol.make_evaluator_forecast,
        "semantic_forecast_seed": old_r_protocol.semantic_forecast_seed,
        "make_time_window_forecast": time_window.make_time_window_forecast,
        "semantic_time_window_seed": time_window.semantic_time_window_seed,
        "time_window_forecast_sha256": time_window.time_window_forecast_sha256,
        "em_select": em.select_em_destination,
        "select_r_action": history_r.select_r_action,
        "eri_select": eri_eg.select_eri20_destination,
        "eri_eg": eri_eg,
        "r_eri_eg": r_eri_eg,
        "truncate_public_frame": cross.truncate_public_frame,
        "run_one_horizon": cross._run_one_horizon,
        "frozen_actor_call": cross.frozen_actor_call,
        "make_public_frame": reference.make_public_frame,
        "private_signature": bank._private_signature,
        "private_physical_stacks": em_runner._private_physical_stacks,
        "public_frame": bank._public_frame,
        "sync_retrieval_events_with_audit": bank._sync_retrieval_events_with_audit,
    }


def _load_gp_rule(runtime: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    acceptance = importlib.import_module("experiments.time_window_shuffle_acceptance")
    config = _read_json(R_ROOT / "configs/r_noise_diagnostics_v1.json")
    rule, identity = acceptance._load_accepted_gp_rule(config)
    identity = dict(identity)
    identity["accepted_identity"] = dict(config["accepted_gp_rule"])
    identity["accepted_artifacts"] = {
        relative: _sha256_file(_workspace_path(relative)) for relative in GP_ACCEPTED_ARTIFACT_PATHS
    }
    return rule, identity


def _explicit_fingerprints(runtime: Mapping[str, Any]) -> tuple[set[str], list[dict[str, Any]]]:
    fingerprints: set[str] = set()
    source_rows: list[dict[str, Any]] = []
    episode_type = runtime["OnlineCRPEpisode"]
    fingerprint_fn = runtime["label_invariant_fingerprint"]
    for relative in KNOWN_EXCLUSION_PATHS:
        path = _workspace_path(relative)
        raw = _read_json(path)
        values: set[str]
        if relative.endswith("M0_EXCLUSION_MANIFEST.json"):
            if raw.get("labels_or_scores_read") is not False:
                raise ROrderStop("047 exclusion manifest is not certified label-blind")
            listed = raw.get("semantic_fingerprints")
            if not isinstance(listed, list) or len(listed) != int(raw.get("excluded_fingerprint_count", -1)):
                raise ROrderStop("047 exclusion fingerprint list is incomplete")
            values = {str(value) for value in listed}
            if len(values) != len(listed):
                raise ROrderStop("047 exclusion fingerprints contain duplicates")
        elif relative.endswith("EXCLUSION_FINGERPRINTS.json"):
            if raw.get("labels_or_scores_read") is not False:
                raise ROrderStop("048 exclusion inventory is not certified label-blind")
            listed = raw.get("union_fingerprints")
            if not isinstance(listed, list):
                raise ROrderStop("048 exclusion union is missing")
            values = {str(value) for value in listed}
            if len(values) != len(listed):
                raise ROrderStop("048 exclusion union contains duplicates")
        else:
            if not isinstance(raw, list) or not raw:
                raise ROrderStop(f"explicit layout source is missing or empty: {relative}")
            episodes = [episode_type.from_record(row) for row in raw]
            values = {str(fingerprint_fn(episode)) for episode in episodes}
            listed = [str(row.get("label_invariant_fingerprint", "")) for row in raw]
            if len(values) != len(episodes) or values != set(listed):
                raise ROrderStop(f"layout fingerprints do not recompute: {relative}")
        if any(len(value) != 64 or any(char not in "0123456789abcdef" for char in value) for value in values):
            raise ROrderStop(f"malformed fingerprint in exclusion source {relative}")
        fingerprints.update(values)
        source_rows.append(
            {
                "path": relative,
                "sha256": _sha256_file(path),
                "unique_fingerprint_count": len(values),
                "labels_or_outcomes_read": False,
            }
        )
    return fingerprints, source_rows


def _frozen_asset_hashes() -> dict[str, str]:
    result: dict[str, str] = {}
    for root in FROZEN_OLD_ROOTS:
        if root.exists():
            for path in sorted(root.rglob("*"), key=lambda item: item.as_posix().casefold()):
                if path.is_file():
                    result[path.relative_to(WORKSPACE).as_posix()] = _sha256_file(path)
    for path in FROZEN_OLD_FILES:
        if path.is_file():
            result[path.relative_to(WORKSPACE).as_posix()] = _sha256_file(path)
    return result


def _source_hash_manifest(
    runtime: Mapping[str, Any], exclusion_rows: Sequence[Mapping[str, Any]], gp_identity: Mapping[str, Any]
) -> dict[str, Any]:
    entries: dict[str, dict[str, Any]] = {}
    for relative in SOURCE_RELATIVE_PATHS:
        path = _workspace_path(relative)
        if not path.is_file():
            raise ROrderStop(f"registered source file is missing: {relative}")
        entries[relative] = {"sha256": _sha256_file(path), "bytes": path.stat().st_size, "role": "code_or_plan"}
    for row in exclusion_rows:
        relative = str(row["path"])
        path = _workspace_path(relative)
        entries[relative] = {"sha256": _sha256_file(path), "bytes": path.stat().st_size, "role": "known_exclusion_manifest"}
    for relative in GP_ACCEPTED_ARTIFACT_PATHS:
        path = _workspace_path(relative)
        if not path.is_file():
            raise ROrderStop(f"accepted GP identity source is missing: {relative}")
        entries[relative] = {"sha256": _sha256_file(path), "bytes": path.stat().st_size, "role": "accepted_frozen_GP_identity"}
    _known, inventory = runtime["source_manifest_inventory"]()
    for row in inventory:
        path = Path(str(row["path"]))
        try:
            relative = path.resolve().relative_to(WORKSPACE.resolve()).as_posix()
        except ValueError:
            relative = str(row["path"])
        entries[relative] = {
            "sha256": str(row["sha256"]),
            "bytes": path.stat().st_size if path.is_file() else None,
            "role": "existing_known_layout_source_manifest",
            "episode_count": int(row["episode_count"]),
            "unique_fingerprint_count": int(row["unique_semantic_fingerprints"]),
        }
    frozen = _frozen_asset_hashes()
    for relative, digest in frozen.items():
        entries[relative] = {
            "sha256": digest,
            "bytes": _workspace_path(relative).stat().st_size,
            "role": "frozen_legacy_asset_unchanged_check",
        }
    return {
        "protocol_id": "r_order_v1",
        "run_id": "r01",
        "recursive_tree_hashing": False,
        "legacy_tree_protection": "frozen prior-run output trees and named legacy handoff/source files were recursively byte-hashed; new PLAN-049 inputs are individually bound",
        "accepted_gp_rule_identity": dict(gp_identity),
        "entries": [{"path": key, **value} for key, value in sorted(entries.items())],
        "frozen_legacy_file_hashes": frozen,
    }


def _condition_specs(config: Mapping[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    conditions: list[dict[str, Any]] = []
    dp = config["conditions"]["DP"]
    for p in dp["p_values"]:
        for repeat in dp["repeats"]:
            conditions.append({
                "condition_id": f"DP_p{float(p):.1f}_r{int(repeat)}",
                "regime": "DP", "family": "delayed_positions", "p": float(p), "repeat": int(repeat),
            })
    tws = config["conditions"]["TWS"]
    for width in tws["window_sizes"]:
        for repeat in tws["repeats"]:
            conditions.append({
                "condition_id": f"TWS_B{int(width)}_r{int(repeat)}",
                "regime": "TWS", "family": "time_window_shuffle", "window_size": int(width), "repeat": int(repeat),
            })
    clean = config["conditions"]["CLEAN"]
    conditions.append({"condition_id": "CLEAN", "regime": "CLEAN", "family": "CLEAN", "repeat": 0})
    if len(conditions) != 15 or len({row["condition_id"] for row in conditions}) != 15:
        raise ROrderStop("frozen 15 unique conditions changed")
    aliases = [dict(row, alias_of="CLEAN", generates_sample=False) for row in clean["aliases"]]
    return conditions, aliases


def _episode_manifest_record(episode: Any, ordinal: int, runtime: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "split": str(episode.metadata["split"]),
        "layout_ordinal": int(ordinal),
        "episode_id": str(episode.episode_id),
        "geometry_id": str(episode.parameter_group),
        "num_stacks": int(episode.num_stacks),
        "max_tiers": int(episode.max_tiers),
        "num_containers": int(episode.num_containers),
        "label_invariant_fingerprint": str(runtime["label_invariant_fingerprint"](episode)),
    }


def _generate_layouts(
    config: Mapping[str, Any], runtime: Mapping[str, Any], exclusion: set[str]
) -> tuple[dict[str, list[Any]], dict[str, Any]]:
    episode_type = runtime["OnlineCRPEpisode"]
    fingerprint_fn = runtime["label_invariant_fingerprint"]
    seen = set(exclusion)
    splits: dict[str, list[Any]] = {}
    collision_log: dict[str, Any] = {}
    for split in SPLITS:
        spec = config["layouts"][split]
        split_episodes: list[Any] = []
        split_retry_rows: dict[str, int] = {}
        for geometry_id, stacks, tiers, count in GEOMETRIES:
            per_geometry = int(spec["per_geometry"])
            for index in range(per_geometry):
                accepted = None
                for retry in range(100):
                    episode_id = f"r-order-r01-{split}-{geometry_id}-{index:03d}"
                    candidate = runtime["rank_layout_episode"](
                        episode_id=episode_id,
                        geometry_id=geometry_id,
                        split=split,
                        num_stacks=stacks,
                        max_tiers=tiers,
                        num_containers=count,
                        master_seed=int(spec["master_seed"]),
                        index=index,
                        retry=retry,
                    )
                    metadata = dict(candidate.metadata)
                    metadata.update({
                        "protocol_id": "r_order_v1",
                        "run_id": "r01",
                        "split": split,
                        "geometry_id": geometry_id,
                        "data_role": "confirm_unscored_layout" if split == "confirm_candidates" else f"{split}_layout",
                    })
                    candidate = replace(candidate, episode_id=episode_id, parameter_group=geometry_id, metadata=metadata)
                    fingerprint = str(fingerprint_fn(candidate))
                    if fingerprint in seen:
                        continue
                    if not isinstance(candidate, episode_type):
                        raise ROrderStop("layout generator changed its episode type")
                    seen.add(fingerprint)
                    accepted = candidate
                    split_retry_rows[f"{geometry_id}/{index:03d}"] = retry
                    break
                if accepted is None:
                    raise ROrderStop(f"100 collision retries exhausted at {split}/{geometry_id}/{index:03d}")
                split_episodes.append(accepted)
        expected_per_geometry = int(spec["per_geometry"])
        counts = {
            geometry_id: sum(str(row.parameter_group) == geometry_id for row in split_episodes)
            for geometry_id, *_ in GEOMETRIES
        }
        expected_count = 2 * expected_per_geometry
        if len(split_episodes) != expected_count or counts != {"G1": expected_per_geometry, "G2": expected_per_geometry}:
            raise ROrderStop(f"{split} layout counts differ from registration: {counts}")
        splits[split] = split_episodes
        collision_log[split] = {
            "count": len(split_episodes),
            "by_geometry": counts,
            "collision_retries": sum(split_retry_rows.values()),
            "retry_by_layout": split_retry_rows,
        }
    fingerprints_by_split = {
        split: {str(fingerprint_fn(episode)) for episode in episodes}
        for split, episodes in splits.items()
    }
    if any(len(rows) != len(splits[split]) for split, rows in fingerprints_by_split.items()):
        raise ROrderStop("duplicate layout fingerprint within a split")
    for left_index, left in enumerate(SPLITS):
        for right in SPLITS[left_index + 1 :]:
            if fingerprints_by_split[left] & fingerprints_by_split[right]:
                raise ROrderStop(f"new layout splits overlap: {left}/{right}")
    if any(values & exclusion for values in fingerprints_by_split.values()):
        raise ROrderStop("a generated layout collides with the known exclusion union")
    collision_log["cross_split_overlap"] = 0
    collision_log["known_source_overlap"] = 0
    return splits, collision_log


def prepare_m0() -> dict[str, Any]:
    config = _load_config()
    if OUTPUT_ROOT.exists():
        raise ROrderStop(f"refusing to overwrite or resume existing M0 output: {OUTPUT_ROOT}")
    preflight_audit = _preflight_audit_binding()
    runtime = _runtime_imports()
    frozen_before = _frozen_asset_hashes()
    gp_rule, gp_identity = _load_gp_rule(runtime)
    del gp_rule
    exclusions, explicit_rows = _explicit_fingerprints(runtime)
    source_bindings = _source_hash_manifest(runtime, explicit_rows, gp_identity)
    known_source_fps, known_source_inventory = runtime["source_manifest_inventory"]()
    exclusions.update(str(value) for value in known_source_fps)
    splits, split_audit = _generate_layouts(config, runtime, exclusions)
    frozen_after = _frozen_asset_hashes()
    if frozen_before != frozen_after:
        changed = sorted(set(frozen_before) ^ set(frozen_after) | {key for key in frozen_before.keys() & frozen_after.keys() if frozen_before[key] != frozen_after[key]})
        raise ROrderStop(f"frozen legacy output/source hashes changed during M0 prepare: {changed[:5]}")

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=False)
    _write_json_new(SOURCE_BINDINGS_PATH, source_bindings)
    _write_json_new(
        OUTPUT_ROOT / "M0_KNOWN_EXCLUSION_MANIFEST.json",
        {
            "protocol_id": "r_order_v1",
            "run_id": "r01",
            "fingerprint_kind": "label_invariant_fingerprint",
            "known_fingerprint_count": len(exclusions),
            "semantic_fingerprints": sorted(exclusions),
            "explicit_sources": explicit_rows,
            "existing_known_source_inventory": known_source_inventory,
            "labels_or_outcomes_read": False,
            "complete_training_universe_overlap_claimed_zero": False,
        },
    )
    condition_rows, aliases = _condition_specs(config)
    _write_json_new(
        OUTPUT_ROOT / "M0_CONDITIONS_EVALUATOR_ONLY.json",
        {"unique_conditions": condition_rows, "aliases": aliases, "forecast_generated": False},
    )
    split_manifest_rows: dict[str, list[dict[str, Any]]] = {}
    for split in SPLITS:
        private_rows = [runtime["episode_to_record"](episode) for episode in splits[split]]
        private_payload = "".join(json.dumps(row, sort_keys=True, ensure_ascii=False, separators=(",", ":")) + "\n" for row in private_rows).encode("utf-8")
        private_path = OUTPUT_ROOT / "layouts_evaluator_private" / f"{split}.jsonl.gz"
        private_path.parent.mkdir(parents=True, exist_ok=True)
        compressed = gzip.compress(private_payload, compresslevel=6, mtime=0)
        _write_new(private_path, compressed)
        manifest = [
            _episode_manifest_record(episode, ordinal, runtime)
            for ordinal, episode in enumerate(splits[split])
        ]
        split_manifest_rows[split] = manifest
        _write_json_new(OUTPUT_ROOT / "manifests" / f"{split}.json", manifest)
    _write_json_new(
        OUTPUT_ROOT / "M0_DATA_SPLIT_AUDIT.json",
        {
            "split_audit": split_audit,
            "generated_total": sum(len(rows) for rows in splits.values()),
            "known_exclusion_count": len(exclusions),
            "known_source_overlap": 0,
            "cross_split_overlap": 0,
            "confirm_outcomes_or_scores_read": False,
            "forecasts_labels_fits_or_rollouts_generated": False,
        },
    )
    output_files = {
        path.relative_to(OUTPUT_ROOT).as_posix(): {
            "sha256": _sha256_file(path),
            "bytes": path.stat().st_size,
        }
        for path in sorted(OUTPUT_ROOT.rglob("*"), key=lambda item: item.as_posix().casefold())
        if path.is_file() and path != PREPARE_RECEIPT_PATH
    }
    prepare_receipt = {
        "protocol_id": "r_order_v1",
        "run_id": "r01",
        "phase": "M0_PREPARE",
        "status": "M0_PREPARE_COMPLETE_PENDING_ROOT_REVIEW",
        "started_utc": None,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "source_bindings_sha256": _sha256_file(SOURCE_BINDINGS_PATH),
        "source_entry_count": len(source_bindings["entries"]),
        "output_sha256": output_files,
        "layout_counts": {split: len(rows) for split, rows in splits.items()},
        "layout_counts_by_geometry": split_audit,
        "known_fingerprint_count": len(exclusions),
        "cross_split_overlap_count": 0,
        "known_source_overlap_count": 0,
        "confirm_scored_or_unsealed": False,
        "labels_fit_bank_cost_or_performance_screen_ran": False,
        "science_ledger_started": False,
        "root_release_present": False,
        "frozen_legacy_hashes_unchanged": frozen_before == frozen_after,
        "preflight_audit": preflight_audit,
    }
    _write_json_new(PREPARE_RECEIPT_PATH, prepare_receipt)
    print(
        "M0_PREPARE_COMPLETE "
        f"layouts={sum(len(rows) for rows in splits.values())} "
        f"known_fingerprints={len(exclusions)} outputs={len(output_files)}",
        flush=True,
    )
    return prepare_receipt


def _source_binding_digest() -> str:
    if not SOURCE_BINDINGS_PATH.is_file():
        raise ROrderStop("M0 source binding manifest is missing")
    manifest = _read_json(SOURCE_BINDINGS_PATH)
    for row in manifest.get("entries", []):
        path = _workspace_path(str(row["path"]))
        if not path.is_file() or _sha256_file(path) != row["sha256"]:
            raise ROrderStop(f"source hash changed after M0: {row['path']}")
    return _sha256_file(SOURCE_BINDINGS_PATH)


def _revision_source_binding_digest() -> str:
    if not M0_REVISION_SOURCE_BINDINGS_PATH.is_file():
        raise ROrderStop("M0 REV02 source binding manifest is missing")
    manifest = _read_json(M0_REVISION_SOURCE_BINDINGS_PATH)
    for row in manifest.get("entries", []):
        path = _workspace_path(str(row["path"]))
        if not path.is_file() or _sha256_file(path) != row["sha256"]:
            raise ROrderStop(f"REV02 source hash changed: {row['path']}")
    return _sha256_file(M0_REVISION_SOURCE_BINDINGS_PATH)


def _verify_rev01_archive_and_data() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    if not REV01_ARCHIVE_MANIFEST_PATH.is_file():
        raise ROrderStop("REV01 immutable archive is absent")
    archive = _read_json(REV01_ARCHIVE_MANIFEST_PATH)
    if (archive.get("archive_id") != "M0_REVIEW_REV01"
            or archive.get("status") != "READ_ONLY_REVISION_BASELINE"
            or archive.get("original_root_review_status") != "REVISE_M0_REVIEW"
            or archive.get("original_outputs_modified") is not False
            or archive.get("layouts_regenerated") is not False
            or archive.get("science_started") is not False):
        raise ROrderStop("REV01 archive metadata does not preserve the reviewed baseline")
    for entry in archive.get("entries", []):
        relative = str(entry["archive_path"]).replace("\\", "/")
        path = (REV01_ARCHIVE_ROOT / Path(*relative.split("/"))).resolve()
        try:
            path.relative_to(REV01_ARCHIVE_ROOT.resolve())
        except ValueError as exc:
            raise ROrderStop("REV01 archive entry escapes its immutable archive") from exc
        if not path.is_file() or _sha256_file(path) != entry.get("sha256") \
                or path.stat().st_size != int(entry.get("bytes", -1)):
            raise ROrderStop(f"REV01 archived bytes changed: {relative}")

    archived_source_entry = next(
        (entry for entry in archive["entries"]
         if entry.get("source_path") == "R/output/r_order_v1/r01/M0_SOURCE_BINDINGS.json"),
        None,
    )
    if archived_source_entry is None:
        raise ROrderStop("REV01 archive does not include its source binding snapshot")
    base_source_path = REV01_ARCHIVE_ROOT / Path(*str(archived_source_entry["archive_path"]).split("/"))
    base_source = _read_json(base_source_path)
    if _sha256_file(base_source_path) != archive.get("original_source_bindings_sha256"):
        raise ROrderStop("REV01 source snapshot does not match its original binding hash")

    prepare_entry = next(
        entry for entry in archive["entries"]
        if entry.get("role") == "M0_RECEIPT"
        and entry.get("source_path") == "R/output/r_order_v1/r01/M0_PREPARE_RECEIPT.json"
    )
    test_entry = next(
        entry for entry in archive["entries"]
        if entry.get("role") == "M0_RECEIPT"
        and entry.get("source_path") == "R/output/r_order_v1/r01/M0_TEST_RECEIPT.json"
    )
    review_entry = next(entry for entry in archive["entries"] if entry.get("role") == "ROOT_REVIEW_REV01")
    prepare_path = OUTPUT_ROOT / "M0_PREPARE_RECEIPT.json"
    test_path = OUTPUT_ROOT / "M0_TEST_RECEIPT.json"
    review_path = OUTPUT_ROOT / "ROOT_M0_REVIEW_REV01.json"
    for live_path, expected_entry in (
        (prepare_path, prepare_entry), (test_path, test_entry), (review_path, review_entry),
    ):
        if not live_path.is_file() or _sha256_file(live_path) != expected_entry["sha256"] \
                or live_path.stat().st_size != int(expected_entry["bytes"]):
            raise ROrderStop(f"original REV01 receipt/review changed: {live_path.name}")
    if _sha256_file(BASE_SOURCE_BINDINGS_PATH) != archive.get("original_source_bindings_sha256"):
        raise ROrderStop("original REV01 source binding bytes changed")

    original_outputs = archive.get("original_m0_output_hashes")
    prepare_receipt = _read_json(prepare_path)
    if prepare_receipt.get("output_sha256") != original_outputs or not isinstance(original_outputs, dict):
        raise ROrderStop("REV01 archive and prepare receipt disagree on the immutable data outputs")
    for relative, detail in original_outputs.items():
        path = (OUTPUT_ROOT / Path(*str(relative).replace("\\", "/").split("/"))).resolve()
        try:
            path.relative_to(OUTPUT_ROOT.resolve())
        except ValueError as exc:
            raise ROrderStop("REV01 data output path escapes the run root") from exc
        if not path.is_file() or _sha256_file(path) != detail.get("sha256") \
                or path.stat().st_size != int(detail.get("bytes", -1)):
            raise ROrderStop(f"registered REV01 data output changed: {relative}")
    return archive, base_source, prepare_receipt


def _validate_rev02_source_inventory(
    entries: Sequence[Mapping[str, Any]], archived_code_rows: Mapping[str, Any],
) -> dict[str, int]:
    registered_paths = set(SOURCE_RELATIVE_PATHS)
    all_bound_paths = {str(row["path"]) for row in entries}
    missing_registered_paths = registered_paths - all_bound_paths
    if missing_registered_paths:
        raise ROrderStop(
            f"REV02 source inventory is missing registered paths: {sorted(missing_registered_paths)[:5]}"
        )
    code_paths = {str(row["path"]) for row in entries if row.get("role") == "code_or_plan"}
    if code_paths != set(archived_code_rows):
        raise ROrderStop("REV02 code-or-plan roles differ from the immutable REV01 code snapshot")
    return {
        "registered_source_path_count": len(registered_paths),
        "complete_bound_entry_count": len(all_bound_paths),
        "frozen_or_transitive_entry_count": len(all_bound_paths - registered_paths),
        "code_or_plan_count": len(code_paths),
    }


def _rev02_prepare_attempt_audit_bindings() -> list[dict[str, Any]]:
    bindings = []
    for path in (REV02_PREPARE_ATTEMPT_AUDIT_PATH, REV02_PREPARE_ATTEMPT_02_AUDIT_PATH):
        if not path.is_file():
            raise ROrderStop(f"a recorded REV02 prepare attempt audit is absent: {path.name}")
        resolved = path.resolve()
        try:
            relative = resolved.relative_to(WORKSPACE.resolve()).as_posix()
        except ValueError as exc:
            raise ROrderStop("REV02 prepare attempt audit path escapes workspace") from exc
        bindings.append({
            "path": relative,
            "sha256": _sha256_file(resolved),
            "bytes": resolved.stat().st_size,
        })
    return bindings


def prepare_m0_revision_rev02() -> dict[str, Any]:
    """Build a new M0 review bundle while reusing the exact REV01 layout bytes."""
    if M0_REVISION_ROOT.exists():
        raise ROrderStop(f"refusing to overwrite or resume M0 revision bundle: {M0_REVISION_ROOT}")
    archive, parent_manifest, parent_prepare = _verify_rev01_archive_and_data()
    entries = []
    archived_rows = {
        str(row["source_path"]): row for row in archive["entries"]
        if row.get("role") == "code_or_plan"
    }
    for original in parent_manifest.get("entries", []):
        row = dict(original)
        relative = str(row["path"])
        path = _workspace_path(relative)
        if not path.is_file():
            raise ROrderStop(f"REV02 registered source disappeared: {relative}")
        digest = _sha256_file(path)
        if row.get("role") == "code_or_plan":
            archived_source = archived_rows.get(relative)
            if archived_source is None:
                raise ROrderStop(f"REV01 source snapshot is missing for {relative}")
            if digest != row.get("sha256") and relative not in M0_REVISION_ALLOWED_CHANGED_SOURCES:
                raise ROrderStop(f"source changed outside the authorized REV02 diagnostic patch: {relative}")
            row["sha256"] = digest
            row["bytes"] = path.stat().st_size
        elif digest != row.get("sha256"):
            raise ROrderStop(f"non-code / protected source changed after REV01: {relative}")
        entries.append(row)
    if set(M0_REVISION_ALLOWED_CHANGED_SOURCES) - {
        str(row["path"]) for row in entries if row.get("role") == "code_or_plan"
    }:
        raise ROrderStop("REV02 allowed-source list and M0 source binding do not align")

    source_manifest = dict(parent_manifest)
    source_manifest.update({
        "revision_id": "M0_REVIEW_REV02",
        "revision_parent_source_bindings_sha256": archive["original_source_bindings_sha256"],
        "revision_parent_root_review_sha256": archive["original_root_review_sha256"],
        "entries": entries,
        "revision_scope": "relation calibration diagnostics only; no model/gate/threshold/trajectory/data changes",
    })
    source_inventory_summary = _validate_rev02_source_inventory(entries, archived_rows)
    preflight = _preflight_audit_binding()
    prepare_attempt_audits = _rev02_prepare_attempt_audit_bindings()
    output_hashes = dict(archive["original_m0_output_hashes"])
    M0_REVISION_ROOT.mkdir(parents=True, exist_ok=False)
    _write_json_new(M0_REVISION_SOURCE_BINDINGS_PATH, source_manifest)
    source_hash = _revision_source_binding_digest()
    receipt = {
        "protocol_id": "r_order_v1", "run_id": "r01", "revision_id": "M0_REVIEW_REV02",
        "phase": "M0_REVISION_PREPARE", "status": "M0_REVISION_PREPARED_PENDING_ROOT_REVIEW",
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "source_bindings_path": M0_REVISION_SOURCE_BINDINGS_PATH.relative_to(OUTPUT_ROOT).as_posix(),
        "source_bindings_sha256": source_hash,
        "parent_root_review_rev01_sha256": archive["original_root_review_sha256"],
        "parent_archive_manifest_sha256": _sha256_file(REV01_ARCHIVE_MANIFEST_PATH),
        "parent_prepare_receipt_sha256": _sha256_file(BASE_PREPARE_RECEIPT_PATH),
        "parent_test_receipt_sha256": _sha256_file(BASE_TEST_RECEIPT_PATH),
        "output_sha256": output_hashes,
        "reused_original_data_output_sha256": output_hashes,
        "original_layout_count": int(sum(parent_prepare.get("layout_counts", {}).values())),
        "original_layout_bytes_verified_unchanged": True,
        "layouts_regenerated": False,
        "labels_fit_or_science_rollouts_generated": False,
        "science_ledger_started": False, "root_release_present": False,
        "confirm_candidates_scored_or_unsealed": False,
        "preflight_audit": preflight,
        "prepare_attempt_audits": prepare_attempt_audits,
        "source_inventory_summary": source_inventory_summary,
        "review_scope": "M0 REV02 calibration/select relation diagnostic implementation only",
    }
    _write_json_new(M0_REVISION_PREPARE_RECEIPT_PATH, receipt)
    print(f"M0_REVISION_PREPARED_REV02 outputs_reused={len(output_hashes)} source_entries={len(entries)}", flush=True)
    return receipt


def _preflight_audit_binding() -> dict[str, Any]:
    """Return the immutable binding for the corrected, output-free M0 preflight attempt."""
    if not PREFLIGHT_AUDIT_PATH.is_file():
        raise ROrderStop("the prior M0 preflight failure audit is absent")
    audit = _read_json(PREFLIGHT_AUDIT_PATH)
    required = {
        "classification": "PREFLIGHT_STATIC_SOURCE_LIST_FAILURE",
        "exit_code": 1,
        "outputs_created": False,
        "labels_fit_or_science_started": False,
        "output_directory_created": False,
        "science_ledger_started": False,
    }
    if any(audit.get(key) != expected for key, expected in required.items()):
        raise ROrderStop("M0 preflight audit is incomplete or inconsistent")
    expected_command = ["py", "-3.12", "-B", "-m", "R.experiments.r_order_protocol", "prepare-m0"]
    if audit.get("command") != expected_command:
        raise ROrderStop("M0 preflight audit does not preserve the observed prepare command")
    if audit.get("stdout") != "" or audit.get("stderr") != "ROrderStop: registered source file is missing: GP/representations.py\n":
        raise ROrderStop("M0 preflight audit does not preserve the observed error streams")
    path = PREFLIGHT_AUDIT_PATH.resolve()
    try:
        relative = path.relative_to(WORKSPACE.resolve()).as_posix()
    except ValueError as exc:
        raise ROrderStop("M0 preflight audit path escapes workspace") from exc
    return {"path": relative, "sha256": _sha256_file(path), "bytes": path.stat().st_size}


def _verify_m0_artifact_bytes() -> dict[str, str]:
    if not PREPARE_RECEIPT_PATH.is_file():
        raise ROrderStop("M0 prepare receipt is absent")
    receipt = _read_json(PREPARE_RECEIPT_PATH)
    expected = receipt.get("output_sha256")
    if not isinstance(expected, dict) or not expected:
        raise ROrderStop("M0 prepare receipt has no immutable output inventory")
    actual: dict[str, str] = {}
    for relative, detail in expected.items():
        path = (OUTPUT_ROOT / Path(*str(relative).replace("\\", "/").split("/"))).resolve()
        try:
            path.relative_to(OUTPUT_ROOT.resolve())
        except ValueError as exc:
            raise ROrderStop(f"M0 artifact inventory path escapes output root: {relative}") from exc
        if not path.is_file():
            raise ROrderStop(f"M0 immutable data artifact disappeared: {relative}")
        digest = _sha256_file(path)
        if digest != detail.get("sha256") or path.stat().st_size != int(detail.get("bytes", -1)):
            raise ROrderStop(f"M0 immutable artifact bytes changed: {relative}")
        actual[str(relative)] = digest
    audit_binding = receipt.get("preflight_audit")
    if not isinstance(audit_binding, dict):
        raise ROrderStop("M0 prepare receipt does not bind the prior preflight audit")
    audit_relative = str(audit_binding.get("path", "")).replace("\\", "/")
    expected_relative = PREFLIGHT_AUDIT_PATH.resolve().relative_to(WORKSPACE.resolve()).as_posix()
    if audit_relative != expected_relative:
        raise ROrderStop("M0 preflight audit path differs from the registered sibling artifact")
    audit_path = _workspace_path(audit_relative).resolve()
    try:
        audit_path.relative_to(WORKSPACE.resolve())
    except ValueError as exc:
        raise ROrderStop("M0 preflight audit path escapes workspace") from exc
    audit_hash = _sha256_file(audit_path) if audit_path.is_file() else None
    if (audit_hash != audit_binding.get("sha256") or not audit_path.is_file()
            or audit_path.stat().st_size != int(audit_binding.get("bytes", -1))):
        raise ROrderStop("M0 preflight audit bytes changed")
    actual[f"@workspace/{audit_relative}"] = str(audit_hash)
    attempt_bindings = receipt.get("prepare_attempt_audits")
    registered_attempt_paths = [
        path.resolve().relative_to(WORKSPACE.resolve()).as_posix()
        for path in (REV02_PREPARE_ATTEMPT_AUDIT_PATH, REV02_PREPARE_ATTEMPT_02_AUDIT_PATH)
    ]
    if not isinstance(attempt_bindings, list) or len(attempt_bindings) != len(registered_attempt_paths):
        raise ROrderStop("M0 prepare receipt must bind both recorded REV02 prepare attempts")
    for binding, expected_attempt_relative in zip(attempt_bindings, registered_attempt_paths):
        if not isinstance(binding, dict):
            raise ROrderStop("M0 prepare attempt audit binding is malformed")
        attempt_relative = str(binding.get("path", "")).replace("\\", "/")
        if attempt_relative != expected_attempt_relative:
            raise ROrderStop("M0 prepare attempt audit path differs from its registered sibling")
        attempt_path = _workspace_path(attempt_relative).resolve()
        try:
            attempt_path.relative_to(WORKSPACE.resolve())
        except ValueError as exc:
            raise ROrderStop("M0 prepare attempt audit path escapes workspace") from exc
        attempt_hash = _sha256_file(attempt_path) if attempt_path.is_file() else None
        if (attempt_hash != binding.get("sha256") or not attempt_path.is_file()
                or attempt_path.stat().st_size != int(binding.get("bytes", -1))):
            raise ROrderStop("M0 prepare attempt audit bytes changed")
        actual[f"@workspace/{attempt_relative}"] = str(attempt_hash)
    return actual


def _verify_m0_test_receipt() -> dict[str, Any]:
    if not TEST_RECEIPT_PATH.is_file():
        raise ROrderStop("M0 synthetic test receipt is absent")
    receipt = _read_json(TEST_RECEIPT_PATH)
    if receipt.get("status") != "PASS" or receipt.get("exit_code") != 0 or receipt.get("synthetic_only") is not True:
        raise ROrderStop("M0 synthetic correctness tests did not pass")
    if receipt.get("prepare_receipt_sha256") != _sha256_file(PREPARE_RECEIPT_PATH):
        raise ROrderStop("M0 test receipt is not bound to the current prepare receipt")
    if receipt.get("source_bindings_sha256") != _source_binding_digest():
        raise ROrderStop("M0 test receipt is not bound to the current source hashes")
    prepare_receipt = _read_json(PREPARE_RECEIPT_PATH)
    if receipt.get("preflight_audit") != prepare_receipt.get("preflight_audit"):
        raise ROrderStop("M0 test receipt and prepare receipt disagree on the preflight audit")
    for key in ("stdout", "stderr", "summary"):
        path_field = f"{key}_path"
        hash_field = f"{key}_sha256"
        size_field = f"{key}_bytes"
        relative = str(receipt.get(path_field, "")).replace("\\", "/")
        path = (OUTPUT_ROOT / Path(*relative.split("/"))).resolve()
        try:
            path.relative_to(OUTPUT_ROOT.resolve())
        except ValueError as exc:
            raise ROrderStop(f"M0 {key} path escapes output root") from exc
        if not path.is_file() or _sha256_file(path) != receipt.get(hash_field):
            raise ROrderStop(f"M0 {key} artifact hash changed")
        if path.stat().st_size != int(receipt.get(size_field, -1)):
            raise ROrderStop(f"M0 {key} artifact byte count changed")
    return receipt


def _stage_artifact_paths(stage: str) -> set[str]:
    paths: set[str] = set()
    for path in OUTPUT_ROOT.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(OUTPUT_ROOT).as_posix()
        if relative == f"{stage}_STAGE_RECEIPT.json":
            continue
        if stage == "M4":
            paths.add(relative)
            continue
        name = path.name
        if name.startswith(f"{stage}_") or relative.startswith(f"private/{stage}_"):
            paths.add(relative)
    return paths


def _stage_artifact_inventory(stage: str) -> dict[str, dict[str, Any]]:
    return {relative: {"sha256": _sha256_file(OUTPUT_ROOT / relative),
                       "bytes": (OUTPUT_ROOT / relative).stat().st_size}
            for relative in sorted(_stage_artifact_paths(stage))}


def _verify_stage_artifacts(receipt: Mapping[str, Any]) -> None:
    stage = str(receipt.get("stage", ""))
    expected = receipt.get("output_artifact_sha256")
    if stage not in ("M1", "M2", "M3", "M4") or not isinstance(expected, dict):
        raise ROrderStop("science-stage receipt has no artifact inventory")
    actual_paths = _stage_artifact_paths(stage)
    if actual_paths != set(str(path) for path in expected):
        raise ROrderStop(f"{stage} stage output inventory changed: added/removed artifacts")
    for relative, detail in expected.items():
        path = (OUTPUT_ROOT / Path(*str(relative).split("/"))).resolve()
        try:
            path.relative_to(OUTPUT_ROOT.resolve())
        except ValueError as exc:
            raise ROrderStop(f"{stage} artifact inventory path escapes output root") from exc
        if not path.is_file() or _sha256_file(path) != detail.get("sha256") \
                or path.stat().st_size != int(detail.get("bytes", -1)):
            raise ROrderStop(f"{stage} stage artifact bytes changed: {relative}")


def verify_root_release(stage: str) -> dict[str, Any]:
    if stage not in RELEASE_PATHS:
        raise ValueError("unknown ROOT release stage")
    if not PREPARE_RECEIPT_PATH.is_file() or not TEST_RECEIPT_PATH.is_file():
        raise ROrderStop("M0 prepare/test receipts are required before any stage release")
    binding_hash = _source_binding_digest()
    immutable_m0_artifacts = _verify_m0_artifact_bytes()
    prepare_hash = _sha256_file(PREPARE_RECEIPT_PATH)
    test_hash = _sha256_file(TEST_RECEIPT_PATH)
    _verify_m0_test_receipt()
    if not ROOT_M0_REVIEW_PATH.is_file():
        raise ROrderStop("ROOT M0 review/release is absent; scientific stage is closed")
    m0 = _read_json(ROOT_M0_REVIEW_PATH)
    if (
        m0.get("protocol_id") != "r_order_v1"
        or m0.get("run_id") != "r01"
        or m0.get("status") != "PASS_M0_REVIEW"
        or m0.get("prepare_receipt_sha256") != prepare_hash
        or m0.get("test_receipt_sha256") != test_hash
        or m0.get("source_bindings_sha256") != binding_hash
    ):
        raise ROrderStop("ROOT M0 review is not bound to the current M0 receipts/source hashes")
    release_path = RELEASE_PATHS[stage]
    previous_path: Path | None = None
    if stage == "M1":
        previous_hash = None
    elif stage == "M4":
        previous_stages = [name for name in ("M3", "M2", "M1") if (OUTPUT_ROOT / f"{name}_STAGE_RECEIPT.json").is_file()]
        if not previous_stages:
            raise ROrderStop("M4 requires a completed or STOP science-stage receipt")
        previous_stage = previous_stages[0]
        previous_path = OUTPUT_ROOT / f"{previous_stage}_STAGE_RECEIPT.json"
        previous_receipt = _read_json(previous_path)
        allowed_m4_status = {
            "M1": ("M1_COMPLETE", "M1_SCIENTIFIC_GATE_FAIL_STOP", "M1_NOT_EVALUATED_STOP",
                   "M1_IMPLEMENTATION_STOP", "M1_BUDGET_STOP", "M1_TIMEOUT_STOP"),
            "M2": ("M2_COMPLETE", "M2_SCIENTIFIC_GATE_FAIL_STOP", "M2_IMPLEMENTATION_STOP",
                   "M2_BUDGET_STOP", "M2_TIMEOUT_STOP"),
            "M3": ("M3_COMPLETE", "M3_SCIENTIFIC_GATE_FAIL_STOP", "M3_INSUFFICIENT_POWER_STOP",
                   "M3_RUNTIME_CAP_STOP", "M3_IMPLEMENTATION_STOP", "M3_BUDGET_STOP", "M3_TIMEOUT_STOP"),
        }
        if previous_receipt.get("stage") != previous_stage or previous_receipt.get("status") not in allowed_m4_status[previous_stage]:
            raise ROrderStop("M4 predecessor is not an auditable complete/STOP stage receipt")
        _verify_stage_artifacts(previous_receipt)
        for ancestor in previous_stages[1:]:
            ancestor_receipt = _read_json(OUTPUT_ROOT / f"{ancestor}_STAGE_RECEIPT.json")
            _verify_stage_artifacts(ancestor_receipt)
        previous_hash = _sha256_file(previous_path)
    else:
        previous_path = OUTPUT_ROOT / f"M{int(stage[1:]) - 1}_STAGE_RECEIPT.json"
        if not previous_path.is_file():
            raise ROrderStop(f"previous completed stage receipt is missing for {stage}")
        previous_receipt = _read_json(previous_path)
        previous_hash = _sha256_file(previous_path)
        if stage == "M2" and (
            previous_receipt.get("stage") != "M1"
            or previous_receipt.get("scientific_gate_status") != "PASS"
            or not str(previous_receipt.get("status", "")).startswith("M1_COMPLETE")
        ):
            raise ROrderStop("M2 requires a completed M1 whose scientific gates passed")
        if stage == "M2":
            _verify_stage_artifacts(previous_receipt)
        if stage == "M3" and (
            previous_receipt.get("stage") != "M2"
            or previous_receipt.get("point_gate_pass") is not True
            or previous_receipt.get("cost_gate_pass") is not True
            or not str(previous_receipt.get("status", "")).startswith("M2_COMPLETE")
        ):
            raise ROrderStop("M3 requires M2 point and cost gates to have passed")
        if stage == "M3":
            _verify_stage_artifacts(previous_receipt)
            m1_receipt_path = OUTPUT_ROOT / "M1_STAGE_RECEIPT.json"
            if not m1_receipt_path.is_file():
                raise ROrderStop("M3 requires the M1 fitted model/tau stage receipt")
            _verify_stage_artifacts(_read_json(m1_receipt_path))
    if not release_path.is_file():
        raise ROrderStop(f"ROOT {stage} release is absent; stage remains closed")
    release = _read_json(release_path)
    expected_status = f"PASS_{stage}_RELEASE"
    if (
        release.get("protocol_id") != "r_order_v1"
        or release.get("run_id") != "r01"
        or release.get("stage") != stage
        or release.get("status") != expected_status
        or release.get("root_m0_review_sha256") != _sha256_file(ROOT_M0_REVIEW_PATH)
        or release.get("source_bindings_sha256") != binding_hash
        or release.get("m0_artifact_sha256") != immutable_m0_artifacts
        or release.get("prepare_receipt_sha256") != prepare_hash
        or release.get("test_receipt_sha256") != test_hash
        or release.get("previous_stage_receipt_sha256") != previous_hash
    ):
        raise ROrderStop(f"ROOT {stage} release is unbound or has an invalid status")
    return {"path": release_path.relative_to(WORKSPACE).as_posix(), "sha256": _sha256_file(release_path), "record": release}


class ScienceLedger:
    """One elapsed interval per stage worker; nested children are not double-counted."""

    def __init__(self, phase: str) -> None:
        if phase not in PHASE_CAP_SECONDS:
            raise ValueError("unknown science phase")
        self.phase = phase
        self.run_token = hashlib.sha256(os.urandom(32)).hexdigest()[:32]
        self.started_utc: datetime | None = None
        self.started_monotonic: float | None = None
        self.available_seconds: float | None = None
        self.release_sha256: str | None = None

    @classmethod
    def attach_running(cls, phase: str, run_token: str) -> "ScienceLedger":
        ledger = cls(phase)
        ledger.run_token = str(run_token)
        starts = [row for row in ledger._events()
                  if row.get("event") == "start" and row.get("run_token") == ledger.run_token]
        stops = [row for row in ledger._events()
                 if row.get("event") == "stop" and row.get("run_token") == ledger.run_token]
        if len(starts) != 1 or stops or starts[0].get("phase") != phase:
            raise ROrderStop("stage worker has no unique active parent ledger interval")
        ledger.started_utc = datetime.fromisoformat(str(starts[0]["timestamp_utc"]))
        ledger.started_monotonic = time.perf_counter()
        ledger.release_sha256 = str(starts[0].get("release_sha256", ""))
        return ledger

    def _events(self) -> list[dict[str, Any]]:
        if not SCIENCE_LEDGER_PATH.is_file():
            return []
        rows = []
        for line_number, line in enumerate(SCIENCE_LEDGER_PATH.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ROrderStop(f"science ledger malformed at line {line_number}") from exc
            if not isinstance(row, dict):
                raise ROrderStop("science ledger rows must be objects")
            rows.append(row)
        return rows

    def _totals(self) -> tuple[float, dict[str, float], set[str]]:
        starts: dict[str, tuple[datetime, str]] = {}
        closed: set[str] = set()
        total = 0.0
        phases: dict[str, float] = defaultdict(float)
        now = datetime.now(timezone.utc)
        for row in self._events():
            token = str(row.get("run_token", ""))
            event = row.get("event")
            if event == "start":
                if not token or token in starts:
                    raise ROrderStop("science ledger contains duplicate or empty run token")
                stamp = datetime.fromisoformat(str(row["timestamp_utc"]))
                starts[token] = (stamp, str(row["phase"]))
            elif event == "stop":
                if token not in starts or token in closed:
                    raise ROrderStop("science ledger stop has no unique start")
                duration = float(row["worker_elapsed_seconds"])
                if not math.isfinite(duration) or duration < 0:
                    raise ROrderStop("science ledger stop duration is invalid")
                phase = starts[token][1]
                phases[phase] += duration
                total += duration
                closed.add(token)
            else:
                raise ROrderStop("science ledger event must be start or stop")
        open_tokens = set(starts) - closed
        for token in open_tokens:
            stamp, phase = starts[token]
            duration = max(0.0, (now - stamp).total_seconds())
            if token == self.run_token and self.started_monotonic is not None:
                duration = max(0.0, time.perf_counter() - self.started_monotonic)
            phases[phase] += duration
            total += duration
        return total, dict(phases), open_tokens

    def start(self, release: Mapping[str, Any]) -> None:
        total, phase_totals, open_tokens = self._totals()
        if open_tokens:
            raise ROrderStop("an earlier science worker has no closed ledger event; inspect partial output")
        if total >= TOTAL_CAP_SECONDS or phase_totals.get(self.phase, 0.0) >= PHASE_CAP_SECONDS[self.phase]:
            raise ROrderStop(f"registered science wall cap reached before {self.phase}")
        self.available_seconds = min(
            TOTAL_CAP_SECONDS - total,
            PHASE_CAP_SECONDS[self.phase] - phase_totals.get(self.phase, 0.0),
        )
        self.release_sha256 = str(release["sha256"])
        self.started_utc = datetime.now(timezone.utc)
        self.started_monotonic = time.perf_counter()
        self._append({
            "event": "start",
            "protocol_id": "r_order_v1",
            "run_id": "r01",
            "phase": self.phase,
            "run_token": self.run_token,
            "timestamp_utc": self.started_utc.isoformat(),
            "prior_total_science_wall_seconds": total,
            "prior_phase_science_wall_seconds": phase_totals.get(self.phase, 0.0),
            "release_sha256": release["sha256"],
            "nested_processes_double_counted": False,
        })

    def check(self) -> None:
        if self.started_monotonic is None:
            raise ROrderStop("science ledger was not started")
        total, phases, _open = self._totals()
        elapsed = max(0.0, time.perf_counter() - self.started_monotonic)
        if total >= TOTAL_CAP_SECONDS or phases.get(self.phase, 0.0) >= PHASE_CAP_SECONDS[self.phase] or elapsed >= PHASE_CAP_SECONDS[self.phase]:
            raise ROrderStop(f"science wall cap reached during {self.phase}")

    def stop(self, *, status: str, exit_code: int) -> dict[str, Any]:
        if self.started_monotonic is None or self.started_utc is None:
            return {"status": "NOT_STARTED", "phase_elapsed_seconds": 0.0}
        elapsed = max(0.0, time.perf_counter() - self.started_monotonic)
        self._append({
            "event": "stop",
            "protocol_id": "r_order_v1",
            "run_id": "r01",
            "phase": self.phase,
            "run_token": self.run_token,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "worker_elapsed_seconds": elapsed,
            "status": status,
            "exit_code": int(exit_code),
            "nested_processes_double_counted": False,
        })
        total, phases, open_tokens = self._totals()
        if open_tokens:
            raise ROrderStop("science ledger interval remained open after stop")
        return {
            "status": status,
            "phase_elapsed_seconds": elapsed,
            "phase_cumulative_seconds": phases.get(self.phase, 0.0),
            "total_science_wall_seconds": total,
            "phase_cap_seconds": PHASE_CAP_SECONDS[self.phase],
            "total_cap_seconds": TOTAL_CAP_SECONDS,
            "nested_processes_double_counted": False,
        }

    @staticmethod
    def _append(row: Mapping[str, Any]) -> None:
        SCIENCE_LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)
        with SCIENCE_LEDGER_PATH.open("a", encoding="utf-8", newline="\n") as destination:
            destination.write(json.dumps(row, sort_keys=True, separators=(",", ":")))
            destination.write("\n")
            destination.flush()
            os.fsync(destination.fileno())


def _load_split(split: str, runtime: Mapping[str, Any]) -> tuple[list[Any], list[dict[str, Any]]]:
    if split not in SPLITS:
        raise ValueError(f"unknown layout split: {split}")
    manifest = _read_json(OUTPUT_ROOT / "manifests" / f"{split}.json")
    private_path = OUTPUT_ROOT / "layouts_evaluator_private" / f"{split}.jsonl.gz"
    with gzip.open(private_path, "rt", encoding="utf-8") as source:
        records = [json.loads(line) for line in source if line.strip()]
    if len(records) != len(manifest):
        raise ROrderStop(f"private layout count differs from public manifest for {split}")
    episodes = [runtime["OnlineCRPEpisode"].from_record(row) for row in records]
    seen = set()
    for ordinal, (episode, row) in enumerate(zip(episodes, manifest, strict=True)):
        fingerprint = str(runtime["label_invariant_fingerprint"](episode))
        if fingerprint != row.get("label_invariant_fingerprint") or fingerprint in seen or int(row["layout_ordinal"]) != ordinal:
            raise ROrderStop(f"layout manifest/fingerprint changed in {split}")
        seen.add(fingerprint)
    return episodes, manifest


def _episode_context(
    episode: Any,
    fingerprint: str,
    condition: Mapping[str, Any],
    runtime: Mapping[str, Any],
) -> tuple[Any, Any, tuple[int, ...], dict[str, Any]]:
    from R.scrp.r_order_public import Q_PRIOR

    true_future = tuple(int(value) for value in episode.retrieval_sequence[1:])
    regime = str(condition["regime"])
    if regime == "CLEAN":
        forecast = true_future
        seed = None
    elif regime == "DP":
        seed = int(runtime["semantic_forecast_seed"](fingerprint, "delayed_positions", int(condition["repeat"])))
        forecast = tuple(runtime["make_evaluator_forecast"](
            true_future,
            family="delayed_positions",
            p_evaluator_only=float(condition["p"]),
            seed=seed,
        ))
    elif regime == "TWS":
        seed = int(runtime["semantic_time_window_seed"](
            fingerprint, window_size=int(condition["window_size"]), repeat=int(condition["repeat"])
        ))
        forecast = tuple(runtime["make_time_window_forecast"](
            true_future, window_size=int(condition["window_size"]), seed=seed
        ))
    else:
        raise ROrderStop(f"unregistered noise regime: {regime}")
    if len(forecast) != len(true_future) or set(forecast) != set(true_future):
        raise ROrderStop("condition forecast is not an immutable permutation of the fixed truth")
    provider = runtime["ForecastProvider"](
        episode.retrieval_sequence,
        p=0.0,
        seed=0 if seed is None else seed,
        q_prior=Q_PRIOR,
        max_horizon=20,
        forecast_future=forecast,
    )
    environment = runtime["OnlineCRPEnvironment"](episode, lookahead_horizon=0)
    reset_count = int(environment.state.retrieval_count)
    reset_events = runtime["sync_retrieval_events_with_audit"](
        provider, episode.retrieval_sequence, 0, reset_count
    )
    if provider.max_horizon != 20 or not math.isclose(provider.history.q_prior, Q_PRIOR, rel_tol=0.0, abs_tol=1e-15):
        raise ROrderStop("provider max horizon/q prior differs from registration")
    return environment, provider, forecast, {
        "regime_evaluator_only": regime,
        "condition_id_evaluator_only": str(condition["condition_id"]),
        "forecast_seed_evaluator_only": seed,
        "forecast_sha256_evaluator_only": _sha256_bytes(_canonical_bytes(list(forecast))),
        "reset_retrieval_count": reset_count,
        "reset_reveal_events_private": reset_events,
    }


def _summary_to_record(summary: Any) -> dict[str, Any]:
    return {
        "q": float(summary.q),
        "displacement": float(summary.displacement),
        "coverage": float(summary.coverage),
        "retrieval_fraction": float(summary.retrieval_fraction),
        "records": [float(value) for value in summary.records],
    }


def _summary_from_record(row: Mapping[str, Any]) -> Any:
    from R.scrp.r_order_public import PublicHistorySummary

    return PublicHistorySummary(
        q=float(row["q"]),
        displacement=float(row["displacement"]),
        coverage=float(row["coverage"]),
        retrieval_fraction=float(row["retrieval_fraction"]),
        records=tuple(float(value) for value in row["records"]),
    )


def _public_frame_record(frame: object, summary: Any | None = None) -> dict[str, Any]:
    observation = frame.observation
    record = {
        "rank_tensor": np.asarray(observation.rank_tensor, dtype=np.int32).tolist(),
        "occupied_mask": np.asarray(observation.occupied_mask, dtype=np.bool_).tolist(),
        "legal_destination_mask": np.asarray(observation.legal_destination_mask, dtype=np.bool_).tolist(),
        "legal_destinations": [int(value) for value in frame.legal_destinations],
        "max_tiers": int(frame.max_tiers),
        "retrieval_count": int(frame.retrieval_count),
        "relocation_count": int(frame.relocation_count),
        "history_hits": [bool(value) for value in observation.history_hits],
        "q_prior": float(observation.q_prior),
        "q_t": float(observation.q_t),
    }
    if summary is not None:
        record["r_order_summary"] = _summary_to_record(summary)
    return record


def _frame_from_record(row: Mapping[str, Any]) -> Any:
    obs = _StoredObservation(
        np.asarray(row["rank_tensor"], dtype=np.int32),
        np.asarray(row["occupied_mask"], dtype=np.bool_),
        np.asarray(row["legal_destination_mask"], dtype=np.bool_),
        tuple(bool(value) for value in row["history_hits"]),
        float(row["q_prior"]),
        float(row["q_t"]),
        _summary_from_record(row["r_order_summary"]) if "r_order_summary" in row else None,
    )
    return _StoredFrame(
        obs,
        tuple(int(value) for value in row["legal_destinations"]),
        int(row["max_tiers"]),
        int(row["retrieval_count"]),
        int(row["relocation_count"]),
    )


@dataclass(frozen=True)
class _StoredObservation:
    rank_tensor: np.ndarray
    occupied_mask: np.ndarray
    legal_destination_mask: np.ndarray
    history_hits: tuple[bool, ...]
    q_prior: float
    q_t: float
    r_order_summary: Any


@dataclass(frozen=True)
class _StoredFrame:
    observation: _StoredObservation
    legal_destinations: tuple[int, ...]
    max_tiers: int
    retrieval_count: int
    relocation_count: int


def _write_jsonl_gz(path: Path, rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = "".join(json.dumps(_json_safe(row), sort_keys=True, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows).encode("utf-8")
    compressed = gzip.compress(raw, compresslevel=6, mtime=0)
    _write_new(path, compressed)
    return {"sha256": _sha256_file(path), "bytes": path.stat().st_size}


def _read_jsonl_gz(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as source:
        rows = [json.loads(line) for line in source if line.strip()]
    if not all(isinstance(row, dict) for row in rows):
        raise ROrderStop(f"expected object JSONL rows in {path}")
    return rows


def _condition_strength(condition: Mapping[str, Any]) -> str:
    regime = str(condition["regime"])
    if regime == "DP":
        return f"p={float(condition['p']):.1f}"
    if regime == "TWS":
        return f"B={int(condition['window_size'])}"
    if regime == "CLEAN":
        return "CLEAN"
    raise ROrderStop(f"unregistered condition strength: {regime}")


def _make_actor(
    method: str,
    algorithm: str,
    runtime: Mapping[str, Any],
    gp_rule: Mapping[str, Any],
    *,
    rates: Any | None = None,
    tau: float = 0.0,
) -> Any:
    from R.scrp.r_order_actor import FixedHorizonActor, FrozenRActor, ROrderActor

    if method.startswith("FIXED"):
        horizon = int(method.removeprefix("FIXED"))
        return FixedHorizonActor(algorithm, horizon, runtime, gp_rule)
    if method == "OLD_FROZEN_R":
        return FrozenRActor(algorithm, runtime, gp_rule)
    if method == "R_ORDER":
        if rates is None:
            raise ValueError("R_ORDER actor requires a frozen fitted rate model")
        return ROrderActor(algorithm, rates, tau, runtime, gp_rule, include_risk=True)
    if method == "NO_EXTRA":
        if rates is None:
            raise ValueError("NO_EXTRA actor requires the main frozen rate model")
        return ROrderActor(algorithm, rates, tau, runtime, gp_rule, include_risk=False)
    if method == "NO_HISTORY":
        if rates is None:
            raise ValueError("NO_HISTORY actor requires the independent control model")
        return ROrderActor(algorithm, rates, tau, runtime, gp_rule, include_risk=True)
    raise ValueError(f"unregistered actor method: {method}")


def _run_trajectory(
    episode: Any,
    fingerprint: str,
    condition: Mapping[str, Any],
    algorithm: str,
    method: str,
    runtime: Mapping[str, Any],
    gp_rule: Mapping[str, Any],
    *,
    rates: Any | None = None,
    tau: float = 0.0,
    records: Sequence[float] = (),
    on_decision: Any | None = None,
    decision_sink: Any | None = None,
    check_budget: Any | None = None,
) -> dict[str, Any]:
    from R.scrp.r_order_actor import HistoryBoundController

    environment, provider, forecast, context = _episode_context(episode, fingerprint, condition, runtime)
    policy = _make_actor(method, algorithm, runtime, gp_rule, rates=rates, tau=tau)
    controller = HistoryBoundController.fresh(policy, tuple(float(value) for value in records))
    decisions = 0
    maximum = int(episode.num_containers) * max(1, int(episode.max_tiers) - 1)
    frame = runtime["public_frame"](episode, environment, provider)
    while not environment.state.terminated:
        if check_budget is not None:
            check_budget()
        if not frame.legal_destinations:
            raise ROrderStop("nonterminal registered episode has no legal relocation")
        actor_group_start = bool(getattr(policy, "is_group_start", True))
        pre_tracker_records = controller.tracker.records
        pre_previous_frame = controller.previous_frame
        pre_previous_action = controller.previous_action
        action, trace, decorated = controller.select(frame)
        if on_decision is not None:
            on_decision(
                decision_index=decisions,
                frame=frame,
                decorated_frame=decorated,
                action=action,
                trace=trace,
                group_start=actor_group_start,
                environment=environment,
                provider=provider,
                policy=policy,
                tracker_records=controller.tracker.records,
                pre_tracker_records=pre_tracker_records,
                pre_previous_frame=pre_previous_frame,
                pre_previous_action=pre_previous_action,
            )
        before = int(environment.state.retrieval_count)
        before_relocations = int(environment.relocation_count)
        environment.step(int(action))
        after = int(environment.state.retrieval_count)
        runtime["sync_retrieval_events_with_audit"](
            provider, episode.retrieval_sequence, before, after
        )
        if int(environment.relocation_count) != before_relocations + 1:
            raise ROrderStop("a legal trajectory action did not add exactly one relocation")
        after_frame = None if environment.state.terminated else runtime["public_frame"](episode, environment, provider)
        if decision_sink is not None:
            decision_sink({
                "algorithm": algorithm,
                "method": method,
                "layout_fingerprint": fingerprint,
                "condition_id": str(condition["condition_id"]),
                "decision_index": decisions,
                "retrieval_count_before": before,
                "retrieval_count_after": int(environment.state.retrieval_count),
                "relocation_count_before": before_relocations,
                "relocation_count_after": int(environment.relocation_count),
                "public_frame_before": _public_frame_record(decorated, decorated.observation.r_order_summary),
                "own_action": int(action),
                "actor_trace": trace,
                "public_frame_after": None if after_frame is None else _public_frame_record(after_frame),
                "terminal_after_action": bool(environment.state.terminated),
            })
        decisions += 1
        if decisions > maximum:
            raise ROrderStop("trajectory exceeded the registered physical relocation bound")
        if after_frame is not None:
            frame = after_frame
    if int(environment.state.retrieval_count) != int(episode.num_containers):
        raise ROrderStop("trajectory terminated before the complete fixed layout was retrieved")
    return {
        "algorithm": algorithm,
        "method": method,
        "geometry": str(episode.parameter_group),
        "layout_fingerprint": fingerprint,
        "condition_id": str(condition["condition_id"]),
        "regime": str(condition["regime"]),
        "strength": _condition_strength(condition),
        "p": condition.get("p"),
        "window_size": condition.get("window_size"),
        "repeat": int(condition.get("repeat", 0)),
        "N": int(episode.num_containers),
        "R_raw": int(environment.relocation_count),
        "r_over_n": float(environment.relocation_count) / float(episode.num_containers),
        "decision_count": decisions,
        "forecast_sha256_evaluator_only": _sha256_bytes(_canonical_bytes(list(forecast))),
        "condition_context_evaluator_only": context,
        "eg_cached_moves": int(getattr(policy, "cached_moves", 0)),
    }


def _capture_behavior_episode(
    episode: Any,
    manifest_row: Mapping[str, Any],
    condition: Mapping[str, Any],
    behavior_algorithm: str,
    runtime: Mapping[str, Any],
    gp_rule: Mapping[str, Any],
    *,
    check_budget: Any | None = None,
    transition_sink: Any | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    from R.scrp.r_order_actor import HistoryBoundController

    fingerprint = str(manifest_row["label_invariant_fingerprint"])
    environment, provider, forecast, context = _episode_context(episode, fingerprint, condition, runtime)
    policy = _make_actor("FIXED20", behavior_algorithm, runtime, gp_rule)
    controller = HistoryBoundController.fresh(policy)
    selected: dict[str, dict[str, Any] | None] = {"first": None, "nearest_050": None, "nearest_085": None}
    selected_keys: dict[str, tuple[int, int] | None] = {key: None for key in selected}
    n = int(episode.num_containers)
    decisions = 0
    maximum = n * max(1, int(episode.max_tiers) - 1)
    frame = None
    while not environment.state.terminated:
        if check_budget is not None:
            check_budget()
        if frame is None:
            frame = runtime["public_frame"](episode, environment, provider)
        if not frame.legal_destinations:
            raise ROrderStop("nonterminal behavior state has no legal relocation")
        group_start = bool(getattr(policy, "is_group_start", True))
        action, trace, decorated = controller.select(frame)
        allowed = behavior_algorithm != "EG" or group_start
        if allowed:
            slots: list[tuple[str, int]] = []
            if decisions == 0:
                slots.append(("first", 0))
            slots.extend((("nearest_050", 10), ("nearest_085", 17)))
            for slot, numerator in slots:
                if slot == "first" and decisions != 0:
                    continue
                key = (abs(20 * int(frame.retrieval_count) - numerator * n), decisions)
                if selected_keys[slot] is None or key < selected_keys[slot]:
                    selected_keys[slot] = key
                    selected[slot] = {
                        "slot": slot,
                        "target_fraction": None if slot == "first" else numerator / 20.0,
                        "decision_index": decisions,
                        "retrieval_count": int(frame.retrieval_count),
                        "group_start": bool(group_start),
                        "public_frame": decorated,
                        "public_frame_record": _public_frame_record(decorated, decorated.observation.r_order_summary),
                        "private_physical_stacks": runtime["private_physical_stacks"](environment),
                        "environment": deepcopy(environment),
                        "provider": deepcopy(provider),
                        "history_records": controller.tracker.records,
                        "true_retrieval_sequence_private": tuple(int(value) for value in episode.retrieval_sequence),
                        "fixed_forecast_private": tuple(int(value) for value in forecast),
                        "trace": trace,
                    }
        before = int(environment.state.retrieval_count)
        before_relocations = int(environment.relocation_count)
        environment.step(int(action))
        after = int(environment.state.retrieval_count)
        runtime["sync_retrieval_events_with_audit"](provider, episode.retrieval_sequence, before, after)
        if int(environment.relocation_count) != before_relocations + 1:
            raise ROrderStop("fit/cal capture behavior did not add exactly one relocation")
        next_frame = None if environment.state.terminated else runtime["public_frame"](episode, environment, provider)
        if transition_sink is not None:
            transition_sink({
                "behavior_algorithm": behavior_algorithm,
                "layout_fingerprint": fingerprint,
                "condition_id": str(condition["condition_id"]),
                "decision_index": decisions,
                "before_public_frame": _public_frame_record(decorated, decorated.observation.r_order_summary),
                "own_action": int(action),
                "actor_trace": trace,
                "after_public_frame": None if next_frame is None else _public_frame_record(next_frame),
                "retrieval_count_before": before,
                "retrieval_count_after": after,
                "relocation_count_before": before_relocations,
                "relocation_count_after": int(environment.relocation_count),
                "terminal_after_action": bool(environment.state.terminated),
            })
        frame = next_frame
        decisions += 1
        if decisions > maximum:
            raise ROrderStop("fit/cal behavior exceeded the registered physical bound")
    if int(environment.state.retrieval_count) != n:
        raise ROrderStop("fit/cal behavior did not finish the immutable layout")

    slots: list[dict[str, Any]] = []
    seen_physical: set[str] = set()
    for slot in ("first", "nearest_050", "nearest_085"):
        capture = selected[slot]
        if capture is None:
            slots.append({"slot": slot, "missing_reason": "no_eligible_legal_decision", "duplicate_physical_state": False})
            continue
        physical_hash = _sha256_bytes(_canonical_bytes(capture["private_physical_stacks"]))
        duplicate = physical_hash in seen_physical
        if not duplicate:
            seen_physical.add(physical_hash)
        slots.append({
            "slot": slot,
            "missing_reason": "duplicate_physical_state" if duplicate else None,
            "duplicate_physical_state": duplicate,
            "physical_state_hash_evaluator_private": physical_hash,
            "capture_id": f"{fingerprint}|{condition['condition_id']}|{slot}",
            "behavior_algorithm": behavior_algorithm,
            "geometry": str(episode.parameter_group),
            "layout_fingerprint": fingerprint,
            "condition_id": str(condition["condition_id"]),
            "regime": str(condition["regime"]),
            "strength": _condition_strength(condition),
            "p": condition.get("p"),
            "window_size": condition.get("window_size"),
            "repeat": int(condition.get("repeat", 0)),
            "N": n,
            "retrieval_count": int(capture["retrieval_count"]),
            "decision_index": int(capture["decision_index"]),
            "target_fraction": capture["target_fraction"],
            "group_start": bool(capture["group_start"]),
            "public_frame": capture["public_frame_record"],
            "physical_stacks_private": capture["private_physical_stacks"],
            "history_records": list(capture["history_records"]),
            "forecast_sha256_evaluator_only": _sha256_bytes(_canonical_bytes(list(forecast))),
            "true_retrieval_sequence_private": list(capture["true_retrieval_sequence_private"]),
            "fixed_forecast_private": list(capture["fixed_forecast_private"]),
            "condition_context_evaluator_only": context,
            "environment": capture["environment"],
            "provider": capture["provider"],
        })
    trajectory = {
        "behavior_algorithm": behavior_algorithm,
        "layout_fingerprint": fingerprint,
        "condition_id": str(condition["condition_id"]),
        "decision_count": decisions,
        "forecast_sha256_evaluator_only": _sha256_bytes(_canonical_bytes(list(forecast))),
        "condition_context_evaluator_only": context,
    }
    return trajectory, slots


def _pair_rows_for_captures(
    captures: Sequence[Mapping[str, Any]], episode_by_fingerprint: Mapping[str, Any], runtime: Mapping[str, Any]
) -> list[Any]:
    from R.scrp.r_order_training import extract_pairwise_labels

    output: list[Any] = []
    for capture in captures:
        if capture.get("duplicate_physical_state") or capture.get("missing_reason"):
            continue
        episode = episode_by_fingerprint[str(capture["layout_fingerprint"])]
        frame = _frame_from_record(capture["public_frame"])
        labels = extract_pairwise_labels(
            frame,
            capture["physical_stacks_private"],
            episode.retrieval_sequence,
            behavior_algorithm=str(capture["behavior_algorithm"]),
            geometry=str(capture["geometry"]),
            layout_fingerprint=str(capture["layout_fingerprint"]),
            regime=str(capture["regime"]),
            strength=str(capture["strength"]),
            repeat=int(capture["repeat"]),
            capture_id=str(capture["capture_id"]),
            condition_id=str(capture["condition_id"]),
        )
        output.extend(labels)
    return output


def _new_relation_capture_state() -> dict[str, Any]:
    return {
        "slots": {"first": None, "nearest_050": None, "nearest_085": None},
        "keys": {"first": None, "nearest_050": None, "nearest_085": None},
    }


def _observe_relation_capture(
    state: dict[str, Any], *, episode: Any, fingerprint: str,
    condition: Mapping[str, Any], algorithm: str, runtime: Mapping[str, Any], **decision: Any,
) -> None:
    """Capture the three registered public states from an existing FIXED20 run."""
    decision_index = int(decision["decision_index"])
    group_start = bool(decision["group_start"])
    if algorithm == "EG" and not group_start:
        return
    frame = decision["decorated_frame"]
    legal = bool(frame.legal_destinations)
    if not legal:
        raise ROrderStop("relation-calibration capture observed a nonlegal state")
    n = int(episode.num_containers)
    retrieval_count = int(frame.retrieval_count)
    slots = state["slots"]
    keys = state["keys"]
    targets = {"nearest_050": 10, "nearest_085": 17}
    eligible: list[str] = []
    if decision_index == 0:
        eligible.append("first")
    for slot, numerator in targets.items():
        key = (abs(20 * retrieval_count - numerator * n), decision_index)
        if keys[slot] is None or key < keys[slot]:
            keys[slot] = key
            eligible.append(slot)
    for slot in eligible:
        summary = frame.observation.r_order_summary
        stacks = runtime["private_physical_stacks"](decision["environment"])
        slots[slot] = {
            "slot": slot,
            "target_fraction": None if slot == "first" else targets[slot] / 20.0,
            "decision_index": decision_index,
            "retrieval_count": retrieval_count,
            "retrieval_fraction": retrieval_count / n,
            "group_start": group_start,
            "public_frame": _public_frame_record(frame, summary),
            "physical_stacks_private": stacks,
            "physical_state_hash_evaluator_private": _sha256_bytes(_canonical_bytes(stacks)),
            "action": int(decision["action"]),
            "actor_trace": decision["trace"],
            "behavior_algorithm": algorithm,
            "geometry": str(episode.parameter_group),
            "layout_fingerprint": fingerprint,
            "condition_id": str(condition["condition_id"]),
            "regime": str(condition["regime"]),
            "strength": _condition_strength(condition),
            "repeat": int(condition.get("repeat", 0)),
            "capture_id": f"{fingerprint}|{condition['condition_id']}|{slot}",
            "status": "SELECTED",
        }


def _finish_relation_capture_state(state: Mapping[str, Any]) -> list[dict[str, Any]]:
    finalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    for slot in ("first", "nearest_050", "nearest_085"):
        capture = state["slots"][slot]
        if capture is None:
            finalized.append({"slot": slot, "status": "MISSING_NO_ELIGIBLE_LEGAL_DECISION"})
            continue
        row = dict(capture)
        physical_hash = str(row["physical_state_hash_evaluator_private"])
        row["duplicate_physical_state"] = physical_hash in seen
        row["status"] = "DUPLICATE_PHYSICAL_STATE" if physical_hash in seen else "MEASURED_CAPTURE"
        if physical_hash not in seen:
            seen.add(physical_hash)
        finalized.append(row)
    return finalized


_RELATION_GROUP_CSV_FIELDS = (
    "dimension", "category", "row_count", "layout_count", "capture_count", "weight_mass",
    "weighted_brier", "mean_prediction", "observed_rate", "calibration_gap",
)
_RELATION_BIN_CSV_FIELDS = (
    "dimension", "category", "probability_bin", "bin_lower", "bin_upper",
    "row_count", "layout_count", "capture_count", "weight_mass", "weighted_brier",
    "mean_prediction", "observed_rate", "calibration_gap",
)


def _persist_relation_calibration_artifacts(
    prefix: str, tables: Mapping[str, Any], *, metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    scored_path = OUTPUT_ROOT / "private" / f"{prefix}_RELATION_CALIBRATION_PAIRS.jsonl.gz"
    group_path = OUTPUT_ROOT / f"{prefix}_RELATION_CALIBRATION_GROUPS.csv"
    bins_path = OUTPUT_ROOT / f"{prefix}_RELATION_CALIBRATION_BINS.csv"
    summary_path = OUTPUT_ROOT / f"{prefix}_RELATION_CALIBRATION_SUMMARY.json"
    _write_jsonl_gz(scored_path, tables["scored_rows"])
    _write_csv_new(group_path, tables["group_rows"], _RELATION_GROUP_CSV_FIELDS)
    _write_csv_new(bins_path, tables["probability_bin_rows"], _RELATION_BIN_CSV_FIELDS)
    summary = {
        "status": "MEASURED" if tables["scored_rows"] else "UNMEASURED_NO_ELIGIBLE_PAIRS",
        "overall": tables["overall"],
        "pair_count": len(tables["scored_rows"]),
        "capture_count": int(tables["overall"]["capture_count"]),
        "weight_definition": tables["weight_definition"],
        "probability_bin_definition": tables["probability_bin_definition"],
        "diagnostic_only_no_gate_or_ci": True,
        "raw_pair_sidecar": scored_path.relative_to(OUTPUT_ROOT).as_posix(),
        "group_table": group_path.relative_to(OUTPUT_ROOT).as_posix(),
        "probability_bin_table": bins_path.relative_to(OUTPUT_ROOT).as_posix(),
    }
    if metadata is not None:
        summary.update(dict(metadata))
    _write_json_new(summary_path, summary)
    return summary


def _fit_record(fit: Any) -> dict[str, Any]:
    return {
        "coefficients": np.asarray(fit.model.coefficients, dtype=np.float64).tolist(),
        "active_feature_indices": list(fit.model.active_feature_indices),
        "model_id": fit.model.model_id,
        "objective": fit.objective,
        "gradient_inf_norm": fit.gradient_inf_norm,
        "iterations": fit.iterations,
        "converged": fit.converged,
        "row_count": fit.row_count,
        "weight_sum": fit.weight_sum,
        "scipy_message": fit.scipy_message,
    }


def _model_from_record(row: Mapping[str, Any]) -> Any:
    from R.scrp.r_order_public import PlackettLuceRates

    return PlackettLuceRates(
        np.asarray(row["coefficients"], dtype=np.float64),
        tuple(int(value) for value in row["active_feature_indices"]),
        str(row["model_id"]),
    )


def _qfit_record(fit: Any) -> dict[str, Any]:
    return asdict(fit)


def _qfit_from_record(row: Mapping[str, Any]) -> Any:
    from R.scrp.r_order_training import QOnlyFit

    return QOnlyFit(**row)


def _run_native_continuation(
    episode: Any, capture: Mapping[str, Any], sequence: Sequence[int], algorithm: str,
    runtime: Mapping[str, Any], gp_rule: Mapping[str, Any], check_budget: Any | None,
    *, branch_id: str | None = None, transition_sink: Any | None = None,
) -> dict[str, Any]:
    from R.scrp.r_order_actor import FixedHorizonActor, HistoryBoundController
    from R.scrp.r_order_public import PublicHistoryTracker

    environment = deepcopy(capture["environment"])
    provider = deepcopy(capture["provider"])
    source_clone_signature_before = runtime["private_signature"](capture["environment"], capture["provider"])
    initial_relocation_count = int(environment.relocation_count)
    initial_retrieval_count = int(environment.state.retrieval_count)
    clone_initial_signature = runtime["private_signature"](environment, provider)
    if environment is capture["environment"] or provider is capture["provider"]:
        raise ROrderStop("candidate branch did not receive independent environment/provider clones")
    continuation = FixedHorizonActor(algorithm, 20, runtime, gp_rule)
    controller = HistoryBoundController(
        continuation, PublicHistoryTracker(tuple(float(value) for value in capture["history_records"]))
    )
    current_frame = runtime["public_frame"](episode, environment, provider)
    candidate_moves = 0
    candidate_trace: list[dict[str, Any]] = []
    continuation_trace: list[dict[str, Any]] = []
    maximum = int(episode.num_containers) * max(1, int(episode.max_tiers) - 1)
    for sequence_index, destination in enumerate(sequence):
        if check_budget is not None:
            check_budget()
        if int(destination) not in current_frame.legal_destinations:
            raise ROrderStop("candidate branch sequence became illegal before completion")
        decorated = controller.observe_action(current_frame, int(destination))
        before = int(environment.state.retrieval_count)
        relocations = int(environment.relocation_count)
        before_public = _public_frame_record(decorated, decorated.observation.r_order_summary)
        environment.step(int(destination))
        after = int(environment.state.retrieval_count)
        runtime["sync_retrieval_events_with_audit"](provider, episode.retrieval_sequence, before, after)
        if int(environment.relocation_count) != relocations + 1:
            raise ROrderStop("candidate sequence failed relocation conservation")
        candidate_moves += 1
        after_frame = None if environment.state.terminated else runtime["public_frame"](episode, environment, provider)
        event = {
            "branch_id": branch_id, "phase": "candidate", "decision_index": candidate_moves - 1,
            "before_public_frame": before_public, "own_action": int(destination),
            "after_public_frame": None if after_frame is None else _public_frame_record(after_frame),
            "retrieval_count_before": before, "retrieval_count_after": after,
            "relocation_count_before": relocations, "relocation_count_after": int(environment.relocation_count),
            "terminal_after_action": bool(environment.state.terminated),
        }
        candidate_trace.append(event)
        if transition_sink is not None:
            transition_sink(event)
        if environment.state.terminated:
            if sequence_index + 1 != len(sequence):
                raise ROrderStop("candidate sequence continued after the environment became terminal")
            break
        current_frame = after_frame
    continuation_rows: list[dict[str, Any]] = []
    decisions = 0
    while not environment.state.terminated:
        if check_budget is not None:
            check_budget()
        frame = runtime["public_frame"](episode, environment, provider)
        action, trace, decorated = controller.select(frame)
        before = int(environment.state.retrieval_count)
        relocations = int(environment.relocation_count)
        before_public = _public_frame_record(decorated, decorated.observation.r_order_summary)
        environment.step(int(action))
        after = int(environment.state.retrieval_count)
        runtime["sync_retrieval_events_with_audit"](provider, episode.retrieval_sequence, before, after)
        if int(environment.relocation_count) != relocations + 1:
            raise ROrderStop("native continuation failed relocation conservation")
        after_frame = None if environment.state.terminated else runtime["public_frame"](episode, environment, provider)
        event = {
            "branch_id": branch_id, "phase": "native_continuation", "decision_index": decisions,
            "before_public_frame": before_public, "own_action": int(action),
            "after_public_frame": None if after_frame is None else _public_frame_record(after_frame),
            "retrieval_count_before": before, "retrieval_count_after": after,
            "relocation_count_before": relocations, "relocation_count_after": int(environment.relocation_count),
            "terminal_after_action": bool(environment.state.terminated), "actor_trace": trace,
        }
        continuation_trace.append(event)
        continuation_rows.append({"action": int(action), "trace": trace})
        if transition_sink is not None:
            transition_sink(event)
        decisions += 1
        if candidate_moves + decisions > maximum:
            raise ROrderStop("native continuation exceeded the physical relocation bound")
    if int(environment.state.retrieval_count) != int(episode.num_containers):
        raise ROrderStop("native continuation did not retrieve every container")
    source_clone_signature_after = runtime["private_signature"](capture["environment"], capture["provider"])
    if source_clone_signature_after != source_clone_signature_before:
        raise ROrderStop("candidate branch mutated its source capture environment/provider")
    clone_terminal_signature = runtime["private_signature"](environment, provider)
    total = int(environment.relocation_count) - int(capture["environment"].relocation_count)
    return {
        "continuation_algorithm_evaluator_only": algorithm,
        "candidate_sequence": [int(value) for value in sequence],
        "candidate_moves_included": candidate_moves,
        "continuation_moves": decisions,
        "branch_R_raw": total,
        "branch_R_over_N": total / float(episode.num_containers),
        "clone_initial_relocation_count_evaluator_only": initial_relocation_count,
        "clone_terminal_relocation_count_evaluator_only": int(environment.relocation_count),
        "clone_initial_retrieval_count_evaluator_only": initial_retrieval_count,
        "clone_final_retrieval_count_evaluator_only": int(environment.state.retrieval_count),
        "source_capture_signature_before_sha256": source_clone_signature_before,
        "source_capture_signature_after_sha256": source_clone_signature_after,
        "branch_clone_initial_signature_sha256": clone_initial_signature,
        "branch_clone_terminal_signature_sha256": clone_terminal_signature,
        "branch_transition_count": len(candidate_trace) + len(continuation_trace),
        "candidate_sequence_completed": candidate_moves == len(sequence),
        "candidate_transition_trace": candidate_trace,
        "continuation_trace": continuation_trace,
    }


def _branch_cache_key(capture: Mapping[str, Any], sequence: Sequence[int], algorithm: str, runtime: Mapping[str, Any]) -> str:
    signature = runtime["private_signature"](capture["environment"], capture["provider"])
    payload = {
        "state": signature,
        "public_history_records": list(capture["history_records"]),
        "forecast_sha256": capture["forecast_sha256_evaluator_only"],
        "candidate_sequence": [int(value) for value in sequence],
        "continuation_algorithm": algorithm,
    }
    return _sha256_bytes(_canonical_bytes(payload))


def _make_pair_decisions(
    capture: Mapping[str, Any], algorithm: str, rates: Any, runtime: Mapping[str, Any],
    gp_rule: Mapping[str, Any],
) -> tuple[Any, dict[str, Any]]:
    from R.scrp.r_order_actor import build_native_proposals
    from R.scrp.r_order_public import build_candidate_pool, build_risk_proposal, select_candidate

    frame = _frame_from_record(capture["public_frame"])
    proposals = build_native_proposals(frame, algorithm, runtime, gp_rule)
    risk = build_risk_proposal(frame, algorithm, rates)
    no_history = capture["no_history_model"]
    no_history_risk = build_risk_proposal(frame, algorithm, no_history)
    full_pool = build_candidate_pool(frame, algorithm, proposals, risk)
    no_history_pool = build_candidate_pool(frame, algorithm, proposals, no_history_risk)
    native_pool = build_candidate_pool(frame, algorithm, proposals, None)
    result: dict[str, Any] = {"frame": frame, "full_pool": full_pool,
                              "no_history_pool": no_history_pool, "native_pool": native_pool,
                              "decisions": {}}
    for control, model, pool in (
        ("main", rates, full_pool),
        ("no_history", no_history, no_history_pool),
        ("no_extra", rates, native_pool),
    ):
        result["decisions"][control] = {
            tau: select_candidate(frame, pool, model, tau=tau) for tau in (0.0, 0.25)
        }
    return frame, result


def _tau_family_summary(rows: Sequence[Mapping[str, Any]], *, control: str, tau: float | None) -> dict[str, Any]:
    """Pool baseline and selected R/N before computing registered relative gain."""
    selected = [row for row in rows if row["control"] == control
                and (tau is None or float(row["tau"]) == float(tau))]
    captures: dict[tuple[Any, ...], list[tuple[float, float]]] = defaultdict(list)
    for row in selected:
        baseline, target = float(row["baseline_R_over_N"]), float(row["target_R_over_N"])
        if not (math.isfinite(baseline) and math.isfinite(target)):
            raise ROrderStop("M1 branch contains a nonfinite native/selected R/N value")
        key = (str(row["algorithm"]), str(row["regime"]), str(row["strength"]), int(row["repeat"]),
               str(row["layout_fingerprint"]), str(row["geometry"]), str(row["condition_id"]))
        captures[key].append((baseline, target))
    # Equal-weight captures within each layout-condition, preserving zero baseline values.
    condition_values = {
        key: (float(np.mean([pair[0] for pair in pairs])), float(np.mean([pair[1] for pair in pairs])))
        for key, pairs in captures.items()
    }
    expected = {"DP": ("p=0.1", "p=0.3", "p=0.5"), "TWS": ("B=2", "B=5", "B=10", "B=20")}
    by_family: dict[str, dict[str, Any]] = {}
    for algorithm in ALGORITHMS:
        for family, strengths in expected.items():
            missing: list[dict[str, Any]] = []
            strength_rows: list[dict[str, Any]] = []
            geometry_strength: dict[str, list[tuple[float, float]]] = {"G1": [], "G2": []}
            for strength in strengths:
                values_by_geometry: dict[str, tuple[float, float]] = {}
                support: dict[str, dict[str, int]] = {"G1": {}, "G2": {}}
                for geometry in ("G1", "G2"):
                    repeats: list[tuple[float, float]] = []
                    for repeat in (0, 1):
                        vals = [value for (alg, reg, level, rep, _fp, geo, _cid), value in condition_values.items()
                                if alg == algorithm and reg == family and level == strength
                                and rep == repeat and geo == geometry]
                        support[geometry][str(repeat)] = len(vals)
                        if vals:
                            repeats.append((float(np.mean([value[0] for value in vals])),
                                            float(np.mean([value[1] for value in vals]))))
                        else:
                            missing.append({"algorithm": algorithm, "family": family, "geometry": geometry,
                                            "strength": strength, "repeat": repeat})
                    if len(repeats) == 2:
                        values_by_geometry[geometry] = (float(np.mean([value[0] for value in repeats])),
                                                        float(np.mean([value[1] for value in repeats])))
                if len(values_by_geometry) == 2:
                    for geometry, value in values_by_geometry.items():
                        geometry_strength[geometry].append(value)
                    baseline_s = float(np.mean([value[0] for value in values_by_geometry.values()]))
                    target_s = float(np.mean([value[1] for value in values_by_geometry.values()]))
                    strength_rows.append({
                        "strength": strength, "baseline_R_over_N": baseline_s, "target_R_over_N": target_s,
                        "gain_pct": 100.0 * (baseline_s - target_s) / baseline_s if baseline_s > 0 else None,
                        "by_geometry_baseline_R_over_N": {geo: value[0] for geo, value in values_by_geometry.items()},
                        "by_geometry_target_R_over_N": {geo: value[1] for geo, value in values_by_geometry.items()},
                        "repeat_layout_condition_support": support,
                    })
            complete = not missing and len(strength_rows) == len(strengths)
            if complete:
                baseline_by_geometry = {geo: float(np.mean([value[0] for value in vals]))
                                        for geo, vals in geometry_strength.items()}
                target_by_geometry = {geo: float(np.mean([value[1] for value in vals]))
                                      for geo, vals in geometry_strength.items()}
                baseline_all = float(np.mean(list(baseline_by_geometry.values())))
                target_all = float(np.mean(list(target_by_geometry.values())))
                gain = 100.0 * (baseline_all - target_all) / baseline_all if baseline_all > 0 else None
                geometry_gain = {
                    geo: 100.0 * (baseline_by_geometry[geo] - target_by_geometry[geo]) / baseline_by_geometry[geo]
                    if baseline_by_geometry[geo] > 0 else None for geo in ("G1", "G2")
                }
            else:
                baseline_by_geometry, target_by_geometry = {}, {}
                baseline_all = target_all = gain = None
                geometry_gain = {}
            by_family[f"{algorithm}|{family}"] = {
                "algorithm": algorithm, "family": family, "gain_pct": gain, "strengths": strength_rows,
                "baseline_R_over_N": baseline_all, "target_R_over_N": target_all,
                "by_geometry_baseline_R_over_N": baseline_by_geometry,
                "by_geometry_target_R_over_N": target_by_geometry,
                "by_geometry_gain_pct": geometry_gain,
                "complete_registered_support": complete, "missing_support": missing,
            }
    return {"control": control, "tau": None if tau is None else float(tau), "families": by_family}


def _choose_tau(rows: Sequence[Mapping[str, Any]], control: str) -> dict[str, Any]:
    from R.experiments.r_order_analysis import ALGORITHMS as _ALG

    summaries = {tau: _tau_family_summary(rows, control=control, tau=tau) for tau in (0.0, 0.25)}
    def vector(tau: float) -> list[float]:
        return [float(summaries[tau]["families"][f"{algorithm}|{family}"]["gain_pct"])
                for algorithm in _ALG for family in ("DP", "TWS")
                if summaries[tau]["families"][f"{algorithm}|{family}"]["gain_pct"] is not None]
    complete = {tau: len(vector(tau)) == 8 for tau in summaries}
    common_scores = []
    for tau in (0.0, 0.25):
        values = vector(tau)
        if complete[tau]:
            common_scores.append((min(values), float(np.mean(values)), tau))
    if not common_scores:
        return {"control": control, "mode": "STOP_UNSUPPORTED", "selected_tau_by_algorithm": {},
                "summaries": {str(tau): summaries[tau] for tau in summaries}}
    best_common = max(common_scores, key=lambda row: (row[0], row[1], row[2]))
    if best_common[0] > 0:
        return {"control": control, "mode": "COMMON_TAU", "selected_tau": best_common[2],
                "minimum_gain_pct": best_common[0], "mean_gain_pct": best_common[1],
                "selected_tau_by_algorithm": {algorithm: best_common[2] for algorithm in ALGORITHMS},
                "summaries": {str(tau): summaries[tau] for tau in summaries}}
    algorithm_tau: dict[str, float] = {}
    algorithm_scores: dict[str, Any] = {}
    for algorithm in ALGORITHMS:
        candidates: list[tuple[float, float, float]] = []
        for tau in (0.0, 0.25):
            values = [summaries[tau]["families"][f"{algorithm}|{family}"]["gain_pct"] for family in ("DP", "TWS")]
            if all(value is not None for value in values):
                candidates.append((min(float(v) for v in values), float(np.mean(values)), tau))
        if not candidates:
            return {"control": control, "mode": "STOP_UNSUPPORTED", "selected_tau_by_algorithm": {},
                    "summaries": {str(tau): summaries[tau] for tau in summaries}}
        chosen = max(candidates, key=lambda row: (row[0], row[1], row[2]))
        algorithm_tau[algorithm] = chosen[2]
        algorithm_scores[algorithm] = {"minimum_gain_pct": chosen[0], "mean_gain_pct": chosen[1]}
    return {"control": control, "mode": "ALGORITHM_TAU_FALLBACK", "selected_tau_by_algorithm": algorithm_tau,
            "algorithm_scores": algorithm_scores, "summaries": {str(tau): summaries[tau] for tau in summaries}}


def _public_capture_slim(capture: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "public_frame": capture["public_frame"],
        "history_records": capture["history_records"],
        "layout_fingerprint_sha256": _sha256_bytes(str(capture["layout_fingerprint"]).encode("utf-8")),
        "forecast_sha256": capture["forecast_sha256_evaluator_only"],
    }


def _run_m1_worker(ledger: ScienceLedger) -> dict[str, Any]:
    from R.scrp.r_order_public import PlackettLuceRates
    from R.scrp.r_order_training import (
        fit_plackett_luce, fit_q_only, predict_pair_probability,
        predict_q_only_probability, relation_calibration_tables, weighted_brier,
    )

    config = _load_config()
    runtime = _runtime_imports()
    gp_rule, gp_identity = _load_gp_rule(runtime)
    conditions, aliases = _condition_specs(config)
    condition_ordinals = {row["condition_id"]: index for index, row in enumerate(conditions)}
    fit_episodes, fit_manifest = _load_split("fit", runtime)
    cal_episodes, cal_manifest = _load_split("cal", runtime)
    episode_map = {
        str(row["label_invariant_fingerprint"]): episode
        for episode, row in zip((*fit_episodes, *cal_episodes), (*fit_manifest, *cal_manifest), strict=True)
    }
    fit_slots: list[dict[str, Any]] = []
    fit_slots_all: list[dict[str, Any]] = []
    fit_public: list[dict[str, Any]] = []
    fit_trajectory_receipts: list[dict[str, Any]] = []
    slot_count = 0
    fit_transition_stream = _JsonlCheckpointStream(
        OUTPUT_ROOT / "M1_FIT_PUBLIC_TRANSITIONS_CHECKPOINT.jsonl.gz"
    ).open()
    for episode, manifest in zip(fit_episodes, fit_manifest, strict=True):
        ordinal = int(manifest["layout_ordinal"])
        for condition in conditions:
            ledger.check()
            behavior = ALGORITHMS[(ordinal + condition_ordinals[condition["condition_id"]]) % 4]
            trajectory, slots = _capture_behavior_episode(
                episode, manifest, condition, behavior, runtime, gp_rule, check_budget=ledger.check,
                transition_sink=fit_transition_stream.append,
            )
            fit_transition_stream.sync()
            trajectory_receipt = {**trajectory, "geometry": str(episode.parameter_group),
                                  "condition_id": condition["condition_id"],
                                  "behavior_algorithm": behavior}
            fit_trajectory_receipts.append(trajectory_receipt)
            _append_jsonl_checkpoint(OUTPUT_ROOT / "M1_FIT_TRAJECTORY_CHECKPOINT.jsonl", trajectory_receipt)
            trajectory_fit_slots: list[dict[str, Any]] = []
            for row in slots:
                fit_public.append(_public_capture_slim(row) if row.get("public_frame") else {"slot": row["slot"], "missing_reason": row.get("missing_reason")})
                fit_slots_all.append({key: value for key, value in row.items() if key not in ("environment", "provider", "trace")})
                _append_jsonl_checkpoint(
                    OUTPUT_ROOT / "M1_FIT_CAPTURE_AUDIT_CHECKPOINT.jsonl",
                    {key: value for key, value in row.items() if key not in ("environment", "provider", "trace")},
                )
                if row.get("public_frame"):
                    _append_jsonl_checkpoint(OUTPUT_ROOT / "M1_FIT_PUBLIC_CAPTURE_CHECKPOINT.jsonl", _public_capture_slim(row))
                    _append_jsonl_checkpoint(OUTPUT_ROOT / "private/M1_FIT_CAPTURE_TRUTH_CHECKPOINT.jsonl", {
                        key: value for key, value in row.items() if key in {
                            "capture_id", "physical_state_hash_evaluator_private", "physical_stacks_private",
                            "true_retrieval_sequence_private", "fixed_forecast_private",
                            "forecast_sha256_evaluator_only", "condition_context_evaluator_only", "history_records",
                            "layout_fingerprint", "condition_id", "behavior_algorithm", "slot", "missing_reason",
                        }
                    })
                if row.get("public_frame") and not row.get("duplicate_physical_state") and not row.get("missing_reason"):
                    fit_slots.append(row)
                    trajectory_fit_slots.append(row)
                slot_count += 1
            for label in _pair_rows_for_captures(trajectory_fit_slots, episode_map, runtime):
                _append_jsonl_checkpoint(OUTPUT_ROOT / "private/M1_FIT_PAIR_LABELS_CHECKPOINT.jsonl", asdict(label))
    fit_transition_stream.close()
    if slot_count > int(config["M1"]["max_capture_slots"]):
        raise ROrderStop("fit behavior exceeded the registered 5400 capture slots")
    fit_labels = _pair_rows_for_captures(fit_slots, episode_map, runtime)
    if not fit_labels:
        raise ROrderStop("fit bank has no eligible public-rank pair labels")
    main_fit = fit_plackett_luce(fit_labels, no_history=False, model_id="r_order_main")
    no_history_fit = fit_plackett_luce(fit_labels, no_history=True, model_id="r_order_no_history")
    q_fit = fit_q_only(fit_labels)
    for model_name, fit in (("main", main_fit), ("no_history", no_history_fit)):
        _append_jsonl_checkpoint(OUTPUT_ROOT / "private/M1_FIT_ATTEMPTS_CHECKPOINT.jsonl",
                                 {"model": model_name, "fit": _fit_attempt_safe(_fit_record(fit))})
    _append_jsonl_checkpoint(OUTPUT_ROOT / "private/M1_FIT_ATTEMPTS_CHECKPOINT.jsonl",
                             {"model": "q_only", "fit": _fit_attempt_safe(_qfit_record(q_fit))})
    for name, fit in (("main", main_fit), ("no_history", no_history_fit)):
        if not fit.converged:
            raise ROrderStop(f"{name} rate model failed its frozen optimizer convergence condition")
    if not q_fit.converged:
        raise ROrderStop("q-only comparator failed its frozen optimizer convergence condition")
    fits = {"main": main_fit, "no_history": no_history_fit}
    loao_fits: dict[str, Any] = {}
    loao_diagnostics: dict[str, dict[str, Any]] = {}
    for heldout in ALGORITHMS:
        ledger.check()
        kept = [row for row in fit_labels if row.behavior_algorithm != heldout]
        if not kept:
            diagnostic = {"status": "INSUFFICIENT_SUPPORT", "eligible_pair_count": 0,
                          "heldout_behavior_algorithm": heldout, "fit_attempt": None}
            loao_diagnostics[heldout] = diagnostic
            _append_jsonl_checkpoint(OUTPUT_ROOT / "private/M1_FIT_ATTEMPTS_CHECKPOINT.jsonl",
                                     {"model": f"loao_without_{heldout}", "diagnostic": diagnostic})
            continue
        fit = fit_plackett_luce(kept, no_history=False, model_id=f"loao_without_{heldout}")
        fit_record = _fit_attempt_safe(_fit_record(fit))
        _append_jsonl_checkpoint(OUTPUT_ROOT / "private/M1_FIT_ATTEMPTS_CHECKPOINT.jsonl",
                                 {"model": f"loao_without_{heldout}", "fit": fit_record})
        if not fit.converged:
            loao_diagnostics[heldout] = {
                "status": "NOT_CONVERGED", "eligible_pair_count": len(kept),
                "heldout_behavior_algorithm": heldout, "fit_attempt": fit_record,
            }
        else:
            loao_fits[heldout] = fit
            loao_diagnostics[heldout] = {
                "status": "FIT_CONVERGED", "eligible_pair_count": len(kept),
                "heldout_behavior_algorithm": heldout,
            }

    # Fit is sealed before calibration paths are generated or labels are read.
    fit_model_payload = {
        "protocol_id": "r_order_v1", "fit_labels_generated_before_cal_labels": True,
        "main": _fit_record(main_fit), "no_history": _fit_record(no_history_fit),
        "q_only": _qfit_record(q_fit),
        "loao": {
            name: ({"status": "FIT_CONVERGED", "fit": _fit_record(loao_fits[name])}
                   if name in loao_fits else loao_diagnostics[name])
            for name in ALGORITHMS
        },
        "loao_generalization_scope": "only the four registered behavior algorithms; unknown algorithms are unsupported",
        "fit_capture_count": len(fit_slots), "fit_pair_count": len(fit_labels),
        "gp_rule_identity": gp_identity,
    }
    # Persist fitted parameters before any calibration trajectory or label is read.
    _write_json_new(OUTPUT_ROOT / "M1_MODELS.json", fit_model_payload)

    cal_slots: list[dict[str, Any]] = []
    cal_slots_all: list[dict[str, Any]] = []
    cal_public: list[dict[str, Any]] = []
    cal_trajectory_receipts: list[dict[str, Any]] = []
    cal_transition_stream = _JsonlCheckpointStream(
        OUTPUT_ROOT / "M1_CAL_PUBLIC_TRANSITIONS_CHECKPOINT.jsonl.gz"
    ).open()
    for episode, manifest in zip(cal_episodes, cal_manifest, strict=True):
        ordinal = int(manifest["layout_ordinal"])
        for condition in conditions:
            ledger.check()
            behavior = ALGORITHMS[(ordinal + condition_ordinals[condition["condition_id"]]) % 4]
            trajectory, slots = _capture_behavior_episode(
                episode, manifest, condition, behavior, runtime, gp_rule, check_budget=ledger.check,
                transition_sink=cal_transition_stream.append,
            )
            cal_transition_stream.sync()
            trajectory_receipt = {**trajectory, "geometry": str(episode.parameter_group),
                                  "condition_id": condition["condition_id"],
                                  "behavior_algorithm": behavior}
            cal_trajectory_receipts.append(trajectory_receipt)
            _append_jsonl_checkpoint(OUTPUT_ROOT / "M1_CAL_TRAJECTORY_CHECKPOINT.jsonl", trajectory_receipt)
            trajectory_cal_slots: list[dict[str, Any]] = []
            for row in slots:
                cal_public.append(_public_capture_slim(row) if row.get("public_frame") else {"slot": row["slot"], "missing_reason": row.get("missing_reason")})
                cal_slots_all.append({key: value for key, value in row.items() if key not in ("environment", "provider", "trace")})
                _append_jsonl_checkpoint(
                    OUTPUT_ROOT / "M1_CAL_CAPTURE_AUDIT_CHECKPOINT.jsonl",
                    {key: value for key, value in row.items() if key not in ("environment", "provider", "trace")},
                )
                if row.get("public_frame"):
                    _append_jsonl_checkpoint(OUTPUT_ROOT / "M1_CAL_PUBLIC_CAPTURE_CHECKPOINT.jsonl", _public_capture_slim(row))
                    _append_jsonl_checkpoint(OUTPUT_ROOT / "private/M1_CAL_CAPTURE_TRUTH_CHECKPOINT.jsonl", {
                        key: value for key, value in row.items() if key in {
                            "capture_id", "physical_state_hash_evaluator_private", "physical_stacks_private",
                            "true_retrieval_sequence_private", "fixed_forecast_private",
                            "forecast_sha256_evaluator_only", "condition_context_evaluator_only", "history_records",
                            "layout_fingerprint", "condition_id", "behavior_algorithm", "slot", "missing_reason",
                        }
                    })
                if row.get("public_frame") and not row.get("duplicate_physical_state") and not row.get("missing_reason"):
                    cal_slots.append(row)
                    trajectory_cal_slots.append(row)
            for label in _pair_rows_for_captures(trajectory_cal_slots, episode_map, runtime):
                _append_jsonl_checkpoint(OUTPUT_ROOT / "private/M1_CAL_PAIR_LABELS_CHECKPOINT.jsonl", asdict(label))
    cal_transition_stream.close()
    cal_labels = _pair_rows_for_captures(cal_slots, episode_map, runtime)
    if not cal_labels:
        raise ROrderStop("calibration bank has no eligible pair labels")
    main_brier = weighted_brier(cal_labels, lambda row: predict_pair_probability(main_fit.model, row))
    no_history_brier = weighted_brier(cal_labels, lambda row: predict_pair_probability(no_history_fit.model, row))
    q_brier = weighted_brier(cal_labels, lambda row: predict_q_only_probability(q_fit, row))
    relation_calibration = relation_calibration_tables(
        cal_labels, lambda row: predict_pair_probability(main_fit.model, row)
    )
    relation_calibration_summary = _persist_relation_calibration_artifacts(
        "M1_CAL", relation_calibration
    )
    loao_brier: dict[str, dict[str, Any]] = {}
    for algorithm in ALGORITHMS:
        heldout_rows = [row for row in cal_labels if row.behavior_algorithm == algorithm]
        if algorithm not in loao_fits or not heldout_rows:
            unavailable = (
                "DIAGNOSTIC_UNAVAILABLE_NO_SUPPORTED_CONVERGED_FIT"
                if algorithm not in loao_fits else "DIAGNOSTIC_UNAVAILABLE_NO_HELDOUT_LABELS"
            )
            loao_brier[algorithm] = {"status": unavailable, "weighted_brier": None,
                                     "eligible_pair_count": len(heldout_rows)}
        else:
            loao_brier[algorithm] = {
                "status": "MEASURED",
                "weighted_brier": weighted_brier(
                    heldout_rows,
                    lambda row, model=loao_fits[algorithm].model: predict_pair_probability(model, row),
                ),
                "eligible_pair_count": len(heldout_rows),
            }

    model_records = {
        "main": _fit_record(main_fit), "no_history": _fit_record(no_history_fit),
        "q_only": _qfit_record(q_fit),
        "loao": {
            key: ({"status": "FIT_CONVERGED", "fit": _fit_record(loao_fits[key])}
                  if key in loao_fits else loao_diagnostics[key])
            for key in ALGORITHMS
        },
    }
    _write_json_new(OUTPUT_ROOT / "M1_CAL_PAIR_METRICS.json", {
        "main_weighted_brier": main_brier, "no_history_weighted_brier": no_history_brier,
        "q_only_weighted_brier": q_brier, "loao_heldout_brier": loao_brier,
        "main_relation_calibration": relation_calibration_summary,
        "fit_model_frozen_before_cal_bank_labels": True,
    })
    for capture in cal_slots:
        capture["no_history_model"] = no_history_fit.model

    branch_cache: dict[str, dict[str, Any]] = {}
    branch_rows: list[dict[str, Any]] = []
    branch_trace_rows: list[dict[str, Any]] = []
    selector_rows: list[dict[str, Any]] = []
    for capture_index, capture in enumerate(cal_slots):
        ledger.check()
        algorithm = str(capture["behavior_algorithm"])
        episode = episode_map[str(capture["layout_fingerprint"])]
        frame, selections = _make_pair_decisions(capture, algorithm, main_fit.model, runtime, gp_rule)
        decisions = selections["decisions"]
        selected_sequences: dict[tuple[str, float], tuple[int, ...]] = {}
        sequence_set = {tuple(selections["full_pool"].anchor_sequence)}
        for control in ("main", "no_history", "no_extra"):
            for tau, decision in decisions[control].items():
                sequence = tuple(int(value) for value in decision.sequence)
                selected_sequences[(control, float(tau))] = sequence
                sequence_set.add(sequence)
                selector_rows.append({
                    "capture_id": str(capture["capture_id"]), "algorithm": algorithm,
                    "control": control, "tau": float(tau), "selected_sequence": list(sequence),
                    "anchor_sequence": list(decision.anchor_sequence), "score": float(decision.score),
                    "anchor_score": float(decision.anchor_score), "used_anchor": bool(decision.used_anchor),
                    "candidate_origins": [
                        {"sequence": list(candidate.destinations), "origins": [
                            {"source": origin.source, "horizon": origin.horizon, "proposal_id": origin.proposal_id}
                            for origin in candidate.origins
                        ]}
                        for candidate in (selections["native_pool"] if control == "no_extra" else selections["full_pool"]).candidates
                    ],
                })
        branch_by_sequence: dict[tuple[int, ...], dict[str, Any]] = {}
        for sequence in sorted(sequence_set):
            cache_key = _branch_cache_key(capture, sequence, algorithm, runtime)
            if cache_key not in branch_cache:
                branch_result = _run_native_continuation(
                    episode, capture, sequence, algorithm, runtime, gp_rule, ledger.check,
                    branch_id=cache_key,
                    transition_sink=lambda event: _append_jsonl_checkpoint(
                        OUTPUT_ROOT / "private/M1_BRANCH_TRANSITIONS_CHECKPOINT.jsonl",
                        {"layout_fingerprint": capture["layout_fingerprint"],
                         "condition_id": capture["condition_id"], "capture_id": capture["capture_id"],
                         "continuation_algorithm_evaluator_only": algorithm, **event},
                    ),
                )
                trace_fields = {"candidate_transition_trace", "continuation_trace"}
                branch_trace_rows.append({
                    "cache_key_sha256": cache_key,
                    "continuation_algorithm_evaluator_only": algorithm,
                    "layout_fingerprint": capture["layout_fingerprint"],
                    "condition_id": capture["condition_id"],
                    "capture_id": capture["capture_id"],
                    **{key: value for key, value in branch_result.items() if key not in trace_fields},
                })
                _append_jsonl_checkpoint(OUTPUT_ROOT / "private/M1_BRANCH_TRACE_CHECKPOINT.jsonl", {
                    "cache_key_sha256": cache_key,
                    "continuation_algorithm_evaluator_only": algorithm,
                    "layout_fingerprint": capture["layout_fingerprint"],
                    "condition_id": capture["condition_id"],
                    "capture_id": capture["capture_id"],
                    **{key: value for key, value in branch_result.items() if key not in trace_fields},
                })
                branch_cache[cache_key] = {key: value for key, value in branch_result.items() if key not in trace_fields}
            branch_by_sequence[sequence] = branch_cache[cache_key]
        anchor_sequence = tuple(int(value) for value in selections["full_pool"].anchor_sequence)
        anchor_metric = float(branch_by_sequence[anchor_sequence]["branch_R_over_N"])
        base = {
            "algorithm": algorithm,
            "geometry": str(capture["geometry"]),
            "layout_fingerprint": str(capture["layout_fingerprint"]),
            "regime": str(capture["regime"]),
            "strength": str(capture["strength"]),
            "repeat": int(capture["repeat"]),
            "condition_id": str(capture["condition_id"]),
            "capture_id": str(capture["capture_id"]),
            "N": int(capture["N"]),
            "baseline_R_raw": int(branch_by_sequence[anchor_sequence]["branch_R_raw"]),
            "baseline_R_over_N": anchor_metric,
        }
        branch_row_start = len(branch_rows)
        for control in ("main", "no_history", "no_extra"):
            for tau, sequence in selected_sequences.items():
                if tau[0] != control:
                    continue
                outcome = branch_by_sequence[sequence]
                gain = (anchor_metric - float(outcome["branch_R_over_N"])) / anchor_metric * 100.0 if anchor_metric > 0 else None
                branch_rows.append({
                    **base,
                    "control": control,
                    "tau": float(tau[1]),
                    "selected_sequence": list(sequence),
                    "candidate_relocations_included": len(sequence),
                    "target_R_raw": int(outcome["branch_R_raw"]),
                    "target_R_over_N": float(outcome["branch_R_over_N"]),
                    "gain_pct": gain,
                    "continuation_algorithm_evaluator_only": algorithm,
                    "branch_cache_key_sha256": _branch_cache_key(capture, sequence, algorithm, runtime),
                })
        for row in branch_rows[branch_row_start:]:
            _append_jsonl_checkpoint(OUTPUT_ROOT / "private/M1_BRANCH_ROWS_CHECKPOINT.jsonl", row)
        if capture_index % 100 == 0:
            print(f"M1 calibration branches {capture_index}/{len(cal_slots)}", flush=True)

    tau_main = _choose_tau(branch_rows, "main")
    tau_no_history = _choose_tau(branch_rows, "no_history")
    if tau_main.get("mode") == "STOP_UNSUPPORTED" or tau_no_history.get("mode") == "STOP_UNSUPPORTED":
        tau_supported = False
        main_tau_by_algorithm: dict[str, float] = {}
        nohistory_tau_by_algorithm: dict[str, float] = {}
    else:
        tau_supported = True
        main_tau_by_algorithm = {str(k): float(v) for k, v in tau_main["selected_tau_by_algorithm"].items()}
        nohistory_tau_by_algorithm = {str(k): float(v) for k, v in tau_no_history["selected_tau_by_algorithm"].items()}
    chosen_main_rows: list[dict[str, Any]] = []
    chosen_nohistory_rows: list[dict[str, Any]] = []
    chosen_noextra_rows: list[dict[str, Any]] = []
    if tau_supported:
        for row in branch_rows:
            selected_tau = main_tau_by_algorithm.get(str(row["algorithm"])) if row["control"] in ("main", "no_extra") else nohistory_tau_by_algorithm.get(str(row["algorithm"]))
            if selected_tau is None or float(row["tau"]) != float(selected_tau):
                continue
            if row["control"] == "main":
                chosen_main_rows.append(row)
            elif row["control"] == "no_history":
                chosen_nohistory_rows.append(row)
            elif row["control"] == "no_extra":
                chosen_noextra_rows.append(row)
    main_tau_summary = _tau_family_summary(chosen_main_rows, control="main", tau=None) if chosen_main_rows else None
    selected_main_gain_points = [] if main_tau_summary is None else [
        main_tau_summary["families"][f"{algorithm}|{family}"]["gain_pct"]
        for algorithm in ALGORITHMS for family in ("DP", "TWS")
    ]
    branch_support = bool(
        tau_supported and len(selected_main_gain_points) == 8
        and all(value is not None and math.isfinite(float(value)) and float(value) > 0.0 for value in selected_main_gain_points)
    )
    brier_pass = bool(math.isfinite(main_brier) and math.isfinite(q_brier) and main_brier <= q_brier)
    gate_pass = bool(branch_support and brier_pass)
    not_evaluated = not tau_supported or any(value is None for value in selected_main_gain_points)
    stage_status = "M1_COMPLETE" if gate_pass else ("M1_NOT_EVALUATED_STOP" if not_evaluated else "M1_SCIENTIFIC_GATE_FAIL_STOP")
    calibration = {
        "main_model": tau_main,
        "no_history_control": tau_no_history,
        "main_tau_by_algorithm": main_tau_by_algorithm,
        "no_history_tau_by_algorithm": nohistory_tau_by_algorithm,
        "NO_EXTRA": "same main model and main tau; no independent calibration",
        "selected_main_branch_points": selected_main_gain_points,
        "main_branch_eight_positive": branch_support,
        "main_brier_le_q_only": brier_pass,
        "main_weighted_brier": main_brier,
        "q_only_weighted_brier": q_brier,
        "no_history_weighted_brier": no_history_brier,
        "loao_heldout_cal_brier": loao_brier,
        "development_only_no_confirmation_inference": True,
    }
    _write_json_new(OUTPUT_ROOT / "M1_TAU_CALIBRATION.json", calibration)
    _write_json_new(OUTPUT_ROOT / "M1_GATE.json", {
        "status": "PASS" if gate_pass else stage_status,
        "brier_gate_pass": brier_pass,
        "eight_branch_points_pass": branch_support,
        "selected_main_branch_points": selected_main_gain_points,
        "selected_main_tau_by_algorithm": main_tau_by_algorithm,
        "no_history_tau_by_algorithm": nohistory_tau_by_algorithm,
        "fit_coverage": {"eligible_pairs": len(fit_labels), "fit_captures": len(fit_slots), "cal_eligible_pairs": len(cal_labels), "cal_captures": len(cal_slots)},
        "stop_rule_applied_without_tuning_or_retry": not gate_pass,
    })
    # Persist complete audit/sidecar data after computing labels; public observations stay separate.
    _write_jsonl_gz(OUTPUT_ROOT / "M1_FIT_PUBLIC_CAPTURE_VIEWS.jsonl.gz", fit_public)
    _write_jsonl_gz(OUTPUT_ROOT / "M1_CAL_PUBLIC_CAPTURE_VIEWS.jsonl.gz", cal_public)
    _write_jsonl_gz(OUTPUT_ROOT / "M1_FIT_TRAJECTORY_RECEIPTS.jsonl.gz", fit_trajectory_receipts)
    _write_jsonl_gz(OUTPUT_ROOT / "M1_CAL_TRAJECTORY_RECEIPTS.jsonl.gz", cal_trajectory_receipts)
    _write_jsonl_gz(OUTPUT_ROOT / "M1_FIT_CAPTURE_SLOT_AUDIT_EVALUATOR.jsonl.gz", fit_slots_all)
    _write_jsonl_gz(OUTPUT_ROOT / "M1_CAL_CAPTURE_SLOT_AUDIT_EVALUATOR.jsonl.gz", cal_slots_all)
    _write_jsonl_gz(OUTPUT_ROOT / "private/M1_FIT_CAPTURE_TRUTH.jsonl.gz", [
        {key: value for key, value in row.items() if key in {
            "capture_id", "physical_state_hash_evaluator_private", "physical_stacks_private",
            "true_retrieval_sequence_private", "fixed_forecast_private",
            "forecast_sha256_evaluator_only", "condition_context_evaluator_only", "history_records",
            "layout_fingerprint", "condition_id", "behavior_algorithm", "slot", "missing_reason",
        }} for row in fit_slots
    ])
    _write_jsonl_gz(OUTPUT_ROOT / "private/M1_CAL_CAPTURE_TRUTH.jsonl.gz", [
        {key: value for key, value in row.items() if key in {
            "capture_id", "physical_state_hash_evaluator_private", "physical_stacks_private",
            "true_retrieval_sequence_private", "fixed_forecast_private",
            "forecast_sha256_evaluator_only", "condition_context_evaluator_only", "history_records",
            "layout_fingerprint", "condition_id", "behavior_algorithm", "slot", "missing_reason",
        }} for row in cal_slots
    ])
    _write_jsonl_gz(OUTPUT_ROOT / "private/M1_FIT_PAIR_LABELS.jsonl.gz", [asdict(row) for row in fit_labels])
    _write_jsonl_gz(OUTPUT_ROOT / "private/M1_CAL_PAIR_LABELS.jsonl.gz", [asdict(row) for row in cal_labels])
    _write_jsonl_gz(OUTPUT_ROOT / "private/M1_BRANCH_ROWS.jsonl.gz", branch_rows)
    _write_csv_new(OUTPUT_ROOT / "private/M1_BRANCH_ROWS.csv", branch_rows, (
        "algorithm", "geometry", "layout_fingerprint", "regime", "strength", "repeat", "condition_id",
        "capture_id", "N", "baseline_R_raw", "baseline_R_over_N", "control", "tau",
        "selected_sequence", "candidate_relocations_included", "target_R_raw", "target_R_over_N",
        "gain_pct", "continuation_algorithm_evaluator_only", "branch_cache_key_sha256",
    ))
    _write_jsonl_gz(OUTPUT_ROOT / "private/M1_BRANCH_LABELS_AND_TRACES.jsonl.gz", branch_trace_rows)
    _write_jsonl_gz(OUTPUT_ROOT / "M1_SELECTOR_DECISIONS_EVALUATOR.jsonl.gz", selector_rows)
    return {
        "status": stage_status,
        "scientific_gate_status": "PASS" if gate_pass else ("NOT_EVALUATED" if not_evaluated else "FAIL_STOP"),
        "brier_gate_pass": brier_pass,
        "branch_gate_pass": branch_support,
        "tau_selection": calibration,
        "fit_label_count": len(fit_labels),
        "cal_label_count": len(cal_labels),
        "fit_capture_count": len(fit_slots),
        "cal_capture_count": len(cal_slots),
        "m2_release_allowed": gate_pass,
        "gp_rule_identity": gp_identity,
    }


def _retrieval_stratum(retrieval_count: int, n: int) -> str:
    fraction = int(retrieval_count) / float(n)
    if fraction < 1.0 / 3.0:
        return "early"
    if fraction < 2.0 / 3.0:
        return "mid"
    return "late"


def _cost_state_snapshot(
    *, fingerprint: str, condition: Mapping[str, Any], decision_index: int,
    geometry: str, frame: object, pre_tracker_records: Sequence[float],
    pre_previous_frame: object | None, pre_previous_action: int | None,
) -> dict[str, Any]:
    key_tuple = (str(fingerprint), str(condition["condition_id"]), int(decision_index))
    hash_key = _sha256_bytes(_canonical_bytes(list(key_tuple)))
    return {
        "sort_key": (hash_key, *key_tuple),
        "state_id": hash_key,
        "geometry": str(geometry),
        "stratum": _retrieval_stratum(int(frame.retrieval_count), int(condition.get("num_containers", 1))),
        "layout_fingerprint": str(fingerprint),
        "condition_id": str(condition["condition_id"]),
        "decision_index": int(decision_index),
        "retrieval_count": int(frame.retrieval_count),
        "public_frame": _public_frame_record(frame),
        "pre_tracker_records": [float(value) for value in pre_tracker_records],
        "pre_previous_frame": None if pre_previous_frame is None else _public_frame_record(pre_previous_frame),
        "pre_previous_action": None if pre_previous_action is None else int(pre_previous_action),
    }


def _insert_cost_state(
    pool: dict[tuple[str, str], list[dict[str, Any]]], row: dict[str, Any], *, limit: int = 20
) -> None:
    key = (str(row["geometry"]), str(row["stratum"]))
    values = pool.setdefault(key, [])
    if any(existing["state_id"] == row["state_id"] for existing in values):
        return
    values.append(row)
    values.sort(key=lambda item: item["sort_key"])
    del values[limit:]


def _write_csv_new(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8-sig", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            clean: dict[str, Any] = {}
            for field in fields:
                value = _json_safe(row.get(field))
                if isinstance(value, (dict, list)):
                    value = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                clean[field] = value
            writer.writerow(clean)
        target.flush()
        os.fsync(target.fileno())
    return {"sha256": _sha256_file(path), "bytes": path.stat().st_size}


def _m2_cost_evaluate(
    cost_candidates: Mapping[tuple[str, str], Sequence[Mapping[str, Any]]],
    runtime: Mapping[str, Any], gp_rule: Mapping[str, Any],
    main_models: Mapping[str, Any], main_tau: Mapping[str, float],
    ledger: ScienceLedger,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    from R.scrp.r_order_actor import HistoryBoundController
    from R.scrp.r_order_public import PublicHistoryTracker

    expected_keys = {(geometry, stratum) for geometry, *_ in GEOMETRIES for stratum in ("early", "mid", "late")}
    if set(cost_candidates) != expected_keys or any(len(cost_candidates[key]) != 20 for key in expected_keys):
        missing = {str(key): len(cost_candidates.get(key, ())) for key in sorted(expected_keys)
                   if len(cost_candidates.get(key, ())) != 20}
        raise ROrderStop(f"M2 cost source lacks one or more registered 20-state strata: {missing}")
    states = [dict(row) for key in sorted(expected_keys) for row in cost_candidates[key]]
    if len(states) != 120 or len({row["state_id"] for row in states}) != 120:
        raise ROrderStop("M2 cost source did not produce 120 unique public states")

    samples: dict[str, dict[str, dict[str, list[float]]]] = {
        algorithm: {method: {str(row["state_id"]): [] for row in states} for method in ("R_ORDER", "FIXED20")}
        for algorithm in ALGORITHMS
    }
    warmup_counts = {method: 0 for method in ("R_ORDER", "FIXED20")}
    warmup_ns: dict[str, dict[str, list[int]]] = {algorithm: {method: [] for method in ("R_ORDER", "FIXED20")} for algorithm in ALGORITHMS}
    for state_index, state in enumerate(states):
        ledger.check()
        frame = _frame_from_record(state["public_frame"])
        previous = None if state["pre_previous_frame"] is None else _frame_from_record(state["pre_previous_frame"])
        for algorithm in ALGORITHMS:
            for call_index in range(36):
                order = ("R_ORDER", "FIXED20") if (state_index + call_index) % 2 == 0 else ("FIXED20", "R_ORDER")
                for method in order:
                    policy = _make_actor(
                        method, algorithm, runtime, gp_rule,
                        rates=main_models[algorithm], tau=float(main_tau[algorithm]),
                    )
                    controller = HistoryBoundController(
                        policy,
                        PublicHistoryTracker(tuple(float(value) for value in state["pre_tracker_records"])),
                        previous_frame=previous,
                        previous_action=state["pre_previous_action"],
                    )
                    # Actor/model and frame copies are prepared before the timer.
                    call_frame = _frame_from_record(state["public_frame"])
                    started = time.perf_counter_ns()
                    action, _trace, _decorated = controller.select(call_frame)
                    elapsed = time.perf_counter_ns() - started
                    if action not in call_frame.legal_destinations:
                        raise ROrderStop("M2 cost actor emitted an illegal destination")
                    if call_index < 5:
                        warmup_counts[method] += 1
                        warmup_ns[algorithm][method].append(int(elapsed))
                    else:
                        samples[algorithm][method][str(state["state_id"])].append(elapsed / 1e6)
        for algorithm in ALGORITHMS:
            for method in ("R_ORDER", "FIXED20"):
                timed = samples[algorithm][method][str(state["state_id"])]
                if len(timed) != 31:
                    raise ROrderStop("M2 cost state checkpoint is missing a timed call")
                _append_jsonl_checkpoint(OUTPUT_ROOT / "M2_COST_STATE_CHECKPOINT.jsonl", {
                    "algorithm": algorithm, "method": method, "state_id": state["state_id"],
                    "geometry": state["geometry"], "stratum": state["stratum"],
                    "warmup_calls": 5, "timed_calls": len(timed),
                    "timed_latency_ms": list(timed), "mean_timed_latency_ms": float(np.mean(timed)),
                })
        if state_index % 20 == 0:
            print(f"M2 cost states {state_index}/{len(states)}", flush=True)
    per_algorithm: dict[str, Any] = {}
    call_rows: list[dict[str, Any]] = []
    for algorithm in ALGORITHMS:
        by_method: dict[str, Any] = {}
        for method in ("R_ORDER", "FIXED20"):
            per_state = {
                state_id: float(np.mean(values))
                for state_id, values in samples[algorithm][method].items()
            }
            if any(len(samples[algorithm][method][state_id]) != 31 for state_id in per_state):
                raise ROrderStop("M2 cost measurement missed a registered timed call")
            equal_state_mean = float(np.mean(list(per_state.values())))
            by_stratum: dict[str, float] = {}
            for stratum in ("early", "mid", "late"):
                matching = [float(per_state[str(row["state_id"])]) for row in states
                            if row["geometry"] in ("G1", "G2") and row["stratum"] == stratum]
                by_stratum[stratum] = float(np.mean(matching)) if matching else math.nan
            by_method[method] = {
                "equal_state_mean_ms": equal_state_mean,
                "by_retrieval_stratum_equal_state_mean_ms": by_stratum,
                "warmup_calls_per_state": 5,
                "timed_calls_per_state": 31,
                "per_state_mean_ms": per_state,
            }
            for state in states:
                call_rows.append({
                    "algorithm": algorithm, "method": method,
                    "state_id": state["state_id"], "geometry": state["geometry"],
                    "stratum": state["stratum"],
                    "mean_timed_latency_ms": per_state[str(state["state_id"])],
                })
        base = by_method["FIXED20"]["equal_state_mean_ms"]
        order = by_method["R_ORDER"]["equal_state_mean_ms"]
        by_method["ratio_R_ORDER_over_FIXED20"] = order / base if base > 0 else None
        by_method["ratio_gate_pass"] = bool(base > 0 and order / base <= 6.0)
        per_algorithm[algorithm] = by_method
    result = {
        "source": "M2 EM_FIXED20 trajectories only",
        "same_120_public_states_all_algorithms": True,
        "state_count": len(states),
        "strata_counts": {f"{key[0]}|{key[1]}": len(cost_candidates[key]) for key in sorted(expected_keys)},
        "warmup_calls_per_policy_state": 5,
        "timed_calls_per_policy_state": 31,
        "alternating_call_order": True,
        "timer_includes_history_update_proposals_scoring_selection": True,
        "timer_excludes_actor_setup_and_frame_snapshot_copy": True,
        "EG_full_plan_each_call_cached_moves_reported_separately": True,
        "warmup_call_counts": warmup_counts,
        "warmup_latency_ns_evaluator_only": warmup_ns,
        "algorithms": per_algorithm,
        "all_algorithm_cost_gates_pass": all(per_algorithm[algorithm]["ratio_gate_pass"] for algorithm in ALGORITHMS),
    }
    return result, call_rows


def _run_m2_worker(ledger: ScienceLedger) -> dict[str, Any]:
    from R.experiments.r_order_analysis import (
        family_strength_levels, main_gates, paired_layout_bootstrap_many,
        select_devbestfixed, validate_m2_trajectory_matrix,
    )
    from R.scrp.r_order_training import predict_pair_probability, relation_calibration_tables

    config = _load_config()
    runtime = _runtime_imports()
    gp_rule, gp_identity = _load_gp_rule(runtime)
    models_json = _read_json(OUTPUT_ROOT / "M1_MODELS.json")
    main_model = _model_from_record(models_json["main"])
    no_history_model = _model_from_record(models_json["no_history"])
    tau_record = _read_json(OUTPUT_ROOT / "M1_TAU_CALIBRATION.json")
    gate_record = _read_json(OUTPUT_ROOT / "M1_GATE.json")
    if gate_record.get("status") != "PASS":
        raise ROrderStop("M2 requires a passing M1 scientific gate")
    main_tau = {str(key): float(value) for key, value in tau_record["main_tau_by_algorithm"].items()}
    no_history_tau = {str(key): float(value) for key, value in tau_record["no_history_tau_by_algorithm"].items()}
    if set(main_tau) != set(ALGORITHMS) or set(no_history_tau) != set(ALGORITHMS):
        raise ROrderStop("M1 does not bind one frozen tau per registered algorithm")
    models_main = {algorithm: main_model for algorithm in ALGORITHMS}
    models_nh = {algorithm: no_history_model for algorithm in ALGORITHMS}
    conditions, aliases = _condition_specs(config)
    methods = ("FIXED0", "FIXED2", "FIXED5", "FIXED10", "FIXED20", "OLD_FROZEN_R", "R_ORDER", "NO_HISTORY", "NO_EXTRA")
    episodes, manifest = _load_split("select", runtime)
    fingerprint_geometry = {str(row["label_invariant_fingerprint"]): str(row["geometry_id"]) for row in manifest}
    condition_ordinal = {condition["condition_id"]: index for index, condition in enumerate(conditions)}
    cost_pool: dict[tuple[str, str], list[dict[str, Any]]] = {}
    trajectory_rows: list[dict[str, Any]] = []
    relation_calibration_labels: list[Any] = []
    forecast_hashes: dict[tuple[str, str], str] = {}
    decision_stream = _JsonlCheckpointStream(
        OUTPUT_ROOT / "M2_SELECT_PUBLIC_DECISIONS_CHECKPOINT.jsonl.gz"
    ).open()
    for episode, layout_row in zip(episodes, manifest, strict=True):
        fingerprint = str(layout_row["label_invariant_fingerprint"])
        ordinal = int(layout_row["layout_ordinal"])
        for condition in conditions:
            ledger.check()
            assigned_behavior = ALGORITHMS[(ordinal + condition_ordinal[condition["condition_id"]]) % 4]
            for algorithm in ALGORITHMS:
                for method in methods:
                    callback_functions: list[Any] = []
                    if algorithm == "EM" and method == "FIXED20":
                        def cost_callback(**state: Any) -> None:
                            condition_for_stratum = dict(condition, num_containers=int(episode.num_containers))
                            snapshot = _cost_state_snapshot(
                                fingerprint=fingerprint, condition=condition_for_stratum,
                                decision_index=int(state["decision_index"]), geometry=str(episode.parameter_group),
                                frame=state["frame"], pre_tracker_records=state["pre_tracker_records"],
                                pre_previous_frame=state["pre_previous_frame"],
                                pre_previous_action=state["pre_previous_action"],
                            )
                            _insert_cost_state(cost_pool, snapshot)
                        callback_functions.append(cost_callback)
                    capture_state = None
                    if algorithm == assigned_behavior and method == "FIXED20":
                        capture_state = _new_relation_capture_state()

                        def relation_callback(**state: Any) -> None:
                            _observe_relation_capture(
                                capture_state,
                                episode=episode,
                                fingerprint=fingerprint,
                                condition=condition,
                                algorithm=algorithm,
                                runtime=runtime,
                                **state,
                            )

                        callback_functions.append(relation_callback)
                    callback = None
                    if callback_functions:
                        def callback(**state: Any) -> None:
                            for function in callback_functions:
                                function(**state)
                    policy_model = main_model if method in ("R_ORDER", "NO_EXTRA") else (no_history_model if method == "NO_HISTORY" else None)
                    tau = main_tau[algorithm] if method in ("R_ORDER", "NO_EXTRA") else (no_history_tau[algorithm] if method == "NO_HISTORY" else 0.0)
                    trajectory_started = time.perf_counter()
                    decision_rows: list[dict[str, Any]] = []
                    row = _run_trajectory(
                        episode, fingerprint, condition, algorithm, method, runtime, gp_rule,
                        rates=policy_model, tau=tau,
                        on_decision=callback, decision_sink=decision_rows.append, check_budget=ledger.check,
                    )
                    row["trajectory_elapsed_seconds"] = max(0.0, time.perf_counter() - trajectory_started)
                    row["decision_trace_count"] = len(decision_rows)
                    cell_key = (fingerprint, str(condition["condition_id"]))
                    previous_hash = forecast_hashes.setdefault(cell_key, str(row["forecast_sha256_evaluator_only"]))
                    if previous_hash != row["forecast_sha256_evaluator_only"]:
                        raise ROrderStop("algorithms/methods received different immutable F for one layout-condition")
                    trajectory_rows.append(row)
                    _append_jsonl_checkpoint(OUTPUT_ROOT / "M2_SELECT_TRAJECTORY_CHECKPOINT.jsonl", row)
                    for decision_row in decision_rows:
                        decision_stream.append({
                            "trajectory_id": [algorithm, method, fingerprint, str(condition["condition_id"])],
                            **decision_row,
                        })
                    decision_stream.sync()
                    if capture_state is not None:
                        capture_rows = _finish_relation_capture_state(capture_state)
                        for capture_row in capture_rows:
                            _append_jsonl_checkpoint(
                                OUTPUT_ROOT / "private/M2_SELECT_RELATION_CALIBRATION_CAPTURE_CHECKPOINT.jsonl",
                                {
                                    "trajectory_id": [algorithm, method, fingerprint, str(condition["condition_id"])],
                                    "preallocated_behavior_algorithm": assigned_behavior,
                                    **capture_row,
                                },
                            )
                        usable_captures = [
                            row for row in capture_rows
                            if row.get("status") == "MEASURED_CAPTURE" and not row.get("duplicate_physical_state")
                        ]
                        labels = _pair_rows_for_captures(usable_captures, {fingerprint: episode}, runtime)
                        for label in labels:
                            _append_jsonl_checkpoint(
                                OUTPUT_ROOT / "private/M2_SELECT_RELATION_CALIBRATION_LABEL_CHECKPOINT.jsonl",
                                asdict(label),
                            )
                        relation_calibration_labels.extend(labels)
            if len(trajectory_rows) and len(trajectory_rows) % 1800 == 0:
                print(f"M2 complete trajectories {len(trajectory_rows)}/{int(config['M2']['complete_trajectories'])}", flush=True)
    decision_stream.close()
    if len(trajectory_rows) != int(config["M2"]["complete_trajectories"]):
        raise ROrderStop("M2 did not produce exactly 43,200 full trajectories")
    if relation_calibration_labels:
        relation_calibration = relation_calibration_tables(
            relation_calibration_labels,
            lambda label: predict_pair_probability(main_model, label),
        )
        relation_calibration_summary = _persist_relation_calibration_artifacts(
            "M2_SELECT", relation_calibration,
            metadata={
                "capture_rule": "preallocated behavior algorithm native FIXED20 trajectory; first legal and nearest public retrieval fractions .50/.85; EG group starts only",
                "capture_trajectory_count": len(episodes) * len(conditions),
                "capture_slots_per_trajectory": 3,
                "model": "frozen M1 main relation model; no fit or trajectory added",
            },
        )
    else:
        relation_calibration_summary = {"status": "UNMEASURED_NO_ELIGIBLE_PAIRS"}
        relation_calibration_summary.update({
            "capture_rule": "preallocated behavior algorithm native FIXED20 trajectory; first legal and nearest public retrieval fractions .50/.85; EG group starts only",
            "capture_trajectory_count": len(episodes) * len(conditions),
            "capture_slots_per_trajectory": 3,
            "model": "frozen M1 main relation model; no fit or trajectory added",
            "diagnostic_only_no_gate_or_ci": True,
        })
        _write_json_new(OUTPUT_ROOT / "M2_SELECT_RELATION_CALIBRATION_SUMMARY.json", relation_calibration_summary)
    matrix_audit = validate_m2_trajectory_matrix(
        trajectory_rows,
        layout_fingerprints=fingerprint_geometry,
        condition_ids=[str(row["condition_id"]) for row in conditions],
        methods=methods,
    )
    gates = main_gates(trajectory_rows)
    devbest = select_devbestfixed(trajectory_rows)
    specs: list[dict[str, Any]] = []
    for algorithm in ALGORITHMS:
        for family in ("DP", "TWS"):
            specs.append({"algorithm": algorithm, "target": "R_ORDER", "baseline": "FIXED20", "family": family, "strength": None,
                          "endpoint_kind": "superiority"})
            for strength in family_strength_levels(family):
                specs.append({"algorithm": algorithm, "target": "R_ORDER", "baseline": "FIXED20", "family": family,
                              "strength": strength, "endpoint_kind": "guard"})
        specs.append({"algorithm": algorithm, "target": "R_ORDER", "baseline": "FIXED20", "family": "CLEAN", "strength": "CLEAN",
                      "endpoint_kind": "guard"})
    ci_rows = paired_layout_bootstrap_many(
        trajectory_rows, specs,
        draws=int(config["statistics"]["M2_bootstrap_replicates"]),
        seed=int(config["statistics"]["M2_bootstrap_seed"]),
    )
    for spec, ci in zip(specs, ci_rows, strict=True):
        spec["bootstrap"] = ci
    point_pass = bool(gates["all_pass"])
    if not point_pass:
        cost_result = None
        cost_rows: list[dict[str, Any]] = []
        cost_pass = False
    else:
        cost_result, cost_rows = _m2_cost_evaluate(
            cost_pool, runtime, gp_rule, models_main, main_tau, ledger
        )
        cost_pass = bool(cost_result["all_algorithm_cost_gates_pass"])
    _write_jsonl_gz(OUTPUT_ROOT / "M2_SELECT_TRAJECTORIES_EVALUATOR.jsonl.gz", trajectory_rows)
    _write_csv_new(
        OUTPUT_ROOT / "M2_SELECT_TRAJECTORIES.csv", trajectory_rows,
        ("algorithm", "method", "geometry", "layout_fingerprint", "condition_id", "regime", "strength",
         "p", "window_size", "repeat", "N", "R_raw", "r_over_n", "decision_count",
         "decision_trace_count", "forecast_sha256_evaluator_only"),
    )
    _write_json_new(OUTPUT_ROOT / "M2_MATRIX_AUDIT.json", {**matrix_audit, "conditions": len(conditions),
                                                             "methods_per_algorithm": len(methods),
                                                             "layout_count": len(episodes), "forecast_condition_count": len(forecast_hashes)})
    _write_json_new(OUTPUT_ROOT / "M2_POINT_GATES.json", gates)
    _write_json_new(OUTPUT_ROOT / "M2_PRIMARY_CI_DEVELOPMENT.json", {
        "draws": int(config["statistics"]["M2_bootstrap_replicates"]),
        "seed": int(config["statistics"]["M2_bootstrap_seed"]),
        "ordinary_two_sided_95_percent_descriptive_only": True,
        "endpoints": specs,
    })
    _write_json_new(OUTPUT_ROOT / "M2_DEVBESTFIXED.json", {
        "horizon_by_algorithm": devbest,
        "selection_rule": "two noise-family equal-weight mean R/N, ties smaller H",
        "primary_comparator_unchanged": "same algorithm native FIXED20",
    })
    if cost_result is not None:
        _write_json_new(OUTPUT_ROOT / "M2_COST.json", cost_result)
        _write_jsonl_gz(OUTPUT_ROOT / "M2_COST_PER_STATE.jsonl.gz", cost_rows)
        _write_csv_new(OUTPUT_ROOT / "M2_COST_PER_STATE.csv", cost_rows,
                       ("algorithm", "method", "state_id", "geometry", "stratum", "mean_timed_latency_ms"))
    status = "M2_COMPLETE" if point_pass and cost_pass else "M2_SCIENTIFIC_GATE_FAIL_STOP"
    return {
        "stage": "M2", "status": status,
        "point_gate_pass": point_pass, "cost_gate_pass": cost_pass,
        "matrix_complete": True, "trajectory_count": len(trajectory_rows),
        "main_gates": gates, "cost": cost_result,
        "devbestfixed_horizon_by_algorithm": devbest,
        "m3_release_allowed": point_pass and cost_pass,
        "gp_rule_identity": gp_identity,
    }


def _m3_formal_specs() -> list[dict[str, Any]]:
    from R.experiments.r_order_analysis import family_strength_levels

    specs: list[dict[str, Any]] = []
    for algorithm in ALGORITHMS:
        for family in ("DP", "TWS"):
            specs.append({"algorithm": algorithm, "target": "R_ORDER", "baseline": "FIXED20",
                          "family": family, "strength": None, "kind": "superiority"})
            for strength in family_strength_levels(family):
                specs.append({"algorithm": algorithm, "target": "R_ORDER", "baseline": "FIXED20",
                              "family": family, "strength": strength, "kind": "guard"})
        specs.append({"algorithm": algorithm, "target": "R_ORDER", "baseline": "FIXED20",
                      "family": "CLEAN", "strength": "CLEAN", "kind": "guard"})
    if len(specs) != 40:
        raise AssertionError("M3 formal endpoint registration must contain exactly 40 rows")
    return specs


def _m3_project_seconds(
    select_rows: Sequence[Mapping[str, Any]], devbest: Mapping[str, int], n_total: int,
    *, conditions_per_layout: int = 15,
) -> dict[str, Any]:
    means: dict[tuple[str, str], float] = {}
    for algorithm in ALGORITHMS:
        for method in ("FIXED20", "R_ORDER", "NO_HISTORY", "FIXED0", "FIXED2", "FIXED5", "FIXED10"):
            values = [float(row["trajectory_elapsed_seconds"]) for row in select_rows
                      if row["algorithm"] == algorithm and row["method"] == method
                      and row.get("trajectory_elapsed_seconds") is not None]
            if values:
                means[(algorithm, method)] = float(np.mean(values))
    method_seconds: dict[str, float] = {}
    unmeasured: list[dict[str, str]] = []
    for algorithm in ALGORITHMS:
        policies = ["FIXED20", "R_ORDER", "NO_HISTORY"]
        horizon = int(devbest[algorithm])
        if horizon != 20:
            policies.append(f"FIXED{horizon}")
        for method in policies:
            value = means.get((algorithm, method))
            if value is None or not math.isfinite(value) or value <= 0:
                unmeasured.append({"algorithm": algorithm, "method": method})
            else:
                method_seconds[f"{algorithm}|{method}"] = value * int(n_total) * conditions_per_layout
    raw = float(sum(method_seconds.values())) if not unmeasured else None
    projected = None if raw is None else raw * 1.2
    return {"measured_mean_seconds_per_trajectory": {f"{a}|{m}": value for (a, m), value in means.items()},
            "formal_policy_seconds": method_seconds, "raw_projected_seconds": raw,
            "projected_seconds_with_20_percent": projected,
            "projected_hours_with_20_percent": None if projected is None else projected / 3600.0,
            "unmeasured_policy_throughput": unmeasured,
            "layout_count_total": int(n_total), "conditions_per_layout": int(conditions_per_layout),
            "devbestfixed_aliases_native20": {algorithm: int(devbest[algorithm]) == 20 for algorithm in ALGORITHMS}}


def _m3_decision_gate(
    rows: Sequence[Mapping[str, Any]], specs: Sequence[Mapping[str, Any]], ci_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    from R.experiments.r_order_analysis import main_gates

    point = main_gates(rows)
    by_endpoint = {
        (str(spec["algorithm"]), str(spec["family"]), None if spec.get("strength") is None else str(spec["strength"])): ci
        for spec, ci in zip(specs, ci_rows, strict=True)
    }
    primary_rows = []
    guard_rows = []
    for item in point["superiority"]:
        key = (str(item["algorithm"]), str(item["family"]), None)
        ci = by_endpoint.get(key, {"status": "UNMEASURED", "bonferroni_lcb": None, "ci95": None})
        primary_rows.append({**item, "bootstrap": ci})
    for item in point["guards"]:
        key = (str(item["algorithm"]), str(item["family"]), str(item["strength"]))
        ci = by_endpoint.get(key, {"status": "UNMEASURED", "bonferroni_lcb": None, "ci95": None})
        guard_rows.append({**item, "bootstrap": ci})
    primary_lcb_pass = all(
        row["bootstrap"].get("status") == "MEASURED"
        and row["bootstrap"].get("bonferroni_lcb") is not None
        and float(row["bootstrap"]["bonferroni_lcb"]) > 0.0
        for row in primary_rows
    )
    guards_lcb_pass = all(
        row["bootstrap"].get("status") == "MEASURED"
        and row["bootstrap"].get("bonferroni_lcb") is not None
        and float(row["bootstrap"]["bonferroni_lcb"]) >= -1.0
        for row in guard_rows
    )
    return {"primary_points_pass": point["superiority_pass"], "guard_points_pass": point["guards_pass"],
            "primary_bonferroni_lcb_pass": primary_lcb_pass, "guard_bonferroni_lcb_pass": guards_lcb_pass,
            "all_formal_gates_pass": bool(point["superiority_pass"] and point["guards_pass"]
                                           and primary_lcb_pass and guards_lcb_pass),
            "primary_comparator": "same algorithm native FIXED20",
            "superiority": primary_rows, "guards": guard_rows}


def _run_m3_worker(ledger: ScienceLedger) -> dict[str, Any]:
    from statistics import NormalDist
    from R.experiments.r_order_analysis import (
        paired_layout_bootstrap_many, formal_power, validate_m2_trajectory_matrix,
    )

    config = _load_config()
    select_rows = _read_jsonl_gz(OUTPUT_ROOT / "M2_SELECT_TRAJECTORIES_EVALUATOR.jsonl.gz")
    devbest_record = _read_json(OUTPUT_ROOT / "M2_DEVBESTFIXED.json")
    devbest = {str(key): int(value) for key, value in devbest_record["horizon_by_algorithm"].items()}
    z_alpha = NormalDist().inv_cdf(1.0 - 0.05 / 40.0)
    power = formal_power(select_rows, z_alpha=z_alpha)
    _write_json_new(OUTPUT_ROOT / "M3_POWER.json", power)
    if power.get("n_total") is None or power.get("stop_reason") is not None:
        return {"stage": "M3", "status": "M3_INSUFFICIENT_POWER_STOP", "power_pass": False,
                "n_total": power.get("n_total"), "stop_reason": power.get("stop_reason"),
                "confirm_outcomes_opened": False}

    projected = _m3_project_seconds(select_rows, devbest, int(power["n_total"]))
    _write_json_new(OUTPUT_ROOT / "M3_RUNTIME_PROJECTION.json", projected)
    if (projected["projected_seconds_with_20_percent"] is None
            or float(projected["projected_seconds_with_20_percent"]) > PHASE_CAP_SECONDS["M3"]):
        return {"stage": "M3", "status": "M3_RUNTIME_CAP_STOP", "power_pass": True,
                "n_total": int(power["n_total"]), "runtime_projection": projected,
                "confirm_outcomes_opened": False}

    # Confirmation candidates are read only after both registered preconditions pass.
    runtime = _runtime_imports()
    gp_rule, gp_identity = _load_gp_rule(runtime)
    conditions, _aliases = _condition_specs(config)
    candidates, candidate_manifest = _load_split("confirm_candidates", runtime)
    per_geometry = int(power["n_per_geometry"])
    selected_layouts: list[tuple[Any, dict[str, Any]]] = []
    for geometry in ("G1", "G2"):
        matching = [(index, episode, row) for index, (episode, row) in enumerate(zip(candidates, candidate_manifest, strict=True))
                    if str(row["geometry_id"]) == geometry]
        matching.sort(key=lambda item: int(item[2]["layout_ordinal"]))
        if len(matching) < per_geometry:
            raise ROrderStop(f"M3 confirm candidate prefix is too short for {geometry}")
        for index, episode, row in matching[:per_geometry]:
            selected_layouts.append((episode, row))
    if len(selected_layouts) != int(power["n_total"]):
        raise ROrderStop("M3 confirmation did not select exactly the powered total n")

    m1_models = _read_json(OUTPUT_ROOT / "M1_MODELS.json")
    main_model = _model_from_record(m1_models["main"])
    no_history_model = _model_from_record(m1_models["no_history"])
    tau_record = _read_json(OUTPUT_ROOT / "M1_TAU_CALIBRATION.json")
    main_tau = {str(key): float(value) for key, value in tau_record["main_tau_by_algorithm"].items()}
    no_history_tau = {str(key): float(value) for key, value in tau_record["no_history_tau_by_algorithm"].items()}
    trajectory_rows: list[dict[str, Any]] = []
    alias_rows: list[dict[str, Any]] = []
    decision_stream = _JsonlCheckpointStream(
        OUTPUT_ROOT / "M3_CONFIRM_PUBLIC_DECISIONS_CHECKPOINT.jsonl.gz"
    ).open()
    for episode, layout_row in selected_layouts:
        fingerprint = str(layout_row["label_invariant_fingerprint"])
        for condition in conditions:
            for algorithm in ALGORITHMS:
                ledger.check()
                per_policy: dict[str, dict[str, Any]] = {}
                for method, rates, tau in (
                    ("FIXED20", None, 0.0),
                    ("R_ORDER", main_model, main_tau[algorithm]),
                    ("NO_HISTORY", no_history_model, no_history_tau[algorithm]),
                ):
                    started = time.perf_counter()
                    decision_rows: list[dict[str, Any]] = []
                    result = _run_trajectory(episode, fingerprint, condition, algorithm, method, runtime, gp_rule,
                                             rates=rates, tau=tau, decision_sink=decision_rows.append,
                                             check_budget=ledger.check)
                    result["trajectory_elapsed_seconds"] = max(0.0, time.perf_counter() - started)
                    result["trajectory_executed"] = True
                    result["horizon_used"] = 20 if method == "FIXED20" else None
                    result["decision_trace_count"] = len(decision_rows)
                    trajectory_rows.append(result)
                    per_policy[method] = result
                    _append_jsonl_checkpoint(OUTPUT_ROOT / "M3_CONFIRM_TRAJECTORY_CHECKPOINT.jsonl", result)
                    for decision_row in decision_rows:
                        decision_stream.append({
                            "trajectory_id": [algorithm, method, fingerprint, str(condition["condition_id"])],
                            **decision_row,
                        })
                    decision_stream.sync()
                best_h = int(devbest[algorithm])
                if best_h == 20:
                    alias = {**per_policy["FIXED20"], "method": "DEV_BEST_FIXED",
                             "trajectory_executed": False, "alias_of": "FIXED20", "horizon_used": 20}
                    trajectory_rows.append(alias)
                    alias_rows.append(alias)
                    _append_jsonl_checkpoint(OUTPUT_ROOT / "M3_FORMAL_METHOD_ALIASES.jsonl", alias)
                else:
                    method = f"FIXED{best_h}"
                    started = time.perf_counter()
                    decision_rows = []
                    result = _run_trajectory(episode, fingerprint, condition, algorithm, method, runtime, gp_rule,
                                             rates=None, tau=0.0, decision_sink=decision_rows.append,
                                             check_budget=ledger.check)
                    result["trajectory_elapsed_seconds"] = max(0.0, time.perf_counter() - started)
                    result["trajectory_executed"] = True
                    result["horizon_used"] = best_h
                    result["method"] = "DEV_BEST_FIXED"
                    result["decision_trace_count"] = len(decision_rows)
                    trajectory_rows.append(result)
                    _append_jsonl_checkpoint(OUTPUT_ROOT / "M3_CONFIRM_TRAJECTORY_CHECKPOINT.jsonl", result)
                    for decision_row in decision_rows:
                        decision_stream.append({
                            "trajectory_id": [algorithm, "DEV_BEST_FIXED", fingerprint, str(condition["condition_id"])],
                            **decision_row,
                        })
                    decision_stream.sync()

    decision_stream.close()

    layout_geometries = {
        str(row["label_invariant_fingerprint"]): str(row["geometry_id"])
        for _episode, row in selected_layouts
    }
    condition_ids = [str(row["condition_id"]) for row in conditions]
    matrix = validate_m2_trajectory_matrix(
        trajectory_rows, layout_fingerprints=layout_geometries, condition_ids=condition_ids,
        methods=("FIXED20", "R_ORDER", "NO_HISTORY", "DEV_BEST_FIXED"),
    )
    specs = _m3_formal_specs()
    ci_rows = paired_layout_bootstrap_many(
        trajectory_rows, specs,
        draws=int(config["M3"]["bootstrap"]["replicates"]),
        seed=int(config["M3"]["bootstrap"]["seed"]), bonferroni_tests=40,
    )
    gate = _m3_decision_gate(trajectory_rows, specs, ci_rows)
    gate.update({"matrix": matrix, "confirm_layout_count": len(selected_layouts),
                 "confirm_trajectory_count": sum(bool(row.get("trajectory_executed")) for row in trajectory_rows),
                 "formal_bootstrap_draws": int(config["M3"]["bootstrap"]["replicates"]),
                 "formal_bootstrap_seed": int(config["M3"]["bootstrap"]["seed"]),
                 "gp_rule_identity": gp_identity, "confirm_outcomes_opened": True})
    _write_json_new(OUTPUT_ROOT / "M3_GATE.json", gate)
    _write_jsonl_gz(OUTPUT_ROOT / "M3_CONFIRM_TRAJECTORIES.jsonl.gz", trajectory_rows)
    _write_csv_new(OUTPUT_ROOT / "M3_CONFIRM_TRAJECTORIES.csv", trajectory_rows,
                   ("algorithm", "method", "geometry", "layout_fingerprint", "condition_id", "regime", "strength",
                    "p", "window_size", "repeat", "N", "R_raw", "r_over_n", "decision_count",
                    "decision_trace_count", "trajectory_elapsed_seconds", "trajectory_executed", "alias_of", "horizon_used"))
    status = "M3_COMPLETE" if gate["all_formal_gates_pass"] else "M3_SCIENTIFIC_GATE_FAIL_STOP"
    return {"stage": "M3", "status": status, "scientific_gate_status": "PASS" if status == "M3_COMPLETE" else "FAIL_STOP",
            "power_pass": True, "runtime_projection_pass": True, "formal_gate_pass": gate["all_formal_gates_pass"],
            "n_total": int(power["n_total"]), "confirm_trajectory_count": gate["confirm_trajectory_count"],
            "m4_release_allowed": True}


def _m4_endpoint_plot_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    m3_path = OUTPUT_ROOT / "M3_GATE.json"
    if m3_path.is_file():
        gate = _read_json(m3_path)
        for kind, items in (("superiority", gate.get("superiority", [])), ("guard", gate.get("guards", []))):
            for item in items:
                rows.append({"kind": kind, "algorithm": str(item["algorithm"]),
                             "family": str(item["family"]), "strength": item.get("strength"),
                             "gain_pct": item.get("gain_pct"),
                             "ci95": item.get("bootstrap", {}).get("ci95"),
                             "lcb": item.get("bootstrap", {}).get("bonferroni_lcb"),
                             "ci_status": item.get("bootstrap", {}).get("status", "UNMEASURED")})
        return rows
    m2_path = OUTPUT_ROOT / "M2_POINT_GATES.json"
    m2_ci_path = OUTPUT_ROOT / "M2_PRIMARY_CI_DEVELOPMENT.json"
    ci_map: dict[tuple[str, str, str | None], Mapping[str, Any]] = {}
    if m2_ci_path.is_file():
        for item in _read_json(m2_ci_path).get("endpoints", []):
            key = (str(item["algorithm"]), str(item["family"]),
                   None if item.get("strength") is None else str(item["strength"]))
            ci_map[key] = item.get("bootstrap", {})
    if m2_path.is_file():
        gate = _read_json(m2_path)
        for kind, name in (("superiority", "superiority"), ("guard", "guards")):
            for item in gate.get(name, []):
                key = (str(item["algorithm"]), str(item["family"]),
                       None if kind == "superiority" else str(item.get("strength")))
                ci = ci_map.get(key, {})
                rows.append({"kind": kind, "algorithm": key[0], "family": key[1], "strength": key[2],
                             "gain_pct": item.get("gain_pct"), "ci95": ci.get("ci95"), "lcb": None,
                             "ci_status": ci.get("status", "UNMEASURED")})
        return rows
    from R.experiments.r_order_analysis import family_strength_levels
    for algorithm in ALGORITHMS:
        for family in ("DP", "TWS"):
            rows.append({"kind": "superiority", "algorithm": algorithm, "family": family,
                         "strength": None, "gain_pct": None, "ci95": None, "lcb": None,
                         "ci_status": "UNMEASURED"})
            for strength in family_strength_levels(family):
                rows.append({"kind": "guard", "algorithm": algorithm, "family": family,
                             "strength": strength, "gain_pct": None, "ci95": None, "lcb": None,
                             "ci_status": "UNMEASURED"})
        rows.append({"kind": "guard", "algorithm": algorithm, "family": "CLEAN", "strength": "CLEAN",
                     "gain_pct": None, "ci95": None, "lcb": None, "ci_status": "UNMEASURED"})
    return rows


def _m4_m1_branch_rows(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    from R.experiments.r_order_analysis import m1_branch_bootstrap_many

    branch_path = OUTPUT_ROOT / "private/M1_BRANCH_ROWS.jsonl.gz"
    specs = [
        {"algorithm": algorithm, "family": family, "control": control, "tau": tau}
        for algorithm in ALGORITHMS for family in ("DP", "TWS")
        for control in ("main", "no_history", "no_extra") for tau in (0.0, 0.25)
    ]
    if not branch_path.is_file():
        return [
            {**spec, "gain_pct": None, "baseline_R_over_N": None, "target_R_over_N": None,
             "ci95": None, "bootstrap_status": "UNMEASURED_NO_M1_BRANCH_DATA",
             "draws": int(config["statistics"]["M1_bootstrap_replicates"]),
             "seed": int(config["statistics"]["M1_bootstrap_seed"]),
             "layout_count_by_geometry": {"G1": 0, "G2": 0}}
            for spec in specs
        ]
    branches = _read_jsonl_gz(branch_path)
    return m1_branch_bootstrap_many(
        branches, specs,
        draws=int(config["statistics"]["M1_bootstrap_replicates"]),
        seed=int(config["statistics"]["M1_bootstrap_seed"]),
    )


def _m4_render_relation_calibration_figures(
    figures_dir: Path, *, pyplot: Any,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    dimensions = {
        "noise_family": ("DP", "TWS", "CLEAN"),
        "rank_distance": ("1-2", "3-5", "6-10", "11-19", "known-tail"),
        "censor_history": ("False", "True"),
        "retrieval_phase": ("early", "mid", "late"),
    }
    plot_receipts: list[dict[str, Any]] = []
    statuses: dict[str, dict[str, Any]] = {}
    plotted_groups: list[dict[str, Any]] = []
    plotted_bins: list[dict[str, Any]] = []
    for split, summary_path in (
        ("M1_CAL", OUTPUT_ROOT / "M1_CAL_RELATION_CALIBRATION_SUMMARY.json"),
        ("M2_SELECT", OUTPUT_ROOT / "M2_SELECT_RELATION_CALIBRATION_SUMMARY.json"),
    ):
        summary = _read_json(summary_path) if summary_path.is_file() else {"status": "UNMEASURED_NOT_RUN"}
        statuses[split] = summary
        group_path = OUTPUT_ROOT / f"{split}_RELATION_CALIBRATION_GROUPS.csv"
        bin_path = OUTPUT_ROOT / f"{split}_RELATION_CALIBRATION_BINS.csv"
        measured = summary.get("status") == "MEASURED" and group_path.is_file() and bin_path.is_file()
        group_rows: list[dict[str, Any]] = []
        bin_rows: list[dict[str, Any]] = []
        if measured:
            with group_path.open("r", encoding="utf-8-sig", newline="") as stream:
                group_rows = [dict(row) for row in csv.DictReader(stream)]
            with bin_path.open("r", encoding="utf-8-sig", newline="") as stream:
                bin_rows = [dict(row) for row in csv.DictReader(stream)]
            for row in group_rows:
                plotted_groups.append({"split": split, "status": "MEASURED", **row})
            for row in bin_rows:
                plotted_bins.append({"split": split, "status": "MEASURED", **row})
        else:
            for dimension, categories in dimensions.items():
                for category in categories:
                    plotted_groups.append({
                        "split": split, "status": str(summary.get("status", "UNMEASURED_NOT_RUN")),
                        "dimension": dimension, "category": category,
                        "row_count": 0, "layout_count": 0, "capture_count": 0,
                        "weight_mass": 0.0, "weighted_brier": None,
                        "mean_prediction": None, "observed_rate": None,
                        "calibration_gap": None,
                    })
                    for bin_index in range(10):
                        plotted_bins.append({
                            "split": split, "status": str(summary.get("status", "UNMEASURED_NOT_RUN")),
                            "dimension": dimension, "category": category,
                            "probability_bin": bin_index, "bin_lower": bin_index / 10.0,
                            "bin_upper": (bin_index + 1) / 10.0,
                            "row_count": 0, "layout_count": 0, "capture_count": 0,
                            "weight_mass": 0.0, "weighted_brier": None,
                            "mean_prediction": None, "observed_rate": None,
                            "calibration_gap": None,
                        })
            for bin_index in range(10):
                plotted_bins.append({
                    "split": split, "status": str(summary.get("status", "UNMEASURED_NOT_RUN")),
                    "dimension": "overall", "category": "all",
                    "probability_bin": bin_index, "bin_lower": bin_index / 10.0,
                    "bin_upper": (bin_index + 1) / 10.0,
                    "row_count": 0, "layout_count": 0, "capture_count": 0,
                    "weight_mass": 0.0, "weighted_brier": None,
                    "mean_prediction": None, "observed_rate": None,
                    "calibration_gap": None,
                })
        if not measured:
            continue

        fig, axes = pyplot.subplots(2, 2, figsize=(12, 8))
        for axis, (dimension, categories) in zip(axes.flat, dimensions.items(), strict=True):
            subset = [row for row in group_rows if row.get("dimension") == dimension]
            by_category = {str(row.get("category")): row for row in subset}
            positions = np.arange(len(categories), dtype=np.float64)
            predicted = [
                np.nan if by_category.get(category, {}).get("mean_prediction") in (None, "")
                else float(by_category[category]["mean_prediction"])
                for category in categories
            ]
            observed = [
                np.nan if by_category.get(category, {}).get("observed_rate") in (None, "")
                else float(by_category[category]["observed_rate"])
                for category in categories
            ]
            axis.plot(positions, predicted, marker="o", label="mean predicted")
            axis.plot(positions, observed, marker="s", label="truth rate")
            axis.set_xticks(positions, categories, rotation=25, ha="right")
            axis.set_ylim(0.0, 1.0)
            axis.set_title(dimension)
            axis.grid(axis="y", alpha=0.2)
            axis.legend(fontsize=8)
            for index, category in enumerate(categories):
                row = by_category.get(category, {})
                if row.get("weighted_brier") in (None, ""):
                    axis.text(index, 0.04, "未测", rotation=90, ha="center", fontsize=7)
        overall = summary.get("overall", {})
        sample_note = (
            f"layouts={overall.get('layout_count', 0)}, captures={summary.get('capture_count', 0)}, "
            f"pairs={summary.get('pair_count', 0)}"
        )
        fig.suptitle(f"{split} relationship calibration groups ({sample_note}); descriptive only, no CI")
        fig.tight_layout()
        group_png, group_pdf = figures_dir / f"{split.lower()}_relation_groups.png", figures_dir / f"{split.lower()}_relation_groups.pdf"
        fig.savefig(group_png, dpi=180)
        fig.savefig(group_pdf)
        pyplot.close(fig)
        plot_receipts.append({
            "name": f"{split.lower()}_relation_groups",
            "png": group_png.relative_to(OUTPUT_ROOT).as_posix(),
            "pdf": group_pdf.relative_to(OUTPUT_ROOT).as_posix(),
            "group_rows": len(group_rows), "confidence_intervals": "NOT_COMPUTED_DIAGNOSTIC",
        })

        pooled_bins = [row for row in bin_rows if row.get("dimension") == "overall"
                       and str(row.get("category")) == "all" and row.get("mean_prediction") not in (None, "")]
        fig, axis = pyplot.subplots(figsize=(7, 6))
        axis.plot([0.0, 1.0], [0.0, 1.0], color="#374151", linestyle="--", label="perfect calibration")
        if pooled_bins:
            axis.plot([float(row["mean_prediction"]) for row in pooled_bins],
                      [float(row["observed_rate"]) for row in pooled_bins],
                      marker="o", label="weighted ten-bin relation")
            for row in pooled_bins:
                axis.annotate(str(row["row_count"]),
                              (float(row["mean_prediction"]), float(row["observed_rate"])),
                              xytext=(3, 3), textcoords="offset points", fontsize=7)
        else:
            axis.text(0.5, 0.5, "未测", ha="center", va="center", transform=axis.transAxes)
        axis.set_xlim(0.0, 1.0)
        axis.set_ylim(0.0, 1.0)
        axis.set_xlabel("weighted mean predicted pair probability")
        axis.set_ylabel("weighted observed truth rate")
        overall = summary.get("overall", {})
        sample_note = (
            f"layouts={overall.get('layout_count', 0)}, captures={summary.get('capture_count', 0)}, "
            f"pairs={summary.get('pair_count', 0)}; point labels are pair counts"
        )
        axis.set_title(f"{split} ten equal-width probability bins ({sample_note}); no CI")
        axis.grid(alpha=0.2)
        axis.legend(loc="best")
        fig.tight_layout()
        bins_png, bins_pdf = figures_dir / f"{split.lower()}_relation_bins.png", figures_dir / f"{split.lower()}_relation_bins.pdf"
        fig.savefig(bins_png, dpi=180)
        fig.savefig(bins_pdf)
        pyplot.close(fig)
        plot_receipts.append({
            "name": f"{split.lower()}_relation_bins",
            "png": bins_png.relative_to(OUTPUT_ROOT).as_posix(),
            "pdf": bins_pdf.relative_to(OUTPUT_ROOT).as_posix(),
            "bin_rows": len(pooled_bins), "confidence_intervals": "NOT_COMPUTED_DIAGNOSTIC",
        })

    group_fields = ("split", "status", *_RELATION_GROUP_CSV_FIELDS)
    bin_fields = ("split", "status", *_RELATION_BIN_CSV_FIELDS)
    _write_csv_new(OUTPUT_ROOT / "M4_PLOTTED_RELATION_GROUPS.csv", plotted_groups, group_fields)
    _write_csv_new(OUTPUT_ROOT / "M4_PLOTTED_RELATION_BINS.csv", plotted_bins, bin_fields)
    return plot_receipts, statuses, plotted_groups, plotted_bins


def _m4_render_figures(
    config: Mapping[str, Any], *, branch_points: Sequence[Mapping[str, Any]] | None = None,
    brier_metrics: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    figures_dir = OUTPUT_ROOT / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    endpoint_rows = _m4_endpoint_plot_rows()
    labels = [f"{row['algorithm']} {row['family']}" +
              ("" if row["strength"] is None else f" {row['strength']}") for row in endpoint_rows]
    values = [np.nan if row["gain_pct"] is None else float(row["gain_pct"]) for row in endpoint_rows]
    fig, axis = plt.subplots(figsize=(max(14, len(labels) * 0.38), 7))
    x = np.arange(len(labels))
    colors = ["#2563eb" if row["kind"] == "superiority" else "#ea580c" for row in endpoint_rows]
    bars = axis.bar(x, np.nan_to_num(np.asarray(values, dtype=np.float64), nan=0.0), color=colors, width=0.82)
    for index, (bar, row, value) in enumerate(zip(bars, endpoint_rows, values, strict=True)):
        ci = row.get("ci95")
        if ci is not None and len(ci) == 2 and all(math.isfinite(float(item)) for item in ci) and math.isfinite(value):
            low, high = float(ci[0]), float(ci[1])
            axis.vlines(index, low, high, color="#111827", linewidth=0.8)
            axis.plot(index, low, marker="_", color="#111827", markersize=4)
            axis.plot(index, high, marker="_", color="#111827", markersize=4)
        if not math.isfinite(value):
            axis.text(index, 0.02, "未测", rotation=90, ha="center", va="bottom", fontsize=7)
    axis.axhline(0.5, color="#1d4ed8", linestyle="--", linewidth=1, label="primary point gate 0.5%")
    axis.axhline(-1.0, color="#c2410c", linestyle=":", linewidth=1, label="guard point gate -1%")
    axis.set_xticks(x, labels, rotation=90, fontsize=7)
    axis.set_ylabel("相对 native H20 的 R/N 降幅 (%)，越高越好")
    axis.set_title("PLAN-049 注册端点；CI 仅在对应阶段实测时绘出")
    axis.grid(axis="y", alpha=0.2)
    axis.legend(loc="best")
    fig.tight_layout()
    endpoint_png, endpoint_pdf = figures_dir / "registered_endpoints.png", figures_dir / "registered_endpoints.pdf"
    fig.savefig(endpoint_png, dpi=180)
    fig.savefig(endpoint_pdf)
    plt.close(fig)
    plot_receipts = [{"name": "registered_endpoints", "png": endpoint_png.relative_to(OUTPUT_ROOT).as_posix(),
                      "pdf": endpoint_pdf.relative_to(OUTPUT_ROOT).as_posix(),
                      "endpoint_rows": len(endpoint_rows), "missing_ci_count": sum(row["ci95"] is None for row in endpoint_rows)}]
    _write_csv_new(OUTPUT_ROOT / "M4_PLOTTED_REGISTERED_ENDPOINTS.csv", endpoint_rows,
                   ("kind", "algorithm", "family", "strength", "gain_pct", "ci95", "lcb", "ci_status"))

    if branch_points is None:
        branch_points = _m4_m1_branch_rows(config)
    _write_csv_new(OUTPUT_ROOT / "M4_PLOTTED_M1_BRANCHES.csv", branch_points,
                   ("algorithm", "family", "control", "tau", "gain_pct", "baseline_R_over_N",
                    "target_R_over_N", "ci95", "bootstrap_status", "layout_count_by_geometry"))
    if branch_points:
        branch_labels = [f"{row['algorithm']} {row['family']} {row['control']} τ={row['tau']}" for row in branch_points]
        branch_values = [np.nan if row.get("gain_pct") is None else float(row["gain_pct"]) for row in branch_points]
        control_colors = {"main": "#2563eb", "no_history": "#ea580c", "no_extra": "#16a34a"}
        fig, axis = plt.subplots(figsize=(max(18, len(branch_labels) * 0.42), 8))
        bx = np.arange(len(branch_labels))
        branch_bars = axis.bar(
            bx, np.nan_to_num(np.asarray(branch_values, dtype=np.float64), nan=0.0),
            color=[control_colors[str(row["control"])] for row in branch_points], width=0.8,
        )
        for index, (bar, row, value) in enumerate(zip(branch_bars, branch_points, branch_values, strict=True)):
            ci = row.get("ci95")
            if ci is not None and math.isfinite(value):
                low, high = float(ci[0]), float(ci[1])
                axis.vlines(index, low, high, color="#111827", linewidth=0.8)
                axis.plot(index, low, marker="_", color="#111827", markersize=4)
                axis.plot(index, high, marker="_", color="#111827", markersize=4)
            if not math.isfinite(value):
                axis.text(index, 0.02, "未测", rotation=90, ha="center", va="bottom", fontsize=7)
        axis.axhline(0.0, color="#111827", linewidth=1)
        axis.set_xticks(bx, branch_labels, rotation=90, fontsize=7)
        axis.set_ylabel("分层池化 R/N 相对降幅 (%)，正值表示下降")
        axis.set_title("M1 development branches；10000次布局 bootstrap 仅作描述")
        axis.grid(axis="y", alpha=0.2)
        fig.tight_layout()
        branch_png, branch_pdf = figures_dir / "m1_all_branch_gains.png", figures_dir / "m1_all_branch_gains.pdf"
        fig.savefig(branch_png, dpi=180)
        fig.savefig(branch_pdf)
        plt.close(fig)
        plot_receipts.append({"name": "m1_all_branch_gains", "png": branch_png.relative_to(OUTPUT_ROOT).as_posix(),
                              "pdf": branch_pdf.relative_to(OUTPUT_ROOT).as_posix(), "endpoint_rows": len(branch_points),
                              "missing_ci_count": sum(row.get("ci95") is None for row in branch_points),
                              "bootstrap_draws": int(config["statistics"]["M1_bootstrap_replicates"])})

    brier_rows: list[dict[str, Any]] = []
    metrics_path = OUTPUT_ROOT / "M1_CAL_PAIR_METRICS.json"
    metrics = (dict(brier_metrics) if brier_metrics is not None
               else (_read_json(metrics_path) if metrics_path.is_file() else {}))
    brier_rows = [
        {"model": name, "weighted_brier": metrics.get(field),
         "status": "MEASURED" if metrics.get(field) is not None else "UNMEASURED"}
        for name, field in (("main", "main_weighted_brier"),
                            ("no_history", "no_history_weighted_brier"),
                            ("q_only", "q_only_weighted_brier"))
    ]
    _write_csv_new(OUTPUT_ROOT / "M4_PLOTTED_M1_BRIER.csv", brier_rows,
                   ("model", "weighted_brier", "status"))
    if brier_rows:
        fig, axis = plt.subplots(figsize=(7, 5))
        labels = [str(row["model"]) for row in brier_rows]
        values = [np.nan if row["weighted_brier"] is None else float(row["weighted_brier"]) for row in brier_rows]
        axis.bar(np.arange(len(labels)), np.nan_to_num(np.asarray(values), nan=0.0),
                 color=["#2563eb", "#ea580c", "#64748b"])
        for index, value in enumerate(values):
            if not math.isfinite(value):
                axis.text(index, 0.01, "未测", rotation=90, ha="center", va="bottom")
        axis.set_xticks(np.arange(len(labels)), labels)
        axis.set_ylabel("cal weighted Brier，越低越好")
        axis.set_title("M1 calibration diagnostics")
        axis.grid(axis="y", alpha=0.2)
        fig.tight_layout()
        brier_png, brier_pdf = figures_dir / "m1_cal_brier.png", figures_dir / "m1_cal_brier.pdf"
        fig.savefig(brier_png, dpi=180)
        fig.savefig(brier_pdf)
        plt.close(fig)
        plot_receipts.append({"name": "m1_cal_brier", "png": brier_png.relative_to(OUTPUT_ROOT).as_posix(),
                              "pdf": brier_pdf.relative_to(OUTPUT_ROOT).as_posix(),
                              "endpoint_rows": len(brier_rows), "missing_ci_count": len(brier_rows)})

    relation_receipts, _relation_statuses, _relation_groups, _relation_bins = _m4_render_relation_calibration_figures(
        figures_dir, pyplot=plt
    )
    plot_receipts.extend(relation_receipts)

    conditions, _aliases = _condition_specs(config)
    methods = ("FIXED0", "FIXED2", "FIXED5", "FIXED10", "FIXED20", "OLD_FROZEN_R", "R_ORDER", "NO_HISTORY", "NO_EXTRA")
    condition_ids = [str(row["condition_id"]) for row in conditions]
    observed: list[dict[str, Any]] = []
    m2_gz = OUTPUT_ROOT / "M2_SELECT_TRAJECTORIES_EVALUATOR.jsonl.gz"
    if m2_gz.is_file():
        observed = _read_jsonl_gz(m2_gz)
    matrix = np.full((len(ALGORITHMS), len(methods), len(condition_ids)), np.nan, dtype=np.float64)
    if observed:
        for ai, algorithm in enumerate(ALGORITHMS):
            for mi, method in enumerate(methods):
                for ci, condition_id in enumerate(condition_ids):
                    baseline_layouts: dict[str, tuple[str, float]] = {}
                    target_layouts: dict[str, tuple[str, float]] = {}
                    for row in observed:
                        if row["algorithm"] != algorithm or row["condition_id"] != condition_id:
                            continue
                        if row["method"] == "FIXED20":
                            baseline_layouts[str(row["layout_fingerprint"])] = (str(row["geometry"]), float(row["r_over_n"]))
                        if row["method"] == method:
                            target_layouts[str(row["layout_fingerprint"])] = (str(row["geometry"]), float(row["r_over_n"]))
                    if set(baseline_layouts) != set(target_layouts) or not baseline_layouts:
                        continue
                    geometry_baselines: list[float] = []
                    geometry_targets: list[float] = []
                    for geometry in ("G1", "G2"):
                        matching = [fp for fp, pair in baseline_layouts.items() if pair[0] == geometry]
                        if not matching:
                            geometry_baselines = []
                            geometry_targets = []
                            break
                        base_mean = float(np.mean([baseline_layouts[fp][1] for fp in matching]))
                        target_mean = float(np.mean([target_layouts[fp][1] for fp in matching]))
                        if base_mean <= 0:
                            geometry_baselines = []
                            geometry_targets = []
                            break
                        geometry_baselines.append(base_mean)
                        geometry_targets.append(target_mean)
                    if len(geometry_baselines) == 2:
                        pooled_baseline = float(np.mean(geometry_baselines))
                        pooled_target = float(np.mean(geometry_targets))
                        matrix[ai, mi, ci] = 100.0 * (pooled_baseline - pooled_target) / pooled_baseline
    fig, axes = plt.subplots(2, 2, figsize=(18, 11), sharex=True, sharey=True,
                             constrained_layout=True)
    limit = float(np.nanmax(np.abs(matrix))) if np.isfinite(matrix).any() else 1.0
    limit = max(limit, 1.0)
    image = None
    for ai, axis in enumerate(axes.flat):
        image = axis.imshow(matrix[ai], aspect="auto", cmap="RdYlGn", vmin=-limit, vmax=limit)
        axis.set_title(ALGORITHMS[ai])
        axis.set_yticks(np.arange(len(methods)), methods)
        axis.set_xticks(np.arange(len(condition_ids)), [row["condition_id"] for row in conditions], rotation=90, fontsize=7)
        for mi in range(len(methods)):
            for ci in range(len(condition_ids)):
                value = matrix[ai, mi, ci]
                axis.text(ci, mi, "未测" if not math.isfinite(value) else f"{value:.1f}",
                          ha="center", va="center", fontsize=5)
    fig.colorbar(image, ax=list(axes.flat), label="相对同算法 FIXED20 的 R/N 降幅 (%)；此条件图无独立 CI")
    fig.suptitle("全部唯一 registered conditions × algorithms × methods")
    condition_png, condition_pdf = figures_dir / "all_registered_conditions.png", figures_dir / "all_registered_conditions.pdf"
    fig.savefig(condition_png, dpi=180)
    fig.savefig(condition_pdf)
    plt.close(fig)
    plot_receipts.append({"name": "all_registered_conditions", "png": condition_png.relative_to(OUTPUT_ROOT).as_posix(),
                          "pdf": condition_pdf.relative_to(OUTPUT_ROOT).as_posix(),
                          "condition_count": len(condition_ids), "method_count": len(methods),
                          "measured_cells": int(np.isfinite(matrix).sum()),
                          "confidence_intervals": "UNMEASURED_FOR_CONDITION_LEVEL_DISPLAY"})
    condition_plot_rows: list[dict[str, Any]] = []
    for ai, algorithm in enumerate(ALGORITHMS):
        for mi, method in enumerate(methods):
            for ci, condition_id in enumerate(condition_ids):
                matching_pairs = [row for row in observed if row["algorithm"] == algorithm
                                  and row["condition_id"] == condition_id
                                  and row["method"] in ("FIXED20", method)]
                baseline_by_geo: dict[str, list[float]] = {"G1": [], "G2": []}
                target_by_geo: dict[str, list[float]] = {"G1": [], "G2": []}
                for row in matching_pairs:
                    geometry = str(row["geometry"])
                    if row["method"] == "FIXED20":
                        baseline_by_geo[geometry].append(float(row["r_over_n"]))
                    if row["method"] == method:
                        target_by_geo[geometry].append(float(row["r_over_n"]))
                if all(baseline_by_geo[geo] and target_by_geo[geo] for geo in ("G1", "G2")):
                    base = float(np.mean([np.mean(baseline_by_geo[geo]) for geo in ("G1", "G2")]))
                    target = float(np.mean([np.mean(target_by_geo[geo]) for geo in ("G1", "G2")]))
                    gain = 100.0 * (base - target) / base if base > 0 else None
                    status = "MEASURED" if gain is not None else "UNMEASURED_NONPOSITIVE_BASELINE"
                else:
                    base = target = gain = None
                    status = "UNMEASURED_MISSING_GEOMETRY"
                condition_plot_rows.append({"algorithm": algorithm, "method": method,
                                            "condition_id": condition_id,
                                            "baseline_R_over_N_equal_geometry": base,
                                            "target_R_over_N_equal_geometry": target,
                                            "gain_pct": gain, "status": status})
    _write_csv_new(OUTPUT_ROOT / "M4_PLOTTED_ALL_CONDITIONS.csv", condition_plot_rows,
                   ("algorithm", "method", "condition_id", "baseline_R_over_N_equal_geometry",
                    "target_R_over_N_equal_geometry", "gain_pct", "status"))
    formal_plot = _m4_render_m3_formal_conditions(config, conditions)
    if formal_plot is not None:
        plot_receipts.append(formal_plot)
    return plot_receipts


def _m4_trajectory_count(final_path: Path, checkpoint_path: Path) -> int:
    if final_path.is_file():
        return len(_read_jsonl_gz(final_path))
    if checkpoint_path.is_file():
        with checkpoint_path.open("r", encoding="utf-8") as source:
            return sum(bool(line.strip()) for line in source)
    return 0


def _m4_fit_statuses() -> dict[str, str]:
    def describe(record: Any) -> str:
        if not isinstance(record, dict):
            return "未拟合"
        state = "收敛" if record.get("converged") is True else "未收敛"
        details = []
        if record.get("iterations") is not None:
            details.append(f"iter={record['iterations']}")
        if record.get("gradient_inf_norm") is not None:
            details.append(f"grad_inf={record['gradient_inf_norm']}")
        if record.get("objective") is not None:
            details.append(f"objective={record['objective']}")
        return state + (" (" + ", ".join(details) + ")" if details else "")

    model_path = OUTPUT_ROOT / "M1_MODELS.json"
    if model_path.is_file():
        models = _read_json(model_path)
        output = {}
        for name in ("main", "no_history", "q_only"):
            record = models.get(name)
            output[name] = describe(record)
        for algorithm in ALGORITHMS:
            record = models.get("loao", {}).get(algorithm, {})
            fit_record = record.get("fit") or record.get("fit_attempt")
            output[f"LOAO-{algorithm}"] = str(record.get("status", "未测")) + \
                ("; " + describe(fit_record) if isinstance(fit_record, dict) else "")
        return output
    attempt_path = OUTPUT_ROOT / "private/M1_FIT_ATTEMPTS_CHECKPOINT.jsonl"
    attempts: dict[str, str] = {}
    if attempt_path.is_file():
        with attempt_path.open("r", encoding="utf-8") as source:
            for line in source:
                if not line.strip():
                    continue
                row = json.loads(line)
                name, fit = str(row.get("model", "")), row.get("fit", {})
                if isinstance(fit, dict):
                    attempts[name] = describe(fit)
                elif row.get("diagnostic"):
                    attempts[name] = str(row["diagnostic"].get("status", "诊断不足"))
    names = {"main": "main", "no_history": "no_history", "q_only": "q_only"}
    output = {name: attempts.get(source, "未拟合") for name, source in names.items()}
    for algorithm in ALGORITHMS:
        output[f"LOAO-{algorithm}"] = attempts.get(f"loao_without_{algorithm}", "未测")
    return output


def _m4_report_detail_lines(
    config: Mapping[str, Any], statuses: Mapping[str, str],
    branch_points: Sequence[Mapping[str, Any]], brier_metrics: Mapping[str, Any],
) -> tuple[list[str], str]:
    m1_receipt_path = OUTPUT_ROOT / "M1_STAGE_RECEIPT.json"
    m1_receipt = _read_json(m1_receipt_path) if m1_receipt_path.is_file() else {}
    worker_result = m1_receipt.get("worker_result") or {}
    expected_fit = int(config["M1"]["fit_behavior_trajectories"])
    expected_cal = int(config["M1"]["cal_behavior_trajectories"])
    fit_count = _m4_trajectory_count(
        OUTPUT_ROOT / "M1_FIT_TRAJECTORY_RECEIPTS.jsonl.gz",
        OUTPUT_ROOT / "M1_FIT_TRAJECTORY_CHECKPOINT.jsonl",
    )
    cal_count = _m4_trajectory_count(
        OUTPUT_ROOT / "M1_CAL_TRAJECTORY_RECEIPTS.jsonl.gz",
        OUTPUT_ROOT / "M1_CAL_TRAJECTORY_CHECKPOINT.jsonl",
    )
    gate_path = OUTPUT_ROOT / "M1_GATE.json"
    gate = _read_json(gate_path) if gate_path.is_file() else {}
    tau_path = OUTPUT_ROOT / "M1_TAU_CALIBRATION.json"
    tau_record = _read_json(tau_path) if tau_path.is_file() else {}
    expected_status = m1_receipt.get("status", "未运行")
    bank_complete = fit_count == expected_fit and cal_count == expected_cal
    fit_statuses = _m4_fit_statuses()
    main_brier = brier_metrics.get("main_weighted_brier")
    q_brier = brier_metrics.get("q_only_weighted_brier")
    nh_brier = brier_metrics.get("no_history_weighted_brier")
    brier_gate = gate.get("brier_gate_pass")
    selected_tau = tau_record.get("main_tau_by_algorithm", {})
    lines = [
        "## M1 拟合与校准结果",
        "",
        f"- 阶段状态：`{statuses.get('M1', '未运行')}`；bank轨迹 fit={fit_count}/{expected_fit}、cal={cal_count}/{expected_cal}；完整性={('完整' if bank_complete else '不完整/未运行')}。",
        f"- 有效 capture/pair：fit={worker_result.get('fit_capture_count', '未测')} captures、{worker_result.get('fit_label_count', '未测')} pairs；cal={worker_result.get('cal_capture_count', '未测')} captures、{worker_result.get('cal_label_count', '未测')} pairs。",
        f"- 拟合状态：main={fit_statuses.get('main')}；NO_HISTORY={fit_statuses.get('no_history')}；q-only={fit_statuses.get('q_only')}。LOAO 为诊断；缺失或未收敛只限制对四个已注册行为算法之外的外推。",
        f"- cal weighted Brier：main={main_brier if main_brier is not None else '未测'}，NO_HISTORY={nh_brier if nh_brier is not None else '未测'}，q-only comparator={q_brier if q_brier is not None else '未测'}；主门槛 main≤q-only：{brier_gate if brier_gate is not None else '未测'}。",
        f"- M1 branch gate：八个所选 main 分支点全部严格大于0：{gate.get('eight_branch_points_pass', '未测')}；所选阈值 τ：{json.dumps(selected_tau, ensure_ascii=False, sort_keys=True) if selected_tau else '未测'}。",
        "",
        "所选 main 配置的八个 family 点（相对同算法 native H20 的 R/N 降幅，%）：",
        "",
        "|算法|family|τ|baseline R/N|selected R/N|降幅 (%)|布局 bootstrap 95% CI|状态|",
        "|---|---|---:|---:|---:|---:|---|---|",
    ]
    by_key = {(str(row.get("algorithm")), str(row.get("family")), float(row.get("tau", -1))): row
              for row in branch_points}
    main_points: list[dict[str, Any]] = []
    for algorithm in ALGORITHMS:
        tau = selected_tau.get(algorithm)
        for family in ("DP", "TWS"):
            row = by_key.get((algorithm, family, float(tau))) if tau is not None else None
            if row is None:
                row = {"gain_pct": None, "baseline_R_over_N": None, "target_R_over_N": None,
                       "ci95": None, "bootstrap_status": "UNMEASURED"}
            main_points.append({"algorithm": algorithm, "family": family,
                                "tau": tau, **row})
            fmt = lambda value: "未测" if value is None else f"{float(value):.6g}"
            lines.append(
                f"|{algorithm}|{family}|{fmt(tau)}|{fmt(row.get('baseline_R_over_N'))}|"
                f"{fmt(row.get('target_R_over_N'))}|{fmt(row.get('gain_pct'))}|"
                f"{json.dumps(row.get('ci95'), ensure_ascii=False) if row.get('ci95') is not None else '未测'}|"
                f"{row.get('bootstrap_status', '未测')}|"
            )
    stop_reason = m1_receipt.get("worker_error")
    if not stop_reason and str(expected_status).endswith("STOP"):
        stop_reason = gate.get("status") or expected_status
    if not stop_reason:
        stop_reason = "未停止" if str(expected_status).startswith("M1_COMPLETE") else "未运行"
    lines.extend([
        "",
        f"- 停止原因：{stop_reason}。所有 M1 分支图的 48 个注册组合（4算法×2 family×3 control×2 tau）见 plotted-data CSV；缺失分支标为未测。",
        ("- M3 formal 条件图：已从完整 confirmation trajectory 生成四方法×15条件的 plotted-data CSV、PNG 和矢量 PDF；条件级区间未测。"
         if (OUTPUT_ROOT / "M3_CONFIRM_TRAJECTORIES.jsonl.gz").is_file()
         else "- M3 formal 条件结果：未测/未执行或未保存完整 confirmation trajectory；不生成 formal 条件数据图。"),
        "- 计划/未执行阶段：" + ", ".join(
            f"{stage}={status}" for stage, status in statuses.items() if status == "未运行"
        ) if any(status == "未运行" for status in statuses.values()) else "- M1/M2/M3 均已有阶段收据。",
        "",
        "开发 bootstrap 为描述性 10000 次、按 geometry 分层布局重采样；它不构成确认推断。",
    ])
    for stage in ("M2", "M3"):
        receipt_path = OUTPUT_ROOT / f"{stage}_STAGE_RECEIPT.json"
        if not receipt_path.is_file():
            continue
        receipt = _read_json(receipt_path)
        result = receipt.get("worker_result") or {}
        stage_status = str(receipt.get("status", "UNKNOWN"))
        if not stage_status.endswith("STOP"):
            continue
        reason = receipt.get("worker_error") or result.get("stop_reason")
        if not reason and stage == "M2":
            reason = (f"point_gate_pass={result.get('point_gate_pass')}, "
                      f"cost_gate_pass={result.get('cost_gate_pass')}")
        if not reason and stage == "M3":
            projection = result.get("runtime_projection", {})
            reason = (f"runtime_projection={projection.get('projected_seconds_with_20_percent')} seconds; "
                      f"formal_gate_pass={result.get('formal_gate_pass')}")
        lines.append(f"- {stage} STOP 原因：{reason or stage_status}。")
    if str(expected_status).endswith("STOP"):
        next_step = "只允许离线审阅本报告与既有失败证据；M2/M3及任何重跑、补样、调参均禁止。"
    elif statuses.get("M1") == "M1_COMPLETE" and statuses.get("M2") == "未运行":
        next_step = "下一项仅可在 ROOT 独立审查并写入 hash-bound M2 release 后运行 M2；M3仍关闭。"
    elif str(statuses.get("M2", "")).endswith("STOP"):
        next_step = "只允许离线审阅已保存证据；M3及任何重跑、补样、调参均禁止。"
    elif statuses.get("M2") == "M2_COMPLETE" and statuses.get("M3") == "未运行":
        next_step = "下一项仅可在 ROOT 独立审查 M2 point/cost gates 并写入 hash-bound M3 release 后运行 M3。"
    elif statuses.get("M3") == "M3_COMPLETE":
        next_step = "PLAN-049 注册阶段已结束；只允许归档和审阅，不再增加实验。"
    else:
        next_step = "只允许按阶段 gate 与 ROOT hash-bound release 推进；禁止跳阶段、重跑、补样或调参。"
    return lines, next_step


def _m4_formal_condition_rows(
    observed: Sequence[Mapping[str, Any]], conditions: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    methods = ("FIXED20", "DEV_BEST_FIXED", "R_ORDER", "NO_HISTORY")
    condition_ids = [str(row["condition_id"]) for row in conditions]
    indexed: dict[tuple[str, str, str, str], tuple[str, float]] = {}
    for row in observed:
        algorithm, method, condition_id = str(row["algorithm"]), str(row["method"]), str(row["condition_id"])
        if algorithm not in ALGORITHMS or method not in methods or condition_id not in condition_ids:
            continue
        key = (algorithm, method, condition_id, str(row["layout_fingerprint"]))
        if key in indexed:
            raise ROrderStop(f"M3 formal condition artifact contains duplicate trajectory row: {key}")
        indexed[key] = (str(row["geometry"]), float(row["r_over_n"]))
    output: list[dict[str, Any]] = []
    for algorithm in ALGORITHMS:
        for method in methods:
            for condition_id in condition_ids:
                baseline = {
                    fingerprint: (geometry, value)
                    for (alg, candidate_method, cid, fingerprint), (geometry, value) in indexed.items()
                    if alg == algorithm and candidate_method == "FIXED20" and cid == condition_id
                }
                target = {
                    fingerprint: (geometry, value)
                    for (alg, candidate_method, cid, fingerprint), (geometry, value) in indexed.items()
                    if alg == algorithm and candidate_method == method and cid == condition_id
                }
                counts = {geometry: 0 for geometry, *_ in GEOMETRIES}
                if set(baseline) == set(target) and baseline:
                    base_by_geometry: dict[str, list[float]] = defaultdict(list)
                    target_by_geometry: dict[str, list[float]] = defaultdict(list)
                    for fingerprint, (geometry, base_value) in baseline.items():
                        target_geometry, target_value = target[fingerprint]
                        if geometry != target_geometry:
                            raise ROrderStop(f"M3 formal paired row geometry changed for {fingerprint}")
                        base_by_geometry[geometry].append(base_value)
                        target_by_geometry[geometry].append(target_value)
                        counts[geometry] += 1
                    if all(base_by_geometry.get(geometry) for geometry in counts):
                        base = float(np.mean([np.mean(base_by_geometry[geometry]) for geometry in counts]))
                        value = float(np.mean([np.mean(target_by_geometry[geometry]) for geometry in counts]))
                        gain = 100.0 * (base - value) / base if base > 0 else None
                        status = "MEASURED" if gain is not None else "UNMEASURED_NONPOSITIVE_BASELINE"
                    else:
                        base = value = gain = None
                        status = "UNMEASURED_MISSING_GEOMETRY"
                else:
                    base = value = gain = None
                    status = "UNMEASURED_PAIRED_LAYOUT_MISMATCH"
                output.append({
                    "algorithm": algorithm, "method": method, "condition_id": condition_id,
                    "baseline_R_over_N_equal_geometry": base,
                    "target_R_over_N_equal_geometry": value, "gain_pct": gain,
                    "layout_count_by_geometry": counts, "status": status,
                    "confidence_interval": "UNMEASURED_FOR_CONDITION_LEVEL_DISPLAY",
                })
    return output


def _m4_render_m3_formal_conditions(
    config: Mapping[str, Any], conditions: Sequence[Mapping[str, Any]],
) -> dict[str, Any] | None:
    trajectory_path = OUTPUT_ROOT / "M3_CONFIRM_TRAJECTORIES.jsonl.gz"
    if not trajectory_path.is_file():
        return None
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = _m4_formal_condition_rows(_read_jsonl_gz(trajectory_path), conditions)
    _write_csv_new(OUTPUT_ROOT / "M4_PLOTTED_M3_FORMAL_CONDITIONS.csv", rows, (
        "algorithm", "method", "condition_id", "baseline_R_over_N_equal_geometry",
        "target_R_over_N_equal_geometry", "gain_pct", "layout_count_by_geometry", "status",
        "confidence_interval",
    ))
    methods = ("FIXED20", "DEV_BEST_FIXED", "R_ORDER", "NO_HISTORY")
    condition_ids = [str(row["condition_id"]) for row in conditions]
    lookup = {(str(row["algorithm"]), str(row["method"]), str(row["condition_id"])): row for row in rows}
    matrix = np.full((len(ALGORITHMS), len(methods), len(condition_ids)), np.nan, dtype=np.float64)
    for ai, algorithm in enumerate(ALGORITHMS):
        for mi, method in enumerate(methods):
            for ci, condition_id in enumerate(condition_ids):
                row = lookup[(algorithm, method, condition_id)]
                if row["gain_pct"] is not None:
                    matrix[ai, mi, ci] = float(row["gain_pct"])
    figures_dir = OUTPUT_ROOT / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(16, 10), sharex=True, sharey=True,
                             constrained_layout=True)
    limit = float(np.nanmax(np.abs(matrix))) if np.isfinite(matrix).any() else 1.0
    limit = max(limit, 1.0)
    image = None
    for ai, axis in enumerate(axes.flat):
        image = axis.imshow(matrix[ai], aspect="auto", cmap="RdYlGn", vmin=-limit, vmax=limit)
        axis.set_title(ALGORITHMS[ai])
        axis.set_yticks(np.arange(len(methods)), methods)
        axis.set_xticks(np.arange(len(condition_ids)), condition_ids, rotation=90, fontsize=7)
        for mi in range(len(methods)):
            for ci in range(len(condition_ids)):
                value = matrix[ai, mi, ci]
                axis.text(ci, mi, "未测" if not math.isfinite(value) else f"{value:.1f}",
                          ha="center", va="center", fontsize=5)
    fig.colorbar(image, ax=list(axes.flat), label="相对同算法 native H20 的 R/N 降幅 (%)；条件级区间未测")
    fig.suptitle("M3 formal：四方法 × 15 个唯一 registered conditions")
    png = figures_dir / "m3_formal_conditions.png"
    pdf = figures_dir / "m3_formal_conditions.pdf"
    fig.savefig(png, dpi=180)
    fig.savefig(pdf)
    plt.close(fig)
    return {
        "name": "m3_formal_conditions", "png": png.relative_to(OUTPUT_ROOT).as_posix(),
        "pdf": pdf.relative_to(OUTPUT_ROOT).as_posix(),
        "condition_count": len(condition_ids), "method_count": len(methods),
        "measured_cells": int(np.isfinite(matrix).sum()),
        "confidence_intervals": "UNMEASURED_FOR_CONDITION_LEVEL_DISPLAY",
        "plotted_data_csv": "M4_PLOTTED_M3_FORMAL_CONDITIONS.csv",
    }


def _m4_relation_report_lines(stage_statuses: Mapping[str, str]) -> list[str]:
    lines = [
        "## 关系校准诊断",
        "",
        "诊断复用完整 pair 表上的 normalized_hierarchical_weights；每个分组按其原权重质量归一，分组与十个等宽概率 bin 可复算 pooled weighted Brier。此处仅作描述，不改变模型、阈值、科学 gate，也不报告 CI。",
        "",
        "|来源|状态|pairs|captures|weighted Brier|原始入口|",
        "|---|---|---:|---:|---:|---|",
    ]
    for split, summary_path in (
        ("M1 cal", OUTPUT_ROOT / "M1_CAL_RELATION_CALIBRATION_SUMMARY.json"),
        ("M2 select", OUTPUT_ROOT / "M2_SELECT_RELATION_CALIBRATION_SUMMARY.json"),
    ):
        summary = _read_json(summary_path) if summary_path.is_file() else {"status": "UNMEASURED_NOT_RUN"}
        measured = summary.get("status") == "MEASURED"
        brier = summary.get("overall", {}).get("weighted_brier") if measured else None
        sidecar = summary.get("raw_pair_sidecar", "未生成")
        groups = summary.get("group_table", "未生成")
        bins = summary.get("probability_bin_table", "未生成")
        lines.append(
            f"|{split}|{'已测' if measured else '未测（无已执行标签）'}|"
            f"{summary.get('pair_count', '—')}|{summary.get('capture_count', '—')}|"
            f"{brier if brier is not None else '—'}|`{sidecar}`；`{groups}`；`{bins}`|"
        )
    formal_status = (
        "未测（M3未运行）" if stage_statuses.get("M3") == "未运行"
        else "未测（本关系校准诊断只注册于 cal 与 select）"
    )
    lines.extend([
        f"|M3 formal|{formal_status}|—|—|—|无正式关系标签/诊断图|",
        "",
        "分组维度为噪声族、公开名次距离、history 是否含 censor、检索阶段；bin 表为各分组及 pooled overall 的十等宽概率区间。M2 只采预分配算法的既有 native FIXED20 轨迹，每条恰取 first legal、最近 0.50、最近 0.85（整数距离并列取早；EG 仅 group-start），未另加轨迹。",
    ])
    return lines


def _run_m4_worker() -> dict[str, Any]:
    """Offline-only report generation from artifacts already on disk."""
    config = _load_config()
    statuses: dict[str, str] = {}
    for stage in ("M1", "M2", "M3"):
        path = OUTPUT_ROOT / f"{stage}_STAGE_RECEIPT.json"
        if path.is_file():
            statuses[stage] = str(_read_json(path).get("status", "UNKNOWN"))
        else:
            statuses[stage] = "未运行"
    branch_points = _m4_m1_branch_rows(config)
    brier_path = OUTPUT_ROOT / "M1_CAL_PAIR_METRICS.json"
    brier_metrics = _read_json(brier_path) if brier_path.is_file() else {}
    plot_receipts = _m4_render_figures(config, branch_points=branch_points, brier_metrics=brier_metrics)
    m1_lines, next_step = _m4_report_detail_lines(config, statuses, branch_points, brier_metrics)
    relation_lines = _m4_relation_report_lines(statuses)
    plan_text = "planned only" if all(value == "未运行" for value in statuses.values()) else "see executed stage receipts"
    report_lines = [
        "# PLAN-049 R-Order 结果报告",
        "",
        "本报告只汇总本目录已存在的 receipts 与原始结果文件；阶段状态明确区分计划和已执行。",
        "",
        "## 阶段状态",
        "",
        "|阶段|状态|",
        "|---|---|",
        *[f"|{stage}|{status}|" for stage, status in statuses.items()],
        "",
        "## 已执行证据入口",
        "",
        *[f"- `{name}`" for name in sorted(path.name for path in OUTPUT_ROOT.iterdir() if path.is_file())],
        "",
        *m1_lines,
        "",
        *relation_lines,
        "",
        "## 图表",
        "",
        *[f"- PNG `{row['png']}`；矢量 PDF `{row['pdf']}`；端点/格数={row.get('endpoint_rows', row.get('measured_cells', 0))}；缺失区间按未测标记。"
          for row in plot_receipts],
        "",
        "未测或缺失 CI 的单元在图中标记为“未测”，不作为通过证据。开发结果与正式确认分开保存。",
        "",
        "## HANDOFF 决策边界",
        "",
        next_step,
    ]
    report = "\n".join(report_lines) + "\n"
    handoff = (
        "# PLAN-049 HANDOFF\n\n"
        f"状态：{statuses}\n\n"
        f"计划边界：{plan_text}。未运行阶段保持未运行；M1/M2/M3 STOP 保持 STOP，不恢复、不补样、不重试。\n\n"
        f"下一步：{next_step}\n\n"
        "入口：`FINAL_REPORT.md`、`m0_review_rev02/M0_SOURCE_BINDINGS_REV02.json`、"
        "`m0_review_rev02/M0_PREPARE_RECEIPT_REV02.json`、`m0_review_rev02/M0_TEST_RECEIPT_REV02.json`、"
        "各阶段 `*_STAGE_RECEIPT.json`、science ledger 和 raw JSONL/CSV。\n\n"
        "所有注册条件保留于完整 trajectory CSV/JSONL；主指标为 R/N，R_raw 在 raw rows 中同时提供。\n"
    )
    _write_new(OUTPUT_ROOT / "FINAL_REPORT.md", report.encode("utf-8"))
    _write_new(OUTPUT_ROOT / "HANDOFF.md", handoff.encode("utf-8"))
    _write_json_new(OUTPUT_ROOT / "M4_FIGURE_INVENTORY.json", {"figures": plot_receipts})
    return {"stage": "M4", "status": "M4_COMPLETE", "offline_only": True,
            "stage_statuses": statuses, "figure_count": len(plot_receipts),
            "next_allowed_step": next_step,
            "missing_confidence_intervals_marked_unmeasured": True}


def run_m0_revision_tests() -> dict[str, Any]:
    """Run only synthetic M0 REV02 tests in a real child and bind all outputs."""
    if not PREPARE_RECEIPT_PATH.is_file():
        raise ROrderStop("M0 preparation must finish before its synthetic test receipt")
    prepare_receipt = _read_json(PREPARE_RECEIPT_PATH)
    bound_source_hash = _source_binding_digest()
    if prepare_receipt.get("source_bindings_sha256") != bound_source_hash:
        raise ROrderStop("REV02 prepare receipt does not bind current source bytes")
    _verify_m0_artifact_bytes()
    if TEST_RECEIPT_PATH.exists():
        raise ROrderStop("refusing to overwrite or rerun the M0 test receipt")
    if M0_COMPACT_SUMMARY_PATH.exists():
        raise ROrderStop("M0 compact summary exists without a final test receipt; preserve partial output")
    revision_root = TEST_RECEIPT_PATH.parent
    stdout_path = revision_root / "M0_TEST_STDOUT_REV02.txt"
    stderr_path = revision_root / "M0_TEST_STDERR_REV02.txt"
    if stdout_path.exists() or stderr_path.exists():
        raise ROrderStop("M0 test log exists without a final receipt; preserve partial output")
    test_paths = [
        "R/tests/test_r_order_public.py",
        "R/tests/test_r_order_training.py",
        "R/tests/test_r_order_analysis.py",
        "R/tests/test_r_order_protocol.py",
    ]
    command = [sys.executable, "-B", "-m", "pytest", "-p", "no:cacheprovider", "-q", *test_paths]
    started = time.perf_counter()
    process: subprocess.Popen[str] | None = None
    try:
        child_env = os.environ.copy()
        child_env["PYTHONDONTWRITEBYTECODE"] = "1"
        process = subprocess.Popen(
            command, cwd=WORKSPACE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, encoding="utf-8", errors="replace", env=child_env,
        )
        stdout_text, stderr_text = process.communicate(timeout=600)
        exit_code = int(process.returncode)
        timed_out = False
    except subprocess.TimeoutExpired as exc:
        if process is not None:
            process.kill()
            stdout_text, stderr_text = process.communicate()
            exit_code = int(process.returncode if process.returncode is not None else 124)
        else:
            stdout_value = exc.stdout or ""
            stderr_value = exc.stderr or ""
            stdout_text = stdout_value.decode("utf-8", "replace") if isinstance(stdout_value, bytes) else str(stdout_value)
            stderr_text = stderr_value.decode("utf-8", "replace") if isinstance(stderr_value, bytes) else str(stderr_value)
            exit_code = 124
        timed_out = True
    except BaseException as exc:
        stdout_text = ""
        stderr_text = f"{type(exc).__name__}: {exc}\n"
        exit_code = 127
        timed_out = False
    duration = max(0.0, time.perf_counter() - started)
    _write_new(stdout_path, stdout_text.encode("utf-8"))
    _write_new(stderr_path, stderr_text.encode("utf-8"))
    prepare_hash = _sha256_file(PREPARE_RECEIPT_PATH)
    source_hash = _source_binding_digest()
    audit = prepare_receipt.get("preflight_audit", {})
    audit_path = str(audit.get("path", ""))
    compact_summary = "\n".join([
        "# PLAN-049 M0 REV02 Compact Summary",
        "",
        f"M0 REV02 synthetic tests: {'PASS' if exit_code == 0 and not timed_out else 'FAIL_STOP'}; real child exit={exit_code}; timeout={str(timed_out).lower()}; elapsed={duration:.3f}s; pid={None if process is None else process.pid}.",
        f"Prepare receipt: `m0_review_rev02/M0_PREPARE_RECEIPT_REV02.json` SHA256 `{prepare_hash}`.",
        f"Source bindings: `m0_review_rev02/M0_SOURCE_BINDINGS_REV02.json` SHA256 `{source_hash}`.",
        f"Prior preflight audit: `{audit_path}` SHA256 `{audit.get('sha256', 'missing')}` ({audit.get('bytes', 'unknown')} bytes).",
        "M0 test receipt: `m0_review_rev02/M0_TEST_RECEIPT_REV02.json` (created after this summary; its receipt hash-binds this file).",
        f"Test child command: `{json.dumps(command, ensure_ascii=False)}`.",
        f"Captured stdout: `{stdout_path.relative_to(OUTPUT_ROOT).as_posix()}` SHA256 `{_sha256_file(stdout_path)}`; captured stderr: `{stderr_path.relative_to(OUTPUT_ROOT).as_posix()}` SHA256 `{_sha256_file(stderr_path)}`.",
        "M0 REV02 reused the original 1,000 registered layouts byte-for-byte. It generated no forecast labels, fit, behavior bank, cost result, performance screen, or confirmation outcome.",
        "No science ledger was started and no ROOT release was present during M0. Science remains closed pending ROOT's independent hash-bound review.",
        "",
    ])
    _write_new(M0_COMPACT_SUMMARY_PATH, compact_summary.encode("utf-8"))
    receipt = {
        "protocol_id": "r_order_v1", "run_id": "r01", "phase": "M0_SYNTHETIC_TESTS",
        "revision_id": "M0_REVIEW_REV02",
        "status": "PASS" if exit_code == 0 and not timed_out else "FAIL_STOP",
        "command": command, "cwd": WORKSPACE.as_posix(), "timeout_seconds": 600,
        "duration_seconds": duration, "exit_code": exit_code, "timed_out": timed_out,
        "child_pid": None if process is None else process.pid,
        "prepare_receipt_sha256": prepare_hash,
        "stdout_path": stdout_path.relative_to(OUTPUT_ROOT).as_posix(),
        "stdout_sha256": _sha256_file(stdout_path),
        "stdout_bytes": stdout_path.stat().st_size,
        "stderr_path": stderr_path.relative_to(OUTPUT_ROOT).as_posix(),
        "stderr_sha256": _sha256_file(stderr_path),
        "stderr_bytes": stderr_path.stat().st_size,
        "summary_path": M0_COMPACT_SUMMARY_PATH.relative_to(OUTPUT_ROOT).as_posix(),
        "summary_sha256": _sha256_file(M0_COMPACT_SUMMARY_PATH),
        "summary_bytes": M0_COMPACT_SUMMARY_PATH.stat().st_size,
        "preflight_audit": audit,
        "source_bindings_sha256": source_hash,
        "synthetic_only": True, "labels_fit_or_science_rollouts": False,
        "science_ledger_started": False, "root_release_present": False,
    }
    _write_json_new(TEST_RECEIPT_PATH, receipt)
    return receipt


def run_m0_tests() -> dict[str, Any]:
    """Compatibility alias; test-m0 always targets the active REV02 review bundle."""
    return run_m0_revision_tests()


def _stage_worker(stage: str, ledger: ScienceLedger) -> dict[str, Any]:
    if stage == "M1":
        return _run_m1_worker(ledger)
    if stage == "M2":
        return _run_m2_worker(ledger)
    if stage == "M3":
        return _run_m3_worker(ledger)
    if stage == "M4":
        return _run_m4_worker()
    raise ValueError("unknown science stage")


def _stage_output_conflicts(stage: str) -> list[str]:
    if stage == "M4":
        expected = [OUTPUT_ROOT / "FINAL_REPORT.md", OUTPUT_ROOT / "HANDOFF.md",
                    OUTPUT_ROOT / "M4_HASH_INVENTORY.json", OUTPUT_ROOT / "figures"]
        conflicts = [path.relative_to(OUTPUT_ROOT).as_posix() for path in expected if path.exists()]
        conflicts.extend(path.relative_to(OUTPUT_ROOT).as_posix() for path in OUTPUT_ROOT.rglob("M4_*") if path.is_file())
        return sorted(set(conflicts))
    return sorted(_stage_artifact_paths(stage))


def _run_stage_child(stage: str, run_token: str) -> dict[str, Any]:
    """Child entrypoint. The parent owns the single wall-ledger start/stop interval."""
    ledger = ScienceLedger.attach_running(stage, run_token)
    release = verify_root_release(stage)
    if release["sha256"] != ledger.release_sha256:
        raise ROrderStop("stage release changed after the supervisor opened its ledger interval")
    result_path = OUTPUT_ROOT / f"{stage}_CHILD_RECEIPT.json"
    if result_path.exists():
        raise ROrderStop("child receipt already exists; stage resume is forbidden")
    try:
        result = _stage_worker(stage, ledger)
        receipt = {
            "protocol_id": "r_order_v1", "run_id": "r01", "stage": stage,
            "status": str(result.get("status", f"{stage}_COMPLETE")), "exit_code": 0,
            "pid": os.getpid(), "worker_result": _json_safe(result), "worker_error": None,
            "root_release_sha256": release["sha256"],
        }
        _write_json_new(result_path, receipt)
        return receipt
    except BaseException as exc:
        receipt = {
            "protocol_id": "r_order_v1", "run_id": "r01", "stage": stage,
            "status": f"{stage}_IMPLEMENTATION_STOP", "exit_code": 1, "pid": os.getpid(),
            "worker_result": None, "worker_error": f"{type(exc).__name__}: {exc}",
            "traceback": "".join(traceback.format_exception(exc)),
            "root_release_sha256": release["sha256"],
        }
        _write_json_new(result_path, receipt)
        traceback.print_exception(exc, file=sys.stderr)
        return receipt


def _science_worker_command(stage: str, run_token: str) -> list[str]:
    return [sys.executable, "-B", "-m", "R.experiments.r_order_protocol",
            "stage-worker", stage, run_token]


def _supervise_child_process(
    command: Sequence[str], *, deadline_monotonic: float, stdout_path: Path, stderr_path: Path,
) -> tuple[subprocess.Popen[Any] | None, bool, str | None, float]:
    """Run one actual child to an absolute deadline and retain both output streams."""
    started = time.perf_counter()
    process: subprocess.Popen[Any] | None = None
    timed_out = False
    launch_error: str | None = None
    try:
        stdout_path.parent.mkdir(parents=True, exist_ok=True)
        stderr_path.parent.mkdir(parents=True, exist_ok=True)
        with stdout_path.open("xb") as stdout_handle, stderr_path.open("xb") as stderr_handle:
            process = subprocess.Popen(list(command), cwd=WORKSPACE,
                                       stdout=stdout_handle, stderr=stderr_handle)
            remaining = max(0.0, float(deadline_monotonic) - time.perf_counter())
            try:
                process.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                timed_out = True
                try:
                    process.kill()
                except OSError:
                    pass
                process.wait()
    except BaseException as exc:
        launch_error = f"{type(exc).__name__}: {exc}"
        if process is not None and process.poll() is None:
            try:
                process.kill()
            except OSError:
                pass
            process.wait()
    return process, timed_out, launch_error, max(0.0, time.perf_counter() - started)


def _finalize_m4_hash_inventory() -> None:
    inventory_path = OUTPUT_ROOT / "M4_HASH_INVENTORY.json"
    if inventory_path.exists():
        raise ROrderStop("M4 hash inventory already exists; refusing to replace a frozen artifact")
    inventory = {
        path.relative_to(OUTPUT_ROOT).as_posix(): {"sha256": _sha256_file(path), "bytes": path.stat().st_size}
        for path in sorted(OUTPUT_ROOT.rglob("*"), key=lambda item: item.as_posix().casefold())
        if path.is_file() and path not in (inventory_path, OUTPUT_ROOT / "M4_STAGE_RECEIPT.json")
    }
    payload = {
        "algorithm_source_binding": _read_json(SOURCE_BINDINGS_PATH),
        "data_receipt_sha256": _sha256_file(PREPARE_RECEIPT_PATH),
        "science_stage_artifacts": inventory,
        "created_after_child_exit_and_closed_ledger": True,
    }
    _write_json_new(inventory_path, payload)


def run_science_stage(stage: str) -> dict[str, Any]:
    """Supervise one released worker subprocess with hard stage/total wall caps."""
    stage = str(stage).upper()
    if stage not in RELEASE_PATHS:
        raise ValueError("stage must be M1, M2, M3, or M4")
    receipt_path = OUTPUT_ROOT / f"{stage}_STAGE_RECEIPT.json"
    if receipt_path.exists():
        raise ROrderStop(f"{stage} stage already has a receipt; resume/retry is forbidden")
    conflicts = _stage_output_conflicts(stage)
    if conflicts:
        raise ROrderStop(f"{stage} partial outputs already exist; preserve them and do not resume: {conflicts[:5]}")
    release = verify_root_release(stage)
    ledger = ScienceLedger(stage)
    ledger.start(release)
    stdout_path = OUTPUT_ROOT / f"{stage}_WORKER_STDOUT.txt"
    stderr_path = OUTPUT_ROOT / f"{stage}_WORKER_STDERR.txt"
    child_receipt_path = OUTPUT_ROOT / f"{stage}_CHILD_RECEIPT.json"
    command = _science_worker_command(stage, ledger.run_token)
    deadline = float(ledger.started_monotonic or time.perf_counter()) + \
        float(ledger.available_seconds or PHASE_CAP_SECONDS[stage]) - 0.1
    process, timed_out, launch_error, elapsed = _supervise_child_process(
        command, deadline_monotonic=deadline, stdout_path=stdout_path, stderr_path=stderr_path,
    )
    actual_exit_code = int(process.returncode) if process is not None and process.returncode is not None else 127
    child_receipt = _read_json(child_receipt_path) if child_receipt_path.is_file() else None
    worker_result = None if child_receipt is None else child_receipt.get("worker_result")
    worker_error = launch_error if launch_error is not None else (None if child_receipt is None else child_receipt.get("worker_error"))
    if timed_out:
        status = f"{stage}_TIMEOUT_STOP"
    elif launch_error is not None or actual_exit_code != 0 or child_receipt is None:
        status = f"{stage}_IMPLEMENTATION_STOP"
    else:
        status = str(child_receipt.get("status", f"{stage}_COMPLETE"))
    ledger_record = ledger.stop(status=status, exit_code=actual_exit_code)
    if stage == "M4":
        _finalize_m4_hash_inventory()
    result_gate = (worker_result or {}).get("scientific_gate_status")
    receipt: dict[str, Any] = {
        "protocol_id": "r_order_v1", "run_id": "r01", "stage": stage,
        "status": status, "exit_code": actual_exit_code,
        "scientific_gate_status": result_gate,
        "root_release_path": release["path"], "root_release_sha256": release["sha256"],
        "previous_stage_receipt_sha256": release["record"].get("previous_stage_receipt_sha256"),
        "source_bindings_sha256": _source_binding_digest(),
        "m0_prepare_receipt_sha256": _sha256_file(PREPARE_RECEIPT_PATH),
        "m0_test_receipt_sha256": _sha256_file(TEST_RECEIPT_PATH),
        "worker_result": _json_safe(worker_result) if worker_result is not None else None,
        "worker_error": worker_error,
        "child_process": {
            "command": command,
            "pid": (child_receipt.get("pid") if child_receipt is not None else
                    (None if process is None else process.pid)),
            "actual_returncode": actual_exit_code, "timed_out": timed_out,
            "supervisor_elapsed_seconds": elapsed,
            "child_receipt_path": child_receipt_path.relative_to(OUTPUT_ROOT).as_posix()
                if child_receipt is not None else None,
            "child_receipt_sha256": _sha256_file(child_receipt_path) if child_receipt_path.is_file() else None,
        },
        "stdout_path": stdout_path.relative_to(OUTPUT_ROOT).as_posix(),
        "stdout_sha256": _sha256_file(stdout_path),
        "stderr_path": stderr_path.relative_to(OUTPUT_ROOT).as_posix(),
        "stderr_sha256": _sha256_file(stderr_path),
        "science_ledger": ledger_record,
        "output_artifact_sha256": _stage_artifact_inventory(stage),
        "output_inventory_created_utc": datetime.now(timezone.utc).isoformat(),
    }
    if stage == "M1":
        receipt["fit_model_sha256"] = _sha256_file(OUTPUT_ROOT / "M1_MODELS.json") if (OUTPUT_ROOT / "M1_MODELS.json").is_file() else None
        receipt["tau_calibration_sha256"] = _sha256_file(OUTPUT_ROOT / "M1_TAU_CALIBRATION.json") if (OUTPUT_ROOT / "M1_TAU_CALIBRATION.json").is_file() else None
    elif stage == "M2":
        receipt["point_gate_pass"] = (worker_result or {}).get("point_gate_pass")
        receipt["cost_gate_pass"] = (worker_result or {}).get("cost_gate_pass")
    elif stage == "M3":
        receipt["power_pass"] = (worker_result or {}).get("power_pass")
        receipt["runtime_projection_pass"] = (worker_result or {}).get("runtime_projection_pass")
        receipt["formal_gate_pass"] = (worker_result or {}).get("formal_gate_pass")
    _write_json_new(receipt_path, receipt)
    return receipt


def _print_json(value: Any) -> None:
    print(json.dumps(_json_safe(value), sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="PLAN-049 one-shot M0 and ROOT-release-gated runner")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("prepare-m0", help="generate registered layouts/manifests only")
    subparsers.add_parser("test-m0", help="run deterministic/synthetic tests; no science bank")
    subparsers.add_parser("prepare-m0-revision", help="prepare the authorized REV02 review bundle from immutable REV01 data")
    subparsers.add_parser("test-m0-revision", help="run real-child synthetic tests for the REV02 review bundle")
    run_parser = subparsers.add_parser("run-stage", help="run one ROOT-released science phase")
    run_parser.add_argument("stage", choices=("M1", "M2", "M3", "M4"))
    child_parser = subparsers.add_parser("stage-worker", help=argparse.SUPPRESS)
    child_parser.add_argument("stage", choices=("M1", "M2", "M3", "M4"))
    child_parser.add_argument("run_token")
    verify_parser = subparsers.add_parser("verify-release", help="verify one ROOT release without running science")
    verify_parser.add_argument("stage", choices=("M1", "M2", "M3", "M4"))
    subparsers.add_parser("status", help="show M0/stage receipt status without running science")
    args = parser.parse_args(argv)
    try:
        if args.command == "stage-worker":
            child_receipt = _run_stage_child(args.stage, args.run_token)
            _print_json(child_receipt)
            return int(child_receipt.get("exit_code", 1))
        if args.command == "prepare-m0":
            result = prepare_m0()
        elif args.command == "prepare-m0-revision":
            result = prepare_m0_revision_rev02()
        elif args.command == "test-m0":
            result = run_m0_revision_tests()
        elif args.command == "test-m0-revision":
            result = run_m0_revision_tests()
        elif args.command == "run-stage":
            result = run_science_stage(args.stage)
        elif args.command == "verify-release":
            result = verify_root_release(args.stage)
        else:
            result = {
                "protocol_id": "r_order_v1", "run_id": "r01",
                "m0_prepare": _read_json(PREPARE_RECEIPT_PATH) if PREPARE_RECEIPT_PATH.is_file() else "未运行",
                "m0_test": _read_json(TEST_RECEIPT_PATH) if TEST_RECEIPT_PATH.is_file() else "未运行",
                "stages": {
                    stage: _read_json(OUTPUT_ROOT / f"{stage}_STAGE_RECEIPT.json")
                    if (OUTPUT_ROOT / f"{stage}_STAGE_RECEIPT.json").is_file() else "未运行"
                    for stage in ("M1", "M2", "M3", "M4")
                },
            }
        _print_json(result)
        if args.command in ("test-m0", "test-m0-revision") and result.get("status") != "PASS":
            return 1
        if args.command == "run-stage" and int(result.get("exit_code", 0)) != 0:
            return 1
        return 0
    except BaseException as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

