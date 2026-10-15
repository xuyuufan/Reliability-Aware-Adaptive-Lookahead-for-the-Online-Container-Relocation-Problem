"""One-shot PLAN-046 time-window-shuffle correctness acceptance bank.

This runner produces no performance selection or ranking. Use ``--prepare``
once to freeze the exact 20-key manifest and evaluator-private forecasts, then
``--run`` once after the code/source review. A started bank cannot be resumed.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import importlib
import json
import os
from pathlib import Path
import platform
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np


R_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = R_ROOT.parent
CONFIG_PATH = R_ROOT / "configs/r_noise_diagnostics_v1.json"
OUTPUT = R_ROOT / "output/r_noise_diagnostics_v1/r01/time_window"
PLAN_PATH = R_ROOT / "docs/planning/PLAN_046_ALGORITHM_AGNOSTIC_R_DIAGNOSTICS.md"
PROTOCOL_PATH = R_ROOT / "experiments/time_window_shuffle_protocol.py"
RUNNER_PATH = Path(__file__).resolve()
TEST_PATH = R_ROOT / "tests/test_time_window_shuffle.py"
PLOT_PATH = OUTPUT / "plot_time_window_correctness.py"
PREIMPLEMENTATION_PATH = OUTPUT / "PREIMPLEMENTATION_SOURCE_BINDINGS.json"
METHODS = (
    "EM_FIXED0", "EM_FIXED2", "EM_FIXED5", "EM_FIXED10", "EM_FIXED20", "R_EM_FROZEN",
    "GP_FIXED0", "GP_FIXED20", "GP_R_TRANSFER",
    "ERI20", "R_ERI_FROZEN", "ERI_FIXED0",
    "EG_FIXED0", "EG20", "R_EG_GROUP_FROZEN",
)
FORBIDDEN_PUBLIC_KEYS = {
    "family", "window_size", "repeat", "seed", "seed_evaluator_only", "forecast",
    "forecast_hash", "true_future", "true_sequence", "p", "p_evaluator_only",
    "raw_ids", "container_ids", "private_labels", "evaluator_only",
}
RAW_NAMES = (
    "rollouts.jsonl", "decisions.jsonl", "private_transitions.jsonl",
    "private_reveals.jsonl", "forecast_bindings_evaluator_only.jsonl", "eg_group_trace.jsonl",
    "RUN_STATUS.json", "REPORT.md", "bank_snapshot.json", "bank_snapshot.sha256",
    "manifest.json", "manifest.sha256", "test_receipt.json", "test_stdout.txt",
    "test_stderr.txt", "acceptance_stdout.txt", "acceptance_stderr.txt",
    "acceptance_execution_receipt.json", "noise_correctness.png", "noise_correctness.pdf",
)
BANK_RESULT_NAMES = (
    "rollouts.jsonl", "decisions.jsonl", "private_transitions.jsonl", "private_reveals.jsonl", "eg_group_trace.jsonl",
    "RUN_STATUS.json", "REPORT.md", "noise_correctness.png", "noise_correctness.pdf",
    "acceptance_stdout.txt", "acceptance_stderr.txt", "acceptance_execution_receipt.json",
)


class AcceptanceStop(RuntimeError):
    """A registered input, information, physics, or completion gate failed."""


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, indent=2) + "\n").encode("utf-8")


def _write_json_new(path: Path, value: Any) -> None:
    with path.open("xb") as handle:
        handle.write(_json_bytes(value))


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _workspace_path(relative: str) -> Path:
    path = (WORKSPACE / Path(*str(relative).replace("\\", "/").split("/"))).resolve()
    try:
        path.relative_to(WORKSPACE.resolve())
    except ValueError as exc:
        raise ValueError(f"workspace path escapes root: {relative}") from exc
    return path


def _load_config() -> dict[str, Any]:
    config = _read_json(CONFIG_PATH)
    if config.get("protocol_id") != "r_noise_diagnostics_v1":
        raise AcceptanceStop("unexpected PLAN-046 protocol id")
    methods = [method for family in ("EM", "GP", "ERI", "EG") for method in config["methods"][family]]
    if tuple(methods) != METHODS or len(METHODS) != 15:
        raise AcceptanceStop("registered 15-method order changed")
    if int(config["methods"]["total_rollout_count"]) != 300:
        raise AcceptanceStop("registered bank size changed")
    if config["acceptance"].get("one_shot_no_resume_or_retry") is not True:
        raise AcceptanceStop("bank must remain one-shot")
    if _sha256_file(PREIMPLEMENTATION_PATH) != config["preimplementation_source_manifest_sha256"]:
        raise AcceptanceStop("preimplementation source manifest hash differs from config")
    return config


def _runtime_imports() -> dict[str, Any]:
    # Keep the established owner-first import contract for the shared EM/R/
    # Mijn_RL packages; the accepted GP tree is loaded separately below.
    roots = (R_ROOT, WORKSPACE, WORKSPACE / "EM", WORKSPACE / "Mijn_RL", WORKSPACE / "GP")
    for root in reversed(roots):
        text = str(root.resolve())
        while text in sys.path:
            sys.path.remove(text)
        sys.path.insert(0, text)
    retry = importlib.import_module("experiments.r_frozen_cross_solver_transfer_evaluate_r02_h20_retry")
    runtime = retry._runtime_imports()
    runtime["_load_select_episodes"] = retry._load_select_episodes
    runtime["sync_events_with_private_trace"] = retry._sync_events_with_private_trace
    runtime["public_frame_record"] = retry._public_frame_record
    runtime["truncate_public_frame"] = importlib.import_module("scrp.cross_solver_reliability_r").truncate_public_frame
    runtime["frozen_actor_call"] = importlib.import_module("scrp.cross_solver_reliability_r").frozen_actor_call
    runtime["em_select"] = importlib.import_module("scrp.noise_aware_em").select_em_destination
    runtime["em_r_select"] = importlib.import_module("scrp.history_reliability_r").select_r_action
    runtime["eri_select"] = importlib.import_module("scrp.eri_eg_public").select_eri20_destination
    runtime["eri_eg"] = importlib.import_module("scrp.eri_eg_public")
    runtime["r_eri_eg"] = importlib.import_module("scrp.r_eri_eg_public")
    return runtime


def _load_accepted_gp_rule(config: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Compile only the already accepted M0 tree; never scan/select 120 rules."""
    identity = config["accepted_gp_rule"]
    m0_path = _workspace_path("R/output/r_frozen_cross_solver_transfer_v1/r01/M0_PREPARATION_RECEIPT.json")
    m1_path = _workspace_path("R/output/r_frozen_cross_solver_transfer_v1/r01/M1_select/GP/M1_RECEIPT.json")
    run_path = _workspace_path("GP/results/h20_four_terminal_sets/runs/H20/C/run_25/run_complete.json")
    train_path = run_path.with_name("train_complete.json")
    train_manifest = _workspace_path("GP/data/caserta_train_840_seed20260920.json")
    expected_paths = {
        run_path.relative_to(WORKSPACE).as_posix(): str(identity["run_receipt_sha256"]),
        train_path.relative_to(WORKSPACE).as_posix(): str(identity["train_receipt_sha256"]),
        train_manifest.relative_to(WORKSPACE).as_posix(): str(identity["train_manifest_sha256"]),
    }
    actual = {name: _sha256_file(_workspace_path(name)) for name in expected_paths}
    if actual != expected_paths:
        raise AcceptanceStop("accepted GP receipt or training-manifest SHA differs")
    if _sha256_file(m1_path) != str(identity["m1_gp_receipt_sha256"]):
        raise AcceptanceStop("accepted GP M1 receipt hash differs")
    m0 = _read_json(m0_path)
    selected = m0["models"]["selected_gp"]
    if any(
        selected.get(key) != identity.get(key)
        for key in ("selected_method", "run_receipt_sha256", "train_receipt_sha256", "train_manifest_sha256", "selected_best_expression")
    ):
        raise AcceptanceStop("accepted GP rule does not match frozen M0 identity")
    if selected.get("selected_method") != "GP-H20-C-run25":
        raise AcceptanceStop("unexpected GP selected method")
    m1 = _read_json(m1_path)
    old_model = m1.get("model_receipt", {})
    bindings_before = m1.get("model_bindings_before", {})
    bindings_after = m1.get("model_bindings_after", {})
    expression = str(selected["selected_best_expression"])
    if (
        m1.get("status") != "COMPLETE"
        or m1.get("solver") != "GP"
        or m1.get("stage") != "select"
        or old_model.get("selected_method") != identity["selected_method"]
        or old_model.get("run_receipt_sha256") != identity["run_receipt_sha256"]
        or old_model.get("train_receipt_sha256") != identity["train_receipt_sha256"]
        or old_model.get("train_manifest_sha256") != identity["train_manifest_sha256"]
        or old_model.get("selected_best_expression") != expression
        or _sha256_bytes(expression.encode("utf-8")) != identity["selected_expression_sha256"]
        or bindings_before != bindings_after
        or bindings_before.get("selected_method") != identity["selected_method"]
        or bindings_before.get("artifact_sha256") != actual
    ):
        raise AcceptanceStop("accepted GP M1 model/rule/before-after binding check failed")
    from GP.expression import Node
    from GP.simulator import select_destination

    tree = Node.from_prefix(tuple(int(value) for value in selected["selected_best_prefix"]))
    if tree.expression() != selected["selected_best_expression"]:
        raise AcceptanceStop("accepted GP prefix/expression mismatch")

    def slow_public_selector(state: Any, legal_destinations: Sequence[int], *, max_tiers: int) -> int:
        return int(select_destination(tree, state, legal_destinations, max_tiers=max_tiers))

    return {"selector": slow_public_selector, "tree": tree}, {
        "selected_method": selected["selected_method"],
        "run_receipt_sha256": actual[run_path.relative_to(WORKSPACE).as_posix()],
        "train_receipt_sha256": actual[train_path.relative_to(WORKSPACE).as_posix()],
        "train_manifest_sha256": actual[train_manifest.relative_to(WORKSPACE).as_posix()],
        "selected_best_expression": tree.expression(),
        "selected_expression_sha256": _sha256_bytes(tree.expression().encode("utf-8")),
        "m1_gp_receipt_sha256": _sha256_file(m1_path),
        "m1_status": str(m1["status"]),
        "selection_or_training_performed": False,
        "path": "R/cross_solver_reliability_r.py -> public_frame_to_legacy_state -> GP.simulator.select_destination (Python slow public path)",
    }


def _current_source_hashes(config: Mapping[str, Any]) -> dict[str, str]:
    baseline = _read_json(PREIMPLEMENTATION_PATH)
    frozen: dict[str, str] = {}
    changed_expected_plan = "R/docs/planning/PLAN_046_ALGORITHM_AGNOSTIC_R_DIAGNOSTICS.md"
    for row in baseline["files"]:
        path_text = str(row["path"])
        if path_text == changed_expected_plan:
            continue  # this task deliberately revised its pre-implementation registration draft
        path = _workspace_path(path_text)
        actual = _sha256_file(path)
        if actual != str(row["sha256"]):
            raise AcceptanceStop(f"frozen source/asset changed since preimplementation binding: {path_text}")
        frozen[path_text] = actual
    owned = {
        "R/docs/planning/PLAN_046_ALGORITHM_AGNOSTIC_R_DIAGNOSTICS.md": PLAN_PATH,
        "R/configs/r_noise_diagnostics_v1.json": CONFIG_PATH,
        "R/experiments/time_window_shuffle_protocol.py": PROTOCOL_PATH,
        "R/experiments/time_window_shuffle_acceptance.py": RUNNER_PATH,
        "R/tests/test_time_window_shuffle.py": TEST_PATH,
        "R/output/r_noise_diagnostics_v1/r01/time_window/plot_time_window_correctness.py": PLOT_PATH,
    }
    for relative, path in owned.items():
        if not path.is_file():
            raise AcceptanceStop(f"registered source missing: {relative}")
        frozen[relative] = _sha256_file(path)
    return dict(sorted(frozen.items()))


def _select_exact_layouts(config: Mapping[str, Any], runtime: Mapping[str, Any]) -> list[Any]:
    retry_config = _read_json(R_ROOT / "configs/r_frozen_cross_solver_transfer_r02_h20_retry_v1.json")
    episodes = runtime["_load_select_episodes"](retry_config, runtime)
    fingerprint = runtime["label_invariant_fingerprint"]
    chosen: list[Any] = []
    expected_by_geometry = {str(row["geometry_id"]): row for row in config["layout_source"]["selected"]}
    for geometry in ("G1", "G2"):
        rows = [
            (str(fingerprint(ep)), ep)
            for ep in episodes
            if str(ep.parameter_group) == geometry
        ]
        if len(rows) != 10:
            raise AcceptanceStop(f"select20 {geometry} count changed")
        rows.sort(key=lambda item: item[0])
        fp, episode = rows[0]
        expected = expected_by_geometry[geometry]
        if (
            fp != str(expected["fingerprint"])
            or str(episode.episode_id) != str(expected["episode_id"])
        ):
            raise AcceptanceStop(f"selected select20 layout identity changed for {geometry}")
        chosen.append(episode)
    selected_manifest = _workspace_path(str(config["layout_source"]["select_manifest"]))
    if _sha256_file(selected_manifest) != "62f9e205f2df0197260629f4b9ad54a35a412c6ab16838709b0a26d224ecf426":
        raise AcceptanceStop("bound select20 bytes differ from accepted source")
    return sorted(chosen, key=lambda episode: str(fingerprint(episode)))


def _condition_records(config: Mapping[str, Any], runtime: Mapping[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    from experiments.time_window_shuffle_protocol import (
        FAMILY,
        make_time_window_forecast,
        semantic_time_window_seed,
        time_window_forecast_sha256,
    )

    episodes = _select_exact_layouts(config, runtime)
    fingerprint_fn = runtime["label_invariant_fingerprint"]
    conditions: list[dict[str, Any]] = []
    private_bindings: list[dict[str, Any]] = []
    for episode in episodes:
        fp = str(fingerprint_fn(episode))
        for width in config["noise"]["window_sizes"]:
            for repeat in config["noise"]["repeats"]:
                seed = semantic_time_window_seed(fp, window_size=int(width), repeat=int(repeat))
                true_sequence = tuple(int(value) for value in episode.retrieval_sequence)
                forecast = make_time_window_forecast(true_sequence[1:], window_size=int(width), seed=seed)
                forecast_hash = time_window_forecast_sha256(forecast)
                condition_id = _sha256_bytes(
                    f"{fp}|{FAMILY}|B={int(width)}|r={int(repeat)}".encode("ascii")
                )[:20]
                rollout_ids = {
                    method: "tw-" + _sha256_bytes(f"{condition_id}|{method}".encode("ascii"))[:20]
                    for method in METHODS
                }
                condition = {
                    "condition_id_evaluator_only": condition_id,
                    "fingerprint": fp,
                    "geometry_id": str(episode.parameter_group),
                    "family": FAMILY,
                    "window_size": int(width),
                    "repeat": int(repeat),
                    "semantic_seed_evaluator_only": int(seed),
                    "forecast_sha256": forecast_hash,
                    "expected_methods": list(METHODS),
                    "rollout_ids_evaluator_only": rollout_ids,
                }
                conditions.append(condition)
                private_bindings.append({
                    **condition,
                    "episode_id_evaluator_only": str(episode.episode_id),
                    "initial_current_target_private": true_sequence[0],
                    "true_sequence_private": list(true_sequence),
                    "true_sequence_sha256_evaluator_only": _sha256_bytes(
                        json.dumps(list(true_sequence), separators=(",", ":")).encode("utf-8")
                    ),
                    "forecast_private": list(forecast),
                    "forecast_length": len(forecast),
                    "forecast_is_fixed_for_all_15_rollouts": True,
                    "provider_p_placeholder": 0.0,
                    "provider_p_placeholder_is_scientific_condition": False,
                    "evaluator_only": True,
                })
    conditions.sort(key=lambda row: (str(row["fingerprint"]), int(row["window_size"]), int(row["repeat"])))
    private_bindings.sort(key=lambda row: (str(row["fingerprint"]), int(row["window_size"]), int(row["repeat"])))
    expected_order = [
        (str(row["fingerprint"]), int(width), int(repeat))
        for row in sorted(config["layout_source"]["selected"], key=lambda row: str(row["fingerprint"]))
        for width in config["noise"]["window_sizes"]
        for repeat in config["noise"]["repeats"]
    ]
    actual_order = [(str(row["fingerprint"]), int(row["window_size"]), int(row["repeat"])) for row in conditions]
    if actual_order != expected_order or len(conditions) != 20 or len(set(actual_order)) != 20:
        raise AcceptanceStop("expanded condition keys are missing, duplicated, extra, or reordered")
    if any(len(row["expected_methods"]) != 15 for row in conditions):
        raise AcceptanceStop("each condition must expect 15 registered methods")
    return conditions, private_bindings


def prepare_bank() -> dict[str, Any]:
    config = _load_config()
    if any((OUTPUT / name).exists() for name in ("manifest.json", "manifest.sha256", "bank_snapshot.json", "bank_snapshot.sha256", "forecast_bindings_evaluator_only.jsonl", *BANK_RESULT_NAMES)):
        raise FileExistsError("one-shot acceptance outputs already exist; refusing to prepare over them")
    runtime = _runtime_imports()
    gp_rule, gp_identity = _load_accepted_gp_rule(config)
    conditions, private_bindings = _condition_records(config, runtime)
    source_hashes = _current_source_hashes(config)
    private_path = OUTPUT / "forecast_bindings_evaluator_only.jsonl"
    with private_path.open("x", encoding="utf-8", newline="\n") as handle:
        for row in private_bindings:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n")
    conditions_by_key = [
        {
            key: row[key]
            for key in ("condition_id_evaluator_only", "fingerprint", "geometry_id", "family", "window_size", "repeat", "semantic_seed_evaluator_only", "forecast_sha256", "expected_methods", "rollout_ids_evaluator_only")
        }
        for row in conditions
    ]
    manifest = {
        "schema": "plan046_time_window_exact_bank_manifest_v1",
        "protocol_id": config["protocol_id"],
        "run_id": config["run_id"],
        "stage": "PREPARED_NOT_STARTED",
        "condition_key_order": ["fingerprint", "family", "window_size", "repeat"],
        "conditions": conditions_by_key,
        "expected_condition_count": 20,
        "expected_methods_per_condition": list(METHODS),
        "expected_rollout_count": 300,
        "actual_rollout_count": 0,
        "forecast_bindings_evaluator_only_path": "forecast_bindings_evaluator_only.jsonl",
        "forecast_bindings_evaluator_only_sha256": _sha256_file(private_path),
        "source_hashes": source_hashes,
        "accepted_gp_identity": gp_identity,
        "method_outputs_are_correctness_evidence_only": True,
        "performance_ranking_or_gate": False,
    }
    _write_json_new(OUTPUT / "manifest.json", manifest)
    manifest_sha = _sha256_file(OUTPUT / "manifest.json")
    (OUTPUT / "manifest.sha256").write_text(manifest_sha + "\n", encoding="ascii", newline="\n")
    snapshot = {
        "schema": "plan046_time_window_prebank_snapshot_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PREPARED_AWAITING_ONE_SHOT_BANK",
        "source_hashes": source_hashes,
        "manifest_sha256": manifest_sha,
        "private_forecast_bindings_sha256": _sha256_file(private_path),
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "wall_limit_seconds": 600,
        "rollout_count": 300,
        "accepted_gp_identity": gp_identity,
        "selection_or_training_performed": False,
        "only_directly_used_frozen_inputs_hashed": True,
    }
    _write_json_new(OUTPUT / "bank_snapshot.json", snapshot)
    (OUTPUT / "bank_snapshot.sha256").write_text(_sha256_file(OUTPUT / "bank_snapshot.json") + "\n", encoding="ascii", newline="\n")
    return {
        "status": "PREPARED_AWAITING_ONE_SHOT_BANK",
        "condition_count": len(conditions),
        "rollout_count": 300,
        "method_count": 15,
        "manifest_sha256": manifest_sha,
        "forecast_binding_count": len(private_bindings),
        "gp_rule": gp_identity["selected_method"],
    }


class _JsonlWriter:
    def __init__(self, path: Path) -> None:
        self.handle = path.open("x", encoding="utf-8", newline="\n")
        self.path = path
        self.count = 0

    def write(self, row: Mapping[str, Any]) -> None:
        self.handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n")
        self.count += 1

    def close(self) -> None:
        self.handle.flush()
        os.fsync(self.handle.fileno())
        self.handle.close()


def _contains_forbidden_key(value: Any) -> str | None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            normalized = str(key).strip().lower()
            forbidden_fragments = (
                "family", "window", "seed", "forecast", "true_", "private", "evaluator",
                "raw_id", "container_id", "noise", "condition", "p_evaluator",
            )
            if normalized in FORBIDDEN_PUBLIC_KEYS or any(fragment in normalized for fragment in forbidden_fragments):
                return str(key)
            found = _contains_forbidden_key(nested)
            if found:
                return found
    elif isinstance(value, (list, tuple)):
        for nested in value:
            found = _contains_forbidden_key(nested)
            if found:
                return found
    return None


def _public_actor_call(
    frame: Any,
    *,
    algorithm: str,
    method: str,
    actor: Any,
    runtime: Mapping[str, Any],
    gp_rule: Mapping[str, Any],
) -> tuple[int, dict[str, Any]]:
    truncate = runtime["truncate_public_frame"]
    if algorithm == "EM":
        if method == "R_EM_FROZEN":
            decision = runtime["em_r_select"](frame.observation, method="R", max_tiers=frame.max_tiers)
            if not decision.candidates or decision.candidates[0].requested_h != 0:
                raise AcceptanceStop("R-EM lost same-algorithm h=0 anchor")
            output = {
                "action": int(decision.destination),
                "selected_h": int(decision.requested_h),
                "predicted_delta": float(decision.predicted_delta),
                "candidate_pool": [asdict(row) for row in decision.candidates],
                "selector": "frozen_R_EM",
            }
        else:
            h = int(method.removeprefix("EM_FIXED"))
            prefix = truncate(frame, h)
            action = int(runtime["em_select"](prefix.observation, max_tiers=frame.max_tiers, h=h))
            output = {"action": action, "selected_h": h, "selector": "frozen_EM"}
    elif algorithm == "GP":
        if method == "GP_FIXED0":
            gp_method = "FIXED0"
            input_frame = truncate(frame, 0)
        elif method == "GP_FIXED20":
            gp_method = "FIXED20"
            input_frame = truncate(frame, 20)
        elif method == "GP_R_TRANSFER":
            gp_method = "R_TRANSFER"
            input_frame = frame
        else:
            raise ValueError(f"unexpected GP method {method}")
        result = runtime["frozen_actor_call"](
            input_frame, solver="GP", frozen_solver=gp_rule, method=gp_method
        )
        if method == "GP_R_TRANSFER" and (
            not result.deduplicated_candidates or result.deduplicated_candidates[0].requested_h != 0
        ):
            raise AcceptanceStop("GP R transfer lost h=0 anchor")
        output = {
            "action": int(result.action),
            "selected_h": int(result.selected_h),
            "predicted_delta": float(result.delta),
            "raw_candidates": [asdict(row) for row in result.raw_candidates],
            "deduplicated_candidates": [asdict(row) for row in result.deduplicated_candidates],
            "solver_invocation_count": int(result.solver_invocation_count),
            "selector": "accepted_GP_public_slow_path",
        }
    elif algorithm == "ERI":
        if method == "R_ERI_FROZEN":
            result = actor.select_destination(frame)
            if not result["deduplicated_candidates"] or result["deduplicated_candidates"][0]["requested_h"] != 0:
                raise AcceptanceStop("R-ERI lost same-algorithm h=0 anchor")
            output = {**result, "selector": "frozen_R_ERI"}
        else:
            h = 0 if method == "ERI_FIXED0" else 20
            action = int(runtime["eri_select"](truncate(frame, h)))
            output = {"action": action, "selected_h": h, "selector": "frozen_ERI"}
    elif algorithm == "EG":
        if method == "R_EG_GROUP_FROZEN":
            prior_plan = actor._selected_plan
            prior_h = actor._selected_h
            prior_group_id = int(actor._group_id)
            result = actor.select_destination(frame)
            if result["group_started"]:
                if not result["deduplicated_candidates"] or result["deduplicated_candidates"][0]["requested_h"] != 0:
                    raise AcceptanceStop("R-EG lost same-algorithm h=0 anchor at group start")
            elif (
                prior_plan is None
                or result["group_started"]
                or int(result["group_id"]) != prior_group_id
                or int(result["selected_h"]) != int(prior_h)
                or result["committed_plan"] != prior_plan
                or result["deduplicated_candidates"]
            ):
                raise AcceptanceStop("R-EG cached output differs from the existing committed group plan")
            output = {**result, "selector": "frozen_R_EG_group"}
        else:
            h = 0 if method == "EG_FIXED0" else 20
            prefix = truncate(frame, h)
            before_groups = int(actor.groups_planned)
            prior_plan = actor._plan
            prior_move_index = int(actor._next_move)
            action = int(actor.select_destination(prefix))
            group_started = int(actor.groups_planned) > before_groups
            plan = actor._plan if group_started else prior_plan
            if plan is None:
                destinations = (action,)
                blocker_ranks = ()
                plan_index = 0
            else:
                destinations = tuple(int(value) for value in plan.destinations_top_down)
                blocker_ranks = tuple(int(value) for value in plan.blocker_ranks_top_down)
                # EG20Actor advances _next_move when it consumes a cached
                # prefix. The value captured before the call is therefore the
                # previous move's index, not the move being returned now.
                plan_index = 0 if group_started else prior_move_index + 1
            if not destinations or plan_index >= len(destinations) or destinations[plan_index] != action:
                raise AcceptanceStop("EG action differs from the committed full-plan order")
            output = {
                "action": action,
                "selected_h": h,
                "group_started": group_started,
                "group_id": int(actor.groups_planned),
                "group_move_index": plan_index + 1,
                "group_plan_length": len(destinations),
                "remaining_plan_moves": max(0, len(destinations) - plan_index - 1),
                "cached_plan_move": not group_started,
                "full_plan_order_index": plan_index,
                "committed_plan": {
                    "blocker_ranks_top_down": list(blocker_ranks),
                    "destinations_top_down": list(destinations),
                },
                "cached_moves": int(actor.cached_moves),
                "selector": "frozen_EG_group_actor",
            }
    else:
        raise ValueError(f"unexpected algorithm {algorithm}")
    action = int(output["action"])
    if action not in frame.legal_destinations:
        raise AcceptanceStop(f"{method} selected illegal destination {action}")
    leaked = _contains_forbidden_key(output)
    if leaked:
        raise AcceptanceStop(f"forbidden evaluator/public field reached actor output: {leaked}")
    return action, output


class _EGPlanAudit:
    """Sidecar that retains and validates each complete public EG commitment."""

    def __init__(self, *, method: str, rollout_id: str, writer: _JsonlWriter) -> None:
        if method not in ("EG_FIXED0", "EG20", "R_EG_GROUP_FROZEN"):
            raise ValueError(f"not an EG method: {method}")
        self.method = method
        self.rollout_id = rollout_id
        self.writer = writer
        self.active: dict[str, Any] | None = None
        self.last_group_id = 0
        self.groups_completed = 0
        self.cached_moves = 0

    def record(self, *, step: int, action: int, output: Mapping[str, Any]) -> None:
        started = bool(output.get("group_started"))
        group_id = int(output["group_id"])
        selected_h = int(output["selected_h"])
        plan_record = output.get("committed_plan")
        if not isinstance(plan_record, Mapping):
            raise AcceptanceStop("EG actor did not expose its complete committed plan")
        destinations = tuple(int(value) for value in plan_record.get("destinations_top_down", ()))
        blocker_ranks = tuple(int(value) for value in plan_record.get("blocker_ranks_top_down", ()))
        length = int(output["group_plan_length"])
        move_index = int(output["group_move_index"])
        if not destinations or len(destinations) != length:
            raise AcceptanceStop("EG committed plan length/ranks differ from the public plan metadata")
        if blocker_ranks and len(blocker_ranks) != length:
            raise AcceptanceStop("EG blocker-rank count differs from the committed plan length")
        if self.method == "R_EG_GROUP_FROZEN" and len(blocker_ranks) != length:
            raise AcceptanceStop("R-EG committed plan omitted its blocker-rank sequence")
        if not 1 <= move_index <= length or destinations[move_index - 1] != int(action):
            raise AcceptanceStop("EG action does not match the indexed move in its full committed plan")

        if started:
            if self.active is not None:
                raise AcceptanceStop("EG started a new group before completing the previous commitment")
            if group_id != self.last_group_id + 1 or move_index != 1:
                raise AcceptanceStop("EG group id or first move index is not consecutive")
            candidate_plans = output.get("candidate_plans")
            if self.method == "R_EG_GROUP_FROZEN":
                if not isinstance(candidate_plans, list) or not candidate_plans:
                    raise AcceptanceStop("R-EG group start omitted its candidate-plan pool")
                if not output.get("deduplicated_candidates"):
                    raise AcceptanceStop("R-EG group start omitted its h=0 anchor pool")
            self.active = {
                "group_id": group_id,
                "selected_h": selected_h,
                "destinations_top_down": destinations,
                "blocker_ranks_top_down": blocker_ranks,
                "group_plan_length": length,
                "next_move_index": 1,
                "source_group_start_step": int(step),
                "candidate_plans_at_start": candidate_plans,
            }
            self.last_group_id = group_id
        else:
            active = self.active
            if active is None:
                raise AcceptanceStop("EG cached move has no active group commitment")
            if (
                group_id != active["group_id"]
                or selected_h != active["selected_h"]
                or destinations != active["destinations_top_down"]
                or blocker_ranks != active["blocker_ranks_top_down"]
                or length != active["group_plan_length"]
                or move_index != active["next_move_index"] + 1
                or not bool(output.get("cached_plan_move"))
            ):
                raise AcceptanceStop("EG cached move differs from the original full group commitment")
            if self.method == "R_EG_GROUP_FROZEN" and (
                output.get("candidate_plans")
                or output.get("deduplicated_candidates")
                or int(output.get("horizon_plan_build_calls", -1)) != 0
            ):
                raise AcceptanceStop("R-EG cached move unexpectedly rebuilt candidates or a horizon plan")
            active["next_move_index"] = move_index
            self.cached_moves += 1

        active = self.active
        assert active is not None
        self.writer.write({
            "rollout_id": self.rollout_id,
            "method": self.method,
            "step": int(step),
            "group_id": group_id,
            "group_started": started,
            "selected_h": selected_h,
            "action": int(action),
            "group_move_index": move_index,
            "group_plan_length": length,
            "remaining_plan_moves": length - move_index,
            "cached_plan_move": bool(output.get("cached_plan_move")),
            "source_group_start_step": int(active["source_group_start_step"]),
            "committed_plan": {
                "blocker_ranks_top_down": list(active["blocker_ranks_top_down"]),
                "destinations_top_down": list(active["destinations_top_down"]),
            },
            "candidate_plans_at_group_start": active["candidate_plans_at_start"] if started else None,
            "group_plan_build_calls": int(output.get("horizon_plan_build_calls", 0)),
        })
        if move_index == length:
            self.active = None
            self.groups_completed += 1

    def finish(self) -> None:
        if self.active is not None:
            raise AcceptanceStop("episode terminated with an unfinished EG group commitment")


def _assert_frame_contract(frame: Any, *, horizon: int | None = None) -> None:
    ranks = np.asarray(frame.observation.rank_tensor)
    occupied = np.asarray(frame.observation.occupied_mask)
    legal = np.asarray(frame.observation.legal_destination_mask)
    if np.count_nonzero(occupied & (ranks == 1)) != 1:
        raise AcceptanceStop("public frame does not have exactly one current target")
    if np.any(ranks[~occupied] != 0):
        raise AcceptanceStop("empty physical padding changed from public rank 0")
    if tuple(np.flatnonzero(legal).tolist()) != tuple(frame.legal_destinations):
        raise AcceptanceStop("public legal mask differs from legal destination tuple")
    if horizon is not None:
        exposed = occupied & (ranks > 1) & (ranks != 22)
        if np.any(ranks[exposed] > horizon + 1):
            raise AcceptanceStop(f"H{horizon} public frame exposes an over-horizon rank")


def _run_one_rollout(
    episode: Any,
    condition: Mapping[str, Any],
    binding: Mapping[str, Any],
    method: str,
    *,
    runtime: Mapping[str, Any],
    gp_rule: Mapping[str, Any],
    writers: Mapping[str, _JsonlWriter],
    deadline: float,
) -> tuple[dict[str, Any], tuple[tuple[int, bool, float], ...]]:
    from scrp.noise_aware_forecast import ForecastProvider
    from scrp.online_crp import OnlineCRPEnvironment

    algorithm = next(name for name in ("EM", "GP", "ERI", "EG") if method in tuple(condition["method_algorithm_map"][name])) if "method_algorithm_map" in condition else (
        "EM" if method.startswith(("EM_", "R_EM")) else "GP" if method.startswith("GP_")
        else "ERI" if method.startswith(("ERI", "R_ERI")) else "EG"
    )
    rollout_id = str(condition["rollout_ids_evaluator_only"][method])
    forecast = tuple(int(value) for value in binding["forecast_private"])
    seed = int(binding["semantic_seed_evaluator_only"])
    # Every rollout receives a new provider instance. The scientific forecast
    # F and seed were derived once and injected from the evaluator-only binding.
    provider = ForecastProvider(
        episode.retrieval_sequence,
        p=0.0,
        seed=seed,
        q_prior=float(runtime["FROZEN_R_Q_PRIOR"]),
        max_horizon=20,
        forecast_future=forecast,
    )
    if provider.max_horizon != 20 or provider.history.window != 8 or provider.history.q_prior != float(runtime["FROZEN_R_Q_PRIOR"]):
        raise AcceptanceStop("ForecastProvider differs from frozen H20/prior/history")
    environment = OnlineCRPEnvironment(
        episode,
        max_steps=int(episode.num_containers * (episode.max_tiers - 1)),
        lookahead_horizon=0,
    )
    if algorithm == "ERI":
        actor = runtime["r_eri_eg"].RERIActor() if method == "R_ERI_FROZEN" else None
    elif algorithm == "EG":
        actor = (
            runtime["r_eri_eg"].REGGroupActor(crane_travel_time=1.2, pick_drop_time=30.0)
            if method == "R_EG_GROUP_FROZEN"
            else runtime["eri_eg"].EG20Actor()
        )
    else:
        actor = None

    seq = tuple(int(value) for value in episode.retrieval_sequence)
    private_condition = {
        "rollout_id_evaluator_only": rollout_id,
        "fingerprint": str(condition["fingerprint"]),
        "geometry_id": str(condition["geometry_id"]),
        "family": str(condition["family"]),
        "window_size": int(condition["window_size"]),
        "repeat": int(condition["repeat"]),
        "evaluator_only": True,
    }
    reveal_rows: list[dict[str, Any]] = []
    initial_events, _ = runtime["sync_events_with_private_trace"](
        provider,
        seq,
        0,
        int(environment.state.retrieval_count),
        runtime=runtime,
        condition=private_condition,
        method=method,
        step=0,
        phase="reset",
    )
    for event in initial_events:
        row = {**event, "rollout_id_evaluator_only": rollout_id}
        writers["reveals"].write(row)
        reveal_rows.append(row)

    decision_count = 0
    zero_reveal_moves = 0
    legal_actions = 0
    eg_audit = (
        _EGPlanAudit(method=method, rollout_id=rollout_id, writer=writers["eg_groups"])
        if algorithm == "EG"
        else None
    )
    maximum_steps = int(episode.num_containers * (episode.max_tiers - 1))
    while not environment.state.terminated:
        if time.perf_counter() >= deadline:
            raise TimeoutError("PLAN-046 one-shot acceptance exceeded 600-second wall ceiling")
        step = decision_count + 1
        before_stacks = runtime["_private_physical_stacks"](environment)
        current_target = int(environment.current_target_id)
        sources = [index for index, stack in enumerate(before_stacks) if current_target in stack]
        if len(sources) != 1:
            raise AcceptanceStop("current target is not conserved in exactly one physical stack")
        source = int(sources[0])
        if before_stacks[source][-1] == current_target:
            raise AcceptanceStop("environment exposed an accessible target before automatic retrieval")
        blocker = int(before_stacks[source][-1])
        before_retrievals = int(environment.state.retrieval_count)
        before_relocations = int(environment.state.relocation_count)
        frame = runtime["_frame_for_fixture"](episode, environment, provider)
        _assert_frame_contract(frame)
        public_frame = runtime["public_frame_record"](frame)
        action, actor_output = _public_actor_call(
            frame,
            algorithm=algorithm,
            method=method,
            actor=actor,
            runtime=runtime,
            gp_rule=gp_rule,
        )
        if eg_audit is not None:
            eg_audit.record(step=step, action=action, output=actor_output)
        if time.perf_counter() >= deadline:
            raise TimeoutError("PLAN-046 one-shot acceptance exceeded 600-second wall ceiling")
        decision_row = {
            "rollout_id": rollout_id,
            "method": method,
            "step": step,
            "public_frame": public_frame,
            "actor_output": actor_output,
        }
        leaked = _contains_forbidden_key(decision_row)
        if leaked:
            raise AcceptanceStop(f"private condition/truth field present in public decision row: {leaked}")
        writers["decisions"].write(decision_row)

        environment.step(action)
        after_stacks = runtime["_private_physical_stacks"](environment)
        after_retrievals = int(environment.state.retrieval_count)
        after_relocations = int(environment.state.relocation_count)
        if after_relocations != before_relocations + 1:
            raise AcceptanceStop("one public decision did not cause exactly one relocation")
        before_live = {int(value) for stack in before_stacks for value in stack}
        after_live = {int(value) for stack in after_stacks for value in stack}
        newly_retrieved = before_live - after_live
        if len(newly_retrieved) != after_retrievals - before_retrievals:
            raise AcceptanceStop("box conservation differs from automatic retrieval count")
        if before_live - newly_retrieved != after_live:
            raise AcceptanceStop("physical stack transition lost or invented a live container")
        if any(len(stack) > int(episode.max_tiers) for stack in after_stacks):
            raise AcceptanceStop("physical state exceeded registered stack capacity")
        if blocker in {int(value) for stack in after_stacks for value in stack}:
            if blocker in after_stacks[source] or blocker not in after_stacks[action]:
                raise AcceptanceStop("top blocker was not moved from source to chosen destination")

        events, _ = runtime["sync_events_with_private_trace"](
            provider,
            seq,
            before_retrievals,
            after_retrievals,
            runtime=runtime,
            condition=private_condition,
            method=method,
            step=step,
            phase="decision",
        )
        for event in events:
            row = {**event, "rollout_id_evaluator_only": rollout_id}
            writers["reveals"].write(row)
            reveal_rows.append(row)
        q_unchanged = not events and float(provider.history.q_t) == float(frame.observation.q_t)
        if not events and not q_unchanged:
            raise AcceptanceStop("q_t changed on a relocation without a reveal")
        transition = {
            **private_condition,
            "method": method,
            "step": step,
            "current_target_id_private": current_target,
            "source_stack_private": source,
            "blocker_id_private": blocker,
            "destination_stack": action,
            "stacks_before_private": [list(stack) for stack in before_stacks],
            "stacks_after_private": [list(stack) for stack in after_stacks],
            "retrieval_count_before": before_retrievals,
            "retrieval_count_after": after_retrievals,
            "relocation_count_before": before_relocations,
            "relocation_count_after": after_relocations,
            "new_reveal_event_indices": [int(row["event_index"]) for row in events],
            "q_t_before": float(frame.observation.q_t),
            "q_t_after": float(provider.history.q_t),
            "q_unchanged_without_reveal": bool(q_unchanged),
            "evaluator_only": True,
        }
        writers["transitions"].write(transition)
        if time.perf_counter() >= deadline:
            raise TimeoutError("PLAN-046 one-shot acceptance exceeded 600-second wall ceiling")
        if q_unchanged:
            zero_reveal_moves += 1
        legal_actions += 1
        decision_count += 1
        if decision_count > maximum_steps:
            raise AcceptanceStop("rollout exceeded physical relocation safety bound")

    if not environment.state.terminated or int(environment.state.retrieval_count) != int(episode.num_containers):
        raise AcceptanceStop("rollout terminated without retrieving every container")
    if eg_audit is not None:
        eg_audit.finish()
    if provider.event_count != len(reveal_rows):
        raise AcceptanceStop("provider reveal-history count differs from private reveal rows")
    event_signature = tuple(
        (int(row["event_index"]), bool(row["hit_private"]), float(row["q_t_before_reveal"]))
        for row in reveal_rows
    )
    row = {
        "rollout_id_evaluator_only": rollout_id,
        "fingerprint": str(condition["fingerprint"]),
        "geometry_id": str(condition["geometry_id"]),
        "family": str(condition["family"]),
        "window_size": int(condition["window_size"]),
        "repeat": int(condition["repeat"]),
        "method": method,
        "num_containers": int(episode.num_containers),
        "forecast_sha256": str(condition["forecast_sha256"]),
        "complete": True,
        "all_actions_legal": legal_actions == decision_count,
        "decision_count": decision_count,
        "relocation_count": int(environment.state.relocation_count),
        "R_raw_termination_checksum_only": int(environment.state.relocation_count),
        "retrieval_count": int(environment.state.retrieval_count),
        "reveal_event_count": len(reveal_rows),
        "zero_reveal_moves_with_unchanged_q": zero_reveal_moves,
        "provider_instance_private_to_rollout": True,
    }
    writers["rollouts"].write(row)
    return row, event_signature


def _verify_prepared_bank(config: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    manifest = _read_json(OUTPUT / "manifest.json")
    snapshot = _read_json(OUTPUT / "bank_snapshot.json")
    private_rows = [json.loads(line) for line in (OUTPUT / "forecast_bindings_evaluator_only.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    if manifest.get("stage") != "PREPARED_NOT_STARTED" or snapshot.get("status") != "PREPARED_AWAITING_ONE_SHOT_BANK":
        raise AcceptanceStop("manifest/snapshot is not in prepared state")
    if (OUTPUT / "bank_snapshot.sha256").read_text(encoding="ascii").strip() != _sha256_file(OUTPUT / "bank_snapshot.json"):
        raise AcceptanceStop("prebank snapshot hash receipt differs")
    if (OUTPUT / "manifest.sha256").read_text(encoding="ascii").strip() != _sha256_file(OUTPUT / "manifest.json"):
        raise AcceptanceStop("bank manifest hash receipt differs")
    current_sources = _current_source_hashes(config)
    if current_sources != manifest.get("source_hashes") or current_sources != snapshot.get("source_hashes"):
        raise AcceptanceStop("source/config/code/test binding changed after prebank snapshot")
    if _sha256_file(OUTPUT / "forecast_bindings_evaluator_only.jsonl") != manifest.get("forecast_bindings_evaluator_only_sha256"):
        raise AcceptanceStop("private forecast binding changed after manifest freeze")
    if len(private_rows) != 20:
        raise AcceptanceStop("private forecast binding count differs from 20 conditions")
    if len(manifest.get("conditions", [])) != 20:
        raise AcceptanceStop("bank manifest does not contain exactly 20 condition keys")
    expected_keys = [
        (str(row["fingerprint"]), "time_window_shuffle", int(row["window_size"]), int(row["repeat"]))
        for row in manifest["conditions"]
    ]
    if len(set(expected_keys)) != 20 or expected_keys != sorted(expected_keys, key=lambda x: (x[0], x[2], x[3])):
        raise AcceptanceStop("exact condition manifest is duplicated or reordered")
    private_by_key = {
        (str(row["fingerprint"]), str(row["family"]), int(row["window_size"]), int(row["repeat"])): row
        for row in private_rows
    }
    if set(private_by_key) != set(expected_keys):
        raise AcceptanceStop("manifest and private forecast bindings have different keys")
    return manifest, snapshot, private_by_key


def run_bank() -> dict[str, Any]:
    config = _load_config()
    manifest, snapshot, private_by_key = _verify_prepared_bank(config)
    if any((OUTPUT / name).exists() for name in BANK_RESULT_NAMES):
        raise FileExistsError("one-shot bank or a prior attempt already wrote raw outputs; refusing resume/retry")
    runtime = _runtime_imports()
    gp_rule, gp_identity = _load_accepted_gp_rule(config)
    if gp_identity != manifest.get("accepted_gp_identity"):
        raise AcceptanceStop("accepted GP rule identity changed after prepare")
    selected = _select_exact_layouts(config, runtime)
    fingerprint_fn = runtime["label_invariant_fingerprint"]
    episode_by_fp = {str(fingerprint_fn(episode)): episode for episode in selected}
    binding_rows = list(private_by_key.values())
    binding_rows.sort(key=lambda row: (str(row["fingerprint"]), int(row["window_size"]), int(row["repeat"])))
    writers: dict[str, _JsonlWriter] = {}
    started_utc = datetime.now(timezone.utc).isoformat()
    started = time.perf_counter()
    deadline = started + float(config["acceptance"]["total_wall_seconds"])
    summary: dict[str, Any] = {
        "schema": "plan046_time_window_acceptance_status_v1",
        "protocol_id": config["protocol_id"],
        "status": "RUNNING",
        "started_utc": started_utc,
        "expected_condition_count": 20,
        "expected_rollout_count": 300,
        "completed_rollout_count": 0,
        "completed_condition_count": 0,
        "source_hashes_before": manifest["source_hashes"],
        "preimplementation_source_manifest_sha256": config["preimplementation_source_manifest_sha256"],
        "prebank_snapshot_sha256": _sha256_file(OUTPUT / "bank_snapshot.json"),
        "manifest_sha256": _sha256_file(OUTPUT / "manifest.json"),
        "accepted_gp_identity": gp_identity,
        "performance_ranking_or_gate": False,
        "selection_or_training_performed": False,
    }
    try:
        writers = {
            "rollouts": _JsonlWriter(OUTPUT / "rollouts.jsonl"),
            "decisions": _JsonlWriter(OUTPUT / "decisions.jsonl"),
            "transitions": _JsonlWriter(OUTPUT / "private_transitions.jsonl"),
            "reveals": _JsonlWriter(OUTPUT / "private_reveals.jsonl"),
            "eg_groups": _JsonlWriter(OUTPUT / "eg_group_trace.jsonl"),
        }
        for binding in binding_rows:
            if time.perf_counter() >= deadline:
                raise TimeoutError("PLAN-046 one-shot acceptance exceeded 600-second wall ceiling")
            key = (str(binding["fingerprint"]), str(binding["family"]), int(binding["window_size"]), int(binding["repeat"]))
            condition = next(
                row for row in manifest["conditions"]
                if (str(row["fingerprint"]), str(row["family"]), int(row["window_size"]), int(row["repeat"])) == key
            )
            episode = episode_by_fp[str(binding["fingerprint"])]
            condition_signatures: dict[str, tuple[tuple[int, bool, float], ...]] = {}
            for method in METHODS:
                row, signature = _run_one_rollout(
                    episode, condition, binding, method,
                    runtime=runtime, gp_rule=gp_rule, writers=writers, deadline=deadline,
                )
                condition_signatures[method] = signature
                summary["completed_rollout_count"] += 1
                if not row["complete"] or not row["all_actions_legal"]:
                    raise AcceptanceStop("rollout completeness or action legality check failed")
            if len(set(condition_signatures.values())) != 1:
                raise AcceptanceStop(f"same condition reveal/hit/q history differs across methods: {key}")
            summary["completed_condition_count"] += 1
            if int(summary["completed_rollout_count"]) != int(summary["completed_condition_count"]) * 15:
                raise AcceptanceStop("completed rollout count is not 15 per completed condition")
        for writer in writers.values():
            writer.close()
        writers = {}
        if int(summary["completed_rollout_count"]) != 300 or int(summary["completed_condition_count"]) != 20:
            raise AcceptanceStop("bank stopped before exact registered count")
        after_sources = _current_source_hashes(config)
        if after_sources != manifest["source_hashes"]:
            raise AcceptanceStop("source/config/code/test binding changed during the one-shot bank")
        summary.update({
            "status": "PASS_CORRECTNESS_ACCEPTANCE_ONLY",
            "finished_utc": datetime.now(timezone.utc).isoformat(),
            "elapsed_seconds": time.perf_counter() - started,
            "source_hashes_after": after_sources,
            "source_hashes_unchanged": True,
            "total_raw_rollout_rows": 300,
            "performance_statistics_or_ranking": False,
        })
        _write_plot()
        output_names = (
            "PREIMPLEMENTATION_SOURCE_BINDINGS.json",
            "manifest.json", "manifest.sha256", "bank_snapshot.json", "bank_snapshot.sha256",
            "forecast_bindings_evaluator_only.jsonl", "rollouts.jsonl", "decisions.jsonl",
            "private_transitions.jsonl", "private_reveals.jsonl", "test_receipt.json",
            "eg_group_trace.jsonl", "test_stdout.txt", "test_stderr.txt",
            "plot_time_window_correctness.py", "noise_correctness.png", "noise_correctness.pdf",
        )
        summary["artifact_sha256"] = {
            name: _sha256_file(OUTPUT / name)
            for name in output_names
            if (OUTPUT / name).is_file()
        }
        _write_report(config, summary)
        summary["artifact_sha256"]["REPORT.md"] = _sha256_file(OUTPUT / "REPORT.md")
        _write_json_new(OUTPUT / "RUN_STATUS.json", summary)
        return summary
    except BaseException as exc:
        for writer in writers.values():
            if not writer.handle.closed:
                writer.close()
        summary.update({
            "status": "NOT_EVALUATED_INCOMPLETE",
            "failure": {"type": type(exc).__name__, "message": str(exc)},
            "finished_utc": datetime.now(timezone.utc).isoformat(),
            "elapsed_seconds": time.perf_counter() - started,
            "completed_rollout_count": int(summary["completed_rollout_count"]),
            "completed_condition_count": int(summary["completed_condition_count"]),
            "no_resume_or_retry": True,
        })
        if not (OUTPUT / "RUN_STATUS.json").exists():
            _write_json_new(OUTPUT / "RUN_STATUS.json", summary)
        if not (OUTPUT / "REPORT.md").exists():
            _write_failure_report(summary)
        raise


def _write_report(config: Mapping[str, Any], status: Mapping[str, Any]) -> None:
    lines = [
        "# PLAN-046 time-window shuffle acceptance",
        "",
        f"Status: `{status['status']}`.",
        "",
        "This is a correctness and public-information acceptance bank. It is not performance evidence; no method ranking, contrast, or aggregate relocation comparison is reported.",
        "",
        "- Conditions: 20 (two frozen select20 layouts × B={1,2,5,10,20} × repeats {0,1}).",
        "- Complete rollouts: 300 (15 registered methods per condition).",
        "- Runtime: " + f"{float(status['elapsed_seconds']):.3f} seconds.",
        "- Physics / legal actions / causal reveals / q history: passed across every completed rollout.",
        "- Shared fixed F: one evaluator-only forecast binding per condition; each rollout used a fresh provider.",
        "- Frozen GP identity: `GP-H20-C-run25`; public censor-to-22 slow bridge; no selection or training was performed.",
        "- Historical EM and GP/ERI/EG streams remain separate and are owned by their respective executors.",
        "",
        "Artifacts: [manifest.json](manifest.json), [rollouts.jsonl](rollouts.jsonl), [decisions.jsonl](decisions.jsonl), [private_transitions.jsonl](private_transitions.jsonl), [private_reveals.jsonl](private_reveals.jsonl), [eg_group_trace.jsonl](eg_group_trace.jsonl), and [forecast_bindings_evaluator_only.jsonl](forecast_bindings_evaluator_only.jsonl).",
        "The public decision rows contain only a rollout handle, method, step, public frame, and action/candidate output; raw IDs, F, seed, B, family, and private truth are evaluator-only.",
        "",
        "See [noise_correctness.png](noise_correctness.png) / [noise_correctness.pdf](noise_correctness.pdf) for the position-window generator illustration and [plot_time_window_correctness.py](plot_time_window_correctness.py) for its reproduction script.",
        "The targeted test command and real stdout/stderr/exit/elapsed receipt are in [test_receipt.json](test_receipt.json), [test_stdout.txt](test_stdout.txt), and [test_stderr.txt](test_stderr.txt). The one-shot bank command receipt and process stdout/stderr are in [acceptance_execution_receipt.json](acceptance_execution_receipt.json), [acceptance_stdout.txt](acceptance_stdout.txt), and [acceptance_stderr.txt](acceptance_stderr.txt); source/input before/after hashes are in [bank_snapshot.json](bank_snapshot.json) and [RUN_STATUS.json](RUN_STATUS.json).",
    ]
    with (OUTPUT / "REPORT.md").open("x", encoding="utf-8", newline="\n") as handle:
        handle.write("\n".join(lines) + "\n")


def _write_failure_report(status: Mapping[str, Any]) -> None:
    lines = [
        "# PLAN-046 time-window shuffle acceptance",
        "",
        f"Status: `{status.get('status', 'NOT_EVALUATED_INCOMPLETE')}`.",
        "",
        "The one-shot correctness bank stopped before all registered work completed. This output is not performance evidence and no method ranking was produced.",
        "",
        f"Completed conditions: {int(status.get('completed_condition_count', 0))}/20.",
        f"Completed full rollouts: {int(status.get('completed_rollout_count', 0))}/300.",
        f"Failure: `{status.get('failure', {})}`.",
        "",
        "Partial raw JSONL files and the failure receipt are preserved. The bank cannot be resumed or retried.",
    ]
    with (OUTPUT / "REPORT.md").open("x", encoding="utf-8", newline="\n") as handle:
        handle.write("\n".join(lines) + "\n")


def _write_plot() -> None:
    script = PLOT_PATH
    if not script.is_file():
        raise AcceptanceStop("registered reproducible plot script missing")
    namespace = {"__name__": "__main__", "__file__": str(script)}
    exec(compile(script.read_text(encoding="utf-8"), str(script), "exec"), namespace)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--prepare", action="store_true", help="freeze exact manifest/F bindings; does not run trajectories")
    action.add_argument("--run", action="store_true", help="run the one-shot 300-rollout bank; never resumes")
    args = parser.parse_args(argv)
    try:
        result = prepare_bank() if args.prepare else run_bank()
    except Exception as exc:
        print(json.dumps({"status": "FAIL_STOP", "type": type(exc).__name__, "message": str(exc)}, sort_keys=True))
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

