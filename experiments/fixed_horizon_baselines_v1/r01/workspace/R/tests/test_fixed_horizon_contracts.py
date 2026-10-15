"""Synthetic-only contracts for fixed-H policies, scheduling, and release gates."""
from __future__ import annotations

import hashlib
import gzip
import importlib
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import sys

import numpy as np

WORKSPACE = Path(__file__).resolve().parents[2]
for _path in (WORKSPACE, WORKSPACE / "R", WORKSPACE / "EM", WORKSPACE / "Mijn_RL"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from R.experiments import fixed_horizon_integrity as integrity
from R.experiments import fixed_horizon_analysis as analysis
from R.experiments.fixed_horizon_schedule import (
    bank_keys, build_m1_bridge_keys, m1_key_digest, parse_fixed_bank_key, remaining_bank_keys,
    registered_schedule, smoke_keys,
)
from R.experiments import fixed_horizon_runner as runner
from R.experiments import fixed_horizon_stage_process as stage_process
from R.experiments import fixed_horizon_trace_audit_entry as audit_entry
from R.experiments.fixed_horizon_trace_audit import (
    AuditFailure, candidate_policy_check, independent_h_prefix,
    replay_fixed_h_proposer,
)
from R.scrp.fixed_horizon_policy import ALGORITHMS, NEW_HORIZONS, FixedHorizonPolicy
from R.scrp.r_five_versions_actor import (
    FiveRPolicy, RidgeScorer, public_frame_record, strict_public_frame_from_record,
)


def _frame():
    return strict_public_frame_from_record({
        "rank_tensor": [[1, 2, 0], [3, 0, 0], [0, 0, 0]],
        "occupied_mask": [[True, True, False], [True, False, False], [False, False, False]],
        "legal_destination_mask": [False, True, True],
        "legal_destinations": [1, 2], "max_tiers": 3,
        "history_hits": [False, False, False], "q_prior": 5183 / 9920,
        "q_t": 0.25, "retrieval_count": 0, "relocation_count": 0,
        "crane_position": 0, "crane_time": 0.0, "r_order_summary": None,
    })


def _eg_frame():
    return strict_public_frame_from_record({
        "rank_tensor": [[1, 2, 3], [4, 0, 0], [0, 0, 0]],
        "occupied_mask": [[True, True, True], [True, False, False], [False, False, False]],
        "legal_destination_mask": [False, True, True],
        "legal_destinations": [1, 2], "max_tiers": 3,
        "history_hits": [False, False, False], "q_prior": 5183 / 9920,
        "q_t": 0.25, "retrieval_count": 0, "relocation_count": 0,
        "crane_position": 0, "crane_time": 0.0, "r_order_summary": None,
    })


def _original_actor_prefix(frame, horizon):
    return frame if int(horizon) == 20 else independent_h_prefix(frame, int(horizon))


def _full_frame_after_move(frame, action):
    row = public_frame_record(frame)
    ranks = np.asarray(row["rank_tensor"], dtype=np.int32)
    occupied = np.asarray(row["occupied_mask"], dtype=np.bool_)
    source = int(np.argwhere(occupied & (ranks == 1))[0, 0])
    source_height = int(occupied[source].sum())
    destination_height = int(occupied[action].sum())
    source_position = source + 1
    destination_position = int(action) + 1
    previous_position = row["crane_position"]
    crane_time = float(row["crane_time"])
    if previous_position is not None:
        crane_time += abs(int(previous_position) - source_position) * 1.2
    crane_time += abs(source_position - destination_position) * 1.2 + 30.0
    moved_rank = int(ranks[source, source_height - 1])
    ranks[source, source_height - 1] = 0
    occupied[source, source_height - 1] = False
    ranks[action, destination_height] = moved_rank
    occupied[action, destination_height] = True
    heights = occupied.sum(axis=1)
    legal_mask = np.asarray([index != source and heights[index] < frame.max_tiers
                             for index in range(len(heights))], dtype=np.bool_)
    row["rank_tensor"] = ranks.tolist()
    row["occupied_mask"] = occupied.tolist()
    row["legal_destination_mask"] = legal_mask.tolist()
    row["legal_destinations"] = np.flatnonzero(legal_mask).tolist()
    row["relocation_count"] = int(frame.relocation_count) + 1
    row["crane_position"] = destination_position
    row["crane_time"] = crane_time
    return strict_public_frame_from_record(row)


class _CacheActor:
    def __init__(self):
        self.cached_moves = 0
        self._plan = None
        self.calls = 0

    def select_destination(self, _frame):
        self.calls += 1
        if self._plan is None:
            self._plan = (1, 2)
            return 1
        self.cached_moves += 1
        self._plan = None
        return 2

    def reset(self):
        self.cached_moves = 0
        self._plan = None
        self.calls = 0


class FixedHorizonContractTests(unittest.TestCase):
    def test_ridge_eg_real_actor_uses_native_sequence_cache_trace_schema(self):
        frame_record = public_frame_record(_eg_frame())
        frame_record["r_order_summary"] = {
            "q": 0.25, "displacement": 0.0, "coverage": 0.0,
            "retrieval_fraction": 0.0, "records": [],
        }
        frame = strict_public_frame_from_record(frame_record)
        runtime = {"truncate_public_frame": _original_actor_prefix,
                   "eri_eg": importlib.import_module("R.scrp.eri_eg_public")}
        coefficients = (0.0,) * 74
        actor = FiveRPolicy("R_UTILITY_RIDGE", "EG", runtime, {},
                            ridge_scorer=RidgeScorer(coefficients))
        row = {"method": "R_UTILITY_RIDGE", "algorithm": "EG"}

        first_action, first_trace = actor.select(frame)
        candidate_policy_check(row, {
            "own_action": first_action,
            "public_frame_before": public_frame_record(frame),
            "actor_trace": first_trace,
        }, coefficients, ())
        self.assertTrue(first_trace["group_start"])
        self.assertIn("committed_sequence", first_trace)
        self.assertNotIn("requested_h", first_trace)

        after_first = _full_frame_after_move(frame, first_action)
        second_action, second_trace = actor.select(after_first)
        candidate_policy_check(row, {
            "own_action": second_action,
            "public_frame_before": public_frame_record(after_first),
            "actor_trace": second_trace,
        }, coefficients, ())
        self.assertTrue(second_trace["cached_move"])
        self.assertIn("sequence_index", second_trace)
        self.assertNotIn("candidates", second_trace)

    def test_native20_eg_real_actor_uses_fixed20_cache_trace_schema(self):
        frame = _eg_frame()
        runtime = {"truncate_public_frame": _original_actor_prefix,
                   "eri_eg": importlib.import_module("R.scrp.eri_eg_public")}
        actor = FiveRPolicy("NATIVE20", "EG", runtime, {})
        row = {"method": "NATIVE20", "algorithm": "EG"}

        first_action, first_trace = actor.select(frame)
        candidate_policy_check(row, {
            "own_action": first_action,
            "public_frame_before": public_frame_record(frame),
            "actor_trace": first_trace,
        }, (), ())
        self.assertEqual(first_trace["policy"], "FIXED20")
        self.assertEqual(first_trace["requested_h"], 20)
        self.assertNotIn("committed_sequence", first_trace)

        after_first = _full_frame_after_move(frame, first_action)
        second_action, second_trace = actor.select(after_first)
        candidate_policy_check(row, {
            "own_action": second_action,
            "public_frame_before": public_frame_record(after_first),
            "actor_trace": second_trace,
        }, (), ())
        self.assertFalse(second_trace["group_start"])
        self.assertEqual(second_trace["cached_moves"], 1)
        self.assertNotIn("cached_move", second_trace)

    def test_plan052_chain_minmax_bridge_policies_keep_their_native_trace_schemas(self):
        from Chain.chain_public import propose as chain_propose
        from Minmax.minmax_public import propose as minmax_propose
        from R.scrp.plan052_policies import Plan052Policy

        for solver, proposer in (("Chain", chain_propose), ("Minmax", minmax_propose)):
            actor = Plan052Policy(solver, proposer, ridge_scorer=lambda _features: 0.0)
            frame = _frame()
            for method in ("R_UTILITY_RIDGE", "NATIVE20"):
                action, trace = actor.select(frame, method)
                candidate_policy_check({"method": method, "algorithm": solver}, {
                    "own_action": action,
                    "public_frame_before": public_frame_record(frame),
                    "actor_trace": trace,
                }, (), ())
                self.assertNotIn("group_start", trace)
                self.assertNotIn("cached_moves", trace)

    def test_prefix_rule_keeps_empty_zero_and_censors_out_of_h_occupied(self):
        frame = _frame()
        projected = independent_h_prefix(frame, 0)
        ranks = np.asarray(projected.observation.rank_tensor)
        occupied = np.asarray(projected.observation.occupied_mask)
        self.assertEqual(int(ranks[0, 0]), 1)
        self.assertEqual(int(ranks[0, 1]), 22)
        self.assertEqual(int(ranks[1, 0]), 22)
        self.assertEqual(int(ranks[0, 2]), 0)
        self.assertEqual(int(ranks[2, 2]), 0)
        self.assertTrue(bool(occupied[0, 1]))
        self.assertFalse(bool(occupied[0, 2]))

    def test_independent_prefix_schema_matches_runtime_and_six_solver_interfaces(self):
        from Chain.chain_public import propose as chain_propose
        from Minmax.minmax_public import propose as minmax_propose
        from R.scrp.cross_solver_reliability_r import truncate_public_frame

        eri_module = importlib.import_module("R.scrp.eri_eg_public")
        em_module = importlib.import_module("scrp.noise_aware_em")
        runtime = {
            "truncate_public_frame": truncate_public_frame,
            "eri_eg": eri_module,
            "em_select": em_module.select_em_destination,
            "eri_select": eri_module.select_eri20_destination,
        }
        base_frame = _frame()
        decorated_row = public_frame_record(base_frame)
        decorated_row["r_order_summary"] = {
            "q": 0.25, "displacement": 0.0, "coverage": 0.0,
            "retrieval_fraction": 0.0, "records": [],
        }
        decorated_frame = strict_public_frame_from_record(decorated_row)
        self.assertTrue(hasattr(decorated_frame.observation, "r_order_summary"))
        base_observation_type = type(base_frame.observation)

        for frame_index, frame in enumerate((base_frame, decorated_frame)):
            for horizon in NEW_HORIZONS:
                prefix = independent_h_prefix(frame, horizon)
                actual = truncate_public_frame(frame, horizon)
                self.assertEqual(public_frame_record(prefix), public_frame_record(actual))
                self.assertIs(type(prefix.observation), base_observation_type)
                self.assertFalse(hasattr(prefix.observation, "r_order_summary"))
                self.assertEqual(
                    np.asarray(prefix.observation.rank_tensor).tolist(),
                    np.asarray(actual.observation.rank_tensor).tolist(),
                )

                for algorithm in ("EM", "ERI", "Chain", "Minmax"):
                    action, trace = replay_fixed_h_proposer(algorithm, horizon, frame, runtime, {})
                    self.assertIn(action, frame.legal_destinations, (frame_index, horizon, algorithm))
                    if trace is not None:
                        self.assertEqual(trace["requested_h"], horizon)

                gp_seen = []
                def fake_gp(candidate_frame, *, requested_h, solver, frozen_solver):
                    gp_seen.append((requested_h, solver, frozen_solver))
                    self.assertFalse(hasattr(candidate_frame.observation, "r_order_summary"))
                    return SimpleNamespace(destination=candidate_frame.legal_destinations[0])
                with patch("R.scrp.cross_solver_reliability_r._run_one_horizon", side_effect=fake_gp):
                    gp_action, _trace = replay_fixed_h_proposer("GP", horizon, frame, runtime, {"synthetic": True})
                self.assertIn(gp_action, frame.legal_destinations)
                self.assertEqual(gp_seen, [(horizon, "GP", {"synthetic": True})])

                eg_prefix = independent_h_prefix(frame, horizon)
                eg_plan = eri_module.build_eg20_plan(eg_prefix).destinations_top_down
                self.assertTrue(eg_plan)
                self.assertIn(eg_plan[0], frame.legal_destinations)

                eg_policy = FixedHorizonPolicy("EG", horizon, runtime, {})
                eg_action, eg_trace = eg_policy.select(frame)
                self.assertEqual(eg_action, eg_plan[0])
                self.assertTrue(eg_trace["group_start"])
                self.assertEqual(eg_trace["requested_h"], horizon)

    def test_fixed_eg_audit_replays_underfull_source_padding_and_cached_group(self):
        from R.experiments import fixed_horizon_trace_audit as core
        from R.scrp.cross_solver_reliability_r import truncate_public_frame
        from R.scrp.r_five_versions_actor import strict_public_frame_from_record

        algorithm = importlib.import_module("R.scrp.eri_eg_public")
        em_module = importlib.import_module("scrp.noise_aware_em")
        runtime = {
            "truncate_public_frame": truncate_public_frame,
            "eri_eg": algorithm,
            "em_select": em_module.select_em_destination,
            "eri_select": algorithm.select_eri20_destination,
        }
        true_sequence = [1, 2, 3]
        forecast = [2, 3]
        initial = [[1, 3, 2], [], []]
        fingerprint = "synthetic-eg-underfull-source"
        key = f"{fingerprint}|CLEAN|EG|FIXED_H0"
        csv_row = {
            "trajectory_key": key, "layout_fingerprint": fingerprint,
            "condition_id": "CLEAN", "algorithm": "EG", "method": "FIXED_H0",
            "forecast_seed_evaluator_only": "", "decision_count": "2",
            "complete": "true", "N": "3", "R_raw": "2", "R_over_N": str(2 / 3),
        }
        start = {
            "layout_fingerprint": fingerprint,
            "true_retrieval_sequence_private": true_sequence,
            "fixed_forecast_private": forecast,
            "initial_stacks_private": initial,
        }
        evaluator_rows = [{
            "trajectory_key": key, "record_type": "forecast_binding",
            "forecast_sha256_evaluator_only": hashlib.sha256(core.canonical(forecast)).hexdigest(),
            "layout_fingerprint": fingerprint, "forecast_seed_evaluator_only": None,
            "condition_evaluator_only": {"condition_id": "CLEAN"},
        }]
        state, retrieved, crane_position, crane_time = core.replay_initial(initial, true_sequence)
        hits = core.forecast_hits(true_sequence, forecast, retrieved)
        relocations = 0
        proxy_records = []
        policy = FixedHorizonPolicy("EG", 0, runtime, {})
        steps = []
        physical_records = []
        for index in range(2):
            frame = core.frame_from_private(
                state, true_sequence, forecast, retrieved, hits,
                crane_position, crane_time, relocations, 4,
            )
            self.assertIsNotNone(frame)
            before = core.decorate_history_summary(frame, proxy_records)
            action, actor_trace = policy.select(strict_public_frame_from_record(before))
            before_retrievals = retrieved
            physical_before = [list(stack) for stack in state]
            retrieval_before, relocation_before = retrieved, relocations

            target = int(true_sequence[retrieved])
            source, target_tier = core.locate(state, target)
            self.assertLess(target_tier, len(state[source]) - 1)
            blocker = state[source].pop()
            source_position, destination_position = source + 1, action + 1
            if crane_position is not None:
                crane_time += abs(crane_position - source_position) * core.CRANE_TRAVEL
            crane_time += abs(source_position - destination_position) * core.CRANE_TRAVEL + core.PICK_DROP
            crane_position = destination_position
            state[action].append(blocker)
            relocations += 1
            while retrieved < len(true_sequence):
                current = int(true_sequence[retrieved])
                current_source, current_tier = core.locate(state, current)
                if current_tier != len(state[current_source]) - 1:
                    break
                source_position = current_source + 1
                if crane_position is not None:
                    crane_time += abs(crane_position - source_position) * core.CRANE_TRAVEL
                crane_time += source_position * core.CRANE_TRAVEL + core.PICK_DROP
                crane_position = 0
                state[current_source].pop()
                retrieved += 1
            hits = core.forecast_hits(true_sequence, forecast, retrieved)
            next_frame = core.frame_from_private(
                state, true_sequence, forecast, retrieved, hits,
                crane_position, crane_time, relocations, 4,
            )
            terminal = retrieved == len(true_sequence)
            after = None if terminal else next_frame
            steps.append({
                "trajectory_key": key, "decision_index": index, "algorithm": "EG",
                "method": "FIXED_H0", "own_action": action,
                "actor_trace": actor_trace, "public_frame_before": before,
                "public_frame_after": after, "terminal_after_action": terminal,
            })
            physical_records.append({
                "trajectory_key": key, "record_type": "physical_step", "decision_index": index,
                "physical_stacks_before_private": physical_before,
                "retrieval_count_before": retrieval_before,
                "relocation_count_before": relocation_before,
                "physical_stacks_after_private": [list(stack) for stack in state],
                "retrieval_count_after": retrieved,
                "relocation_count_after": relocations,
                "provider_history_hits_private": hits,
            })
            if not terminal:
                proxy = core.public_history_record(before, action, after)
                if proxy is not None:
                    proxy_records.append(float(proxy))
            if index == 0:
                self.assertEqual(retrieved, before_retrievals)
                self.assertFalse(terminal)
            else:
                self.assertGreater(retrieved, before_retrievals)
                self.assertTrue(terminal)

        self.assertEqual(len(initial[0]), 3)
        self.assertLess(len(initial[0]), 4)  # padded rank_tensor cells are zero, not boxes
        self.assertEqual(steps[0]["actor_trace"]["group_start"], True)
        self.assertEqual(steps[1]["actor_trace"]["group_start"], False)
        self.assertEqual(steps[1]["actor_trace"]["cached_moves"], 1)
        native_eg_audit = {}
        replayed_steps = core.replay_trajectory(
            csv_row, start, steps, evaluator_rows + physical_records,
            (), (), 4, native_eg_audit, runtime, {},
        )
        self.assertEqual(replayed_steps, 2)
        self.assertEqual(native_eg_audit["fixed_h_group_count"], 1)
        self.assertEqual(native_eg_audit["fixed_h_action_count"], 2)
        self.assertEqual(native_eg_audit["fixed_h_cached_move_count"], 1)

    def test_policy_uses_exact_single_registered_h_for_all_solvers(self):
        frame = _frame()
        for algorithm in ALGORITHMS:
            for horizon in NEW_HORIZONS:
                calls = []
                if algorithm in ("Chain", "Minmax"):
                    policy = FixedHorizonPolicy(
                        algorithm, horizon, {}, {},
                        proposer=lambda _frame, h, calls=calls: (calls.append(h) or 1, {"requested_h": h}),
                    )
                    action, trace = policy.select(frame)
                    self.assertEqual(calls, [horizon])
                else:
                    class Actor:
                        def __init__(self, alg, h, *_args):
                            self.alg, self.h = alg, h
                            self.cached_moves = 0
                            self.is_group_start = True
                            calls.append(("init", alg, h))
                        def select(self, _frame):
                            calls.append(("propose", self.alg, self.h))
                            return 1, {"requested_h": self.h, "group_start": True, "cached_moves": 0}
                        def reset(self):
                            pass
                    with patch("R.scrp.r_order_actor.FixedHorizonActor", Actor):
                        policy = FixedHorizonPolicy(algorithm, horizon, {
                            "truncate_public_frame": lambda item, _h: item,
                            "eri_eg": object(), "em_select": object(), "eri_select": object(),
                        }, {})
                        action, trace = policy.select(frame)
                    self.assertEqual([x for x in calls if x[0] == "propose"], [("propose", algorithm, horizon)])
                self.assertIn(action, frame.legal_destinations)
                self.assertEqual(trace["requested_h"], horizon)
                self.assertTrue(trace["fixed_h_only"])
                self.assertEqual(trace["policy"], f"FIXED_H{horizon}")

    def test_eg_wrapper_keeps_native_cache_and_calls_proposer_once_per_group(self):
        actor = _CacheActor()
        runtime = {"truncate_public_frame": lambda item, _h: item,
                   "eri_eg": SimpleNamespace(EG20Actor=lambda: actor),
                   "em_select": None, "eri_select": None}
        policy = FixedHorizonPolicy("EG", 5, runtime, {})
        first, first_trace = policy.select(_frame())
        second, second_trace = policy.select(_frame())
        self.assertEqual((first, second), (1, 2))
        self.assertEqual(policy.proposal_calls, 1)
        self.assertEqual(actor.calls, 2)
        self.assertTrue(first_trace["group_start"])
        self.assertFalse(second_trace["group_start"])
        self.assertNotIn("committed_sequence", first_trace)
        self.assertEqual(second_trace["cached_moves"], 1)
        policy.reset()
        self.assertEqual(policy.proposal_calls, 0)
        self.assertEqual(policy.cached_moves, 0)

    def test_native_eg_reconstructs_and_commits_the_complete_fixed_h_group(self):
        from R.scrp.eri_eg_public import build_eg20_plan
        from R.experiments.fixed_horizon_trace_audit import independent_h_prefix
        frame = _eg_frame()
        runtime = {"truncate_public_frame": independent_h_prefix,
                   "eri_eg": __import__("R.scrp.eri_eg_public", fromlist=["EG20Actor"]),
                   "em_select": None, "eri_select": None}
        expected = build_eg20_plan(independent_h_prefix(frame, 0)).destinations_top_down
        self.assertEqual(len(expected), 2)
        policy = FixedHorizonPolicy("EG", 0, runtime, {})
        first, first_trace = policy.select(frame)
        self.assertEqual(first, expected[0])
        second_frame = _full_frame_after_move(frame, first)
        second, second_trace = policy.select(second_frame)
        self.assertEqual(second, expected[1])
        self.assertTrue(first_trace["group_start"])
        self.assertFalse(second_trace["group_start"])
        self.assertEqual(second_trace["cached_moves"], 1)
        self.assertEqual(policy.proposal_calls, 1)
        self.assertNotIn("committed_sequence", first_trace)

    def test_em_and_eri_audit_replay_use_the_registered_h_once(self):
        calls = []
        runtime = {
            "em_select": lambda observation, *, max_tiers, h: (calls.append(("EM", h)) or 1),
            "eri_select": lambda frame: (calls.append(("ERI", 5)) or 2),
        }
        em_action, _ = replay_fixed_h_proposer("EM", 0, _frame(), runtime, {})
        eri_action, _ = replay_fixed_h_proposer("ERI", 5, _frame(), runtime, {})
        self.assertEqual((em_action, eri_action), (1, 2))
        self.assertEqual(calls, [("EM", 0), ("ERI", 5)])

    def test_minmax_audit_replay_is_legal_and_keeps_requested_h(self):
        action, trace = replay_fixed_h_proposer("Minmax", 2, _frame(), {}, {})
        self.assertIn(action, _frame().legal_destinations)
        self.assertEqual(trace["solver"], "Minmax")
        self.assertEqual(trace["requested_h"], 2)

    def test_gp_replay_uses_public_slow_horizon_adapter(self):
        seen = []
        candidate = SimpleNamespace(destination=1)
        with patch("R.scrp.cross_solver_reliability_r._run_one_horizon",
                   side_effect=lambda frame, **kwargs: (seen.append(kwargs) or candidate)):
            action, _trace = replay_fixed_h_proposer("GP", 2, _frame(), {}, {"frozen": True})
        self.assertEqual(action, 1)
        self.assertEqual(seen, [{"requested_h": 2, "solver": "GP", "frozen_solver": {"frozen": True}}])

    def test_chain_h0_preserves_same_frame_minmax_fallback(self):
        action, trace = replay_fixed_h_proposer("Chain", 0, _frame(), {}, {})
        self.assertIn(action, _frame().legal_destinations)
        self.assertEqual(trace["requested_h"], 0)
        self.assertEqual(trace["solver"], "Chain")
        self.assertEqual(trace["fallback_reason"], "unknown_current_rank")
        self.assertEqual(trace["rule"], "minmax_fallback")

    def test_fixed_h_action_trace_mismatch_fails_closed(self):
        row = {"method": "FIXED_H2", "algorithm": "EM"}
        trace = {"own_action": 1,
                 "public_frame_before": {"legal_destinations": [1, 2]},
                 "actor_trace": {"policy": "FIXED_H2", "solver": "EM", "requested_h": 2, "fixed_h_only": True}}
        candidate_policy_check(row, trace, (), ())
        trace["actor_trace"]["requested_h"] = 5
        with self.assertRaises(AuditFailure):
            candidate_policy_check(row, trace, (), ())

    def test_stage_schedules_have_exact_registered_counts_and_disjoint_remainder(self):
        bridge = build_m1_bridge_keys()
        smoke, all_bank, remaining = smoke_keys(), bank_keys(), remaining_bank_keys()
        self.assertEqual((len(bridge), len(smoke), len(all_bank), len(remaining)), (24, 96, 10080, 9984))
        self.assertTrue(smoke.isdisjoint(remaining))
        self.assertEqual(smoke | remaining, all_bank)
        self.assertEqual(len(bridge | smoke), 120)
        self.assertEqual(len(m1_key_digest()), 64)
        parsed = [parse_fixed_bank_key(key) for key in all_bank]
        self.assertEqual({condition_id for _, condition_id, _, _ in parsed},
                         {str(row["condition_id"]) for row in registered_schedule()[1]})
        self.assertEqual(len({condition_id for _, condition_id, _, _ in parsed}), 21)
        self.assertEqual({method for _, _, _, method in parsed}, {f"FIXED_H{h}" for h in NEW_HORIZONS})

    def test_failed_science_audit_and_analysis_attempts_all_consume_budget(self):
        with tempfile.TemporaryDirectory() as temp:
            log_root = Path(temp)
            rows = (
                ("M1", "science", 11.0, "M1"),
                ("M2", "science", 17.0, "M2"),
                ("M1_AUDIT", "audit", 19.0, "M1"),
                ("M2_AUDIT", "audit", 23.0, "M2"),
                ("M3_ANALYSIS", "analysis", 29.0, "M3"),
            )
            for index, (kind, budget_class, elapsed, stage) in enumerate(rows, 1):
                path = log_root / f"attempt{index}_attempt_01.process_receipt.json"
                path.write_text(json.dumps({"kind": kind, "budget_class": budget_class,
                                            "elapsed_seconds": elapsed, "stage": stage,
                                            "actual_exit_code": 1, "status": "FAIL"}), encoding="utf-8")
            with patch.object(stage_process, "LOGS", log_root):
                self.assertEqual(stage_process._prior_elapsed("science", "M1"), 11.0)
                self.assertEqual(stage_process._prior_elapsed("science", "M2"), 17.0)
                self.assertEqual(stage_process._prior_elapsed("audit", "M1_AUDIT"), 19.0)
                self.assertEqual(stage_process._prior_elapsed("audit", "M2_AUDIT"), 23.0)
                self.assertEqual(stage_process._prior_elapsed("analysis", "M3_ANALYSIS"), 29.0)
            self.assertEqual(stage_process.remaining_budget_seconds(500.0, 25.0, 100.0, 7200.0), 475.0)
            self.assertEqual(stage_process.remaining_budget_seconds(400.0, 25.0, 100.0, 7200.0), 375.0)
            self.assertEqual(stage_process.remaining_budget_seconds(1800.0, 29.0, 29.0, 1800.0), 1771.0)
            self.assertEqual(stage_process.remaining_budget_seconds(400.0, 25.0, 300.0, 7200.0), 375.0)
            self.assertEqual(stage_process.remaining_budget_seconds(400.0, 25.0, 7000.0, 7200.0), 200.0)

    def test_stage_gate_fails_without_release_and_returns_complete_snapshot_when_released(self):
        frozen = {
            "plan_sha256": "a" * 64, "authorization_file_sha256": "b" * 64,
            "source_bindings_file_sha256": "c" * 64, "source_binding_sha256": "d" * 64,
            "reuse_receipt_file_sha256": "e" * 64, "synthetic_receipt_file_sha256": "f" * 64,
            "m0_preparation_receipt_file_sha256": "1" * 64, "root_m0_review_file_sha256": "2" * 64,
            "historical_reuse_only_binding_sha256": "3" * 64,
        }
        with tempfile.TemporaryDirectory() as temp:
            release_path = Path(temp) / "ROOT_M1.json"
            with patch.object(integrity, "verify_frozen_m0", return_value=frozen), \
                 patch.dict(integrity.RELEASE_FILES, {"M1_BRIDGE_SMOKE": release_path}), \
                 patch("R.experiments.fixed_horizon_schedule.m1_key_digest", return_value="4" * 64):
                with self.assertRaises(integrity.IntegrityError):
                    integrity.require_stage_release("M1_BRIDGE_SMOKE")
                release = {
                    "protocol_id": "ridge_fixed_horizon_baselines_v1", "run_id": "r01",
                    "stage": "M1_BRIDGE_SMOKE", "status": "ROOT_RELEASED", "issued_by": "ROOT",
                    "release_id": "synthetic", **frozen, "scope": "registered_bridge_24_plus_smoke_96",
                    "authorized_trajectories": 120, "science_budget_cap_seconds": 7200,
                    "authorized_key_whitelist_sha256": "4" * 64,
                }
                release_path.write_text(json.dumps(release), encoding="utf-8")
                payload = integrity.require_stage_release("M1_BRIDGE_SMOKE")
                self.assertEqual(payload["source_binding_sha256"], frozen["source_binding_sha256"])
                self.assertEqual(payload["release_scope"], release["scope"])
                self.assertEqual(payload["release_file_sha256"], integrity.sha256_file(release_path))

    def test_m1_entry_snapshots_from_runner_and_both_audit_paths_pass_exit_check(self):
        frozen = {
            "plan_sha256": "a" * 64, "authorization_file_sha256": "b" * 64,
            "source_bindings_file_sha256": "c" * 64, "source_binding_sha256": "d" * 64,
            "reuse_receipt_file_sha256": "e" * 64, "synthetic_receipt_file_sha256": "f" * 64,
            "m0_preparation_receipt_file_sha256": "1" * 64, "root_m0_review_file_sha256": "2" * 64,
            "historical_reuse_only_binding_sha256": "3" * 64,
        }
        expected_keys = (*integrity.FROZEN_HASH_KEYS, "release_file_sha256")
        with tempfile.TemporaryDirectory() as temp:
            release_path = Path(temp) / "ROOT_M1.json"
            release = {
                "protocol_id": "ridge_fixed_horizon_baselines_v1", "run_id": "r01",
                "stage": "M1_BRIDGE_SMOKE", "status": "ROOT_RELEASED", "issued_by": "ROOT",
                "release_id": "synthetic-snapshot", **frozen,
                "scope": "registered_bridge_24_plus_smoke_96",
                "authorized_trajectories": 120, "science_budget_cap_seconds": 7200,
                "authorized_key_whitelist_sha256": "4" * 64,
            }
            release_path.write_text(json.dumps(release), encoding="utf-8")
            with patch.object(integrity, "verify_frozen_m0", return_value=frozen), \
                 patch.dict(integrity.RELEASE_FILES, {"M1_BRIDGE_SMOKE": release_path}), \
                 patch("R.experiments.fixed_horizon_schedule.m1_key_digest", return_value="4" * 64):
                gate = integrity.require_stage_release("M1_BRIDGE_SMOKE")
                with patch.object(runner, "require_stage_release", return_value=gate):
                    _context, _verified, runner_snapshot = runner.open_stage_context("M1_BRIDGE_SMOKE")
                bridge_audit_snapshot = audit_entry._capture_stage_entry_snapshot(gate)
                combined_audit_snapshot = audit_entry._capture_stage_entry_snapshot(gate)

                snapshots = (runner_snapshot, bridge_audit_snapshot, combined_audit_snapshot)
                for snapshot in snapshots:
                    self.assertEqual(tuple(snapshot), expected_keys)
                    result = integrity.verify_stage_unchanged("M1_BRIDGE_SMOKE", snapshot)
                    self.assertTrue(result["m0_hashes_unchanged"])
                    self.assertEqual(result["release_file_sha256"], gate["release_file_sha256"])

                # Both audit entrypoints must use the shared capture helper;
                # the analyzer has its own M3 helper but the same frozen-key contract.
                audit_source = Path(audit_entry.__file__).read_text(encoding="utf-8")
                self.assertEqual(audit_source.count("start_snapshot = _capture_stage_entry_snapshot(gate)"), 2)

                def verify_synthetic_m3(stage, snapshot):
                    self.assertEqual(stage, "M3_ANALYSIS")
                    result = integrity.verify_stage_unchanged("M1_BRIDGE_SMOKE", snapshot)
                    return {**result, "stage": "M3_ANALYSIS"}

                fake_m3_integrity = SimpleNamespace(
                    FROZEN_HASH_KEYS=integrity.FROZEN_HASH_KEYS,
                    require_stage_release=lambda stage: gate,
                    verify_stage_unchanged=verify_synthetic_m3,
                )
                _payload, analyzer_snapshot = analysis._capture_m3_entry_snapshot(fake_m3_integrity)
                self.assertEqual(tuple(analyzer_snapshot), expected_keys)
                analyzer_exit = analysis._verify_m3_snapshot_unchanged(fake_m3_integrity, analyzer_snapshot)
                self.assertTrue(analyzer_exit["m0_hashes_unchanged"])
                self.assertEqual(analyzer_exit["release_file_sha256"], gate["release_file_sha256"])

    def test_m3_gate_and_exit_check_accept_only_complete_bound_release(self):
        frozen = {
            "plan_sha256": "a" * 64, "authorization_file_sha256": "b" * 64,
            "source_bindings_file_sha256": "c" * 64, "source_binding_sha256": "d" * 64,
            "reuse_receipt_file_sha256": "e" * 64, "synthetic_receipt_file_sha256": "f" * 64,
            "m0_preparation_receipt_file_sha256": "1" * 64, "root_m0_review_file_sha256": "2" * 64,
            "historical_reuse_only_binding_sha256": "3" * 64,
        }
        def sealed_receipt(path: Path, status: str):
            record = {"status": status}
            record["receipt_sha256"] = integrity.canonical_sha(record)
            path.write_text(json.dumps(record), encoding="utf-8")
            return integrity.sha256_file(path), record["receipt_sha256"]

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "bank").mkdir()
            (root / "audit").mkdir()
            (root / "analysis").mkdir()
            m2_file, m2_self = sealed_receipt(root / "bank/M2_BANK_RECEIPT.json", "PASS_M2_COMPLETE_BANK")
            audit_file, audit_self = sealed_receipt(root / "audit/M2_TRACE_AUDIT_RECEIPT.json", "PASS_M2_TRACE_AUDIT")
            m2 = json.loads((root / "bank/M2_BANK_RECEIPT.json").read_text())
            m2.update({"runner_process_exit_code": 0, "science_wall_seconds": 123.5})
            m2["receipt_sha256"] = integrity.canonical_sha({k: v for k, v in m2.items() if k != "receipt_sha256"})
            (root / "bank/M2_BANK_RECEIPT.json").write_text(json.dumps(m2), encoding="utf-8")
            m2_file, m2_self = integrity.sha256_file(root / "bank/M2_BANK_RECEIPT.json"), m2["receipt_sha256"]
            audit = json.loads((root / "audit/M2_TRACE_AUDIT_RECEIPT.json").read_text())
            audit.update({"audit_process_exit_code": 0, "audit_cumulative_seconds": 456.0})
            audit["receipt_sha256"] = integrity.canonical_sha({k: v for k, v in audit.items() if k != "receipt_sha256"})
            (root / "audit/M2_TRACE_AUDIT_RECEIPT.json").write_text(json.dumps(audit), encoding="utf-8")
            audit_file, audit_self = integrity.sha256_file(root / "audit/M2_TRACE_AUDIT_RECEIPT.json"), audit["receipt_sha256"]
            merged = root / "analysis/MERGED_RAW_TRAJECTORIES.csv"
            merged.write_text("synthetic-only\n", encoding="utf-8")
            release_path = root / "ROOT_M3.json"
            release = {
                "protocol_id": "ridge_fixed_horizon_baselines_v1", "run_id": "r01",
                "stage": "M3_ANALYSIS", "status": "ROOT_RELEASED", "issued_by": "ROOT",
                "release_id": "synthetic", **frozen,
                "scope": "registered_complete_15120_row_development_analysis",
                "analysis_budget_cap_seconds": 1800,
                "m2_bank_receipt_file_sha256": m2_file, "m2_bank_receipt_sha256": m2_self,
                "m2_audit_receipt_file_sha256": audit_file, "m2_audit_receipt_sha256": audit_self,
                "merged_raw_sha256": integrity.sha256_file(merged),
            }
            release_path.write_text(json.dumps(release), encoding="utf-8")
            with patch.object(integrity, "verify_frozen_m0", return_value=frozen), \
                 patch.object(integrity, "OUT", root), \
                 patch.dict(integrity.RELEASE_FILES, {"M3_ANALYSIS": release_path}):
                entry = integrity.require_stage_release("M3_ANALYSIS")
                snapshot = {key: entry[key] for key in (*integrity.FROZEN_HASH_KEYS, "release_file_sha256")}
                self.assertEqual(entry["analysis_budget_cap_seconds"], 1800)
                self.assertEqual(entry["plan_sha256"], frozen["plan_sha256"])
                self.assertTrue(integrity.verify_stage_unchanged("M3_ANALYSIS", snapshot)["m0_hashes_unchanged"])
                release_path.write_text(json.dumps({**release, "note": "drift"}), encoding="utf-8")
                with self.assertRaises(integrity.IntegrityError):
                    integrity.verify_stage_unchanged("M3_ANALYSIS", snapshot)

    def test_m1_combined_and_m2_bank_audit_cli_output_contracts(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            audit_dir = root / "audit"
            audit_dir.mkdir()
            bridge_dir = root / "bridge"
            smoke_dir = root / "smoke"
            bridge_dir.mkdir()
            smoke_dir.mkdir()

            def write_receipt(path, record):
                value = dict(record)
                value["receipt_sha256"] = hashlib.sha256(audit_entry._canonical(value)).hexdigest()
                path.write_text(json.dumps(value), encoding="utf-8")
                return value

            bridge_runner = write_receipt(bridge_dir / "M1_BRIDGE_RECEIPT.json", {
                "status": "PASS_M1_BRIDGE", "scope": "bridge", "trajectory_count": 24,
                "unique_trajectory_keys": 24, "runner_process_exit_code": 0,
                "runner_process_seconds": 42.5, "whole_M1_process_scope": True,
            })
            smoke_runner = write_receipt(smoke_dir / "M1_SMOKE_RECEIPT.json", {
                "status": "PASS_M1_SMOKE", "scope": "smoke", "trajectory_count": 96,
                "unique_trajectory_keys": 96, "runner_process_exit_code": 0,
                "runner_process_seconds": 42.5, "whole_M1_process_scope": True,
            })
            write_receipt(audit_dir / "M1_BRIDGE_SMOKE_RECEIPT.json", {
                "status": "PASS_M1_BRIDGE_AND_SMOKE", "runner_process_exit_code": 0,
                "runner_process_seconds": 42.5, "whole_M1_process_scope": True,
                "bridge_receipt_sha256": bridge_runner["receipt_sha256"],
                "smoke_receipt_sha256": smoke_runner["receipt_sha256"],
            })
            write_receipt(audit_dir / "M1_BRIDGE_TRACE_AUDIT.json", {
                "status": "PASS", "scope": "bridge", "trajectory_count": 24, "step_count": 120,
                "runner_receipt_sha256": bridge_runner["receipt_sha256"],
                "runner_receipt_file_sha256": hashlib.sha256((bridge_dir / "M1_BRIDGE_RECEIPT.json").read_bytes()).hexdigest(),
            })
            write_receipt(audit_dir / "M1_SMOKE_TRACE_AUDIT.json", {
                "status": "PASS", "scope": "smoke", "trajectory_count": 96, "step_count": 480,
                "runner_receipt_sha256": smoke_runner["receipt_sha256"],
                "runner_receipt_file_sha256": hashlib.sha256((smoke_dir / "M1_SMOKE_RECEIPT.json").read_bytes()).hexdigest(),
            })
            gate = {key: f"{index:064x}" for index, key in enumerate((
                "plan_sha256", "authorization_file_sha256", "source_bindings_file_sha256",
                "source_binding_sha256", "reuse_receipt_file_sha256", "synthetic_receipt_file_sha256",
                "m0_preparation_receipt_file_sha256", "root_m0_review_file_sha256",
                "historical_reuse_only_binding_sha256",
            ), 1)}
            gate["release_file_sha256"] = "r" * 64

            stdout = StringIO()
            with patch.object(audit_entry, "OUT", root), \
                 patch.object(audit_entry, "require_stage_release", return_value=gate), \
                 patch.object(audit_entry, "verify_stage_unchanged", return_value={
                     "m0_hashes_unchanged": True, "release_file_sha256": gate["release_file_sha256"],
                 }), \
                 patch.object(sys, "argv", ["fixed_horizon_trace_audit_entry.py", "--scope", "m1"]), \
                 redirect_stdout(stdout):
                exit_code = audit_entry.main()
            self.assertEqual(exit_code, 0)
            m1_cli = json.loads(stdout.getvalue())
            self.assertEqual(m1_cli, {
                "status": "PASS_M1_TRACE_AUDIT", "scope": "m1",
                "trajectory_count": 120, "step_count": 600,
            })
            m1_report = json.loads((audit_dir / "M1_TRACE_AUDIT_RECEIPT.json").read_text(encoding="utf-8"))
            self.assertEqual(m1_report["step_count"], 600)

            stdout = StringIO()
            with patch.object(audit_entry, "OUT", root), \
                 patch.object(audit_entry, "run_audit", return_value={
                     "status": "PASS_M2_TRACE_AUDIT", "trajectory_count": 10080, "step_count": 54321,
                 }), \
                 patch.object(sys, "argv", ["fixed_horizon_trace_audit_entry.py", "--scope", "bank"]), \
                 redirect_stdout(stdout):
                exit_code = audit_entry.main()
            self.assertEqual(exit_code, 0)
            m2_cli = json.loads(stdout.getvalue())
            self.assertEqual(m2_cli["status"], "PASS_M2_TRACE_AUDIT")
            self.assertEqual(m2_cli["scope"], "bank")
            self.assertEqual(m2_cli["trajectory_count"], 10080)
            self.assertEqual(m2_cli["step_count"], 54321)

    def test_m1_science_finalizer_links_actual_process_to_component_audit_receipts(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "bridge").mkdir()
            (root / "smoke").mkdir()
            (root / "audit").mkdir()

            def write_receipt(path, record):
                value = dict(record)
                value["receipt_sha256"] = stage_process._canonical_sha(value)
                path.write_text(json.dumps(value), encoding="utf-8")
                return value

            bridge_path = root / "bridge/M1_BRIDGE_RECEIPT.json"
            smoke_path = root / "smoke/M1_SMOKE_RECEIPT.json"
            combined_path = root / "audit/M1_BRIDGE_SMOKE_RECEIPT.json"
            write_receipt(bridge_path, {
                "status": "PASS_M1_BRIDGE", "scope": "bridge", "trajectory_count": 24,
                "unique_trajectory_keys": 24, "files_sha256": {},
            })
            write_receipt(smoke_path, {
                "status": "PASS_M1_SMOKE", "scope": "smoke", "trajectory_count": 96,
                "unique_trajectory_keys": 96, "files_sha256": {},
            })
            write_receipt(combined_path, {
                "status": "PASS_M1_BRIDGE_AND_SMOKE", "trajectory_count": 120,
                "bridge_receipt_sha256": "old-bridge", "smoke_receipt_sha256": "old-smoke",
                "runner_process_exit_code": None, "runner_process_seconds": None,
                "science_wall_seconds": 42.0,
            })

            with patch.object(stage_process, "OUT", root):
                stage_process._finalize_science_receipt("M1", code=0, elapsed=42.5, prior=8.0)

            bridge = json.loads(bridge_path.read_text(encoding="utf-8"))
            smoke = json.loads(smoke_path.read_text(encoding="utf-8"))
            combined = json.loads(combined_path.read_text(encoding="utf-8"))
            for scope, runner in (("bridge", bridge), ("smoke", smoke)):
                audit_entry._selfhash(runner)
                audit_entry._check_runner_process_fields(runner, scope)
                self.assertEqual(runner["runner_process_exit_code"], 0)
                self.assertEqual(runner["runner_process_seconds"], 42.5)
                self.assertIs(runner["whole_M1_process_scope"], True)
            audit_entry._selfhash(combined)
            audit_entry._check_runner_process_fields(combined, "m1")
            self.assertEqual(combined["science_wall_seconds"], 50.5)
            self.assertEqual(combined["bridge_receipt_sha256"], bridge["receipt_sha256"])
            self.assertEqual(combined["smoke_receipt_sha256"], smoke["receipt_sha256"])

    def test_trace_audit_streams_join_by_trajectory_and_fail_on_split_or_eof(self):
        def write_jsonl_gz(path, records):
            with gzip.open(path, "wt", encoding="utf-8") as stream:
                for record in records:
                    stream.write(json.dumps(record) + "\n")

        raw_rows = [{"trajectory_key": "layout-A"}, {"trajectory_key": "layout-B"}]
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            public = root / "public.jsonl.gz"
            evaluator = root / "evaluator.jsonl.gz"
            physical = root / "physical.jsonl.gz"
            write_jsonl_gz(public, [
                {"trajectory_key": "layout-A", "decision_index": 0},
                {"trajectory_key": "layout-A", "decision_index": 1},
                {"trajectory_key": "layout-B", "decision_index": 0},
            ])
            write_jsonl_gz(evaluator, [
                {"trajectory_key": "layout-A", "record_type": "forecast_binding"},
                {"trajectory_key": "layout-A", "record_type": "trajectory_start"},
                {"trajectory_key": "layout-B", "record_type": "forecast_binding"},
                {"trajectory_key": "layout-B", "record_type": "trajectory_start"},
            ])
            write_jsonl_gz(physical, [
                {"trajectory_key": "layout-A", "decision_index": 0},
                {"trajectory_key": "layout-A", "decision_index": 1},
                {"trajectory_key": "layout-B", "decision_index": 0},
            ])
            groups = list(audit_entry._aligned_trajectory_groups(
                raw_rows, audit_entry._jsonl_gz_groups(public),
                audit_entry._jsonl_gz_groups(evaluator), audit_entry._jsonl_gz_groups(physical),
            ))
            self.assertEqual([item[0] for item in groups], ["layout-A", "layout-B"])
            self.assertEqual([len(item[2]) for item in groups], [2, 1])
            self.assertEqual([len(item[3]) for item in groups], [2, 2])
            self.assertEqual([len(item[4]) for item in groups], [2, 1])

            write_jsonl_gz(public, [
                {"trajectory_key": "layout-A"}, {"trajectory_key": "layout-B"},
                {"trajectory_key": "layout-A"},
            ])
            with self.assertRaises(AuditFailure):
                list(audit_entry._jsonl_gz_groups(public))

            write_jsonl_gz(public, [
                {"trajectory_key": "layout-A"}, {"trajectory_key": "layout-B"},
            ])
            write_jsonl_gz(physical, [{"trajectory_key": "layout-A"}])
            with self.assertRaises(AuditFailure):
                list(audit_entry._aligned_trajectory_groups(
                    raw_rows, audit_entry._jsonl_gz_groups(public),
                    audit_entry._jsonl_gz_groups(evaluator), audit_entry._jsonl_gz_groups(physical),
                ))

            write_jsonl_gz(physical, [
                {"trajectory_key": "layout-B"}, {"trajectory_key": "layout-A"},
            ])
            with self.assertRaises(AuditFailure):
                list(audit_entry._aligned_trajectory_groups(
                    raw_rows, audit_entry._jsonl_gz_groups(public),
                    audit_entry._jsonl_gz_groups(evaluator), audit_entry._jsonl_gz_groups(physical),
                ))


if __name__ == "__main__":
    unittest.main()
