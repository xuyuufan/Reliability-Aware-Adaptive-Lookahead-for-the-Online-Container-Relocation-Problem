#!/usr/bin/env python3
"""Independent fail-closed audit of PLAN-051 raw trajectories and action traces.

This script replays the registered physical transition and public forecast view
from the private evaluator sidecars. It does not import the science runner or
actors and does not compute or rank performance results.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict, deque
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


SCRIPT = Path(__file__).resolve()
WORKSPACE = SCRIPT.parents[2]
if str(WORKSPACE) not in sys.path:
    sys.path.insert(0, str(WORKSPACE))
from R.scrp.eri_eg_public import build_eg20_plan

OUT = WORKSPACE.parent
RUN_ROOT = OUT
LAYOUT_PATH = WORKSPACE / "R/output/r_frozen_cross_solver_transfer_v1/r01/data/select20.json"
CONFIG_PATH = WORKSPACE / "R/configs/r_fixed_horizon_baselines_v1.json"
BINDINGS_PATH = OUT / "preparation/SOURCE_BINDINGS.json"
RIDGE_PATH = WORKSPACE / "R/output/universal_r_utility_v1/r02/M1_FIT_01/RIDGE_MODEL.json"
ORDER_MODEL_PATH = WORKSPACE / "R/output/r_order_v1/r01/M1_MODELS.json"
ERI_EG_BUILDER_PATH = WORKSPACE / "R/scrp/eri_eg_public.py"

Q_PRIOR = 5183 / 9920
HORIZONS = (0, 2, 5, 10, 20)
NEW_HORIZONS = (0, 2, 5, 10)
ALGORITHMS = ("EM", "GP", "ERI", "EG", "Chain", "Minmax")
FIXED_METHODS = tuple(f"FIXED_H{h}" for h in NEW_HORIZONS)
BRIDGE_METHODS = ("R_UTILITY_RIDGE", "NATIVE20")
METHODS = FIXED_METHODS + BRIDGE_METHODS
R_COEFFICIENTS = (-0.10860203113738334, -0.7425766885981918,
                  0.20742547985980178, 1.1399906546560534)
CRANE_TRAVEL = 1.2
PICK_DROP = 30.0
HISTORY_WINDOW = 8
UNKNOWN = 22
TOL = 1e-9


class AuditFailure(RuntimeError):
    pass


def canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, ensure_ascii=False, separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_jsonl_gz(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise AuditFailure(f"{path.name}:{line_number}: JSONL row is not an object")
            rows.append(row)
    return rows


def close(left: Any, right: Any, *, tol: float = TOL) -> bool:
    if isinstance(left, bool) or isinstance(right, bool):
        return left is right
    try:
        return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=tol)
    except (TypeError, ValueError, OverflowError):
        return left == right


def same(left: Any, right: Any, *, path: str = "value") -> None:
    if isinstance(left, float) or isinstance(right, float):
        if not close(left, right):
            raise AuditFailure(f"{path}: {left!r} != {right!r}")
        return
    if type(left) is not type(right) and not isinstance(left, (int, float)):
        raise AuditFailure(f"{path}: type {type(left).__name__} != {type(right).__name__}")
    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            raise AuditFailure(f"{path}: lengths {len(left)} != {len(right)}")
        for index, (a, b) in enumerate(zip(left, right, strict=True)):
            same(a, b, path=f"{path}[{index}]")
        return
    if isinstance(left, dict) and isinstance(right, dict):
        if set(left) != set(right):
            raise AuditFailure(f"{path}: keys differ {sorted(left)} != {sorted(right)}")
        for key in left:
            same(left[key], right[key], path=f"{path}.{key}")
        return
    if left != right:
        raise AuditFailure(f"{path}: {left!r} != {right!r}")


def fail_if(condition: bool, message: str) -> None:
    if not condition:
        raise AuditFailure(message)


def condition_registry() -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for family in ("subset_shuffle", "delayed_positions"):
        for p in (0.1, 0.3, 0.5):
            for repeat in (0, 1):
                cid = f"{family}|p={p:.1f}|r={repeat}"
                rows[cid] = {
                    "family": family, "p": p, "window_size": None,
                    "repeat": repeat, "strength": f"p={p:.1f}",
                }
    for width in (2, 5, 10, 20):
        for repeat in (0, 1):
            cid = f"time_window_shuffle|B={width}|r={repeat}"
            rows[cid] = {
                "family": "time_window_shuffle", "p": None,
                "window_size": width, "repeat": repeat, "strength": f"B={width}",
            }
    rows["CLEAN"] = {
        "family": "CLEAN", "p": None, "window_size": None,
        "repeat": 0, "strength": "CLEAN",
    }
    return rows


def stacks_copy(stacks: Sequence[Sequence[int]]) -> list[list[int]]:
    return [[int(item) for item in stack] for stack in stacks]


def locate(stacks: Sequence[Sequence[int]], item: int) -> tuple[int, int]:
    matches = [(i, j) for i, stack in enumerate(stacks) for j, value in enumerate(stack) if value == item]
    if len(matches) != 1:
        raise AuditFailure(f"private item {item} occurs {len(matches)} times")
    return matches[0]


def replay_initial(
    initial_stacks: Sequence[Sequence[int]], true_sequence: Sequence[int],
) -> tuple[list[list[int]], int, int | None, float]:
    stacks = stacks_copy(initial_stacks)
    position: int | None = None
    crane_time = 0.0
    retrieved = 0
    while retrieved < len(true_sequence):
        target = int(true_sequence[retrieved])
        source, tier = locate(stacks, target)
        if tier != len(stacks[source]) - 1:
            break
        source_position = source + 1
        if position is not None:
            crane_time += abs(position - source_position) * CRANE_TRAVEL
        crane_time += source_position * CRANE_TRAVEL + PICK_DROP
        position = 0
        stacks[source].pop()
        retrieved += 1
    return stacks, retrieved, position, crane_time


def forecast_hits(
    true_sequence: Sequence[int], forecast: Sequence[int], retrieved: int,
) -> list[bool]:
    hits: list[bool] = []
    for event_index in range(1, min(retrieved, len(true_sequence) - 1) + 1):
        excluded = set(int(item) for item in true_sequence[:event_index])
        predicted = next((int(item) for item in forecast if int(item) not in excluded), None)
        if predicted is None:
            raise AuditFailure("forecast has no next item for a completed public reveal")
        hits.append(predicted == int(true_sequence[event_index]))
    return hits[-HISTORY_WINDOW:]


def frame_from_private(
    stacks: Sequence[Sequence[int]], true_sequence: Sequence[int],
    forecast: Sequence[int], retrieved: int, hits: Sequence[bool],
    crane_position: int | None, crane_time: float, relocations: int,
    max_tiers: int,
) -> dict[str, Any] | None:
    if retrieved >= len(true_sequence):
        return None
    current = int(true_sequence[retrieved])
    future = [
        int(item) for item in forecast
        if int(item) not in set(int(value) for value in true_sequence[:retrieved])
        and int(item) != current
    ][:20]
    rank_by_item = {current: 1}
    rank_by_item.update({item: offset + 2 for offset, item in enumerate(future)})
    max_tiers = int(max_tiers)
    ranks = [[0] * max_tiers for _ in stacks]
    occupied = [[False] * max_tiers for _ in stacks]
    for stack_index, stack in enumerate(stacks):
        if len(stack) > max_tiers:
            raise AuditFailure("private stack exceeds max_tiers")
        for tier, item in enumerate(stack):
            occupied[stack_index][tier] = True
            ranks[stack_index][tier] = rank_by_item.get(int(item), UNKNOWN)
    source, _target_tier = locate(stacks, current)
    legal = [
        index for index, stack in enumerate(stacks)
        if index != source and len(stack) < max_tiers
    ]
    if not legal:
        raise AuditFailure("nonterminal state has no legal destination")
    q = (2.0 * Q_PRIOR + sum(bool(value) for value in hits)) / (2.0 + len(hits))
    return {
        "rank_tensor": ranks,
        "occupied_mask": occupied,
        "legal_destination_mask": [index in legal for index in range(len(stacks))],
        "legal_destinations": legal,
        "max_tiers": max_tiers,
        "history_hits": list(hits),
        "q_prior": Q_PRIOR,
        "q_t": q,
        "retrieval_count": retrieved,
        "relocation_count": relocations,
        "crane_position": crane_position,
        "crane_time": crane_time,
        "r_order_summary": None,
    }


def public_history_record(before: Mapping[str, Any], action: int, after: Mapping[str, Any]) -> float | None:
    before_ranks = before["rank_tensor"]
    before_occ = before["occupied_mask"]
    after_ranks = after["rank_tensor"]
    after_occ = after["occupied_mask"]
    target_locations = [
        (s, t) for s, row in enumerate(before_ranks)
        for t, rank in enumerate(row) if before_occ[s][t] and rank == 1
    ]
    fail_if(len(target_locations) == 1, "public history source frame lacks exactly one target")
    source, target_tier = target_locations[0]
    heights = [sum(bool(value) for value in row) for row in before_occ]
    fail_if(heights[source] > target_tier + 1, "history update saw an unblocked target")
    fail_if(action in before["legal_destinations"], "history update action is not legal")
    ranks = [list(map(int, row)) for row in before_ranks]
    occ = [list(map(bool, row)) for row in before_occ]
    moved = ranks[source][heights[source] - 1]
    ranks[source][heights[source] - 1] = 0
    occ[source][heights[source] - 1] = False
    ranks[action][heights[action]] = moved
    occ[action][heights[action]] = True
    retrieval_delta = int(after["retrieval_count"]) - int(before["retrieval_count"])
    if retrieval_delta < 0:
        raise AuditFailure("public retrieval count moved backwards")
    if retrieval_delta == 0:
        same(occ, after_occ, path="history.no_reveal_occupancy")
        return None
    new_targets = [
        (s, t) for s, row in enumerate(after_ranks)
        for t, rank in enumerate(row) if after_occ[s][t] and rank == 1
    ]
    fail_if(len(new_targets) == 1, "after frame lacks exactly one revealed target")
    new_source, new_tier = new_targets[0]
    old_rank = ranks[new_source][new_tier]
    if old_rank == UNKNOWN:
        return 1.0
    if not 2 <= old_rank <= 21:
        raise AuditFailure(f"revealed target had invalid prior forecast rank {old_rank}")
    return (old_rank - 2) / 20.0


def decorate_history_summary(frame: Mapping[str, Any], records: Sequence[float]) -> dict[str, Any]:
    result = dict(frame)
    occ_count = sum(sum(bool(value) for value in row) for row in frame["occupied_mask"])
    retrieval_count = int(frame["retrieval_count"])
    count = len(records)
    result["r_order_summary"] = {
        "q": float(frame["q_t"]),
        "displacement": 0.0 if count == 0 else sum(records) / count,
        "coverage": count / float(HISTORY_WINDOW),
        "retrieval_fraction": (
            0.0 if retrieval_count + occ_count == 0
            else retrieval_count / float(retrieval_count + occ_count)
        ),
        "records": list(records),
    }
    return result


def check_frame(actual: Mapping[str, Any], expected: Mapping[str, Any], where: str) -> None:
    for key in (
        "rank_tensor", "occupied_mask", "legal_destination_mask",
        "legal_destinations", "max_tiers", "history_hits", "retrieval_count",
        "relocation_count", "crane_position",
    ):
        same(actual.get(key), expected.get(key), path=f"{where}.{key}")
    same(actual.get("q_prior"), expected.get("q_prior"), path=f"{where}.q_prior")
    same(actual.get("q_t"), expected.get("q_t"), path=f"{where}.q_t")
    same(actual.get("crane_time"), expected.get("crane_time"), path=f"{where}.crane_time")
    summary = expected.get("r_order_summary")
    if summary is not None:
        same(actual.get("r_order_summary"), summary, path=f"{where}.r_order_summary")


def rebuild_native_eg_plan(public_frame: Mapping[str, Any]):
    """Rebuild FIXED20 EG's group plan from the recorded public start frame.

    This deliberately reuses the M0-hash-bound pure EG plan builder. It does
    not instantiate an actor or environment and does not expose evaluator data.
    """
    observation = SimpleNamespace(
        rank_tensor=np.asarray(public_frame["rank_tensor"], dtype=np.int32),
        occupied_mask=np.asarray(public_frame["occupied_mask"], dtype=np.bool_),
        legal_destination_mask=np.asarray(
            public_frame["legal_destination_mask"], dtype=np.bool_,
        ),
        history_hits=tuple(bool(value) for value in public_frame["history_hits"]),
        q_prior=float(public_frame["q_prior"]),
        q_t=float(public_frame["q_t"]),
    )
    frame = SimpleNamespace(
        observation=observation,
        legal_destinations=tuple(int(value) for value in public_frame["legal_destinations"]),
        max_tiers=int(public_frame["max_tiers"]),
        retrieval_count=int(public_frame["retrieval_count"]),
        relocation_count=int(public_frame["relocation_count"]),
        crane_position=public_frame.get("crane_position"),
        crane_time=float(public_frame["crane_time"]),
    )
    return build_eg20_plan(frame)


def forbidden_actor_fields(value: Any, path: str = "actor_trace") -> list[str]:
    denied = (
        "true_retrieval_sequence", "fixed_forecast", "forecast_future",
        "forecast_seed", "condition_evaluator_only", "physical_stacks",
        "private_state", "reveal_events", "p_private",
    )
    found: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            key_lower = str(key).lower()
            if any(fragment in key_lower for fragment in denied):
                found.append(f"{path}.{key}")
            found.extend(forbidden_actor_fields(child, f"{path}.{key}"))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found.extend(forbidden_actor_fields(child, f"{path}[{index}]"))
    return found


def horizon_proxy(h: int, q: float, *, quad: bool) -> float:
    if h == 0:
        return 0.0
    x = h / 20.0
    error = 1.0 - q
    b0, b1, b2, b3 = R_COEFFICIENTS
    score = b0 + b1 * x + b2 * error + b3 * error * x
    if quad:
        score += b3 * error * x * x
    return score


def choose_horizon(scores: Mapping[int, float]) -> tuple[int, float]:
    best_h, best_score = min(
        ((h, score) for h, score in scores.items() if h != 0),
        key=lambda pair: (pair[1], pair[0]),
    )
    return (best_h, best_score) if best_score < 0.0 else (0, 0.0)


def c1_relations(frame: Mapping[str, Any]) -> dict[int, tuple[float, ...]]:
    ranks = frame["rank_tensor"]
    occ = frame["occupied_mask"]
    tiers = int(frame["max_tiers"])
    target = [
        (s, t) for s, row in enumerate(ranks)
        for t, rank in enumerate(row) if occ[s][t] and rank == 1
    ]
    fail_if(len(target) == 1, "candidate feature frame lacks one target")
    source, target_tier = target[0]
    heights = [sum(bool(v) for v in row) for row in occ]
    source_height = heights[source]
    blocker = int(ranks[source][source_height - 1])
    blocker_known = 2 <= blocker <= 21
    relations: dict[int, tuple[float, ...]] = {}
    for destination in frame["legal_destinations"]:
        values = [int(ranks[destination][t]) for t in range(heights[destination])]
        known = [value for value in values if 2 <= value <= 21]
        unknown_count = sum(value == UNKNOWN for value in values)
        min_known = min(known) if known else UNKNOWN
        safe_known = blocker_known and all(2 <= value <= 21 and value > blocker for value in values)
        earlier_count = sum(2 <= value < blocker for value in values) if blocker_known else 0
        denominator = max(heights[destination], 1)
        relations[int(destination)] = (
            (tiers - heights[destination]) / tiers,
            float(heights[destination] == 0),
            unknown_count / denominator,
            min_known / UNKNOWN,
            float(safe_known),
            earlier_count / denominator,
        )
    return relations


def move_public(frame: Mapping[str, Any], destination: int) -> dict[str, Any]:
    state = dict(frame)
    ranks = [list(map(int, row)) for row in frame["rank_tensor"]]
    occ = [list(map(bool, row)) for row in frame["occupied_mask"]]
    heights = [sum(row) for row in occ]
    targets = [
        (s, t) for s, row in enumerate(ranks)
        for t, rank in enumerate(row) if occ[s][t] and rank == 1
    ]
    fail_if(len(targets) == 1, "candidate move lacks one target")
    source, target_tier = targets[0]
    fail_if(destination in frame["legal_destinations"], f"candidate action {destination} illegal")
    fail_if(heights[source] > target_tier + 1, "candidate move sees target unblocked")
    moved_rank = ranks[source][heights[source] - 1]
    ranks[source][heights[source] - 1] = 0
    occ[source][heights[source] - 1] = False
    ranks[destination][heights[destination]] = moved_rank
    occ[destination][heights[destination]] = True
    next_heights = [sum(row) for row in occ]
    legal = [
        i for i, height in enumerate(next_heights)
        if i != source and height < int(frame["max_tiers"])
    ]
    state.update({
        "rank_tensor": ranks,
        "occupied_mask": occ,
        "legal_destinations": legal,
        "legal_destination_mask": [i in legal for i in range(len(occ))],
        "relocation_count": int(frame["relocation_count"]) + 1,
    })
    return state


def utility_features(frame: Mapping[str, Any], sequence: Sequence[int]) -> list[float]:
    occ = frame["occupied_mask"]
    ranks = frame["rank_tensor"]
    stacks, tiers = len(occ), int(frame["max_tiers"])
    heights = [sum(row) for row in occ]
    occupied_count = sum(heights)
    unknown_count = sum(
        1 for s in range(stacks) for t in range(heights[s]) if int(ranks[s][t]) == UNKNOWN
    )
    known_future_count = sum(
        1 for s in range(stacks) for t in range(heights[s])
        if 2 <= int(ranks[s][t]) <= 21
    )
    root_global = [
        1.0, occupied_count / float(stacks * tiers),
        (heights[next(s for s, row in enumerate(ranks) if 1 in row)] -
         next(t for t, rank in enumerate(ranks[next(s for s, row in enumerate(ranks) if 1 in row)]) if rank == 1) - 1)
        / float(tiers),
        unknown_count / float(max(occupied_count, 1)),
        known_future_count / float(max(occupied_count, 1)),
        sum(height == 0 for height in heights) / float(stacks),
    ]
    virtual = dict(frame)
    matrix: list[tuple[float, ...]] = []
    for destination in sequence:
        relation = c1_relations(virtual)
        if int(destination) not in relation:
            raise AuditFailure(f"ridge candidate destination {destination} has no relation features")
        matrix.append(relation[int(destination)])
        virtual = move_public(virtual, int(destination))
    weights = list(range(1, len(matrix) + 1))
    denominator = float(sum(weights))
    weighted = [
        sum(matrix[i][j] * weights[i] for i in range(len(matrix))) / denominator
        for j in range(6)
    ]
    mean = [sum(row[j] for row in matrix) / len(matrix) for j in range(6)]
    maximum = [max(row[j] for row in matrix) for j in range(6)]
    base = root_global + list(matrix[0]) + list(matrix[-1]) + mean + maximum + weighted + [len(sequence) / float(tiers)]
    fail_if(len(base) == 37, f"ridge base feature width {len(base)} != 37")
    q = float(frame["q_t"])
    return base + [(1.0 - q) * value for value in base]


def ridge_scores(
    frame: Mapping[str, Any], trace: Mapping[str, Any], coefficients: Sequence[float],
) -> tuple[list[tuple[tuple[int, ...], float]], tuple[int, ...], float]:
    candidates = trace.get("candidates")
    fail_if(isinstance(candidates, list) and candidates, "ridge trace omitted candidate pool")
    scored = []
    anchors = []
    for candidate in candidates:
        seq = tuple(int(value) for value in candidate["sequence"])
        if candidate.get("is_h20_anchor"):
            anchors.append(seq)
        features = utility_features(frame, seq)
        value = sum(float(a) * float(b) for a, b in zip(coefficients, features, strict=True))
        scored.append((seq, value))
    fail_if(len(anchors) == 1, "ridge pool does not have exactly one H20 anchor")
    return scored, anchors[0], next(score for seq, score in scored if seq == anchors[0])


def r_order_features(summary: Mapping[str, Any]) -> list[float]:
    return [
        1.0, float(summary["q"]), float(summary["displacement"]),
        float(summary["coverage"]), float(summary["retrieval_fraction"]),
    ]


def expected_blocking(frame: Mapping[str, Any], rates: Sequence[Sequence[float]]) -> float:
    ranks = frame["rank_tensor"]
    occ = frame["occupied_mask"]
    heights = [sum(row) for row in occ]
    summary = frame.get("r_order_summary")
    fail_if(isinstance(summary, dict), "R-Order scoring frame lacks public history summary")
    features = r_order_features(summary)

    def log_rate(public_rank: int) -> float:
        if public_rank == UNKNOWN:
            return 0.0
        if not 1 <= public_rank <= 20:
            raise AuditFailure(f"unexpected future rank in R-Order scorer: {public_rank}")
        row = rates[public_rank - 1]
        return sum(float(a) * float(b) for a, b in zip(row, features, strict=True))

    targets = [
        (s, t) for s, row in enumerate(ranks)
        for t, rank in enumerate(row) if occ[s][t] and rank == 1
    ]
    fail_if(len(targets) == 1, "R-Order scoring frame lacks one target")
    target_source, target_tier = targets[0]
    terms: list[float] = []
    for stack in range(len(ranks)):
        for tier in range(heights[stack]):
            public_rank = int(ranks[stack][tier])
            if public_rank == 1:
                continue
            if stack == target_source and tier > target_tier:
                terms.append(1.0)
                continue
            if 2 <= public_rank <= 21:
                rank = public_rank - 1
            elif public_rank == UNKNOWN:
                rank = UNKNOWN
            else:
                raise AuditFailure(f"invalid public rank {public_rank} in R-Order scorer")
            lower = [
                int(ranks[stack][i]) for i in range(tier)
                if int(ranks[stack][i]) != 1
            ]
            if not lower:
                terms.append(0.0)
                continue
            log_i = log_rate(rank)
            denominator_values = [log_i]
            for lower_rank in lower:
                if 2 <= lower_rank <= 21:
                    denominator_values.append(log_rate(lower_rank - 1))
                elif lower_rank == UNKNOWN:
                    denominator_values.append(log_rate(UNKNOWN))
                else:
                    raise AuditFailure(f"invalid lower public rank {lower_rank}")
            max_log = max(denominator_values)
            log_denominator = max_log + math.log(sum(math.exp(x - max_log) for x in denominator_values))
            terms.append(1.0 - math.exp(log_i - log_denominator))
    return max(0.0, sum(terms))


def sequence_score(frame: Mapping[str, Any], sequence: Sequence[int], rates: Sequence[Sequence[float]]) -> float:
    virtual = dict(frame)
    for index, destination in enumerate(sequence):
        virtual = move_public(virtual, int(destination))
        ranks = virtual["rank_tensor"]
        occ = virtual["occupied_mask"]
        target = [
            (s, t) for s, row in enumerate(ranks)
            for t, rank in enumerate(row) if occ[s][t] and rank == 1
        ]
        fail_if(len(target) == 1, "candidate sequence lost current target")
        source, tier = target[0]
        height = sum(occ[source])
        remains_blocked = height > tier + 1
        if not remains_blocked and index < len(sequence) - 1:
            raise AuditFailure("candidate sequence continues after exposing target")
        if remains_blocked and index == len(sequence) - 1 and len(sequence) > 1:
            raise AuditFailure("EG candidate did not complete target blocker group")
    return len(sequence) + expected_blocking(virtual, rates)


def candidate_policy_check(
    row: Mapping[str, str], step: Mapping[str, Any],
    ridge_coefficients: Sequence[float], order_rates: Sequence[Sequence[float]],
) -> None:
    method = row["method"]
    algorithm = row["algorithm"]
    action = int(step["own_action"])
    frame = step["public_frame_before"]
    trace = step.get("actor_trace") or {}
    fail_if(action in frame["legal_destinations"], f"{method}/{algorithm}: illegal recorded action")
    if method in FIXED_METHODS:
        horizon = int(method.removeprefix("FIXED_H"))
        same(trace.get("policy"), method, path=f"{method}.policy")
        same(trace.get("solver"), algorithm, path=f"{method}.solver")
        same(int(trace.get("requested_h", -1)), horizon, path=f"{method}.requested_h")
        same(trace.get("fixed_h_only"), True, path=f"{method}.fixed_h_only")
        if algorithm == "EG":
            fail_if(isinstance(trace.get("group_start"), bool), f"{method}/EG trace lacks group_start")
            fail_if(isinstance(trace.get("cached_moves"), int), f"{method}/EG trace lacks cached_moves")
        return
    if method == "R_UTILITY_RIDGE":
        if algorithm in ("Chain", "Minmax"):
            # PLAN-052's Ridge wrapper records a destination candidate pool,
            # not PLAN-051's full sequence fields. The bridge entry separately
            # compares every frame, action, and actor_trace field to that exact
            # frozen PLAN-052 trace.
            same(trace.get("policy"), "R_UTILITY_RIDGE", path="ridge.plan052.policy")
            same(trace.get("solver"), algorithm, path="ridge.plan052.solver")
            same(int(trace.get("selected_destination", -1)), action,
                 path="ridge.plan052.selected_destination")
            fail_if(isinstance(trace.get("candidate_pool"), list) and trace["candidate_pool"],
                    "PLAN-052 Ridge trace lacks its candidate pool")
            return
        if algorithm == "EG" and trace.get("cached_move"):
            # PLAN-051 EG cache rows carry the committed sequence/index and
            # omit the group-start scoring fields. Check their native schema.
            committed = tuple(map(int, trace.get("committed_sequence") or ()))
            sequence_index = trace.get("sequence_index")
            fail_if(bool(committed), "PLAN-051 Ridge EG cache trace lacks committed_sequence")
            fail_if(isinstance(sequence_index, int) and 1 <= sequence_index <= len(committed),
                    "PLAN-051 Ridge EG cache trace has invalid sequence_index")
            same(trace.get("group_start"), False, path="ridge.EG_cache.group_start")
            same(action, committed[int(sequence_index) - 1], path="ridge.EG_cache.action")
            return
        # PLAN-051 EM/GP/ERI/EG group-start traces are checked below using the
        # frozen Ridge feature/scoring contract. They have no requested_h field.
    elif method == "NATIVE20":
        if algorithm in ("EM", "GP", "ERI", "EG"):
            # PLAN-051 uses the original FixedHorizonActor trace schema.
            same(trace.get("policy"), "FIXED20", path="native20.plan051.policy")
            same(int(trace.get("requested_h", -1)), 20, path="native20.plan051.requested_h")
            if algorithm == "EG":
                fail_if(isinstance(trace.get("group_start"), bool),
                        "PLAN-051 Native20 EG trace lacks group_start")
                fail_if(isinstance(trace.get("cached_moves"), int),
                        "PLAN-051 Native20 EG trace lacks cached_moves")
        else:
            # PLAN-052 Chain/Minmax use the native proposer trace schema.
            same(trace.get("policy"), "NATIVE20", path="native20.plan052.policy")
            same(trace.get("solver"), algorithm, path="native20.plan052.solver")
            same(int(trace.get("requested_h", -1)), 20, path="native20.plan052.requested_h")
            same(int(trace.get("selected_h", -1)), 20, path="native20.plan052.selected_h")
            same(int(trace.get("destination", -1)), action, path="native20.plan052.destination")
        return
    if algorithm == "EG" and trace.get("cached_move"):
        sequence = trace.get("committed_sequence")
        sequence_index = trace.get("sequence_index")
        if sequence is None:
            nested = trace.get("native_trace") or {}
            plan = nested.get("committed_plan") or {}
            sequence = plan.get("destinations_top_down")
            sequence_index = nested.get("group_move_index")
        sequence = tuple(map(int, sequence or []))
        fail_if(bool(sequence), f"{method}/EG cached trace lacks its committed sequence")
        fail_if(isinstance(sequence_index, int) and 1 <= sequence_index <= len(sequence),
                f"{method}/EG cached trace has invalid sequence index")
        same(action, sequence[int(sequence_index) - 1], path=f"{method}.EG_cached_action")
        return
    if method == "R_SOURCE_FULL":
        candidates = trace.get("source_candidates")
        fail_if(isinstance(candidates, list) and len(candidates) == 5, "source-full trace missing five proposals")
        by_h = {int(item["h"]): tuple(map(int, item["sequence"])) for item in candidates}
        fail_if(set(by_h) == set(HORIZONS), "source-full trace has wrong horizon grid")
        q = float(frame["q_t"])
        deltas = {
            h: sum(a * b for a, b in zip(
                R_COEFFICIENTS,
                (1.0, h / 20.0, 1.0 - q, (h / 20.0) * (1.0 - q)),
                strict=True,
            ))
            for h in HORIZONS if h != 0
        }
        selected_h, selected_delta = choose_horizon(deltas)
        same(int(trace.get("selected_h")), selected_h, path="R_SOURCE_FULL.selected_h")
        same(float(trace.get("delta")), selected_delta, path="R_SOURCE_FULL.delta")
        same(action, by_h[selected_h][0], path="R_SOURCE_FULL.first_action")
        if algorithm == "EG":
            same(tuple(map(int, trace.get("committed_sequence", []))), by_h[selected_h], path="R_SOURCE_FULL.EG_commitment")
        return
    if method == "R_QUAD_H":
        q = float(frame["q_t"])
        scores = {h: horizon_proxy(h, q, quad=True) for h in HORIZONS}
        selected_h, selected_score = choose_horizon(scores)
        same(int(trace.get("selected_h")), selected_h, path="R_QUAD_H.selected_h")
        same(float(trace.get("proxy_score")), selected_score, path="R_QUAD_H.proxy_score")
        listed = {int(h): float(score) for h, score in trace.get("all_proxy_scores", [])}
        same(listed, scores, path="R_QUAD_H.all_proxy_scores")
        if algorithm == "EG":
            committed = tuple(map(int, trace.get("committed_sequence", [])))
            fail_if(bool(committed), "R_QUAD_H EG trace lacks full group commitment")
            same(action, committed[0], path="R_QUAD_H.EG_first_action")
        return
    if method == "R_UTILITY_RIDGE":
        scored, anchor, anchor_score = ridge_scores(frame, trace, ridge_coefficients)
        better = [(seq, score) for seq, score in scored if seq != anchor and score < anchor_score]
        if better:
            chosen, chosen_score = min(better, key=lambda item: (item[1], item[0]))
            used_anchor = False
        else:
            chosen, chosen_score, used_anchor = anchor, anchor_score, True
        same(tuple(map(int, trace.get("anchor_sequence", []))), anchor, path="ridge.anchor")
        same(tuple(map(int, trace.get("selected_sequence", []))), chosen, path="ridge.selected_sequence")
        same(bool(trace.get("used_anchor")), used_anchor, path="ridge.used_anchor")
        same(float(trace.get("anchor_score")), anchor_score, path="ridge.anchor_score")
        same(float(trace.get("score")), chosen_score, path="ridge.selected_score")
        if algorithm != "EG":
            same(action, chosen[0], path="ridge.first_action")
        else:
            committed = tuple(map(int, trace.get("committed_sequence", [])))
            same(committed, chosen, path="ridge.EG_commitment")
            same(action, committed[int(trace.get("sequence_index", 1)) - 1], path="ridge.EG_action")
        return
    if method == "R_ORDER_MAIN_TAU0_FULL":
        same(float(trace.get("tau")), 0.0, path="R_ORDER.tau")
        pool = trace.get("candidate_pool")
        fail_if(isinstance(pool, list) and pool, "R-Order trace missing candidate pool")
        sequences = [tuple(map(int, item["sequence"])) for item in pool]
        fail_if(len(sequences) == len(set(sequences)), "R-Order pool duplicates full sequences")
        anchors = [tuple(map(int, item["sequence"])) for item in pool if item.get("is_h20_anchor")]
        fail_if(len(anchors) == 1, "R-Order pool lacks unique H20 anchor")
        risk_origins = [
            origin for item in pool for origin in item.get("origins", [])
            if origin.get("source") == "risk"
        ]
        fail_if(len(risk_origins) == 1, "R-Order full pool does not retain one risk proposal")
        scored = [(seq, sequence_score(frame, seq, order_rates)) for seq in sequences]
        anchor = anchors[0]
        anchor_score = next(score for seq, score in scored if seq == anchor)
        better = [(seq, score) for seq, score in scored if seq != anchor and score < anchor_score]
        if better:
            chosen, chosen_score = min(better, key=lambda item: (item[1], item[0]))
            used_anchor = False
        else:
            chosen, chosen_score, used_anchor = anchor, anchor_score, True
        same(tuple(map(int, trace.get("anchor_sequence", []))), anchor, path="R_ORDER.anchor")
        same(tuple(map(int, trace.get("selected_sequence", []))), chosen, path="R_ORDER.selected_sequence")
        same(bool(trace.get("used_anchor")), used_anchor, path="R_ORDER.used_anchor")
        same(float(trace.get("anchor_score")), anchor_score, path="R_ORDER.anchor_score")
        same(float(trace.get("score")), chosen_score, path="R_ORDER.selected_score")
        if algorithm != "EG":
            same(action, chosen[0], path="R_ORDER.first_action")
        else:
            committed = tuple(map(int, trace.get("committed_sequence", [])))
            same(committed, chosen, path="R_ORDER.EG_commitment")
            same(action, committed[int(trace.get("sequence_index", 1)) - 1], path="R_ORDER.EG_action")
        return
    if method == "OLD_FROZEN_R":
        fail_if(trace.get("policy") == method, "old frozen R trace policy name mismatch")
        fail_if(trace.get("solver") == algorithm, "old frozen R trace solver mismatch")
        fail_if(int(trace.get("requested_h", -1)) in HORIZONS, "old frozen R horizon is outside frozen grid")
        if algorithm == "EG":
            nested = trace.get("native_trace") or {}
            plan = nested.get("committed_plan") or {}
            committed = tuple(map(int, plan.get("destinations_top_down", [])))
            fail_if(bool(committed), "OLD_FROZEN_R EG start trace lacks committed full plan")
            same(action, committed[0], path="OLD_FROZEN_R.EG_first_action")
        return
    if method == "NATIVE20":
        # The runner's CSV/method label is NATIVE20; the underlying actor's
        # registered fixed-H20 policy name remains FIXED20.
        fail_if(trace.get("policy") == "FIXED20", "native baseline trace policy name mismatch")
        fail_if(int(trace.get("requested_h", -1)) == 20, "native baseline is not fixed H20")
        if algorithm == "EG":
            # FIXED20's public EG trace exposes group_start/cached_moves but
            # intentionally omits the complete committed destination list.
            fail_if(isinstance(trace.get("group_start"), bool),
                    "NATIVE20 EG trace lacks group_start marker")
            fail_if(isinstance(trace.get("cached_moves"), int),
                    "NATIVE20 EG trace lacks cached_moves counter")
        return
    raise AuditFailure(f"unregistered method {method!r}")


def independent_h_prefix(frame: object, horizon: int) -> object:
    """Apply the registered visibility formula without calling the runtime truncator."""
    if horizon not in NEW_HORIZONS:
        raise AuditFailure(f"unregistered fixed horizon {horizon}")
    # Match the frozen base observation schema explicitly. R-Order public
    # frames use a subclass carrying r_order_summary, while the registered
    # runtime truncator constructs PublicObservation and drops that extension.
    from scrp.noise_aware_forecast import PublicObservation

    observation = frame.observation
    ranks = np.asarray(observation.rank_tensor, dtype=np.int32).copy()
    occupied = np.asarray(observation.occupied_mask, dtype=np.bool_)
    if ranks.shape != occupied.shape or ranks.ndim != 2:
        raise AuditFailure("public prefix input rank/occupancy shape is invalid")
    if np.any(ranks[~occupied] != 0):
        raise AuditFailure("empty padding is not rank 0 before prefixing")
    if np.count_nonzero(occupied & (ranks == 1)) != 1:
        raise AuditFailure("prefix input must contain one current rank-1 target")
    visible = occupied & ((ranks <= horizon + 1) | (ranks == 1))
    ranks[occupied & ~visible] = UNKNOWN
    ranks[~occupied] = 0
    base_observation = PublicObservation(
        rank_tensor=ranks,
        occupied_mask=occupied,
        legal_destination_mask=np.asarray(observation.legal_destination_mask, dtype=np.bool_),
        history_hits=tuple(bool(value) for value in observation.history_hits),
        q_prior=float(observation.q_prior),
        q_t=float(observation.q_t),
    )
    return replace(frame, observation=base_observation)


def check_runtime_prefix(frame: object, horizon: int, runtime: Mapping[str, Any], key: str, index: int) -> object:
    """Compare the frozen truncator with the independently constructed prefix."""
    independent = independent_h_prefix(frame, horizon)
    runtime_prefix = runtime["truncate_public_frame"](frame, horizon)
    from R.scrp.r_five_versions_actor import public_frame_record
    same(public_frame_record(runtime_prefix), public_frame_record(independent),
         path=f"{key}.prefix_H{horizon}[{index}]")
    return independent


def replay_fixed_h_proposer(
    algorithm: str, horizon: int, frame: object, runtime: Mapping[str, Any], gp_rule: Mapping[str, Any],
) -> tuple[int, dict[str, Any] | None]:
    """Call the frozen public proposer once on an independently constructed H prefix."""
    prefix = independent_h_prefix(frame, horizon)
    if algorithm == "EM":
        return int(runtime["em_select"](prefix.observation, max_tiers=prefix.max_tiers, h=horizon)), None
    if algorithm == "GP":
        from R.scrp.cross_solver_reliability_r import _run_one_horizon
        candidate = _run_one_horizon(
            prefix, requested_h=horizon, solver="GP", frozen_solver=gp_rule,
        )
        return int(candidate.destination), None
    if algorithm == "ERI":
        return int(runtime["eri_select"](prefix)), None
    if algorithm == "Chain":
        from Chain.chain_public import propose
        action, trace = propose(prefix, horizon)
        return int(action), dict(trace)
    if algorithm == "Minmax":
        from Minmax.minmax_public import propose
        action, trace = propose(prefix, horizon)
        return int(action), dict(trace)
    raise AuditFailure(f"no single-step fixed-H proposer registered for {algorithm}")


def replay_trajectory(
    csv_row: Mapping[str, str], start: Mapping[str, Any],
    steps: Sequence[Mapping[str, Any]], evaluator_rows: Sequence[Mapping[str, Any]],
    ridge_coefficients: Sequence[float], order_rates: Sequence[Sequence[float]],
    max_tiers: int, native_eg_audit: dict[str, int], runtime: Mapping[str, Any] | None = None,
    gp_rule: Mapping[str, Any] | None = None,
) -> int:
    key = csv_row["trajectory_key"]
    true_sequence = [int(value) for value in start["true_retrieval_sequence_private"]]
    forecast = [int(value) for value in start["fixed_forecast_private"]]
    initial = stacks_copy(start["initial_stacks_private"])
    fail_if(len(true_sequence) == len(set(true_sequence)), f"{key}: repeated truth target")
    fail_if(len(forecast) == len(true_sequence) - 1, f"{key}: forecast length mismatch")
    fail_if(set(forecast) == set(true_sequence[1:]), f"{key}: forecast does not permute true future")
    fail_if(start.get("layout_fingerprint") == csv_row["layout_fingerprint"], f"{key}: start layout mismatch")
    fail_if(set(map(int, (item for stack in initial for item in stack))) == set(true_sequence), f"{key}: initial stack/truth set mismatch")
    forecast_hash = hashlib.sha256(canonical(forecast)).hexdigest()
    by_key = [item for item in evaluator_rows if item.get("trajectory_key") == key]
    forecast_binding = [item for item in by_key if item.get("record_type") == "forecast_binding"]
    fail_if(len(forecast_binding) == 1, f"{key}: missing/duplicate forecast binding")
    binding = forecast_binding[0]
    same(binding.get("forecast_sha256_evaluator_only"), forecast_hash, path=f"{key}.forecast_hash")
    same(binding.get("layout_fingerprint"), csv_row["layout_fingerprint"], path=f"{key}.binding_fingerprint")
    same(binding.get("forecast_seed_evaluator_only"), None if csv_row["forecast_seed_evaluator_only"] == "" else int(csv_row["forecast_seed_evaluator_only"]), path=f"{key}.forecast_seed")
    cond = binding.get("condition_evaluator_only")
    same(cond.get("condition_id"), csv_row["condition_id"], path=f"{key}.condition_id")
    fail_if(csv_row["algorithm"] in ALGORITHMS and csv_row["method"] in METHODS, f"{key}: unregistered solver or method")
    same(key, f"{csv_row['layout_fingerprint']}|{csv_row['condition_id']}|{csv_row['algorithm']}|{csv_row['method']}", path="trajectory_key")

    state, retrieved, crane_position, crane_time = replay_initial(initial, true_sequence)
    relocations = 0
    hits = forecast_hits(true_sequence, forecast, retrieved)
    expected_initial = frame_from_private(
        state, true_sequence, forecast, retrieved, hits, crane_position, crane_time,
        relocations, max_tiers,
    )
    proxy_records: deque[float] = deque(maxlen=HISTORY_WINDOW)
    nsteps = len(steps)
    same(int(csv_row["decision_count"]), nsteps, path=f"{key}.decision_count")
    fail_if(nsteps > 0, f"{key}: empty action trace")
    physical_by_index = {
        int(item["decision_index"]): item for item in evaluator_rows
        if item.get("record_type") == "physical_step" and item.get("trajectory_key") == key
    }
    fail_if(len(physical_by_index) == nsteps, f"{key}: physical step count does not match public trace")
    pending_eg_sequence: tuple[int, ...] | None = None
    pending_eg_index = 0
    pending_native_eg_sequence: tuple[int, ...] | None = None
    pending_native_eg_index = 0
    native_eg_cached_moves = 0
    fixed_h_eg_cached_moves = 0

    for index, step in enumerate(steps):
        same(int(step.get("decision_index", -1)), index, path=f"{key}.step_index")
        same(step.get("algorithm", step.get("solver")), csv_row["algorithm"], path=f"{key}.step_algorithm_or_solver")
        same(step.get("method"), csv_row["method"], path=f"{key}.step_method")
        for field in forbidden_actor_fields(step.get("actor_trace") or {}):
            raise AuditFailure(f"{key}: private/evaluator key inside actor trace: {field}")
        actual_before = step["public_frame_before"]
        expected_before = decorate_history_summary(expected_initial, list(proxy_records))
        check_frame(actual_before, expected_before, f"{key}.before[{index}]")
        phys = physical_by_index[index]
        same(phys.get("physical_stacks_before_private"), state, path=f"{key}.physical_before[{index}]")
        same(int(phys.get("retrieval_count_before", -1)), retrieved, path=f"{key}.retrieval_before[{index}]")
        same(int(phys.get("relocation_count_before", -1)), relocations, path=f"{key}.relocation_before[{index}]")
        candidate_policy_check(csv_row, step, ridge_coefficients, order_rates)
        action = int(step["own_action"])
        legal = list(actual_before["legal_destinations"])
        if action not in legal:
            raise AuditFailure(f"{key}: step {index} chose illegal destination {action}")
        if csv_row["method"] in FIXED_METHODS and csv_row["algorithm"] != "EG":
            fail_if(runtime is not None and gp_rule is not None,
                    "fixed-H proposer consistency replay requires the isolated frozen runtime and GP rule")
            horizon = int(csv_row["method"].removeprefix("FIXED_H"))
            from R.scrp.r_five_versions_actor import strict_public_frame_from_record
            full_frame = strict_public_frame_from_record(actual_before)
            prefix = check_runtime_prefix(full_frame, horizon, runtime, key, index)
            replayed, proposer_trace = replay_fixed_h_proposer(
                csv_row["algorithm"], horizon, prefix, runtime, gp_rule,
            )
            same(action, replayed, path=f"{key}.{csv_row['method']}.frozen_proposer_action[{index}]")
            if proposer_trace is not None:
                observed_trace = step.get("actor_trace") or {}
                for trace_field, expected_value in proposer_trace.items():
                    same(observed_trace.get(trace_field), expected_value,
                         path=f"{key}.{csv_row['method']}.proposer_trace.{trace_field}[{index}]")
        if csv_row["algorithm"] == "EG":
            actor_trace = step.get("actor_trace") or {}
            cached = bool(actor_trace.get("cached_move"))
            group_start = bool(actor_trace.get("group_start"))
            if csv_row["method"] in FIXED_METHODS:
                fail_if(runtime is not None, "fixed-H EG audit requires the isolated frozen runtime")
                horizon = int(csv_row["method"].removeprefix("FIXED_H"))
                from R.scrp.r_five_versions_actor import strict_public_frame_from_record
                full_frame = strict_public_frame_from_record(actual_before)
                prefix = check_runtime_prefix(full_frame, horizon, runtime, key, index)
                same(group_start, pending_eg_sequence is None,
                     path=f"{key}.{csv_row['method']}.EG_group_boundary[{index}]")
                if group_start:
                    target = int(true_sequence[retrieved])
                    target_source, target_tier = locate(state, target)
                    blockers = len(state[target_source]) - target_tier - 1
                    fail_if(blockers > 0, f"{key}: fixed-H EG group start has no blocked target")
                    plan = build_eg20_plan(prefix)
                    committed = tuple(map(int, plan.destinations_top_down))
                    same(int(plan.source), target_source, path=f"{key}.{csv_row['method']}.EG_source[{index}]")
                    source_height = int(np.count_nonzero(
                        np.asarray(prefix.observation.occupied_mask[target_source], dtype=np.bool_)
                    ))
                    expected_blockers = tuple(
                        int(value) for value in
                        prefix.observation.rank_tensor[target_source, target_tier + 1 : source_height][::-1]
                    )
                    same(tuple(map(int, plan.blocker_ranks_top_down)), expected_blockers,
                         path=f"{key}.{csv_row['method']}.EG_blockers[{index}]")
                    same(len(committed), blockers, path=f"{key}.{csv_row['method']}.EG_commit_length[{index}]")
                    same(action, committed[0], path=f"{key}.{csv_row['method']}.EG_commit_first_action[{index}]")
                    # The actor intentionally records only group_start/cached_moves;
                    # this is an audit-side reconstruction from the public prefix.
                    pending_eg_sequence = committed if len(committed) > 1 else None
                    pending_eg_index = 1
                    native_eg_audit["fixed_h_group_count"] = native_eg_audit.get("fixed_h_group_count", 0) + 1
                    native_eg_audit["fixed_h_action_count"] = native_eg_audit.get("fixed_h_action_count", 0) + 1
                else:
                    fail_if(pending_eg_sequence is not None,
                            f"{key}: fixed-H EG cached move has no pending reconstructed commitment")
                    same(action, pending_eg_sequence[pending_eg_index],
                         path=f"{key}.{csv_row['method']}.EG_cached_action[{index}]")
                    pending_eg_index += 1
                    fixed_h_eg_cached_moves += 1
                    native_eg_audit["fixed_h_action_count"] = native_eg_audit.get("fixed_h_action_count", 0) + 1
                    native_eg_audit["fixed_h_cached_move_count"] = native_eg_audit.get("fixed_h_cached_move_count", 0) + 1
                    if pending_eg_index == len(pending_eg_sequence):
                        pending_eg_sequence = None
                        pending_eg_index = 0
                same(int(actor_trace.get("cached_moves", -1)), fixed_h_eg_cached_moves,
                     path=f"{key}.{csv_row['method']}.EG_cached_moves[{index}]")
            elif csv_row["method"] == "NATIVE20":
                if group_start:
                    fail_if(pending_native_eg_sequence is None,
                            f"{key}: NATIVE20 EG started a new group before the prior blocker group ended")
                    target = int(true_sequence[retrieved])
                    target_source, target_tier = locate(state, target)
                    blockers = len(state[target_source]) - target_tier - 1
                    fail_if(blockers > 0, f"{key}: NATIVE20 EG group start has no blocked target")
                    plan = rebuild_native_eg_plan(actual_before)
                    native_sequence = tuple(map(int, plan.destinations_top_down))
                    same(int(plan.source), target_source, path=f"{key}.NATIVE20.EG_source[{index}]")
                    source_height = sum(
                        bool(value) for value in actual_before["occupied_mask"][target_source]
                    )
                    expected_blocker_ranks = tuple(
                        int(value)
                        for value in actual_before["rank_tensor"][target_source][target_tier + 1 : source_height][::-1]
                    )
                    same(tuple(map(int, plan.blocker_ranks_top_down)), expected_blocker_ranks,
                         path=f"{key}.NATIVE20.EG_blocker_ranks[{index}]")
                    same(len(native_sequence), blockers,
                         path=f"{key}.NATIVE20.EG_group_length[{index}]")
                    fail_if(bool(native_sequence), f"{key}: NATIVE20 EG builder returned an empty plan")
                    same(action, native_sequence[0], path=f"{key}.NATIVE20.EG_group_first_action[{index}]")
                    pending_native_eg_sequence = native_sequence if len(native_sequence) > 1 else None
                    pending_native_eg_index = 1
                    native_eg_audit["group_count"] += 1
                    native_eg_audit["action_count"] += 1
                else:
                    fail_if(pending_native_eg_sequence is not None,
                            f"{key}: NATIVE20 EG continuation has no pending blocker group")
                    same(action, pending_native_eg_sequence[pending_native_eg_index],
                         path=f"{key}.NATIVE20.EG_cached_action[{index}]")
                    pending_native_eg_index += 1
                    native_eg_audit["action_count"] += 1
                    native_eg_audit["cached_move_count"] += 1
                    if pending_native_eg_index == len(pending_native_eg_sequence):
                        pending_native_eg_sequence = None
                        pending_native_eg_index = 0
                    native_eg_cached_moves += 1
                same(int(actor_trace.get("cached_moves", -1)), native_eg_cached_moves,
                     path=f"{key}.NATIVE20.EG_cached_moves[{index}]")
            elif group_start:
                fail_if(pending_eg_sequence is None, f"{key}: EG replaced an unfinished committed group")
                sequence = actor_trace.get("committed_sequence")
                if sequence is None:
                    nested = actor_trace.get("native_trace") or {}
                    sequence = (nested.get("committed_plan") or {}).get("destinations_top_down")
                committed = tuple(map(int, sequence or []))
                fail_if(bool(committed), f"{key}: EG group start has no full sequence")
                targets = [
                    (s, t) for s, ranks in enumerate(actual_before["rank_tensor"])
                    for t, rank in enumerate(ranks)
                    if actual_before["occupied_mask"][s][t] and rank == 1
                ]
                fail_if(len(targets) == 1, f"{key}: EG frame lacks one current target")
                source, target_tier = targets[0]
                blocker_count = sum(actual_before["occupied_mask"][source]) - target_tier - 1
                same(len(committed), blocker_count, path=f"{key}.EG_group_length[{index}]")
                same(action, committed[0], path=f"{key}.EG_group_first_action[{index}]")
                if csv_row["method"] in FIXED_METHODS:
                    fail_if(runtime is not None, "fixed-H EG audit requires the isolated frozen runtime")
                    horizon = int(csv_row["method"].removeprefix("FIXED_H"))
                    from R.scrp.r_five_versions_actor import strict_public_frame_from_record
                    full_frame = strict_public_frame_from_record(actual_before)
                    projected = check_runtime_prefix(full_frame, horizon, runtime, key, index)
                    rebuilt = build_eg20_plan(projected)
                    same(committed, tuple(map(int, rebuilt.destinations_top_down)),
                         path=f"{key}.{csv_row['method']}.EG_rebuilt_commit[{index}]")
                    same(int(rebuilt.source), source, path=f"{key}.{csv_row['method']}.EG_rebuilt_source[{index}]")
                pending_eg_sequence = committed if len(committed) > 1 else None
                pending_eg_index = 1
            else:
                fail_if(cached and pending_eg_sequence is not None,
                        f"{key}: EG cached move has no pending full plan")
                fail_if(pending_eg_sequence is not None,
                        f"{key}: EG omitted group-start marker for a new plan")
                if cached:
                    sequence = actor_trace.get("committed_sequence")
                    sequence_index = actor_trace.get("sequence_index")
                    if sequence is None:
                        nested = actor_trace.get("native_trace") or {}
                        sequence = (nested.get("committed_plan") or {}).get("destinations_top_down")
                        sequence_index = nested.get("group_move_index")
                    same(tuple(map(int, sequence or [])), pending_eg_sequence, path=f"{key}.EG_cached_sequence[{index}]")
                    same(int(sequence_index), pending_eg_index + 1, path=f"{key}.EG_sequence_index[{index}]")
                    same(action, pending_eg_sequence[pending_eg_index], path=f"{key}.EG_cached_action[{index}]")
                    pending_eg_index += 1
                    if pending_eg_index == len(pending_eg_sequence):
                        pending_eg_sequence = None
                        pending_eg_index = 0
                else:
                    fail_if(False, f"{key}: EG continuation has no cached_move marker")
        target = int(true_sequence[retrieved])
        source, target_tier = locate(state, target)
        if target_tier >= len(state[source]) - 1:
            raise AuditFailure(f"{key}: step {index} acts on already accessible target")
        blocker = state[source].pop()
        source_position, destination_position = source + 1, action + 1
        if crane_position is not None:
            crane_time += abs(crane_position - source_position) * CRANE_TRAVEL
        crane_time += abs(source_position - destination_position) * CRANE_TRAVEL + PICK_DROP
        crane_position = destination_position
        state[action].append(blocker)
        relocations += 1
        before_retrieved = retrieved
        while retrieved < len(true_sequence):
            current = int(true_sequence[retrieved])
            current_source, current_tier = locate(state, current)
            if current_tier != len(state[current_source]) - 1:
                break
            source_position = current_source + 1
            if crane_position is not None:
                crane_time += abs(crane_position - source_position) * CRANE_TRAVEL
            crane_time += source_position * CRANE_TRAVEL + PICK_DROP
            crane_position = 0
            state[current_source].pop()
            retrieved += 1
        same(int(phys.get("retrieval_count_after", -1)), retrieved, path=f"{key}.retrieval_after[{index}]")
        same(int(phys.get("relocation_count_after", -1)), relocations, path=f"{key}.relocation_after[{index}]")
        same(phys.get("physical_stacks_after_private"), state, path=f"{key}.physical_after[{index}]")
        hits = forecast_hits(true_sequence, forecast, retrieved)
        same(phys.get("provider_history_hits_private"), hits, path=f"{key}.private_hits_after[{index}]")
        terminal = retrieved == len(true_sequence)
        same(bool(step.get("terminal_after_action")), terminal, path=f"{key}.terminal[{index}]")
        next_frame = frame_from_private(
            state, true_sequence, forecast, retrieved, hits,
            crane_position, crane_time, relocations, max_tiers,
        )
        actual_after = step.get("public_frame_after")
        if terminal:
            same(actual_after, None, path=f"{key}.terminal_frame[{index}]")
        else:
            fail_if(isinstance(actual_after, dict), f"{key}: missing nonterminal after frame")
            check_frame(actual_after, next_frame, f"{key}.after[{index}]")
            proxy = public_history_record(actual_before, action, actual_after)
            if proxy is not None:
                proxy_records.append(float(proxy))
            expected_initial = next_frame
        fail_if(retrieved >= before_retrieved, f"{key}: retrieval count moved backwards at action")
    fail_if(pending_eg_sequence is None, f"{key}: trajectory ended with an incomplete EG group")
    if csv_row["method"] == "NATIVE20" and csv_row["algorithm"] == "EG":
        fail_if(pending_native_eg_sequence is None,
                f"{key}: NATIVE20 EG ended with an incomplete blocker group")
    same(bool(csv_row["complete"].lower() == "true"), True, path=f"{key}.complete")
    same(int(csv_row["N"]), len(true_sequence), path=f"{key}.N")
    same(int(csv_row["R_raw"]), relocations, path=f"{key}.R_raw")
    same(float(csv_row["R_over_N"]), relocations / float(len(true_sequence)), path=f"{key}.R_over_N")
    same(int(csv_row["decision_count"]), relocations, path=f"{key}.actions_equal_relocations")
    same(int(csv_row["decision_count"]), nsteps, path=f"{key}.steps")
    return nsteps


def run_audit(scope: str, run_dir: Path, output_path: Path) -> dict[str, Any]:
    expected_count = 12 if scope == "smoke" else 10080
    raw_path = run_dir / "raw_trajectories.csv"
    public_path = run_dir / "public_decision_trace.jsonl.gz"
    evaluator_path = run_dir / "private/evaluator_sidecar.jsonl.gz"
    physical_path = run_dir / "private/physical_steps.jsonl.gz"
    release_path = RUN_ROOT / ("ROOT_M1_SMOKE_RELEASE.json" if scope == "smoke" else "ROOT_M1_BANK_RELEASE.json")
    receipt_path = run_dir / ("M1_SMOKE_RECEIPT.json" if scope == "smoke" else "M1_BANK_RECEIPT.json")
    smoke_raw_path = RUN_ROOT / "smoke_01/raw_trajectories.csv"
    paths = [raw_path, public_path, evaluator_path, physical_path, LAYOUT_PATH, CONFIG_PATH,
             BINDINGS_PATH, RIDGE_PATH, ORDER_MODEL_PATH, ERI_EG_BUILDER_PATH,
             release_path, receipt_path]
    if scope == "bank":
        paths.append(smoke_raw_path)
    for path in paths:
        fail_if(path.is_file(), f"required audit input missing: {path}")
    config = read_json(CONFIG_PATH)
    manifest = read_json(LAYOUT_PATH)
    bindings = read_json(BINDINGS_PATH)
    release = read_json(release_path)
    runner_receipt = read_json(receipt_path)
    m0_test_receipt = read_json(RUN_ROOT / "m0/tests/M0_TEST_RECEIPT.json")
    same(release.get("scope"), scope, path="root_release.scope")
    same(release.get("authorized"), True, path="root_release.authorized")
    same(int(release.get("trajectory_count", -1)), expected_count, path="root_release.trajectory_count")
    same(release.get("m0_binding_sha256"), bindings.get("binding_sha256"), path="root_release.m0_binding_sha256")
    same(release.get("m0_test_receipt_sha256"), m0_test_receipt.get("receipt_sha256"), path="root_release.m0_test_receipt_sha256")
    same(runner_receipt.get("protocol_id"), "r_five_versions_comparison_v1", path="runner_receipt.protocol_id")
    same(runner_receipt.get("scope"), scope, path="runner_receipt.scope")
    same(runner_receipt.get("status"), "PASS", path="runner_receipt.status")
    same(int(runner_receipt.get("trajectory_count", -1)), expected_count, path="runner_receipt.trajectory_count")
    same(int(runner_receipt.get("unique_trajectory_keys", -1)), expected_count, path="runner_receipt.unique_trajectory_keys")
    same(runner_receipt.get("authorized_by_release_sha256"), sha256(release_path), path="runner_receipt.release_sha256")
    receipt_payload = dict(runner_receipt)
    recorded_receipt_sha = receipt_payload.pop("receipt_sha256", None)
    same(recorded_receipt_sha, hashlib.sha256(canonical(receipt_payload)).hexdigest(), path="runner_receipt.receipt_sha256")
    m0_test_payload = dict(m0_test_receipt)
    recorded_m0_test_sha = m0_test_payload.pop("receipt_sha256", None)
    same(recorded_m0_test_sha, hashlib.sha256(canonical(m0_test_payload)).hexdigest(), path="m0_test_receipt.receipt_sha256")
    model_doc = read_json(ORDER_MODEL_PATH)
    ridge_doc = read_json(RIDGE_PATH)
    ridge_coefficients = [float(value) for value in ridge_doc["coefficients"]]
    order_rates = [[float(value) for value in row] for row in model_doc["main"]["coefficients"]]
    fail_if(len(ridge_coefficients) == 74, "frozen Ridge coefficient width is not 74")
    fail_if(len(order_rates) == 20 and all(len(row) == 5 for row in order_rates), "R-Order coefficient shape is not [20,5]")
    fail_if(bindings["binding_sha256"] == "5788b3e4c8489c37e724b267241a92d913836895a87865961d044e250831007f", "M0 binding is not the released frozen binding")
    expected_model_hashes = {
        "R/output/universal_r_utility_v1/r02/M1_FIT_01/RIDGE_MODEL.json": sha256(RIDGE_PATH),
        "R/output/r_order_v1/r01/M1_MODELS.json": sha256(ORDER_MODEL_PATH),
    }
    for rel, digest in expected_model_hashes.items():
        same(bindings["source_sha256"].get(rel), digest, path=f"binding.{rel}")
    for rel, expected_hash in bindings.get("source_sha256", {}).items():
        source_path = WORKSPACE / Path(rel)
        fail_if(source_path.is_file(), f"bound source missing: {rel}")
        same(sha256(source_path), expected_hash, path=f"live_source_hash.{rel}")
    same(
        bindings["source_sha256"].get("R/scrp/eri_eg_public.py"),
        sha256(ERI_EG_BUILDER_PATH),
        path="binding.R/scrp/eri_eg_public.py_for_native_EG_rebuild",
    )

    layouts = {row["label_invariant_fingerprint"]: row for row in manifest}
    fail_if(len(layouts) == 20, "selected layout manifest does not contain 20 unique layouts")
    conditions = condition_registry()
    with raw_path.open("r", encoding="utf-8-sig", newline="") as stream:
        raw_rows = list(csv.DictReader(stream))
    fail_if(len(raw_rows) == expected_count, f"raw row count {len(raw_rows)} != {expected_count}")
    raw_by_key: dict[str, dict[str, str]] = {}
    for row in raw_rows:
        key = row.get("trajectory_key", "")
        fail_if(key and key not in raw_by_key, f"duplicate/empty raw trajectory key {key!r}")
        raw_by_key[key] = row
        fail_if(row.get("layout_fingerprint") in layouts, f"{key}: layout fingerprint absent from frozen manifest")
        fail_if(row.get("condition_id") in conditions, f"{key}: condition is not registered")
        condition = conditions[row["condition_id"]]
        for field, value in condition.items():
            actual: Any = row[field]
            if field == "p":
                actual = None if actual == "" else float(actual)
            elif field == "window_size":
                actual = None if actual == "" else int(actual)
            elif field == "repeat":
                actual = int(actual)
            same(actual, value, path=f"{key}.{field}")
        layout = layouts[row["layout_fingerprint"]]
        same(row["layout_id"], layout["episode_id"], path=f"{key}.layout_id")
        same(row["geometry"], layout["parameter_group"], path=f"{key}.geometry")
    if scope == "smoke":
        expected_keys = {
            f"{fingerprint}|delayed_positions|p=0.3|r=0|{algorithm}|{method}"
            for fingerprint, layout in layouts.items()
            if layout["episode_id"] == "noise-r01-select-G1-000"
            for algorithm in ALGORITHMS
            for method in ("OLD_FROZEN_R", "R_UTILITY_RIDGE", "R_ORDER_MAIN_TAU0_FULL")
        }
    else:
        expected_keys = {
            f"{fingerprint}|{condition_id}|{algorithm}|{method}"
            for fingerprint in layouts for condition_id in conditions
            for algorithm in ALGORITHMS for method in METHODS
        }
    same(set(raw_by_key), expected_keys, path="registered_complete_key_bank")

    public_rows = read_jsonl_gz(public_path)
    evaluator_rows = read_jsonl_gz(evaluator_path)
    physical_rows = read_jsonl_gz(physical_path)
    starts = [row for row in evaluator_rows if row.get("record_type") == "trajectory_start"]
    bindings_rows = [row for row in evaluator_rows if row.get("record_type") == "forecast_binding"]
    steps_by_key: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in public_rows:
        fail_if(row.get("trajectory_key") in raw_by_key, "public trace has an unknown trajectory key")
        steps_by_key[row["trajectory_key"]].append(row)
    physical_by_key: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in physical_rows:
        fail_if(row.get("trajectory_key") in raw_by_key, "physical sidecar has an unknown trajectory key")
        physical_by_key[row["trajectory_key"]].append(row)
    start_by_key: dict[str, dict[str, Any]] = {}
    evaluator_by_key: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in evaluator_rows:
        key = row.get("trajectory_key")
        fail_if(key in raw_by_key, "evaluator sidecar has an unknown trajectory key")
        evaluator_by_key[key].append(row)
    for start in starts:
        key = start.get("trajectory_key")
        fail_if(key in raw_by_key and key not in start_by_key, f"duplicate/unknown trajectory start {key}")
        start_by_key[key] = start
    fail_if(set(start_by_key) == expected_keys, "evaluator trajectory_start key set does not match CSV")
    fail_if(set(steps_by_key) == expected_keys, "public trace key set does not match CSV")
    fail_if(set(physical_by_key) == expected_keys, "physical sidecar key set does not match CSV")
    fail_if(len(bindings_rows) == expected_count, "forecast binding count differs from trajectory count")

    total_steps = 0
    solver_method_steps: Counter[tuple[str, str]] = Counter()
    native_eg_audit = {"group_count": 0, "action_count": 0, "cached_move_count": 0}
    for key, row in raw_by_key.items():
        steps = sorted(steps_by_key.get(key, []), key=lambda item: int(item["decision_index"]))
        evaluator_for_key = evaluator_by_key[key] + physical_by_key[key]
        physical_for_key = [
            item for item in physical_by_key.get(key, [])
        ]
        if len(physical_for_key) != len(steps):
            raise AuditFailure(f"{key}: physical/public step record counts differ")
        total_steps += replay_trajectory(
            row, start_by_key[key], steps, evaluator_for_key,
            ridge_coefficients, order_rates,
            int(layouts[row["layout_fingerprint"]]["max_tiers"]),
            native_eg_audit,
        )
        solver_method_steps[(row["algorithm"], row["method"])] += len(steps)

    native_eg_steps = solver_method_steps.get(("EG", "NATIVE20"), 0)
    native_eg_csv_cached_moves = sum(
        int(row["eg_cached_moves"])
        for row in raw_by_key.values()
        if row["algorithm"] == "EG" and row["method"] == "NATIVE20"
    )
    same(native_eg_audit["action_count"], native_eg_steps,
         path="native20_EG.reconstructed_action_count")
    same(native_eg_audit["cached_move_count"], native_eg_csv_cached_moves,
         path="native20_EG.cached_move_summary")
    same(native_eg_audit["action_count"],
         native_eg_audit["group_count"] + native_eg_audit["cached_move_count"],
         path="native20_EG.group_plus_cached_action_count")
    fail_if(scope == "smoke" or native_eg_audit["group_count"] > 0,
            "bank audit did not reconstruct any Native20 EG group plans")

    if scope == "bank":
        smoke_path = smoke_raw_path
        with smoke_path.open("r", encoding="utf-8-sig", newline="") as stream:
            smoke_rows = {row["trajectory_key"]: row for row in csv.DictReader(stream)}
        for key, smoke_row in smoke_rows.items():
            same(raw_by_key.get(key), smoke_row, path=f"bank.smoke_reuse.{key}")
        fail_if(len(smoke_rows) == 12, "bank smoke reuse comparison did not find 12 rows")

    report = {
        "protocol_id": "r_five_versions_comparison_v1",
        "run_id": "r01",
        "scope": scope,
        "status": "PASS",
        "performance_scoring": False,
        "trajectory_count": len(raw_by_key),
        "unique_trajectory_keys": len(raw_by_key),
        "step_count": total_steps,
        "native20_EG_reconstructed_group_count": native_eg_audit["group_count"],
        "native20_EG_reconstructed_action_count": native_eg_audit["action_count"],
        "native20_EG_cached_move_count": native_eg_audit["cached_move_count"],
        "public_trace_records": len(public_rows),
        "physical_step_records": len(physical_rows),
        "trajectory_start_records": len(starts),
        "forecast_binding_records": len(bindings_rows),
        "registered_conditions": len(conditions),
        "registered_methods": list(METHODS),
        "registered_solvers": list(ALGORITHMS),
        "checks": {
            "root_release_runner_receipt_and_receipt_hash_bindings": True,
            "all_M0_bound_source_hashes_still_match": True,
            "complete_registered_key_bank": True,
            "layout_geometry_and_condition_registry": True,
            "private_physical_replay_including_auto_retrievals": True,
            "public_rank_occupancy_legality_reconstruction": True,
            "shared_forecast_hash_and_truth_permutation": True,
            "forecast_history_hits_and_q_t": True,
            "public_action_history_summary": True,
            "crane_position_and_time": True,
            "R_sourcefull_quad_ridge_and_order_selection_recomputed": True,
            "old_frozen_R_horizon_metadata_checked_M0_delegate_regression_covers_four_solvers": True,
            "eg_full_sequence_commitment_checked_for_R_policies": True,
            "native20_EG_group_boundaries_and_cached_move_count_reconstructed": True,
            "native20_EG_full_committed_sequence_trace_visible": False,
            "native20_EG_full_sequence_rebuilt_from_public_group_start_frames": True,
            "native20_EG_every_reconstructed_group_action_matches": True,
            "native20_EG_rebuild_uses_M0_hash_bound_pure_builder": True,
            "private_fields_absent_from_actor_trace": True,
            "R_raw_RN_and_decision_count": True,
            "bank_reuses_exact_smoke_rows": scope == "bank",
        },
        "method_solver_step_counts": {
            f"{algorithm}/{method}": count
            for (algorithm, method), count in sorted(solver_method_steps.items())
        },
        "scope_notes": {
            "physics": "Every public action is joined to a private physical step; source blocker removal, destination insertion, automatic retrieval cascade, crane position/time, relocation count, and termination are replayed from true_sequence.",
            "public_forecast_and_history": "For every frame, ranks, occupied/legal masks, shared F, causal reveal hits, q_t, public history summary, and private provider hits are independently reconstructed from trajectory_start true_sequence plus fixed F.",
            "candidate_scoring": "R_SOURCE_FULL delta/h choice, R_QUAD_H proxy scores, frozen Ridge scores, and R-Order sequence scores/anchor/tau=0 selection are recomputed. OLD_FROZEN_R checks registered policy/solver/horizon metadata; its selector is separately covered by the hash-bound M0 four-solver delegate regression.",
            "native20_EG_reconstruction": "FIXED20/NATIVE20 EG omits the destination plan from actor_trace. For every public group-start frame, this audit calls the M0-hash-bound pure build_eg20_plan(frame) from R/scrp/eri_eg_public.py, then checks every recorded action against the rebuilt sequence and verifies group boundaries and cumulative cached_moves. This reuses the frozen strategy generator rather than independently reimplementing EG mathematics; it instantiates no actor/environment and runs no trajectory.",
            "performance": "This is correctness/provenance auditing only; it does not compute treatment effects or a winner."
        },
        "input_sha256": {str(path.relative_to(WORKSPACE)): sha256(path) for path in paths},
        "audit_script_sha256": sha256(SCRIPT),
        "m0_binding_sha256": bindings["binding_sha256"],
        "m0_test_receipt_sha256": m0_test_receipt["receipt_sha256"],
        "root_release_sha256": sha256(release_path),
        "runner_receipt_sha256": runner_receipt["receipt_sha256"],
        "bound_source_sha256": bindings["source_sha256"],
        "model_hashes": expected_model_hashes,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope", choices=("smoke", "bank"), required=True)
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    run_dir = args.run_dir or RUN_ROOT / ("smoke_01" if args.scope == "smoke" else "bank_01")
    if not run_dir.is_absolute():
        run_dir = WORKSPACE / run_dir
    output_path = args.output or RUN_ROOT / f"analysis/{args.scope}_raw_physics_public_audit.json"
    if not output_path.is_absolute():
        output_path = WORKSPACE / output_path
    try:
        report = run_audit(args.scope, run_dir, output_path)
    except BaseException as exc:
        failure = {
            "protocol_id": "r_five_versions_comparison_v1",
            "run_id": "r01",
            "scope": args.scope,
            "status": "FAIL",
            "performance_scoring": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "audit_script_sha256": sha256(SCRIPT),
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(failure, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(json.dumps(failure, ensure_ascii=False))
        return 1
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
