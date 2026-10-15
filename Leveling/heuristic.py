"""Leveling heuristic L for the restricted Online Container Relocation Problem.

This is an independent implementation of the rule in Zehendner, Feillet, and
Jaillet (2017): relocate each blocker above the current target to the legal
stack with minimum height; break height ties by the leftmost stack.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence


@dataclass(frozen=True)
class Episode:
    episode_id: str
    canonical_id: str
    num_stacks: int
    max_tiers: int
    initial_stacks: tuple[tuple[int, ...], ...]
    retrieval_sequence: tuple[int, ...]
    parameter_group: str
    metadata: Mapping[str, object]

    @classmethod
    def from_record(cls, record: Mapping[str, object]) -> "Episode":
        return cls(
            episode_id=str(record["episode_id"]),
            canonical_id=str(record["canonical_id"]),
            num_stacks=int(record["num_stacks"]),
            max_tiers=int(record["max_tiers"]),
            initial_stacks=tuple(
                tuple(int(container) for container in stack)
                for stack in record["initial_stacks"]
            ),
            retrieval_sequence=tuple(
                int(container) for container in record["retrieval_sequence"]
            ),
            parameter_group=str(record["parameter_group"]),
            metadata=dict(record.get("metadata", {})),
        )

    def validate(self) -> None:
        if self.num_stacks < 2:
            raise ValueError("an OCRP episode requires at least two stacks")
        if len(self.initial_stacks) != self.num_stacks:
            raise ValueError("num_stacks does not match initial_stacks")
        if any(len(stack) > self.max_tiers for stack in self.initial_stacks):
            raise ValueError("an initial stack exceeds max_tiers")
        containers = tuple(container for stack in self.initial_stacks for container in stack)
        if len(containers) != len(set(containers)):
            raise ValueError("container identifiers must be unique")
        if set(containers) != set(self.retrieval_sequence):
            raise ValueError("retrieval_sequence must contain every container exactly once")


@dataclass(frozen=True)
class LevelingResult:
    episode_id: str
    canonical_id: str
    scale_id: str
    relocations: int


def load_episode_manifest(path: str | Path) -> tuple[Episode, ...]:
    records = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(records, list) or not records:
        raise ValueError("episode manifest must be a non-empty JSON list")
    episodes = tuple(Episode.from_record(record) for record in records)
    for episode in episodes:
        episode.validate()
    return episodes


def select_destination(
    stack_heights: Sequence[int],
    source_stack: int,
    max_tiers: int,
) -> int:
    """Return the lowest legal stack, breaking ties by leftmost index."""

    legal = [
        stack_id
        for stack_id, height in enumerate(stack_heights)
        if stack_id != source_stack and height < max_tiers
    ]
    if not legal:
        raise RuntimeError("no legal destination exists for the blocking container")
    return min(legal, key=lambda stack_id: (stack_heights[stack_id], stack_id))


def run_leveling(episode: Episode) -> LevelingResult:
    """Retrieve one episode using the paper's restricted Leveling heuristic."""

    episode.validate()
    stacks = [list(stack) for stack in episode.initial_stacks]
    relocations = 0

    for target in episode.retrieval_sequence:
        source_stack = next(
            (stack_id for stack_id, stack in enumerate(stacks) if target in stack),
            None,
        )
        if source_stack is None:
            raise RuntimeError(f"target {target} is absent from the current bay")

        while stacks[source_stack][-1] != target:
            destination = select_destination(
                [len(stack) for stack in stacks],
                source_stack,
                episode.max_tiers,
            )
            stacks[destination].append(stacks[source_stack].pop())
            relocations += 1
        stacks[source_stack].pop()

    if any(stacks):
        raise AssertionError("the bay is not empty after the retrieval sequence")

    initial_height = max(len(stack) for stack in episode.initial_stacks)
    return LevelingResult(
        episode_id=episode.episode_id,
        canonical_id=episode.canonical_id,
        scale_id=f"S{initial_height}-W{episode.num_stacks}",
        relocations=relocations,
    )
