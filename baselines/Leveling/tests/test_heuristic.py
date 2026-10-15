from __future__ import annotations

import json
from pathlib import Path

try:
    from baselines.Leveling.evaluate_caserta import (
        PAPER_TABLE2_LEVELING_MEANS,
        round_half_up,
        summarize,
    )
    from baselines.Leveling.heuristic import (
        Episode,
        load_episode_manifest,
        run_leveling,
        select_destination,
    )
except ModuleNotFoundError:
    from evaluate_caserta import PAPER_TABLE2_LEVELING_MEANS, round_half_up, summarize
    from heuristic import Episode, load_episode_manifest, run_leveling, select_destination


PACKAGE_DIR = Path(__file__).resolve().parents[1]
CASERTA = PACKAGE_DIR / "data" / "caserta_table2_840.json"


def test_selects_lowest_stack_then_leftmost() -> None:
    assert select_destination([3, 1, 1, 2], source_stack=0, max_tiers=5) == 1
    assert select_destination([3, 2, 1, 2], source_stack=0, max_tiers=5) == 2


def test_restricted_relocation_count() -> None:
    episode = Episode(
        episode_id="small",
        canonical_id="small",
        num_stacks=3,
        max_tiers=3,
        initial_stacks=((1, 3, 2), (), ()),
        retrieval_sequence=(1, 2, 3),
        parameter_group="test",
        metadata={},
    )
    assert run_leveling(episode).relocations == 2


def test_exact_caserta_corpus_reproduces_paper_table2() -> None:
    episodes = load_episode_manifest(CASERTA)
    results = tuple(run_leveling(episode) for episode in episodes)
    summary = summarize(results, runtime_seconds=1.0)
    assert len(results) == 840
    assert summary["all_21_paper_means_reproduced_at_2_decimals"] is True
    for scale_id, values in summary["by_scale"].items():
        paper_group = values["paper_group_W-S"]
        assert round_half_up(values["mean_relocations"]) == PAPER_TABLE2_LEVELING_MEANS[paper_group]


def test_manifest_is_not_modified_by_evaluation(tmp_path: Path) -> None:
    before = CASERTA.read_bytes()
    episodes = load_episode_manifest(CASERTA)
    tuple(run_leveling(episode) for episode in episodes[:2])
    assert CASERTA.read_bytes() == before
