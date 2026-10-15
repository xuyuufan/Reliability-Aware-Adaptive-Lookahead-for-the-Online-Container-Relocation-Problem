"""Evaluate paper Leveling L on the exact Caserta Table 2 benchmark."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
import time
from collections import Counter, defaultdict
from dataclasses import asdict
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Sequence

if __package__:
    from .heuristic import Episode, LevelingResult, load_episode_manifest, run_leveling
else:
    from heuristic import Episode, LevelingResult, load_episode_manifest, run_leveling


PACKAGE_DIR = Path(__file__).resolve().parent
DEFAULT_BENCHMARK = PACKAGE_DIR / "data" / "caserta_table2_840.json"
DEFAULT_OUTPUT_DIR = PACKAGE_DIR / "results" / "caserta_table2_840"


PAPER_TABLE2_GROUPS = (
    "3-3", "3-4", "3-5", "3-6", "3-7", "3-8",
    "4-4", "4-5", "4-6", "4-7",
    "5-4", "5-5", "5-6", "5-7", "5-8", "5-9", "5-10",
    "6-6", "6-10", "10-6", "10-10",
)

PAPER_TABLE2_LEVELING_MEANS = {
    "3-3": 6.18, "3-4": 7.43, "3-5": 8.68, "3-6": 10.58,
    "3-7": 11.75, "3-8": 13.03, "4-4": 13.75, "4-5": 17.93,
    "4-6": 18.98, "4-7": 22.38, "5-4": 21.88, "5-5": 27.98,
    "5-6": 33.13, "5-7": 37.53, "5-8": 42.60, "5-9": 47.48,
    "5-10": 52.68, "6-6": 48.70, "6-10": 78.35,
    "10-6": 149.38, "10-10": 230.28,
}


def round_half_up(value: float, places: str = "0.01") -> float:
    return float(Decimal(str(value)).quantize(Decimal(places), rounding=ROUND_HALF_UP))


def validate_exact_caserta(episodes: Sequence[Episode]) -> None:
    if len(episodes) != 840:
        raise ValueError("the exact Caserta Table 2 benchmark must contain 840 episodes")
    if len({episode.canonical_id for episode in episodes}) != 840:
        raise ValueError("the benchmark must contain 840 unique canonical IDs")
    if any(
        episode.metadata.get("exact_original_benchmark") is not True
        for episode in episodes
    ):
        raise ValueError("every episode must be marked as an exact original benchmark")
    counts = Counter(episode.parameter_group for episode in episodes)
    if counts != Counter({group: 40 for group in PAPER_TABLE2_GROUPS}):
        raise ValueError("the benchmark must contain the 21 paper groups, 40 instances each")


def summarize(results: Sequence[LevelingResult], runtime_seconds: float) -> dict[str, object]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for result in results:
        grouped[result.scale_id].append(result.relocations)

    by_scale: dict[str, dict[str, object]] = {}
    matches = True
    for paper_group in PAPER_TABLE2_GROUPS:
        initial_height, num_stacks = (int(value) for value in paper_group.split("-"))
        scale_id = f"S{initial_height}-W{num_stacks}"
        values = grouped[scale_id]
        observed = float(statistics.fmean(values))
        paper = PAPER_TABLE2_LEVELING_MEANS[paper_group]
        matched = round_half_up(observed) == paper
        matches = matches and matched
        by_scale[scale_id] = {
            "paper_group_W-S": paper_group,
            "episode_count": len(values),
            "mean_relocations": observed,
            "sample_std_relocations": float(statistics.stdev(values)),
            "total_relocations": int(sum(values)),
            "paper_table2_mean_relocations": paper,
            "paper_value_reproduced_at_2_decimals": matched,
        }

    scale_means = [float(entry["mean_relocations"]) for entry in by_scale.values()]
    all_values = [result.relocations for result in results]
    macro = float(statistics.fmean(scale_means))
    return {
        "episode_count": len(results),
        "macro_mean_relocations": macro,
        "macro_mean_reward": -macro,
        "micro_mean_relocations": float(statistics.fmean(all_values)),
        "sample_std_relocations": float(statistics.stdev(all_values)),
        "median_relocations": float(statistics.median(all_values)),
        "total_relocations": int(sum(all_values)),
        "evaluation_runtime_seconds": runtime_seconds,
        "mean_runtime_seconds_per_episode": runtime_seconds / len(results),
        "all_21_paper_means_reproduced_at_2_decimals": matches,
        "by_scale": by_scale,
    }


def write_episode_results(
    path: Path, results: Sequence[LevelingResult], horizons: Sequence[int]
) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for horizon in horizons:
            for result in results:
                record = asdict(result)
                record.update({
                    "algorithm": "paper_leveling_L",
                    "relocations_reward": -result.relocations,
                    "comparison_horizon": horizon,
                    "future_information_used": 0,
                })
                handle.write(json.dumps(record, sort_keys=True) + "\n")


def write_scale_csv(
    path: Path, baseline: dict[str, object], horizons: Sequence[int]
) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "scale_id", "paper_group_W-S", "paper_table2_mean_relocations",
            *[f"H{h}_leveling_mean_relocations" for h in horizons],
        ])
        by_scale = baseline["by_scale"]
        for scale_id, entry in by_scale.items():
            writer.writerow([
                scale_id,
                entry["paper_group_W-S"],
                entry["paper_table2_mean_relocations"],
                *[entry["mean_relocations"] for _ in horizons],
            ])


def load_rl_summary(path: Path, horizons: Sequence[int]) -> dict[int, dict[str, object]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    indexed = {
        int(entry["lookahead_horizon"]): entry for entry in payload["horizons"]
    }
    missing = set(horizons) - set(indexed)
    if missing:
        raise ValueError(f"RL summary is missing horizons: {sorted(missing)}")
    return indexed


def write_comparison(
    path: Path,
    baseline: dict[str, object],
    rl_by_horizon: dict[int, dict[str, object]],
    horizons: Sequence[int],
) -> dict[str, object]:
    leveling_mean = float(baseline["macro_mean_relocations"])
    comparisons = []
    for horizon in horizons:
        rl_mean = float(rl_by_horizon[horizon]["macro_mean_relocations"])
        comparisons.append({
            "lookahead_horizon": horizon,
            "Mijn_RL_macro_mean_relocations": rl_mean,
            "Leveling_macro_mean_relocations": leveling_mean,
            "Mijn_RL_minus_Leveling": rl_mean - leveling_mean,
            "Mijn_RL_relocation_reduction_percent":
                100.0 * (leveling_mean - rl_mean) / leveling_mean,
        })
    payload = {
        "metric": "macro mean relocations; lower is better",
        "leveling_uses_future_information": False,
        "comparisons": comparisons,
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def plot_comparison(
    path: Path,
    baseline: dict[str, object],
    rl_by_horizon: dict[int, dict[str, object]],
    horizons: Sequence[int],
) -> None:
    # Matplotlib is optional because it is only needed for the Mijn_RL comparison.
    import matplotlib.pyplot as plt

    leveling_mean = float(baseline["macro_mean_relocations"])
    rl_means = [float(rl_by_horizon[h]["macro_mean_relocations"]) for h in horizons]
    leveling_means = [leveling_mean] * len(horizons)

    plt.style.use("seaborn-v0_8-whitegrid")
    figure, axis = plt.subplots(figsize=(10, 6), dpi=160)
    axis.plot(horizons, rl_means, marker="o", linewidth=2.5, label="Mijn_RL")
    axis.plot(
        horizons, leveling_means, marker="s", linewidth=2.5,
        label="Leveling L (uses H=0 information)",
    )
    for horizon, value in zip(horizons, rl_means):
        axis.annotate(f"{value:.3f}", (horizon, value), xytext=(0, 8),
                      textcoords="offset points", ha="center")
    axis.annotate(
        f"{leveling_mean:.3f}", (horizons[-1], leveling_mean),
        xytext=(0, 8), textcoords="offset points", ha="center",
    )
    axis.set_xticks(horizons)
    axis.set_xlabel("Mijn_RL lookahead horizon (future targets beyond current)")
    axis.set_ylabel("Macro mean relocations")
    axis.set_title("840 original Caserta instances: Mijn_RL vs. paper Leveling L")
    axis.text(0.01, 0.03, "Lower is better", transform=axis.transAxes, color="#475569")
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def parse_horizons(value: str) -> tuple[int, ...]:
    horizons = tuple(int(item.strip()) for item in value.split(","))
    if not horizons or any(horizon < 0 for horizon in horizons):
        raise argparse.ArgumentTypeError("horizons must be non-negative integers")
    if len(set(horizons)) != len(horizons):
        raise argparse.ArgumentTypeError("horizons must not contain duplicates")
    return horizons


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--benchmark",
        type=Path,
        default=DEFAULT_BENCHMARK,
        help=f"Caserta manifest (default: {DEFAULT_BENCHMARK})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"result directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument("--horizons", type=parse_horizons, default=(0, 5, 10, 20))
    parser.add_argument("--rl-summary", type=Path)
    args = parser.parse_args()

    episodes = load_episode_manifest(args.benchmark)
    validate_exact_caserta(episodes)
    started = time.perf_counter()
    results = tuple(run_leveling(episode) for episode in episodes)
    runtime_seconds = time.perf_counter() - started
    baseline = summarize(results, runtime_seconds)
    if not baseline["all_21_paper_means_reproduced_at_2_decimals"]:
        raise AssertionError("implementation does not reproduce every paper Table 2 mean")

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "protocol": "Zehendner2017_OCRP_0_restricted",
        "algorithm": "paper_leveling_L",
        "source": {
            "authors": "Zehendner, Feillet, and Jaillet",
            "title": "An algorithm with performance guarantee for the Online Container Relocation Problem",
            "journal": "European Journal of Operational Research 259 (2017) 48-62",
            "doi": "10.1016/j.ejor.2016.09.011",
        },
        "decision_rule": "lowest legal stack; leftmost stack breaks height ties",
        "relocation_restriction": "only blockers above the current target may move",
        "future_information_used": 0,
        "comparison_horizons": list(args.horizons),
        "horizon_note": "Leveling L is H=0 and therefore identical in every comparison column",
        "benchmark": str(args.benchmark.resolve()),
        "benchmark_sha256": hashlib.sha256(args.benchmark.read_bytes()).hexdigest(),
        "exact_original_caserta_benchmark": True,
        "baseline": baseline,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    write_episode_results(output_dir / "episode_results.jsonl", results, args.horizons)
    write_scale_csv(output_dir / "mean_relocations_by_scale.csv", baseline, args.horizons)

    if args.rl_summary:
        rl_by_horizon = load_rl_summary(args.rl_summary, args.horizons)
        write_comparison(
            output_dir / "comparison_with_mijn_rl.json",
            baseline,
            rl_by_horizon,
            args.horizons,
        )
        plot_comparison(
            output_dir / "leveling_vs_mijn_rl.png",
            baseline,
            rl_by_horizon,
            args.horizons,
        )

    print(
        f"Leveling macro mean relocations: {baseline['macro_mean_relocations']:.6f}"
    )
    print("All 21 Table 2 means reproduced at two decimals: True")
    print(f"Saved results to {output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
