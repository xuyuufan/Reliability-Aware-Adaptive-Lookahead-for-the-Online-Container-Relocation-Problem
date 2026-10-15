from __future__ import annotations

import csv
from pathlib import Path
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"
FIGURES = HERE
FAMILY_INPUT = DATA / "ridge_six_solver_family_rows.csv"
CONDITION_INPUT = DATA / "ridge_six_solver_condition_rows.csv"

ALGORITHMS = ["EM", "GP", "ERI", "EG", "Chain", "Minmax"]
FAMILIES = ["subset_shuffle", "delayed_positions", "time_window_shuffle"]
FAMILY_LABELS = {
    "subset_shuffle": "Subset shuffle",
    "delayed_positions": "Delayed positions",
    "time_window_shuffle": "Time-window shuffle",
}
COLORS = {
    "subset_shuffle": "#0072B2",       # Okabe-Ito blue
    "delayed_positions": "#D55E00",    # vermilion
    "time_window_shuffle": "#009E73",  # bluish green
    "CLEAN": "#777777",
}
RIDGE = "R_UTILITY_RIDGE"
BASELINE = "NATIVE20"


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def join_comparison(rows: list[dict[str, str]], *, by_condition: bool) -> list[dict[str, object]]:
    key_fields = ["algorithm", "condition_id"] if by_condition else ["algorithm", "scope"]
    grouped: dict[tuple[str, ...], dict[str, dict[str, str]]] = defaultdict(dict)
    for row in rows:
        if row["algorithm"] not in ALGORITHMS or row["method"] not in (RIDGE, BASELINE):
            continue
        key = tuple(row[field] for field in key_fields)
        if row["method"] in grouped[key]:
            raise ValueError(f"duplicate method row for {key}: {row['method']}")
        grouped[key][row["method"]] = row

    expected = 126 if by_condition else 18
    if len(grouped) != expected:
        raise ValueError(f"expected {expected} joined comparisons, got {len(grouped)}")

    output: list[dict[str, object]] = []
    for key, methods in grouped.items():
        if set(methods) != {RIDGE, BASELINE}:
            raise ValueError(f"missing paired method for {key}: {sorted(methods)}")
        ridge = methods[RIDGE]
        native = methods[BASELINE]
        row: dict[str, object] = {
            "metric": ridge["metric"],
            "algorithm": ridge["algorithm"],
            "unit": ridge["unit"],
            "direction": ridge["direction"],
            "ridge_r_raw_estimate": ridge["estimate"],
            "ridge_r_raw_ci95_low": ridge["ci95_low"],
            "ridge_r_raw_ci95_high": ridge["ci95_high"],
            "native20_r_raw_estimate": native["estimate"],
            "native20_r_raw_ci95_low": native["ci95_low"],
            "native20_r_raw_ci95_high": native["ci95_high"],
            "improvement_percent": ridge["improvement_percent"],
            "improvement_ci95_low_percent": ridge["improvement_ci95_low_percent"],
            "improvement_ci95_high_percent": ridge["improvement_ci95_high_percent"],
            "layout_n": ridge["layout_n"],
            "bootstrap_replicates": ridge["bootstrap_replicates"],
        }
        if by_condition:
            row.update({
                "condition_id": ridge["condition_id"],
                "family": ridge["family"],
                "strength": ridge["strength"],
                "repeat": ridge["repeat"],
            })
        else:
            row.update({
                "scope": ridge["scope"],
                "layouts_per_geometry": ridge["layouts_per_geometry"],
            })
        output.append(row)

    if by_condition:
        output.sort(key=lambda row: (condition_order(str(row["condition_id"])), ALGORITHMS.index(str(row["algorithm"]))))
    else:
        output.sort(key=lambda row: (FAMILIES.index(str(row["scope"])), ALGORITHMS.index(str(row["algorithm"]))))
    return output


def condition_order(condition_id: str) -> tuple[int, float, int]:
    if condition_id == "CLEAN":
        return (3, 0.0, 0)
    family, *parts = condition_id.split("|")
    family_index = {"subset_shuffle": 0, "delayed_positions": 1, "time_window_shuffle": 2}[family]
    attrs = dict(part.split("=", 1) for part in parts)
    repeat = int(attrs.get("r", "0"))
    if family == "time_window_shuffle":
        strength = float(attrs["B"])
    else:
        strength = float(attrs["p"])
    return (family_index, strength, repeat)


def condition_tick(condition_id: str) -> str:
    if condition_id == "CLEAN":
        return "CLEAN"
    family, *parts = condition_id.split("|")
    attrs = dict(part.split("=", 1) for part in parts)
    repeat = attrs["r"]
    short = {
        "subset_shuffle": "SS",
        "delayed_positions": "DP",
        "time_window_shuffle": "TW",
    }[family]
    if family == "time_window_shuffle":
        detail = f"B={attrs['B']}"
    else:
        detail = f"p={attrs['p'].replace('0.', '.')}"
    return f"{short}\n{detail} r{repeat}"


def style_plot() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "figure.titlesize": 15,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def main_family_figure(rows: list[dict[str, str]]) -> None:
    selected = [r for r in rows if r["method"] == RIDGE and r["scope"] in FAMILIES and r["algorithm"] in ALGORITHMS]
    if len(selected) != 18:
        raise ValueError(f"expected 18 Ridge family cells; found {len(selected)}")

    values = [float(r["improvement_percent"]) for r in selected]
    lows = [float(r["improvement_ci95_low_percent"]) for r in selected]
    highs = [float(r["improvement_ci95_high_percent"]) for r in selected]
    if min(lows) < -8 or max(highs) > 17:
        raise ValueError("family CI endpoints fell outside the checked plotting range")

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 6.6), sharex=True, sharey=True)
    for ax, family in zip(axes, FAMILIES):
        family_rows = {r["algorithm"]: r for r in selected if r["scope"] == family}
        if set(family_rows) != set(ALGORITHMS):
            raise ValueError(f"incomplete family: {family}")
        y_positions = list(range(len(ALGORITHMS) - 1, -1, -1))
        color = COLORS[family]
        for algorithm, y in zip(ALGORITHMS, y_positions):
            row = family_rows[algorithm]
            estimate = float(row["improvement_percent"])
            low = float(row["improvement_ci95_low_percent"])
            high = float(row["improvement_ci95_high_percent"])
            ax.errorbar(
                estimate, y,
                xerr=[[estimate - low], [high - estimate]],
                fmt="o", color=color, markersize=6,
                elinewidth=1.6, capsize=3, capthick=1.2, zorder=3,
            )
            ax.annotate(
                f"{estimate:+.1f}%",
                xy=(high, y), xytext=(4, 0), textcoords="offset points",
                ha="left", va="center", fontsize=8, color="#222222",
            )
        ax.axvline(0, color="#333333", lw=1.0, ls=(0, (4, 3)), zorder=1)
        ax.grid(axis="x", color="#dddddd", lw=0.7, zorder=0)
        ax.set_title(FAMILY_LABELS[family], pad=10)
        ax.set_xlim(-8, 17)
        ax.set_xticks([-5, 0, 5, 10, 15])
        ax.set_yticks(y_positions, ALGORITHMS)
        ax.tick_params(axis="y", length=0)
    axes[0].set_ylabel("Algorithm")
    for ax in axes:
        ax.set_xlabel("Improvement in R_raw vs NATIVE20 (%)\npositive = fewer relocations")
    fig.suptitle("Ridge-R utility change across the three noisy families", y=0.98)
    fig.text(
        0.5, 0.105,
        "18 family-level paired comparisons; points and existing paired 95% CIs. "
        "20 layouts (10 per geometry), 10,000 layout-cluster bootstrap replicates.",
        ha="center", va="center", fontsize=9,
    )
    fig.text(
        0.5, 0.055,
        "Leveling reference: fixed H=0 uses no forecast; Ridge leaves its action unchanged "
        "(analytic identity only, no empirical CI).",
        ha="center", va="center", fontsize=8.5, color="#444444",
    )
    fig.subplots_adjust(left=0.09, right=0.99, top=0.87, bottom=0.24, wspace=0.18)
    fig.savefig(FIGURES / "fig_ridge_six_solver_three_noise_families.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIGURES / "fig_ridge_six_solver_three_noise_families.pdf", bbox_inches="tight")
    plt.close(fig)


def all_condition_figure(rows: list[dict[str, str]]) -> None:
    selected = [r for r in rows if r["method"] == RIDGE and r["algorithm"] in ALGORITHMS]
    conditions = sorted({r["condition_id"] for r in selected}, key=condition_order)
    if len(conditions) != 21 or len(selected) != 126:
        raise ValueError(f"expected 21 conditions x 6 algorithms; got {len(conditions)} conditions, {len(selected)} cells")
    row_index = {(r["algorithm"], r["condition_id"]): r for r in selected}
    if len(row_index) != 126:
        raise ValueError("duplicate condition/algorithm cells")

    fig, axes = plt.subplots(3, 2, figsize=(18, 12.2), sharex=True, sharey=True)
    axes_flat = list(axes.flat)
    ticks = list(range(len(conditions)))
    labels = [condition_tick(c) for c in conditions]
    boundaries = []
    prior_family = None
    for index, condition in enumerate(conditions):
        family = "CLEAN" if condition == "CLEAN" else condition.split("|", 1)[0]
        if prior_family is not None and family != prior_family:
            boundaries.append(index - 0.5)
        prior_family = family

    for ax, algorithm in zip(axes_flat, ALGORITHMS):
        for x, condition in enumerate(conditions):
            row = row_index[(algorithm, condition)]
            family = "CLEAN" if condition == "CLEAN" else row["family"]
            color = COLORS[family]
            estimate = float(row["improvement_percent"])
            low = float(row["improvement_ci95_low_percent"])
            high = float(row["improvement_ci95_high_percent"])
            ax.errorbar(
                x, estimate,
                yerr=[[estimate - low], [high - estimate]],
                fmt="o", color=color, markersize=4.5,
                elinewidth=1.1, capsize=2.2, capthick=1.0, zorder=3,
            )
        for boundary in boundaries:
            ax.axvline(boundary, color="#999999", lw=0.8, ls=":", zorder=1)
        clean_index = conditions.index("CLEAN")
        ax.axvspan(clean_index - 0.48, clean_index + 0.48, color="#eeeeee", zorder=0)
        ax.axhline(0, color="#333333", lw=0.9, ls=(0, (4, 3)), zorder=1)
        ax.grid(axis="y", color="#e1e1e1", lw=0.6, zorder=0)
        ax.set_title(algorithm, loc="left", fontweight="bold", pad=5)
        ax.set_xlim(-0.6, len(conditions) - 0.4)
        ax.set_ylim(-22, 25)
        ax.set_yticks([-20, -10, 0, 10, 20])
        ax.set_ylabel("Improvement in R_raw (%)")

    for ax in axes[-1, :]:
        ax.set_xticks(ticks, labels, rotation=45, ha="right", fontsize=7.2)
        ax.tick_params(axis="x", length=0, pad=4)
    legend_handles = [
        Patch(facecolor=COLORS["subset_shuffle"], label="Subset shuffle"),
        Patch(facecolor=COLORS["delayed_positions"], label="Delayed positions"),
        Patch(facecolor=COLORS["time_window_shuffle"], label="Time-window shuffle"),
        Patch(facecolor=COLORS["CLEAN"], label="CLEAN (separate control)"),
    ]
    fig.suptitle("Ridge-R vs NATIVE20 across all 21 registered conditions", y=0.98)
    fig.legend(handles=legend_handles, loc="upper center", bbox_to_anchor=(0.5, 0.945), ncol=4, frameon=False)
    fig.text(
        0.5, 0.035,
        "All 20 noisy conditions plus CLEAN; per-condition paired 95% CIs (20 layouts, "
        "10,000 layout-cluster bootstrap replicates). CLEAN is not part of the three noisy families.",
        ha="center", va="center", fontsize=9,
    )
    fig.subplots_adjust(left=0.075, right=0.99, top=0.88, bottom=0.20, hspace=0.3, wspace=0.13)
    fig.savefig(FIGURES / "fig_ridge_six_solver_all21_conditions.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIGURES / "fig_ridge_six_solver_all21_conditions.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    style_plot()
    family_rows = read_rows(FAMILY_INPUT)
    condition_rows = read_rows(CONDITION_INPUT)
    family_output = join_comparison(family_rows, by_condition=False)
    condition_output = join_comparison(condition_rows, by_condition=True)

    family_fields = [
        "metric", "scope", "algorithm", "unit", "direction", "layout_n", "layouts_per_geometry",
        "bootstrap_replicates", "ridge_r_raw_estimate", "ridge_r_raw_ci95_low", "ridge_r_raw_ci95_high",
        "native20_r_raw_estimate", "native20_r_raw_ci95_low", "native20_r_raw_ci95_high",
        "improvement_percent", "improvement_ci95_low_percent", "improvement_ci95_high_percent",
    ]
    condition_fields = [
        "metric", "condition_id", "family", "strength", "repeat", "algorithm", "unit", "direction",
        "layout_n", "bootstrap_replicates", "ridge_r_raw_estimate", "ridge_r_raw_ci95_low",
        "ridge_r_raw_ci95_high", "native20_r_raw_estimate", "native20_r_raw_ci95_low",
        "native20_r_raw_ci95_high", "improvement_percent", "improvement_ci95_low_percent",
        "improvement_ci95_high_percent",
    ]
    write_rows(DATA / "ridge_vs_native20_three_noise_families_18_cells.csv", family_output, family_fields)
    write_rows(DATA / "ridge_vs_native20_all21_conditions_126_cells.csv", condition_output, condition_fields)
    main_family_figure(family_rows)
    all_condition_figure(condition_rows)
    print("Generated:")
    for path in sorted(FIGURES.glob("fig_ridge_six_solver_*")):
        print(f"{path} ({path.stat().st_size} bytes)")
    for path in sorted(DATA.glob("ridge_vs_native20_*.csv")):
        print(f"{path} ({path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()



