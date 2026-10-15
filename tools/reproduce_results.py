"""Reanalyze published data; never run trajectories or historical release gates."""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "results/two_noise_families"
WORKSPACE = ROOT / "baselines/_snapshot"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(WORKSPACE))
from R.experiments import fixed_horizon_analysis as a

INPUT_HASHES = {
    a.LAYOUT_PATH: "62f9e205f2df0197260629f4b9ad54a35a412c6ab16838709b0a26d224ecf426",
}


def compare_csv(actual: Path, expected: Path) -> tuple[int, float]:
    def read(p: Path):
        with p.open(encoding="utf-8-sig", newline="") as f:
            return list(csv.reader(f))
    left, right = read(actual), read(expected)
    if len(left) != len(right) or left[0] != right[0]:
        raise ValueError(f"CSV shape/header mismatch: {actual.name}")
    count, maximum = 0, 0.0
    for i, (lr, rr) in enumerate(zip(left[1:], right[1:]), start=2):
        if len(lr) != len(rr):
            raise ValueError(f"CSV width mismatch: {actual.name}:{i}")
        for column, lv, rv in zip(left[0], lr, rr):
            count += 1
            if lv == rv:
                continue
            try:
                x, y = float(lv), float(rv)
            except ValueError:
                raise ValueError(f"Text mismatch: {actual.name}:{i}:{column}") from None
            if not math.isfinite(x) or not math.isfinite(y) or abs(x-y) > 1e-10:
                raise ValueError(f"Numeric mismatch: {actual.name}:{i}:{column}: {lv} != {rv}")
            maximum = max(maximum, abs(x-y))
    return count, maximum


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "reproduced")
    parser.add_argument("--input", type=Path, default=ROOT / "data/reference_trajectories.csv",
                        help="complete merged bank: published input or a fresh 10,800-row rerun")
    parser.add_argument("--publish-reference", action="store_true", help="create the two-family reference analysis without a previous-table comparison")
    args = parser.parse_args()
    target = args.output_dir.resolve()
    if target.exists():
        raise ValueError(f"Output directory already exists: {target}; choose a new directory")
    for path, expected in INPUT_HASHES.items():
        if a._sha256_file(path) != expected:
            raise ValueError(f"Frozen input SHA-256 mismatch: {path}")
    start = time.perf_counter()
    input_path = args.input.resolve()
    bank = a.validate_rows(a._read_csv(input_path), a.load_layout_registry())
    raw, rn = a.scope_layout_values(bank.raw), a.scope_layout_values(bank.rn)
    indices = a.make_bootstrap_indices(
        bank.layouts, bank.geometry_by_layout, replicates=a.BOOTSTRAP_REPLICATES,
        seed=a.BOOTSTRAP_SEED, batch_size=a.BOOTSTRAP_BATCH_SIZE,
    )
    deadline = start + a.ANALYSIS_BUDGET_SECONDS
    boots = {
        scope: a.bootstrap_scope_pair(raw[scope], rn[scope], indices,
                                     batch_size=a.BOOTSTRAP_BATCH_SIZE, deadline=deadline)
        for scope in raw
    }
    point = a.geometry_mean(raw["noisy"], bank.geometry_by_layout, bank.layouts)
    rnpoint = a.geometry_mean(rn["noisy"], bank.geometry_by_layout, bank.layouts)
    primary, overall = a.primary_comparisons(point, boots["noisy"][0], replicates=a.BOOTSTRAP_REPLICATES)
    fixed = a._fixed_h_vs_h20_rows(point, rnpoint, *boots["noisy"], replicates=a.BOOTSTRAP_REPLICATES)
    best = a._best_fixed_rows(point, boots["noisy"][0], primary)
    curves = a._fixed_curve_rows(point, boots["noisy"][0], a.BOOTSTRAP_REPLICATES)
    target.mkdir(parents=True)
    names = a._write_analysis_outputs(
        target, bank, raw, rn, boots, primary, overall, fixed, best, curves,
        input_sha256=a._sha256_file(input_path), analyzer_sha256=a._sha256_file(a.SCRIPT),
        replicates=a.BOOTSTRAP_REPLICATES,
    )
    comparisons = {}
    for name in names:
        if name.endswith(".csv") and not args.publish_reference:
            fields, error = compare_csv(target / name, RUN / name)
            comparisons[name] = {"cells_compared": fields, "max_absolute_error": error}
    receipt = {
        "status": "PASS_PUBLISHED_DATA_REANALYSIS", "new_trajectories": 0, "new_fits": 0,
        "merged_rows": a.EXPECTED_ROWS, "forecast_pairs": bank.forecast_pair_count,
        "bootstrap_replicates": a.BOOTSTRAP_REPLICATES, "bootstrap_seed": a.BOOTSTRAP_SEED,
        "elapsed_seconds": time.perf_counter()-start, "csv_comparisons": comparisons,
        "figure_pairs_generated": 4, "formal_stage_acceptance": False,
        "input_sha256": a._sha256_file(input_path),
        "input_origin": "published_two_family_reference" if input_path == (ROOT / "data/reference_trajectories.csv").resolve() else "fresh_complete_bank",
        "compared_to_published_tables": not args.publish_reference,
    }
    (target / "REPRODUCTION_RECEIPT.json").write_text(json.dumps(receipt, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
