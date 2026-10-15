"""Report the revised two-family scope without carrying over old macro values."""

def build_report(primary, overall, best_rows, bank, *, clean_point, solvers, policies,
                 input_sha256, analyzer_sha256):
    supported = sum(float(r["simultaneous_low_percent"]) > 0 for r in primary)
    reverse = sum(float(r["simultaneous_high_percent"]) < 0 for r in primary)
    lines = [
        "# Ridge-R: two-family fixed-horizon comparison", "",
        "Scope: subset_shuffle and time_window_shuffle; CLEAN is separate. "
        "20 exposed development layouts (10 per geometry), 15 conditions, "
        "six solvers, six policy slots, and 10,800 trajectory observations.", "",
        "The evaluation scope was revised at the user's request after the historical "
        "results were available. These recomputed results are descriptive development "
        "evidence, not an independently confirmed or newly preregistered study.", "",
        "## Equal-six noisy macro", "",
        "The two noise families have equal weight. Within each family, strengths "
        "and repeats are averaged hierarchically. Geometries have equal weight. "
        "Each solver's improvement is a ratio of its aggregated means; the six "
        "improvement percentages are then averaged equally.", "",
        "| Fixed H | Ridge improvement | Ordinary 95% CI | Simultaneous 95% CI |",
        "|---:|---:|---:|---:|",
    ]
    for r in overall:
        lines.append(f'| {r["horizon"]} | {float(r["equal6_improvement_percent"]):.3f}% '
                     f'| [{float(r["ci95_low_percent"]):.3f}%, {float(r["ci95_high_percent"]):.3f}%] '
                     f'| [{float(r["simultaneous_low_percent"]):.3f}%, {float(r["simultaneous_high_percent"]):.3f}%] |')
    lines += ["", f"Of 30 solver-horizon comparisons: {supported} support Ridge, "
              f"{reverse} support the reverse direction, and {30-supported-reverse} are unresolved.",
              "Intervals use 100,000 paired geometry-stratified bootstrap draws, "
              "NumPy PCG64 seed 202610155301, method=linear percentile quantiles, "
              "and Bonferroni simultaneous intervals over the declared 35-comparison family.",
              "", "## Best fixed H on the same development data", "",
              "| Solver | Selected H | Fixed mean relocations | Ridge mean relocations | Ridge improvement | Simultaneous interval |",
              "|---|---:|---:|---:|---:|---:|"]
    for r in best_rows:
        if r["algorithm"] not in solvers:
            continue
        lines.append(f'| {r["algorithm"]} | {r["development_best_horizon"]} '
                     f'| {float(r["development_best_fixed_mean_R_raw"]):.3f} '
                     f'| {float(r["ridge_mean_R_raw"]):.3f} '
                     f'| {float(r["ridge_improvement_percent"]):.3f}% '
                     f'| [{float(r["simultaneous_low_percent"]):.3f}%, {float(r["simultaneous_high_percent"]):.3f}%] |')
    lines += ["", "Selection of H on the same data is descriptive; its interval "
              "does not establish independent confirmation.", "", "## CLEAN negative results", "",
              "| Solver | H20 mean relocations | Ridge mean relocations | Ridge improvement |",
              "|---|---:|---:|---:|"]
    ridge, fixed = policies.index("RIDGE"), policies.index("FIXED_H20")
    for i, solver in enumerate(solvers):
        rv, fv = float(clean_point[i, ridge]), float(clean_point[i, fixed])
        lines.append(f"| {solver} | {fv:.3f} | {rv:.3f} | {100*(fv-rv)/fv:.3f}% |")
    lines += ["", "Negative improvement means Ridge requires more relocations. "
              "CLEAN is excluded from the noisy macro. Full descriptive ordinary "
              "intervals are in [CLEAN_RESULTS.csv](CLEAN_RESULTS.csv).", "",
              "## Data and reproducibility", "",
              "- [Individual comparisons](RIDGE_VS_FIXED_H_MACRO.csv)",
              "- [Equal-six comparisons](OVERALL_EQUAL6_RESULTS.csv)",
              "- [Family results](FAMILY_RESULTS.csv)",
              "- [All 15 conditions](CONDITION_RESULTS.csv)",
              "- [Best development fixed H](DEVELOPMENT_BEST_FIXED_H.csv)",
              "- [Fixed H versus H20](FIXED_H_VS_H20.csv)", "",
              "R_raw is the primary relocation metric; R/N is retained separately. "
              "Fixed GP and Ridge models are used without fitting, selection, or calibration. "
              "No independent confirmation, sealed evaluation, OOD result, or deployment "
              "benefit is established. The noise-family scope change does not alter these limits.", "",
              f"Input SHA-256: `{input_sha256}`", "",
              f"Analyzer SHA-256: `{analyzer_sha256}`", "",
              f"Input origins: `{dict(bank.source_origin_counts)}`; "
              f"forecast-paired layout-condition cells: {bank.forecast_pair_count}.", ""]
    return "\n".join(lines)
