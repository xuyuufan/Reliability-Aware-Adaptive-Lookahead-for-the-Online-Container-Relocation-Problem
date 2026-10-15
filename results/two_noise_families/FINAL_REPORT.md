# Ridge-R: two-family fixed-horizon comparison

Scope: subset_shuffle and time_window_shuffle; CLEAN is separate. 20 exposed development layouts (10 per geometry), 15 conditions, six solvers, six policy slots, and 10,800 trajectory observations.

The evaluation scope was revised at the user's request after the historical results were available. These recomputed results are descriptive development evidence, not an independently confirmed or newly preregistered study.

## Equal-six noisy macro

The two noise families have equal weight. Within each family, strengths and repeats are averaged hierarchically. Geometries have equal weight. Each solver's improvement is a ratio of its aggregated means; the six improvement percentages are then averaged equally.

| Fixed H | Ridge improvement | Ordinary 95% CI | Simultaneous 95% CI |
|---:|---:|---:|---:|
| 0 | 14.686% | [10.713%, 18.312%] | [8.180%, 20.325%] |
| 2 | 12.373% | [9.976%, 14.544%] | [8.470%, 15.650%] |
| 5 | 9.603% | [7.697%, 11.685%] | [6.763%, 13.135%] |
| 10 | 6.490% | [4.893%, 8.160%] | [4.016%, 9.209%] |
| 20 | 5.486% | [3.812%, 7.164%] | [2.827%, 8.168%] |

Of 30 solver-horizon comparisons: 29 support Ridge, 0 support the reverse direction, and 1 are unresolved.
Intervals use 100,000 paired geometry-stratified bootstrap draws, NumPy PCG64 seed 202610155301, method=linear percentile quantiles, and Bonferroni simultaneous intervals over the declared 35-comparison family.

## Best fixed H on the same development data

| Solver | Selected H | Fixed mean relocations | Ridge mean relocations | Ridge improvement | Simultaneous interval |
|---|---:|---:|---:|---:|---:|
| EM | 20 | 31.374 | 29.891 | 4.728% | [1.831%, 7.832%] |
| GP | 20 | 31.503 | 29.811 | 5.370% | [1.911%, 8.440%] |
| ERI | 10 | 31.236 | 30.349 | 2.841% | [-1.257%, 6.516%] |
| EG | 20 | 31.557 | 28.877 | 8.493% | [6.386%, 10.526%] |
| Chain | 20 | 31.076 | 29.614 | 4.706% | [1.503%, 7.893%] |
| Minmax | 20 | 31.323 | 29.538 | 5.700% | [2.150%, 9.423%] |

Selection of H on the same data is descriptive; its interval does not establish independent confirmation.

## CLEAN negative results

| Solver | H20 mean relocations | Ridge mean relocations | Ridge improvement |
|---|---:|---:|---:|
| EM | 23.900 | 26.550 | -11.088% |
| GP | 23.900 | 26.050 | -8.996% |
| ERI | 26.550 | 28.000 | -5.461% |
| EG | 24.550 | 25.800 | -5.092% |
| Chain | 24.450 | 26.350 | -7.771% |
| Minmax | 23.900 | 26.250 | -9.833% |

Negative improvement means Ridge requires more relocations. CLEAN is excluded from the noisy macro. Full descriptive ordinary intervals are in [CLEAN_RESULTS.csv](CLEAN_RESULTS.csv).

## Data and reproducibility

- [Individual comparisons](RIDGE_VS_FIXED_H_MACRO.csv)
- [Equal-six comparisons](OVERALL_EQUAL6_RESULTS.csv)
- [Family results](FAMILY_RESULTS.csv)
- [All 15 conditions](CONDITION_RESULTS.csv)
- [Best development fixed H](DEVELOPMENT_BEST_FIXED_H.csv)
- [Fixed H versus H20](FIXED_H_VS_H20.csv)

R_raw is the primary relocation metric; R/N is retained separately. Fixed GP and Ridge models are used without fitting, selection, or calibration. No independent confirmation, sealed evaluation, OOD result, or deployment benefit is established. The noise-family scope change does not alter these limits.

Input SHA-256: `a892558c90fb0cce35b8c9ef527cf08cad78f97889fe110fa004ff6e39761668`

Analyzer SHA-256: `0eba026dc14c6218332136c1ee3d47d92f002e5e94902b8593de3d59b0251d78`

Input origins: `{'historical_reuse': 3600, 'fixed_h_bank': 7200}`; forecast-paired layout-condition cells: 300.
