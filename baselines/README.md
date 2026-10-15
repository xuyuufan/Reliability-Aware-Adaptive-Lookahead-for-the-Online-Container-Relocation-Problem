# Baselines and solver adapters

All seven algorithms have a visible entry here. Six are included in the paired Ridge/fixed-horizon experiment; Leveling is a separate forecast-independent reference.

| Algorithm | Public entrypoint | Actual experiment implementation |
|---|---|---|
| [EM](EM/) | `baselines.EM.make_policy` | [_snapshot/EM/scrp/noise_aware_em.py](_snapshot/EM/scrp/noise_aware_em.py) |
| [GP](GP/) | `baselines.GP.make_policy` | [_snapshot/GP/simulator.py](_snapshot/GP/simulator.py), [expression.py](_snapshot/GP/expression.py) |
| [ERI](ERI/) | `baselines.ERI.make_policy` | [_snapshot/R/scrp/eri_eg_public.py](_snapshot/R/scrp/eri_eg_public.py) |
| [EG](EG/) | `baselines.EG.make_policy` | Same ERI/EG implementation; native committed group planning |
| [Chain](Chain/) | `baselines.Chain.make_policy` | [_snapshot/Chain/chain_public.py](_snapshot/Chain/chain_public.py) |
| [Minmax](Minmax/) | `baselines.Minmax.make_policy` | [_snapshot/Minmax/minmax_public.py](_snapshot/Minmax/minmax_public.py) |
| [Leveling](Leveling/) | `baselines.Leveling.run_leveling` | [Leveling/heuristic.py](Leveling/heuristic.py) |

Example: `from baselines.EM import make_policy; actor = make_policy("RIDGE")`.
The six solver APIs support `RIDGE` and `FIXED_H0`, `FIXED_H2`, `FIXED_H5`, `FIXED_H10`, `FIXED_H20`.
They accept only detached public frames; dataset/noise metadata belong to the evaluator.

`_snapshot/` is the single authoritative copy of the frozen runtime. The solver/policy sources are preserved byte-for-byte; the statistical analyzer is adapted to the current two-family evaluation. Small entrypoint modules delegate to it; algorithms are not reimplemented in the public folders. Historical release-gated scripts remain archival, while [the portable entrypoint](../tools/reproduce_experiments.py) performs new trajectories in a new directory.

Leveling has no Ridge-versus-native comparison because its actions are independent of forecasts. Its standalone Caserta evaluation is available as:

```powershell
py -3 -B baselines/Leveling/evaluate_caserta.py --output-dir runs/leveling-caserta
```

See [full reproduction instructions](../docs/REPRODUCTION.md), including dependencies, smoke scope, full scope, comparison criteria, and output files.
