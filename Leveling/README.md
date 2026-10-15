# Leveling L baseline

This folder independently reproduces the Leveling heuristic from:

E. Zehendner, D. Feillet, and P. Jaillet, "An algorithm with performance
guarantee for the Online Container Relocation Problem," *European Journal of
Operational Research*, 259 (2017), 48-62. DOI: 10.1016/j.ejor.2016.09.011.

## Paper rule

The implementation follows the paper's restricted online CRP setting:

1. Only the current retrieval target is known (`H=0`).
2. Only containers above that target may be relocated.
3. A blocking container is sent to the legal stack with the lowest height.
4. If multiple stacks have the same height, the leftmost stack is selected.

The rule never reads future retrieval targets. Consequently its result is the
same when displayed beside Mijn_RL at H=0, 5, 10, or 20. Those horizon columns
are comparison labels, not different Leveling variants.

## Requirements

Only Python 3.10 or newer is required for the Leveling evaluation. The exact
840-instance Caserta benchmark is included in `data/caserta_table2_840.json`.
No files from Mijn_RL are required.

## Run the exact Caserta benchmark

Open a terminal in this `Leveling` folder and run:

```powershell
py -3.12 evaluate_caserta.py
```

The script uses the included benchmark and writes its outputs to
`results/caserta_table2_840` by default. Optional paths can still be supplied:

```powershell
py -3.12 evaluate_caserta.py `
  --benchmark data\caserta_table2_840.json `
  --output-dir results\caserta_table2_840
```

The evaluator rejects a manifest unless it contains all 840 unique original
Caserta episodes (21 physical scales, 40 instances each). It also fails unless
the reproduced mean relocation count for every scale matches Table 2 of the
paper to two decimal places.

## Tests

```powershell
py -3.12 -m pip install pytest
py -3.12 -m pytest tests -q
```
