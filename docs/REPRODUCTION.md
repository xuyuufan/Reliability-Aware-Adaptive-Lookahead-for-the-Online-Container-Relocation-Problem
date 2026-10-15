# Complete experiment reproduction

The portable entrypoint regenerates **all 10,800 trajectories** from the included 20 layouts, deterministic forecast generators, frozen GP rule, Ridge coefficients, and original six solver implementations. It covers six solvers × six policy slots × 20 layouts × 15 conditions. No historical rows are reused in the new bank. The current noisy evaluation includes only subset_shuffle and time_window_shuffle; CLEAN is separate. Previous-scope data and results are archived locally.

## Environment

Python 3.12 is the tested version. Install in a virtual environment:

```powershell
py -3 -m venv .venv
.venv/Scripts/python -m pip install -r requirements-experiments.txt
```

The commands below use `py -3` for readability; substitute `.venv/Scripts/python` on Windows or `.venv/bin/python` on Linux when using that environment. CPU execution is sufficient. Torch is required by the original shared runtime's imports; no neural inference, GPU, or neural checkpoint is used by this experiment. Version pins record the tested environment, not performance requirements.

## 1. Check inputs and dependencies

```powershell
py -3 -B tools/reproduce_experiments.py --preflight --scope full
```

This loads the local runtime, verifies source/model/data hashes, validates layout fingerprints, and checks all 300 forecast hashes/seeds against the included two-family reference without running a trajectory. Imports must stay under `baselines/_snapshot/`. The input lock contains only repository-relative paths.

## 2. Run the bounded smoke matrix

```powershell
py -3 -B tools/reproduce_experiments.py --scope smoke --workers 4 --output-dir runs/smoke
```

This executes **216 fresh trajectories**: one layout from each geometry, one representative condition from each noise family plus CLEAN, six solvers, and all six policy slots. It compares 18 deterministic fields per row, including relocations, N, normalized relocations, decisions, EG cached moves, completion, and forecast bindings.

For a single solver, use `--algorithms EM` (or GP, ERI, EG, Chain, Minmax). Its smoke scope has 36 trajectories. This is a correctness check, not evidence over the complete experimental matrix.

## 3. Run the complete experiment and statistical analysis

```powershell
py -3 -B tools/reproduce_experiments.py --scope full --workers 4 --output-dir runs/full --analyze
```

Every `RIDGE` and `FIXED_H0/2/5/10/20` trajectory is simulated afresh. Workers share the same full forecast within each layout-condition cell; each trajectory starts with a new environment, provider, and history tracker. Actor inputs contain no noise family, p, window size, seed, private truth, or raw dataset identity.

The comparison stops with a failure receipt if any deterministic field differs from the included two-family reference. Timings and source-origin fields are recorded freshly and are not required to equal the reference execution. After the full bank passes, the analyzer regenerates **12 CSV tables and four PNG/PDF figure pairs** from the new bank, comparing every CSV cell with the current-scope published results at absolute tolerance `1e-10`.

The trajectory wall cap defaults to 7,200 seconds for the entire job, with 30 seconds per trajectory; `--cap-seconds` may lower that cap. Statistics have a separate 1,800-second deadline. `--workers 1` is supported; worker count changes scheduling, not seeds or policy choices. The full simulation and analysis use the already selected models without training or tuning.

## Outputs and failure behavior

`runs/full/` contains:

- `PREFLIGHT.json`: planned matrix, hash checks, and local module locations.
- `MERGED_RAW_TRAJECTORIES.csv`: newly simulated observations with `source_origin=fresh_portable_rerun` and per-trajectory action-sequence hashes.
- `public_decision_trace.jsonl.gz`: detached public frames and actions.
- `evaluator_physics.jsonl.gz`: evaluator-only truth, forecasts, and physical transitions, kept separate from actors.
- `cells/`: individual layout-condition output shards, retained for diagnostics.
- `RUN_RECEIPT.json`: scope, execution count, comparison results, timings, input hashes, and success/failure.
- `analysis/`: reconstructed tables, figures, and `REPRODUCTION_RECEIPT.json` when `--analyze` is used.

Trace files are concatenated gzip members and can be read normally with Python's `gzip.open`. Use `--no-traces` to omit new trace copies while retaining raw rows and per-trajectory action hashes; the receipt records this reduced output scope.

Existing output directories are refused. Choose a new directory for another attempt. Partial files and failure receipts are retained; the tool does not implicitly resume, delete, or overwrite a run. Generated `runs/` directories are ignored by Git.

## Independently verify the physical traces

The supplied fresh run can be checked without using the simulation environment implementation:

```powershell
py -3 -B tools/verify_traces.py
```

The checker reconstructs blocker relocations and automatic target retrieval from the layout and action records, checks every before/after physical state, pairs public/private steps, validates public occupancy and legal destinations, and checks action-sequence hashes. For another trace-enabled run, use `--run-dir runs/full`.

## Reanalyze published data only

For statistics and plots without simulation:

```powershell
py -3 -m pip install -r requirements.txt
py -3 -B tools/reproduce_results.py --output-dir reproduced
```

To analyze a previously generated complete bank separately:

```powershell
py -3 -B tools/reproduce_results.py --input runs/full/MERGED_RAW_TRAJECTORIES.csv --output-dir runs/full-analysis
```

## Standalone Leveling

```powershell
py -3 -B baselines/Leveling/evaluate_caserta.py --output-dir runs/leveling-caserta
py -3 -B -m pytest -q baselines/Leveling/tests
```

Leveling is forecast-independent. It is not counted as an empirical Ridge comparison in the six-solver bank. Its own 840-instance Caserta evaluator is included separately.

## Scope of reproduction

This reproduces the published six-solver evaluation with fixed trained models. It does not recreate the previous execution history or stage approvals, GP evolutionary training, other R variants, neural training, sealed testing, or OOD evaluation. The historical shared helper scripts remain in the snapshot; prior-scope receipts and result data are preserved in the local archive. the portable runner has its own input binding and receipts and does not alter or impersonate historical stage releases. A rerun on the same layouts verifies implementation and reported values; it does not provide new generalization evidence.
