# Experiment inputs

| Input | Location | Role |
|---|---|---|
| 20 exposed development layouts | [select20.json](layouts/select20.json) | Complete six-solver experiment; G1/G2, 10 layouts each |
| 10,800 reference trajectories | [reference_trajectories.csv](reference_trajectories.csv) | Two noise families plus CLEAN; historical observations retained solely within the current evaluation scope |
| 840 original Caserta episodes | [Caserta benchmark](../baselines/Leveling/data/caserta_table2_840.json) | Standalone Leveling evaluation |

The current noise families are subset_shuffle and time_window_shuffle. CLEAN is separate. The complete matrix has 15 conditions, six solvers, and Ridge plus five fixed horizons.

The portable runner generates fresh trajectories from the layouts and deterministic forecast functions. It does not copy reference measurements into new results. Reference observations are used only to compare deterministic outcomes and forecast pairing. New rows are marked `fresh_portable_rerun`; timings and source metadata are recorded anew.
