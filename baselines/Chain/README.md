# Chain

Public-prefix Chain heuristic. The implementation used by the accepted experiment is retained byte-for-byte in the shared snapshot; this folder provides its public entrypoint.

## Implementation

- [Chain/chain_public.py](../_snapshot/Chain/chain_public.py)
- [Chain/chain_offline.py](../_snapshot/Chain/chain_offline.py)

## Verify this solver

From the repository root:

```powershell
py -3 -B tools/reproduce_experiments.py --scope smoke --algorithms Chain --output-dir runs/chain-smoke
```

This runs 36 fresh trajectories: both geometries, three representative conditions, and Ridge plus H=0,2,5,10,20. It compares all trajectory counts and forecast bindings with the archive.

```python
from baselines.Chain import make_policy
actor = make_policy("FIXED_H20")  # or RIDGE, FIXED_H0/2/5/10
action, trace = actor.select(public_frame)
```

The actor accepts a detached public frame. Evaluator truth, noise parameters, forecast seeds, and dataset identifiers are not actor inputs. See [complete reproduction instructions](../../docs/REPRODUCTION.md) for the full six-solver run.
