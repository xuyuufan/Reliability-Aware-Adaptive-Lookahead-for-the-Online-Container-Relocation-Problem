# Frozen models

- [RIDGE_MODEL.json](RIDGE_MODEL.json): 74 coefficients, float64 Ridge, lambda=1, no intercept or standardization; original SHA-256 `07bf606c2a7ab1ac1bffa7d39449c34fc719ad8e454eba1243761af44fd7effe`.
- [GP_RULE.json](GP_RULE.json): the exact accepted `GP-H20-C-run25` prefix tree and expression, extracted from the original model receipt. It retains source/training receipt hashes and historical selection metadata without including old evaluation data. The loader verifies prefix/expression identity and evaluates only through the public-information slow adapter.

Neither model is refitted, recalibrated, or reselected for the current evaluation scope. The earlier GP evolutionary training pipeline is outside the scope of this experiment reproduction.
