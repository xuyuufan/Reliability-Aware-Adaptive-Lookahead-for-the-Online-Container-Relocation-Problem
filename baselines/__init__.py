"""Public baseline entrypoint, backed by the single frozen source snapshot."""

ALGORITHMS = ("EM", "GP", "ERI", "EG", "Chain", "Minmax", "Leveling")


def make_policy(algorithm, method="FIXED_H20", *, context=None):
    """Create an actor accepting only a public frame; no evaluator metadata."""
    from reproduction.runtime import load_runtime, make_policy as create
    return create(algorithm, method, context or load_runtime())
