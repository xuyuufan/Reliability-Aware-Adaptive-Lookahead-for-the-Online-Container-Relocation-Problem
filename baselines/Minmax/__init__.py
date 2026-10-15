"""Public entrypoint for the frozen Minmax solver."""


def make_policy(method="FIXED_H20", *, context=None):
    from baselines import make_policy as create
    return create("Minmax", method, context=context)
