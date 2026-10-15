"""PLAN-046's algorithm-agnostic, evaluator-private time-window noise rule."""

from __future__ import annotations

import hashlib
import json
import operator
import re
from typing import Iterable, Sequence

import numpy as np


FAMILY = "time_window_shuffle"
PROTOCOL_ID = "r_noise_diagnostics_v1"
_FINGERPRINT_RE = re.compile(r"^[0-9a-f]{64}$")


def _strict_int(
    value: object, *, name: str, minimum: int | None = None, maximum: int | None = None
) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be an integer, not a boolean")
    try:
        result = operator.index(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if (minimum is not None and result < minimum) or (maximum is not None and result > maximum):
        lower = "unbounded" if minimum is None else str(minimum)
        upper = "unbounded" if maximum is None else str(maximum)
        raise ValueError(f"{name} must be in [{lower}, {upper}]")
    return int(result)


def _integer_tokens(values: Iterable[object]) -> tuple[int, ...]:
    result = tuple(
        _strict_int(value, name=f"true_future[{index}]")
        for index, value in enumerate(values)
    )
    if len(set(result)) != len(result):
        raise ValueError("true_future must contain unique tokens")
    return result


def make_time_window_forecast(
    true_future: Sequence[int], *, window_size: int, seed: int
) -> tuple[int, ...]:
    """Uniformly permute independent consecutive chunks of a fixed future.

    Identity permutations are legal. The final short chunk is permuted as its
    own window. No value can move across a window boundary.
    """
    width = _strict_int(window_size, name="window_size", minimum=1)
    seed_value = _strict_int(seed, name="seed", minimum=0)
    values = _integer_tokens(true_future)
    if not values:
        return ()
    rng = np.random.Generator(np.random.PCG64(seed_value))
    result: list[int] = []
    for start in range(0, len(values), width):
        chunk = values[start : start + width]
        order = rng.permutation(len(chunk))
        result.extend(chunk[int(index)] for index in order)
    forecast = tuple(result)
    if len(forecast) != len(values) or set(forecast) != set(values):
        raise RuntimeError("time-window forecast is not a legal full permutation")
    for start in range(0, len(values), width):
        stop = min(start + width, len(values))
        if set(forecast[start:stop]) != set(values[start:stop]):
            raise RuntimeError("time-window forecast crossed a synthetic window boundary")
    return forecast


def semantic_time_window_seed(
    fingerprint: str, *, window_size: int, repeat: int
) -> int:
    """Return the frozen seed keyed only by layout, family, B, and repeat."""
    if not isinstance(fingerprint, str) or not _FINGERPRINT_RE.fullmatch(fingerprint):
        raise ValueError("fingerprint must be a lowercase 64-character SHA-256 hex string")
    width = _strict_int(window_size, name="window_size", minimum=1)
    repetition = _strict_int(repeat, name="repeat", minimum=0)
    payload = {
        "family": FAMILY,
        "fingerprint": fingerprint,
        "protocol_id": PROTOCOL_ID,
        "repeat": repetition,
        "window_size": width,
    }
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return int(hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:16], 16)


def time_window_forecast_sha256(forecast: Sequence[int]) -> str:
    """Hash the canonical compact-JSON integer list used in evaluator bindings."""
    values = _integer_tokens(forecast)
    canonical = json.dumps(list(values), separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


__all__ = [
    "FAMILY",
    "PROTOCOL_ID",
    "make_time_window_forecast",
    "semantic_time_window_seed",
    "time_window_forecast_sha256",
]
