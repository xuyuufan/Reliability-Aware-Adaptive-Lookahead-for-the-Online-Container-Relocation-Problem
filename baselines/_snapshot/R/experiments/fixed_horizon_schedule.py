"""Only the registered M1/M2 fixed-H trajectory keys are schedulable."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

WORKSPACE = Path(__file__).resolve().parents[2]
HORIZONS = (0, 2, 5, 10)
ALGORITHMS = ("EM", "GP", "ERI", "EG", "Chain", "Minmax")


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def registered_schedule() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    from R.experiments.r_five_versions_comparison import condition_specs
    layout_path = WORKSPACE / "R/output/r_frozen_cross_solver_transfer_v1/r01/data/select20.json"
    layouts = json.loads(layout_path.read_text(encoding="utf-8"))
    conditions = condition_specs()
    if len(layouts) != 20 or len(conditions) != 21:
        raise ValueError("fixed-H schedule requires the registered 20x21 layout/condition matrix")
    if sum(row.get("parameter_group") == "G1" for row in layouts) != 10 or sum(row.get("parameter_group") == "G2" for row in layouts) != 10:
        raise ValueError("fixed-H schedule geometry split differs from registration")
    return layouts, conditions


def _key(layout: dict[str, Any], condition: dict[str, Any], algorithm: str, method: str) -> str:
    return "|".join((str(layout["label_invariant_fingerprint"]), str(condition["condition_id"]), algorithm, method))


def parse_fixed_bank_key(key: str) -> tuple[str, str, str, str]:
    """Parse a fixed-H key while preserving pipes inside condition identifiers."""
    parts = str(key).split("|")
    if len(parts) < 4 or not all(parts):
        raise ValueError(f"malformed fixed-H key: {key!r}")
    fingerprint, algorithm, method = parts[0], parts[-2], parts[-1]
    condition_id = "|".join(parts[1:-2])
    if not condition_id:
        raise ValueError(f"fixed-H key has an empty condition id: {key!r}")
    return fingerprint, condition_id, algorithm, method


def smoke_keys() -> set[str]:
    layouts, conditions = registered_schedule()
    selected_layouts = [row for row in layouts if row.get("parameter_group") in ("G1", "G2")]
    selected_layouts = [next(row for row in selected_layouts if row.get("parameter_group") == geometry) for geometry in ("G1", "G2")]
    selected_conditions = [
        next(row for row in conditions if row["family"] == "CLEAN"),
        next(row for row in conditions if row["family"] == "delayed_positions" and row["p"] == 0.3 and row["repeat"] == 0),
    ]
    keys = {
        _key(layout, condition, algorithm, f"FIXED_H{horizon}")
        for layout in selected_layouts for condition in selected_conditions
        for algorithm in ALGORITHMS for horizon in HORIZONS
    }
    if len(keys) != 96:
        raise ValueError(f"smoke key count {len(keys)} != 96")
    return keys


def bank_keys() -> set[str]:
    layouts, conditions = registered_schedule()
    keys = {
        _key(layout, condition, algorithm, f"FIXED_H{horizon}")
        for layout in layouts for condition in conditions
        for algorithm in ALGORITHMS for horizon in HORIZONS
    }
    if len(keys) != 10080:
        raise ValueError(f"registered full new-H bank count {len(keys)} != 10080")
    return keys


def remaining_bank_keys() -> set[str]:
    keys = bank_keys() - smoke_keys()
    if len(keys) != 9984:
        raise ValueError(f"remaining M2 key count {len(keys)} != 9984")
    return keys


def whitelist_digest(keys: set[str]) -> str:
    return sha256_bytes(_canonical(sorted(keys)))


def build_m1_bridge_keys() -> set[str]:
    layouts, conditions = registered_schedule()
    layout = next(row for row in layouts if row.get("parameter_group") == "G1")
    selected_conditions = [
        next(row for row in conditions if row["family"] == "CLEAN"),
        next(row for row in conditions if row["family"] == "delayed_positions" and row["p"] == 0.3 and row["repeat"] == 0),
    ]
    keys = {
        "BRIDGE|" + _key(layout, condition, algorithm, method)
        for condition in selected_conditions for algorithm in ALGORITHMS
        for method in ("R_UTILITY_RIDGE", "NATIVE20")
    }
    if len(keys) != 24:
        raise ValueError(f"bridge key count {len(keys)} != 24")
    return keys


def m1_key_digest() -> str:
    return whitelist_digest(build_m1_bridge_keys() | smoke_keys())

