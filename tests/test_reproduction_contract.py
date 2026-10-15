"""Check that the replay verifier rejects wrong observations and partial banks."""
from copy import deepcopy
import pytest

from reproduction.runner import compare_rows, read_csv, RAW


def test_archive_comparator_rejects_relocation_changes():
    row = deepcopy(read_csv(RAW)[0])
    row["R_raw"] = str(int(row["R_raw"])+1)
    with pytest.raises(ValueError, match="Archive mismatch"):
        compare_rows([row], 1)


def test_archive_comparator_rejects_duplicates_and_incomplete_matrices():
    row = read_csv(RAW)[0]
    with pytest.raises(ValueError, match="Duplicate"):
        compare_rows([row, row], 2)
    with pytest.raises(ValueError, match="incomplete"):
        compare_rows([row], 2)


def test_comparison_allows_fresh_timing_and_provenance():
    row = deepcopy(read_csv(RAW)[0])
    row.update(wall_seconds="1234.5", actor_time_seconds="999", source_origin="fresh_portable_rerun")
    assert compare_rows([row], 1)["deterministic_mismatches"] == 0


def test_reference_contains_only_active_families_and_all_expected_rows():
    rows = read_csv(RAW)
    assert len(rows) == 10800
    assert {r["family"] for r in rows} == {"subset_shuffle", "time_window_shuffle", "CLEAN"}
    assert len({r["condition_id"] for r in rows}) == 15
