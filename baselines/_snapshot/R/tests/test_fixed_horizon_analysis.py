"""Synthetic-only contract tests for the fixed-horizon statistics analyzer."""
from __future__ import annotations

import math
import csv
import tempfile
import sys
import unittest
from pathlib import Path

import numpy as np

WORKSPACE = Path(__file__).resolve().parents[2]
if str(WORKSPACE) not in sys.path:
    sys.path.insert(0, str(WORKSPACE))

from R.experiments import fixed_horizon_analysis as analysis


class FixedHorizonAnalysisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        # The full fixture is generated in memory; no trajectory file is read.
        cls.rows = analysis.build_synthetic_rows()
        cls.registry = analysis.build_synthetic_layout_registry()
        cls.bank = analysis.validate_rows(cls.rows, cls.registry)

    def test_full_matrix_and_common_frozen_forecast_pairing(self) -> None:
        self.assertEqual(len(self.rows), 10800)
        self.assertEqual(len(analysis.CONDITIONS), 15)
        self.assertEqual(self.bank.forecast_pair_count, 20 * 15)
        self.assertEqual(self.bank.raw.shape, (20, 15, 6, 6))
        self.assertEqual(self.bank.rn.shape, self.bank.raw.shape)
        self.assertEqual(sum(v == "G1" for v in self.bank.geometry_by_layout.values()), 10)
        self.assertEqual(sum(v == "G2" for v in self.bank.geometry_by_layout.values()), 10)

    def test_hierarchical_weights_and_ratio_direction(self) -> None:
        raw_scopes = analysis.scope_layout_values(self.bank.raw)
        macro = analysis.geometry_mean(raw_scopes["noisy"], self.bank.geometry_by_layout, self.bank.layouts)
        em = analysis.SOLVERS.index("EM")
        ridge = analysis.RIDGE_INDEX
        fixed_h2 = analysis.POLICY_FOR_H[2]

        # Per layout: repeat means are 32 / 244 for the two families;
        # family means are equal, and G1/G2 layout means are then equal.
        # The method deltas are -20 for Ridge and +6 for FIXED_H2.
        self.assertAlmostEqual(float(macro[em, ridge]), 277.5)
        self.assertAlmostEqual(float(macro[em, fixed_h2]), 303.5)
        self.assertAlmostEqual(float(macro[em, fixed_h2] - macro[em, ridge]), 26.0)
        rate = 100 * (macro[em, fixed_h2] - macro[em, ridge]) / macro[em, fixed_h2]
        self.assertAlmostEqual(float(rate), 100 * 26 / 303.5)
        naive_all_conditions = (6 * 32 + 8 * 244) / 14
        self.assertNotAlmostEqual(naive_all_conditions, 138.0)

    def test_duplicate_and_missing_keys_fail_closed(self) -> None:
        duplicate = list(self.rows)
        duplicate[-1] = dict(duplicate[0])
        with self.assertRaisesRegex(analysis.AnalysisFailure, "duplicate"):
            analysis.validate_rows(duplicate, self.registry)
        with self.assertRaisesRegex(analysis.AnalysisFailure, "expected 10800"):
            analysis.validate_rows(self.rows[:-1], self.registry)

    def test_forecast_mismatch_within_layout_condition_fails(self) -> None:
        changed = [dict(row) for row in self.rows]
        first = changed[0]
        pair = (first["layout_fingerprint"], first["condition_id"])
        other = next(
            row for row in changed
            if (row["layout_fingerprint"], row["condition_id"]) == pair
            and row["trajectory_key"] != first["trajectory_key"]
        )
        other["F_hash"] = "SYNTHETIC_FORECAST_MISMATCH"
        with self.assertRaisesRegex(analysis.AnalysisFailure, "forecast"):
            analysis.validate_rows(changed, self.registry)

    def test_m3_entry_exit_gate_snapshot_contract_is_synthetic_only(self) -> None:
        # Mocked release data/path exercise the same entry and exit helpers used
        # by formal M3, without creating ROOT artifacts or reading merged data.
        keys = tuple(
            "plan_sha256 authorization_file_sha256 source_bindings_file_sha256 "
            "source_binding_sha256 reuse_receipt_file_sha256 "
            "synthetic_receipt_file_sha256 m0_preparation_receipt_file_sha256 "
            "root_m0_review_file_sha256 historical_reuse_only_binding_sha256".split()
        )
        release_payload = {key: f"SYNTHETIC_{key.upper()}" for key in keys}
        release_payload.update({
            "release_path": "SYNTHETIC_MOCK_ROOT_M3_RELEASE__NOT_CREATED.json",
            "release_file_sha256": "SYNTHETIC_RELEASE_SHA256",
            "release_scope": "synthetic_scope_only",
            "analysis_budget_cap_seconds": 1800,
            "status": "ROOT_RELEASED",
            "authorized_trajectories": 0,
        })

        class MockIntegrity:
            FROZEN_HASH_KEYS = keys

            def __init__(self) -> None:
                self.events: list[str] = []
                self.snapshots: list[dict[str, str]] = []
                self.current_release = dict(release_payload)

            def require_stage_release(self, stage: str) -> dict[str, object]:
                if stage != "M3_ANALYSIS":
                    raise AssertionError(f"unexpected stage: {stage}")
                self.events.append("require")
                return dict(self.current_release)

            def verify_stage_unchanged(self, stage: str, snapshot: dict[str, str]) -> dict[str, object]:
                if stage != "M3_ANALYSIS":
                    raise AssertionError(f"unexpected stage: {stage}")
                self.events.append("verify")
                self.snapshots.append(dict(snapshot))
                current = self.require_stage_release(stage)
                if current["release_file_sha256"] != snapshot["release_file_sha256"]:
                    raise analysis.AnalysisFailure("ROOT M3 release file changed while the stage was running")
                return {
                    "stage": stage,
                    "m0_hashes_unchanged": True,
                    "release_file_sha256": current["release_file_sha256"],
                }

        mock = MockIntegrity()
        payload, snapshot = analysis._capture_m3_entry_snapshot(mock)
        self.assertEqual(analysis._validate_release_analysis_cap(payload, 1800.0), 1800.0)
        self.assertEqual(tuple(snapshot), (*keys, "release_file_sha256"))
        self.assertNotIn("release_path", snapshot)
        self.assertNotIn("analysis_budget_cap_seconds", snapshot)
        self.assertEqual(payload["release_path"], "SYNTHETIC_MOCK_ROOT_M3_RELEASE__NOT_CREATED.json")
        self.assertFalse(Path(str(payload["release_path"])).exists())

        entry_result = analysis._verify_m3_snapshot_unchanged(mock, snapshot)
        exit_result = analysis._verify_m3_snapshot_unchanged(mock, snapshot)
        self.assertEqual(entry_result["release_file_sha256"], "SYNTHETIC_RELEASE_SHA256")
        self.assertEqual(exit_result["release_file_sha256"], "SYNTHETIC_RELEASE_SHA256")
        self.assertEqual(mock.events, ["require", "verify", "require", "verify", "require"])
        self.assertEqual(mock.snapshots, [snapshot, snapshot])

        mock.current_release["release_file_sha256"] = "CHANGED_SYNTHETIC_RELEASE_SHA256"
        with self.assertRaisesRegex(analysis.AnalysisFailure, "release file changed"):
            analysis._verify_m3_snapshot_unchanged(mock, snapshot)

    def test_analysis_cap_is_bounded_before_any_formal_input_access(self) -> None:
        with self.assertRaisesRegex(analysis.AnalysisFailure, "exceeds the registered 1800s maximum"):
            analysis.run_formal_analysis(
                Path("<SYNTHETIC_INPUT_MUST_NOT_BE_OPENED>"),
                replicates=analysis.BOOTSTRAP_REPLICATES,
                cap_seconds=1800.001,
            )
        with self.assertRaisesRegex(analysis.AnalysisFailure, "finite positive"):
            analysis.run_formal_analysis(
                Path("<SYNTHETIC_INPUT_MUST_NOT_BE_OPENED>"),
                replicates=analysis.BOOTSTRAP_REPLICATES,
                cap_seconds=float("inf"),
            )
        self.assertEqual(analysis._validate_requested_analysis_cap(1800), 1800.0)
        self.assertEqual(analysis._validate_requested_analysis_cap(900), 900.0)
        with self.assertRaisesRegex(analysis.AnalysisFailure, "must exactly match"):
            analysis._validate_release_analysis_cap(
                {"analysis_budget_cap_seconds": 1799}, 1799.0,
            )

    def test_development_best_checks_h20_and_breaks_exact_ties_downward(self) -> None:
        point = np.full((len(analysis.SOLVERS), len(analysis.POLICIES)), 120.0, dtype=np.float64)
        point[:, analysis.RIDGE_INDEX] = 180.0
        for si in range(len(analysis.SOLVERS)):
            for h, value in zip(analysis.HORIZONS, (100.0, 105.0, 110.0, 115.0, 125.0)):
                point[si, analysis.POLICY_FOR_H[h]] = value

        # EM's only best policy is H20; GP has an exact H0/H20 tie, so the
        # smaller registered horizon must win. Other solvers prefer H0.
        point[0, analysis.POLICY_FOR_H[20]] = 50.0
        point[1, analysis.POLICY_FOR_H[0]] = 40.0
        point[1, analysis.POLICY_FOR_H[20]] = 40.0
        boot = np.repeat(point[np.newaxis, :, :], repeats=32, axis=0)
        primary, _ = analysis.primary_comparisons(point, boot, replicates=len(boot))
        best_rows = analysis._best_fixed_rows(point, boot, primary)
        best_by_solver = {row["algorithm"]: row for row in best_rows if row["algorithm"] in analysis.SOLVERS}
        self.assertEqual(best_by_solver["EM"]["development_best_horizon"], 20)
        self.assertEqual(best_by_solver["EM"]["development_best_fixed_mean_R_raw"], 50.0)
        self.assertEqual(best_by_solver["GP"]["development_best_horizon"], 0)
        self.assertEqual(best_by_solver["GP"]["development_best_fixed_mean_R_raw"], 40.0)

        # H-vs-H20 remains a separate descriptive table of the four shorter
        # fixed horizons; H20 is its reference, not a redundant self-contrast.
        rn_point = point / 100.0
        rn_boot = boot / 100.0
        h20_rows = analysis._fixed_h_vs_h20_rows(
            point, rn_point, boot, rn_boot, replicates=len(boot),
        )
        self.assertEqual(set(row["horizon"] for row in h20_rows), set(analysis.FIXED_HORIZONS))
        self.assertEqual(len(h20_rows), 2 * len(analysis.FIXED_HORIZONS) * (len(analysis.SOLVERS) + 1))

    def test_common_layout_pairing_and_35_simultaneous_intervals(self) -> None:
        raw_scopes = analysis.scope_layout_values(self.bank.raw)
        rn_scopes = analysis.scope_layout_values(self.bank.rn)
        indices = analysis.make_bootstrap_indices(
            self.bank.layouts, self.bank.geometry_by_layout, replicates=256,
            seed=analysis.BOOTSTRAP_SEED, batch_size=64,
        )
        rng = np.random.Generator(np.random.PCG64(analysis.BOOTSTRAP_SEED))
        expected_indices = []
        g1 = np.asarray([i for i, fp in enumerate(self.bank.layouts)
                         if self.bank.geometry_by_layout[fp] == "G1"], dtype=np.int16)
        g2 = np.asarray([i for i, fp in enumerate(self.bank.layouts)
                         if self.bank.geometry_by_layout[fp] == "G2"], dtype=np.int16)
        for batch_start in range(0, 256, 64):
            count = min(64, 256 - batch_start)
            first = g1[rng.integers(0, 10, size=(count, 10))]
            second = g2[rng.integers(0, 10, size=(count, 10))]
            expected_indices.extend(np.concatenate((first, second), axis=1))
        self.assertTrue((indices == np.asarray(expected_indices)).all())
        self.assertEqual(analysis.BOOTSTRAP_BATCH_SIZE, 1000)
        self.assertEqual(analysis.BOOTSTRAP_QUANTILE_METHOD, "linear")
        for draw in indices:
            self.assertTrue(all(self.bank.geometry_by_layout[self.bank.layouts[i]] == "G1" for i in draw[:10]))
            self.assertTrue(all(self.bank.geometry_by_layout[self.bank.layouts[i]] == "G2" for i in draw[10:]))

        noisy = raw_scopes["noisy"]
        fixed = noisy[:, :, analysis.POLICY_FOR_H[2]]
        ridge = noisy[:, :, analysis.RIDGE_INDEX]
        paired_delta = fixed[indices].mean(axis=1) - ridge[indices].mean(axis=1)
        self.assertTrue(math.isclose(float(paired_delta.min()), 26.0, abs_tol=1e-12))
        self.assertTrue(math.isclose(float(paired_delta.max()), 26.0, abs_tol=1e-12))

        raw_boot, _ = analysis.bootstrap_scope_pair(
            raw_scopes["noisy"], rn_scopes["noisy"], indices, batch_size=64,
        )
        point = analysis.geometry_mean(noisy, self.bank.geometry_by_layout, self.bank.layouts)
        main_rows, overall_rows = analysis.primary_comparisons(point, raw_boot, replicates=len(indices))
        self.assertEqual(len(main_rows), 30)
        self.assertEqual(len(overall_rows), 5)
        self.assertEqual(len(main_rows) + len(overall_rows), 35)
        h0_row = next(row for row in main_rows if row["algorithm"] == "EM" and row["horizon"] == 0)
        fixed_index = analysis.POLICY_FOR_H[0]
        rate_draws = 100 * (raw_boot[:, 0, fixed_index] - raw_boot[:, 0, analysis.RIDGE_INDEX]) / raw_boot[:, 0, fixed_index]
        expected_lo, expected_hi = analysis._quantiles(
            rate_draws, (analysis.SIMULTANEOUS_TAIL, 1 - analysis.SIMULTANEOUS_TAIL),
        )
        self.assertAlmostEqual(h0_row["simultaneous_low_percent"], expected_lo)
        self.assertAlmostEqual(h0_row["simultaneous_high_percent"], expected_hi)
        solver_rates = [
            100 * (point[si, fixed_index] - point[si, analysis.RIDGE_INDEX]) / point[si, fixed_index]
            for si in range(len(analysis.SOLVERS))
        ]
        self.assertAlmostEqual(overall_rows[0]["equal6_improvement_percent"], sum(solver_rates) / 6)

    def test_synthetic_chart_exports_png_and_vector_pdf(self) -> None:
        raw_scopes = analysis.scope_layout_values(self.bank.raw)
        rn_scopes = analysis.scope_layout_values(self.bank.rn)
        indices = analysis.make_bootstrap_indices(
            self.bank.layouts, self.bank.geometry_by_layout, replicates=64,
            seed=analysis.BOOTSTRAP_SEED, batch_size=32,
        )
        raw_boot, rn_boot = analysis.bootstrap_scope_pair(
            raw_scopes["noisy"], rn_scopes["noisy"], indices, batch_size=32,
        )
        raw_point = analysis.geometry_mean(raw_scopes["noisy"], self.bank.geometry_by_layout, self.bank.layouts)
        primary, _ = analysis.primary_comparisons(raw_point, raw_boot, replicates=len(indices))
        curve = analysis._fixed_curve_rows(raw_point, raw_boot, len(indices))

        family_rows = []
        for family in analysis.FAMILIES:
            scope = f"family:{family}"
            raw_point_family = analysis.geometry_mean(raw_scopes[scope], self.bank.geometry_by_layout, self.bank.layouts)
            rn_point_family = analysis.geometry_mean(rn_scopes[scope], self.bank.geometry_by_layout, self.bank.layouts)
            rb, nb = analysis.bootstrap_scope_pair(raw_scopes[scope], rn_scopes[scope], indices, batch_size=32)
            family_scope_rows = analysis._descriptive_rows(scope, raw_point_family, rn_point_family, rb, nb, replicates=len(indices))
            for row in family_scope_rows:
                row["family"] = family
            family_rows.extend(family_scope_rows)
        clean_raw = analysis.geometry_mean(raw_scopes["clean"], self.bank.geometry_by_layout, self.bank.layouts)
        clean_rn = analysis.geometry_mean(rn_scopes["clean"], self.bank.geometry_by_layout, self.bank.layouts)
        clean_rb, clean_nb = analysis.bootstrap_scope_pair(raw_scopes["clean"], rn_scopes["clean"], indices, batch_size=32)
        clean_rows = analysis._descriptive_rows("CLEAN", clean_raw, clean_rn, clean_rb, clean_nb, replicates=len(indices))

        with tempfile.TemporaryDirectory(prefix="fixed_horizon_plot_selftest_") as temp:
            out_dir = Path(temp)
            names = analysis._render_figures(out_dir, curve, primary, family_rows, clean_rows)
            self.assertEqual(len(names), 8)
            for name in names:
                artifact = out_dir / name
                self.assertTrue(artifact.is_file())
                self.assertGreater(artifact.stat().st_size, 1000)
            for name in names:
                if name.endswith(".pdf"):
                    self.assertTrue((out_dir / name).read_bytes().startswith(b"%PDF"))

    def test_synthetic_output_tables_and_report_contract(self) -> None:
        raw_scopes = analysis.scope_layout_values(self.bank.raw)
        rn_scopes = analysis.scope_layout_values(self.bank.rn)
        indices = analysis.make_bootstrap_indices(
            self.bank.layouts, self.bank.geometry_by_layout, replicates=32,
            seed=analysis.BOOTSTRAP_SEED, batch_size=16,
        )
        scope_boots = {
            scope: analysis.bootstrap_scope_pair(raw_scopes[scope], rn_scopes[scope], indices, batch_size=16)
            for scope in raw_scopes
        }
        point = analysis.geometry_mean(raw_scopes["noisy"], self.bank.geometry_by_layout, self.bank.layouts)
        primary, overall = analysis.primary_comparisons(point, scope_boots["noisy"][0], replicates=len(indices))
        rn_point = analysis.geometry_mean(rn_scopes["noisy"], self.bank.geometry_by_layout, self.bank.layouts)
        h20 = analysis._fixed_h_vs_h20_rows(point, rn_point, scope_boots["noisy"][0], scope_boots["noisy"][1], replicates=len(indices))
        best = analysis._best_fixed_rows(point, scope_boots["noisy"][0], primary)
        curve = analysis._fixed_curve_rows(point, scope_boots["noisy"][0], len(indices))
        with tempfile.TemporaryDirectory(prefix="fixed_horizon_output_selftest_") as temp:
            out_dir = Path(temp)
            names = analysis._write_analysis_outputs(
                out_dir, self.bank, raw_scopes, rn_scopes, scope_boots, primary, overall,
                h20, best, curve, input_sha256="SYNTHETIC_INPUT_SHA256",
                analyzer_sha256="SYNTHETIC_ANALYZER_SHA256", replicates=len(indices),
            )
            self.assertEqual(len(names), 21)
            with (out_dir / "RIDGE_VS_FIXED_H_MACRO.csv").open("r", encoding="utf-8", newline="") as stream:
                self.assertEqual(len(list(csv.DictReader(stream))), 30)
            with (out_dir / "OVERALL_EQUAL6_RESULTS.csv").open("r", encoding="utf-8", newline="") as stream:
                self.assertEqual(len(list(csv.DictReader(stream))), 5)
            with (out_dir / "CONDITION_RESULTS.csv").open("r", encoding="utf-8", newline="") as stream:
                self.assertEqual(len(list(csv.DictReader(stream))), 15 * 6 * 6 * 2)
            report = (out_dir / "FINAL_REPORT.md").read_text(encoding="utf-8")
            self.assertIn("two-family", report)
            self.assertIn("method=linear", report)
            self.assertIn("Bonferroni", report)


if __name__ == "__main__":
    unittest.main(verbosity=2)
