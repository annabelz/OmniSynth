"""
Unit tests for bivariate fidelity metrics.

Covers
------
SpearmanCorrelation          — score range, perfect copy, shifted data, fewer than
                               2 numerical cols edge case, pair_differences details
ContingencyMatrix            — score range, perfect copy, shifted data, high-cardinality
                               skip, mixed (num × cat) pairs, disjoint categories,
                               NaN handling
PairwiseCorrelationDifference — score range, perfect copy, NaN pair filtering,
                                t-test details present, fewer than 2 cols edge case
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stdg_eval.metrics.fidelity.bivariate import (
    ContingencyMatrix,
    PairwiseCorrelationDifference,
    SpearmanCorrelation,
)
from stdg_eval.utils.data_utils import ColumnTypes


# ===========================================================================
# SpearmanCorrelation
# ===========================================================================

class TestSpearmanCorrelation:
    metric = SpearmanCorrelation()

    def test_perfect_copy_scores_one(self, real_df, synth_perfect, col_types):
        result = self.metric.evaluate(real_df, synth_perfect, col_types)
        assert result.score == pytest.approx(1.0, abs=1e-9)

    def test_score_in_unit_interval(self, real_df, synth_shifted, col_types):
        result = self.metric.evaluate(real_df, synth_shifted, col_types)
        assert 0.0 <= result.score <= 1.0

    def test_shifted_scores_below_perfect(self, real_df, synth_shifted, col_types):
        perfect = self.metric.evaluate(real_df, real_df.copy(), col_types)
        shifted = self.metric.evaluate(real_df, synth_shifted, col_types)
        assert shifted.score < perfect.score

    def test_fewer_than_two_numerical_cols_returns_one(self):
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0], "cat": ["A", "B", "A"]})
        ct: ColumnTypes = {"x": "numerical", "cat": "categorical"}
        result = SpearmanCorrelation().evaluate(df, df.copy(), ct)
        assert result.score == pytest.approx(1.0)
        assert "message" in result.details

    def test_pair_differences_keys_are_column_pairs(self, real_df, synth_perfect, col_types):
        result = self.metric.evaluate(real_df, synth_perfect, col_types)
        # Each key should be "col1|col2"
        for key in result.details["pair_differences"]:
            assert "|" in key
            c1, c2 = key.split("|")
            assert c1 in col_types
            assert c2 in col_types

    def test_pair_differences_near_zero_for_perfect_copy(self, real_df, synth_perfect, col_types):
        result = self.metric.evaluate(real_df, synth_perfect, col_types)
        for pair, diff in result.details["pair_differences"].items():
            assert diff == pytest.approx(0.0, abs=1e-9), f"non-zero diff for {pair!r}"

    def test_details_contain_both_correlation_matrices(self, real_df, synth_shifted, col_types):
        result = self.metric.evaluate(real_df, synth_shifted, col_types)
        assert "real_correlation_matrix" in result.details
        assert "synth_correlation_matrix" in result.details

    def test_nan_rows_dropped_before_correlation(self, real_with_missing, synth_with_missing, col_types):
        result = self.metric.evaluate(real_with_missing, synth_with_missing, col_types)
        assert 0.0 <= result.score <= 1.0

    def test_metric_name(self):
        assert self.metric.name == "Spearman Correlation"

    def test_metric_axis(self):
        assert self.metric.axis == "fidelity"


# ===========================================================================
# ContingencyMatrix
# ===========================================================================

class TestContingencyMatrix:
    metric = ContingencyMatrix()

    def test_perfect_copy_scores_one(self, real_df, synth_perfect, col_types):
        result = self.metric.evaluate(real_df, synth_perfect, col_types)
        assert result.score == pytest.approx(1.0, abs=1e-9)

    def test_score_in_unit_interval(self, real_df, synth_shifted, col_types):
        result = self.metric.evaluate(real_df, synth_shifted, col_types)
        assert 0.0 <= result.score <= 1.0

    def test_shifted_scores_below_perfect(self, real_df, synth_shifted, col_types):
        perfect = self.metric.evaluate(real_df, real_df.copy(), col_types)
        shifted = self.metric.evaluate(real_df, synth_shifted, col_types)
        assert shifted.score < perfect.score

    def test_high_cardinality_pair_is_skipped(self):
        """Column with > max_categories unique values should be excluded from pair_tvds."""
        rng = np.random.default_rng(20)
        n = 200
        # 'id_col' has 100 unique values — exceeds default max_categories=30
        real = pd.DataFrame({
            "id_col": [f"id_{i}" for i in range(n)],
            "grade":  rng.choice(["A", "B", "C"], n),
        })
        synth = pd.DataFrame({
            "id_col": [f"id_{i}" for i in range(n)],
            "grade":  rng.choice(["A", "B", "C"], n),
        })
        ct: ColumnTypes = {"id_col": "categorical", "grade": "categorical"}
        result = ContingencyMatrix(max_categories=30).evaluate(real, synth, ct)
        # The only pair id_col|grade should be skipped — no valid pairs
        assert "message" in result.details or result.score == pytest.approx(1.0)

    def test_custom_max_categories_threshold(self):
        """max_categories=5 should skip a 10-category column."""
        rng = np.random.default_rng(21)
        n = 100
        real = pd.DataFrame({
            "many_cats": rng.choice([f"cat_{i}" for i in range(10)], n),
            "few_cats":  rng.choice(["X", "Y"], n),
        })
        synth = real.copy()
        ct: ColumnTypes = {"many_cats": "categorical", "few_cats": "categorical"}
        result_strict = ContingencyMatrix(max_categories=5).evaluate(real, synth, ct)
        result_loose = ContingencyMatrix(max_categories=15).evaluate(real, synth, ct)
        # Strict should have no valid pairs (many_cats excluded), loose should include it
        assert "message" in result_strict.details or len(result_strict.details.get("pair_tvds", {})) == 0
        assert result_loose.score == pytest.approx(1.0, abs=1e-9)

    def test_mixed_num_cat_pairs_included(self, real_df, synth_perfect, col_types):
        """Numerical × categorical pairs should appear in pair_tvds."""
        result = self.metric.evaluate(real_df, synth_perfect, col_types)
        pair_keys = set(result.details.get("pair_tvds", {}).keys())
        # e.g. "age|sex" should be present
        mixed_pairs = [k for k in pair_keys if "|" in k]
        assert len(mixed_pairs) > 0, "Expected mixed num×cat pairs in pair_tvds"

    def test_numerical_numerical_pair_not_included(self, only_numerical_df, col_types_num_only):
        """Pure numerical × numerical pairs are skipped (handled by Spearman)."""
        result = self.metric.evaluate(only_numerical_df, only_numerical_df.copy(), col_types_num_only)
        # No cat columns → no qualifying pairs
        assert "message" in result.details

    def test_disjoint_categorical_pair_tvd_is_one(self):
        real = pd.DataFrame({"a": ["X", "X"], "b": ["Y", "Y"]})
        synth = pd.DataFrame({"a": ["A", "A"], "b": ["B", "B"]})
        ct: ColumnTypes = {"a": "categorical", "b": "categorical"}
        result = ContingencyMatrix().evaluate(real, synth, ct)
        tvds = result.details.get("pair_tvds", {})
        assert len(tvds) == 1
        assert list(tvds.values())[0] == pytest.approx(1.0, abs=1e-9)

    def test_identical_joint_distributions_tvd_is_zero(self, real_df, synth_perfect, col_types):
        result = self.metric.evaluate(real_df, synth_perfect, col_types)
        for pair, tvd in result.details["pair_tvds"].items():
            assert tvd == pytest.approx(0.0, abs=1e-9), f"TVD should be 0 for {pair!r}"

    def test_pair_tvds_values_in_unit_interval(self, real_df, synth_shifted, col_types):
        result = self.metric.evaluate(real_df, synth_shifted, col_types)
        for pair, tvd in result.details["pair_tvds"].items():
            assert 0.0 <= tvd <= 1.0, f"TVD out of [0,1] for {pair!r}: {tvd}"

    def test_reindex_handles_unseen_category_in_synth(self):
        """Synth has a category not in real — should be handled via union reindex."""
        real = pd.DataFrame({"a": ["X", "X", "Y"], "b": ["P", "Q", "P"]})
        synth = pd.DataFrame({"a": ["X", "Z", "Z"], "b": ["P", "P", "Q"]})
        ct: ColumnTypes = {"a": "categorical", "b": "categorical"}
        result = ContingencyMatrix().evaluate(real, synth, ct)
        assert 0.0 <= result.score <= 1.0

    def test_nan_rows_dropped_before_crosstab(self, real_with_missing, synth_with_missing, col_types):
        result = self.metric.evaluate(real_with_missing, synth_with_missing, col_types)
        assert 0.0 <= result.score <= 1.0


# ===========================================================================
# PairwiseCorrelationDifference
# ===========================================================================

class TestPairwiseCorrelationDifference:
    metric = PairwiseCorrelationDifference()

    def test_perfect_copy_score_near_one(self, real_df, synth_perfect, col_types):
        result = self.metric.evaluate(real_df, synth_perfect, col_types)
        assert result.score == pytest.approx(1.0, abs=0.05)

    def test_score_in_unit_interval(self, real_df, synth_shifted, col_types):
        result = self.metric.evaluate(real_df, synth_shifted, col_types)
        assert 0.0 <= result.score <= 1.0

    def test_shifted_scores_below_perfect(self, real_df, synth_shifted, col_types):
        perfect = self.metric.evaluate(real_df, real_df.copy(), col_types)
        shifted = self.metric.evaluate(real_df, synth_shifted, col_types)
        assert shifted.score <= perfect.score

    def test_fewer_than_two_columns_returns_one(self):
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
        ct: ColumnTypes = {"x": "numerical"}
        result = PairwiseCorrelationDifference().evaluate(df, df.copy(), ct)
        assert result.score == pytest.approx(1.0)
        assert "message" in result.details

    def test_nan_pairs_filtered_no_crash(self, real_with_missing, synth_with_missing, col_types):
        result = self.metric.evaluate(real_with_missing, synth_with_missing, col_types)
        assert 0.0 <= result.score <= 1.0

    def test_details_contain_pcd_value(self, real_df, synth_shifted, col_types):
        result = self.metric.evaluate(real_df, synth_shifted, col_types)
        assert "pcd" in result.details
        assert 0.0 <= result.details["pcd"] <= 1.0

    def test_details_contain_ttest_results(self, real_df, synth_shifted, col_types):
        result = self.metric.evaluate(real_df, synth_shifted, col_types)
        assert "t_statistic" in result.details
        assert "p_value" in result.details
        assert "significant_difference" in result.details

    def test_pair_diffs_keys_are_column_pairs(self, real_df, synth_perfect, col_types):
        result = self.metric.evaluate(real_df, synth_perfect, col_types)
        for key in result.details.get("pair_differences", {}):
            assert "|" in key

    def test_pair_diffs_near_zero_for_perfect_copy(self, real_df, synth_perfect, col_types):
        result = self.metric.evaluate(real_df, synth_perfect, col_types)
        for pair, diff in result.details.get("pair_differences", {}).items():
            assert diff == pytest.approx(0.0, abs=0.05), f"large diff for {pair!r}: {diff}"

    def test_pcd_is_mean_of_pair_diffs(self, real_df, synth_shifted, col_types):
        result = self.metric.evaluate(real_df, synth_shifted, col_types)
        diffs = list(result.details.get("pair_differences", {}).values())
        if diffs:
            expected_pcd = float(np.mean(diffs))
            assert result.details["pcd"] == pytest.approx(expected_pcd, rel=1e-6)

    def test_score_equals_one_minus_pcd(self, real_df, synth_shifted, col_types):
        result = self.metric.evaluate(real_df, synth_shifted, col_types)
        pcd = result.details["pcd"]
        assert result.score == pytest.approx(max(0.0, 1.0 - pcd), rel=1e-6)

    def test_metric_name(self):
        assert self.metric.name == "Pairwise Correlation Difference"

    def test_metric_axis(self):
        assert self.metric.axis == "fidelity"
