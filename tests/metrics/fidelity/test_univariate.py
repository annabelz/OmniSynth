"""
Unit tests for univariate fidelity metrics.

Covers
------
WassersteinDistance  — score range, perfect copy, shifted data, NaN handling,
                       no numerical columns edge case
TotalVariationDistance — score range, perfect copy, shifted data, NaN handling,
                         no categorical columns edge case, disjoint categories
HellingerDistance    — score range, perfect copy, shifted data, NaN handling,
                       numerical histogram path, constant column edge case
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stdg_eval.metrics.fidelity.univariate import (
    HellingerDistance,
    TotalVariationDistance,
    WassersteinDistance,
)
from stdg_eval.utils.data_utils import ColumnTypes


# ===========================================================================
# WassersteinDistance
# ===========================================================================

class TestWassersteinDistance:
    metric = WassersteinDistance()

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

    def test_no_numerical_columns_returns_one(self, only_categorical_df, col_types_cat_only):
        result = self.metric.evaluate(only_categorical_df, only_categorical_df.copy(), col_types_cat_only)
        assert result.score == pytest.approx(1.0)
        assert "message" in result.details

    def test_column_scores_keys_match_numerical_cols(self, real_df, synth_perfect, col_types):
        result = self.metric.evaluate(real_df, synth_perfect, col_types)
        num_cols = {"age", "bmi", "score"}
        assert set(result.column_scores.keys()) == num_cols

    def test_column_scores_in_unit_interval(self, real_df, synth_shifted, col_types):
        result = self.metric.evaluate(real_df, synth_shifted, col_types)
        for col, s in result.column_scores.items():
            assert 0.0 <= s <= 1.0, f"column_score out of range for {col!r}: {s}"

    def test_all_nan_column_skipped_no_crash(self, real_df, col_types):
        synth = real_df.copy()
        synth["age"] = np.nan
        result = self.metric.evaluate(real_df, synth, col_types)
        # 'age' should be absent from column_scores (skipped), rest computed
        assert "age" not in result.column_scores
        assert "bmi" in result.column_scores

    def test_details_contain_raw_and_normalised_distances(self, real_df, synth_shifted, col_types):
        result = self.metric.evaluate(real_df, synth_shifted, col_types)
        assert "raw_distances" in result.details
        assert "normalised_distances" in result.details

    def test_identical_distribution_raw_distance_near_zero(self, real_df, col_types):
        result = self.metric.evaluate(real_df, real_df.copy(), col_types)
        for col, d in result.details["raw_distances"].items():
            assert d == pytest.approx(0.0, abs=1e-9), f"raw distance should be 0 for {col!r}"


# ===========================================================================
# TotalVariationDistance
# ===========================================================================

class TestTotalVariationDistance:
    metric = TotalVariationDistance()

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

    def test_no_categorical_columns_returns_one(self, only_numerical_df, col_types_num_only):
        result = self.metric.evaluate(only_numerical_df, only_numerical_df.copy(), col_types_num_only)
        assert result.score == pytest.approx(1.0)
        assert "message" in result.details

    def test_column_scores_keys_match_categorical_cols(self, real_df, synth_perfect, col_types):
        result = self.metric.evaluate(real_df, synth_perfect, col_types)
        cat_cols = {"sex", "diagnosis", "smoker"}
        assert set(result.column_scores.keys()) == cat_cols

    def test_column_scores_in_unit_interval(self, real_df, synth_shifted, col_types):
        result = self.metric.evaluate(real_df, synth_shifted, col_types)
        for col, s in result.column_scores.items():
            assert 0.0 <= s <= 1.0, f"column_score out of range for {col!r}: {s}"

    def test_disjoint_categories_score_near_zero(self):
        """Synthetic with entirely different categories → TVD = 1, score = 0."""
        real = pd.DataFrame({"cat": ["A", "B", "A", "B"]})
        synth = pd.DataFrame({"cat": ["C", "D", "C", "D"]})
        ct: ColumnTypes = {"cat": "categorical"}
        result = TotalVariationDistance().evaluate(real, synth, ct)
        assert result.score == pytest.approx(0.0, abs=1e-9)

    def test_unseen_synth_category_handled(self):
        """Category present in synth but not real — union is used, no crash."""
        real = pd.DataFrame({"cat": ["A", "A", "B", "B"]})
        synth = pd.DataFrame({"cat": ["A", "B", "C", "C"]})
        ct: ColumnTypes = {"cat": "categorical"}
        result = TotalVariationDistance().evaluate(real, synth, ct)
        assert 0.0 <= result.score <= 1.0

    def test_details_contain_tvd_values_and_frequencies(self, real_df, synth_shifted, col_types):
        result = self.metric.evaluate(real_df, synth_shifted, col_types)
        assert "tvd_values" in result.details
        assert "real_frequencies" in result.details
        assert "synth_frequencies" in result.details

    def test_nan_values_excluded_before_comparison(self):
        real = pd.DataFrame({"cat": ["A", "A", None, "B"]})
        synth = pd.DataFrame({"cat": [None, "A", "A", "B"]})
        ct: ColumnTypes = {"cat": "categorical"}
        result = TotalVariationDistance().evaluate(real, synth, ct)
        assert 0.0 <= result.score <= 1.0


# ===========================================================================
# HellingerDistance
# ===========================================================================

class TestHellingerDistance:
    metric = HellingerDistance()

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

    def test_column_scores_cover_all_columns(self, real_df, synth_perfect, col_types):
        result = self.metric.evaluate(real_df, synth_perfect, col_types)
        assert set(result.column_scores.keys()) == set(col_types.keys())

    def test_column_scores_in_unit_interval(self, real_df, synth_shifted, col_types):
        result = self.metric.evaluate(real_df, synth_shifted, col_types)
        for col, s in result.column_scores.items():
            assert 0.0 <= s <= 1.0, f"column_score out of range for {col!r}: {s}"

    def test_numerical_path_uses_shared_bins(self):
        """Both datasets are binned on the same shared grid — identical data should score 1."""
        rng = np.random.default_rng(5)
        n = 100
        vals = rng.normal(0, 1, n)
        real = pd.DataFrame({"x": vals})
        synth = pd.DataFrame({"x": vals.copy()})
        ct: ColumnTypes = {"x": "numerical"}
        result = HellingerDistance().evaluate(real, synth, ct)
        assert result.score == pytest.approx(1.0, abs=1e-9)

    def test_categorical_path_handles_unseen_category(self):
        """Category in synth not in real → prob = 0 for real, handled via union."""
        real = pd.DataFrame({"cat": ["A", "A", "B", "B"]})
        synth = pd.DataFrame({"cat": ["A", "B", "C", "C"]})
        ct: ColumnTypes = {"cat": "categorical"}
        result = HellingerDistance().evaluate(real, synth, ct)
        assert 0.0 <= result.score <= 1.0

    def test_constant_numerical_column_scores_one(self):
        """When lo == hi (constant column), Hellinger = 0, score = 1."""
        real = pd.DataFrame({"x": [5.0] * 50})
        synth = pd.DataFrame({"x": [5.0] * 50})
        ct: ColumnTypes = {"x": "numerical"}
        result = HellingerDistance().evaluate(real, synth, ct)
        assert result.score == pytest.approx(1.0, abs=1e-9)

    def test_disjoint_distributions_score_near_zero(self):
        """Completely non-overlapping distributions → Hellinger = 1, score = 0."""
        real = pd.DataFrame({"cat": ["A"] * 50})
        synth = pd.DataFrame({"cat": ["B"] * 50})
        ct: ColumnTypes = {"cat": "categorical"}
        result = HellingerDistance().evaluate(real, synth, ct)
        assert result.score == pytest.approx(0.0, abs=1e-9)

    def test_details_contain_hellinger_values(self, real_df, synth_shifted, col_types):
        result = self.metric.evaluate(real_df, synth_shifted, col_types)
        assert "hellinger_values" in result.details

    def test_nan_values_dropped_before_computation(self, real_with_missing, synth_with_missing, col_types):
        result = self.metric.evaluate(real_with_missing, synth_with_missing, col_types)
        assert 0.0 <= result.score <= 1.0

    def test_metric_name(self):
        assert self.metric.name == "Hellinger Distance"

    def test_metric_axis(self):
        assert self.metric.axis == "fidelity"
