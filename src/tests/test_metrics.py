from datetime import datetime, timezone

import polars as pl
import pytest

from src.model._metrics import compute_metrics


class TestComputeMetrics:
    def _make_eval_df(self, actuals, q05s, q50s, q95s, feeders=None):
        n = len(actuals)
        if feeders is None:
            feeders = [1] * n
        return pl.DataFrame({
            "feeder": feeders,
            "timestamp_UTC": [
                datetime(2024, 1, 1, i, tzinfo=timezone.utc) for i in range(n)
            ],
            "target": actuals,
            "q05": q05s,
            "q50": q50s,
            "q95": q95s,
        })

    def test_perfect_forecast(self):
        df = self._make_eval_df(
            actuals=[10.0, 20.0, 30.0],
            q05s=[10.0, 20.0, 30.0],
            q50s=[10.0, 20.0, 30.0],
            q95s=[10.0, 20.0, 30.0],
        )
        metrics = compute_metrics(df, "target")
        row = metrics.filter(pl.col("feeder") == "1")
        assert row["mae"][0] == pytest.approx(0.0)
        assert row["pinball_q50"][0] == pytest.approx(0.0)
        assert row["winkler_score"][0] == pytest.approx(0.0)

    def test_mae_computation(self):
        df = self._make_eval_df(
            actuals=[10.0, 20.0],
            q05s=[5.0, 15.0],
            q50s=[12.0, 18.0],
            q95s=[15.0, 25.0],
        )
        metrics = compute_metrics(df, "target")
        row = metrics.filter(pl.col("feeder") == "1")
        expected_mae = (abs(10 - 12) + abs(20 - 18)) / 2
        assert row["mae"][0] == pytest.approx(expected_mae)

    def test_pinball_q05_actual_above(self):
        """When actual >= q05, pinball = alpha * (actual - q05)."""
        df = self._make_eval_df(
            actuals=[20.0],
            q05s=[10.0],
            q50s=[15.0],
            q95s=[25.0],
        )
        metrics = compute_metrics(df, "target")
        row = metrics.filter(pl.col("feeder") == "1")
        assert row["pinball_q05"][0] == pytest.approx(0.05 * (20.0 - 10.0))

    def test_pinball_q05_actual_below(self):
        """When actual < q05, pinball = (1 - alpha) * (q05 - actual)."""
        df = self._make_eval_df(
            actuals=[5.0],
            q05s=[10.0],
            q50s=[15.0],
            q95s=[25.0],
        )
        metrics = compute_metrics(df, "target")
        row = metrics.filter(pl.col("feeder") == "1")
        assert row["pinball_q05"][0] == pytest.approx(0.95 * (10.0 - 5.0))

    def test_pinball_q95_actual_above(self):
        """When actual >= q95, pinball = alpha * (actual - q95)."""
        df = self._make_eval_df(
            actuals=[30.0],
            q05s=[10.0],
            q50s=[15.0],
            q95s=[25.0],
        )
        metrics = compute_metrics(df, "target")
        row = metrics.filter(pl.col("feeder") == "1")
        assert row["pinball_q95"][0] == pytest.approx(0.95 * (30.0 - 25.0))

    def test_pinball_q95_actual_below(self):
        """When actual < q95, pinball = (1 - alpha) * (q95 - actual)."""
        df = self._make_eval_df(
            actuals=[20.0],
            q05s=[10.0],
            q50s=[15.0],
            q95s=[25.0],
        )
        metrics = compute_metrics(df, "target")
        row = metrics.filter(pl.col("feeder") == "1")
        assert row["pinball_q95"][0] == pytest.approx(0.05 * (25.0 - 20.0))

    def test_winkler_score_inside_interval(self):
        """When actual is inside [q05, q95], winkler = q95 - q05."""
        df = self._make_eval_df(
            actuals=[15.0],
            q05s=[10.0],
            q50s=[15.0],
            q95s=[20.0],
        )
        metrics = compute_metrics(df, "target")
        row = metrics.filter(pl.col("feeder") == "1")
        assert row["winkler_score"][0] == pytest.approx(10.0)

    def test_winkler_score_below_interval(self):
        """When actual < q05, winkler = (q95-q05) + 20*(q05-actual)."""
        df = self._make_eval_df(
            actuals=[5.0],
            q05s=[10.0],
            q50s=[15.0],
            q95s=[20.0],
        )
        metrics = compute_metrics(df, "target")
        row = metrics.filter(pl.col("feeder") == "1")
        expected = (20.0 - 10.0) + 20.0 * (10.0 - 5.0)
        assert row["winkler_score"][0] == pytest.approx(expected)

    def test_winkler_score_above_interval(self):
        """When actual > q95, winkler = (q95-q05) + 20*(actual-q95)."""
        df = self._make_eval_df(
            actuals=[25.0],
            q05s=[10.0],
            q50s=[15.0],
            q95s=[20.0],
        )
        metrics = compute_metrics(df, "target")
        row = metrics.filter(pl.col("feeder") == "1")
        expected = (20.0 - 10.0) + 20.0 * (25.0 - 20.0)
        assert row["winkler_score"][0] == pytest.approx(expected)

    def test_average_row_present(self):
        df = self._make_eval_df(
            actuals=[10.0, 20.0],
            q05s=[8.0, 18.0],
            q50s=[10.0, 20.0],
            q95s=[12.0, 22.0],
            feeders=[1, 2],
        )
        metrics = compute_metrics(df, "target")
        avg = metrics.filter(pl.col("feeder") == "average")
        assert len(avg) == 1

    def test_multiple_feeders_averaged(self):
        df = self._make_eval_df(
            actuals=[10.0, 20.0],
            q05s=[10.0, 20.0],
            q50s=[10.0, 20.0],
            q95s=[10.0, 20.0],
            feeders=[1, 2],
        )
        metrics = compute_metrics(df, "target")
        avg = metrics.filter(pl.col("feeder") == "average")
        assert avg["mae"][0] == pytest.approx(0.0)

    def test_null_quantiles_filtered(self):
        df = pl.DataFrame({
            "feeder": [1, 1],
            "target": [10.0, 20.0],
            "q05": [8.0, None],
            "q50": [10.0, None],
            "q95": [12.0, None],
        })
        metrics = compute_metrics(df, "target")
        row = metrics.filter(pl.col("feeder") == "1")
        assert row["mae"][0] == pytest.approx(0.0)
