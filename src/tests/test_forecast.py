from datetime import date, datetime, timezone

import polars as pl
import pytest

from src.model.forecast import _parse_split_date, _validate_forecast_inputs, _compute_daily_forecast


class TestParseSplitDate:
    def test_string_input(self):
        result = _parse_split_date("2024-03-01")
        assert result == date(2024, 3, 1)

    def test_date_input(self):
        result = _parse_split_date(date(2024, 3, 1))
        assert result == date(2024, 3, 1)


class TestValidateForecastInputs:
    def _make_data(self, dates):
        return pl.DataFrame({
            "feeder": [1] * len(dates),
            "timestamp_UTC": [
                datetime(d.year, d.month, d.day, tzinfo=timezone.utc) for d in dates
            ],
            "target": [1.0] * len(dates),
        })

    def test_missing_columns(self):
        df = pl.DataFrame({"feeder": [1], "timestamp_UTC": [datetime(2024, 1, 1, tzinfo=timezone.utc)]})
        with pytest.raises(ValueError, match="Missing columns"):
            _validate_forecast_inputs(df, date(2024, 1, 1), "target")

    def test_empty_train(self):
        df = self._make_data([date(2024, 3, 1), date(2024, 3, 2)])
        with pytest.raises(ValueError, match="Training set is empty"):
            _validate_forecast_inputs(df, date(2024, 3, 1), "target")

    def test_empty_test(self):
        df = self._make_data([date(2024, 1, 1), date(2024, 1, 2)])
        with pytest.raises(ValueError, match="Test set is empty"):
            _validate_forecast_inputs(df, date(2025, 1, 1), "target")

    def test_valid_split(self):
        df = self._make_data([date(2024, 1, 1), date(2024, 3, 1)])
        _validate_forecast_inputs(df, date(2024, 2, 1), "target")


class TestComputeDailyForecast:
    def test_returns_quantile_columns(self):
        history = pl.DataFrame({
            "feeder": [1] * 24,
            "timestamp_UTC": [
                datetime(2024, 1, 1, h, tzinfo=timezone.utc) for h in range(24)
            ],
            "target": list(range(24)),
        })
        test_day_data = pl.DataFrame({
            "feeder": [1] * 24,
            "timestamp_UTC": [
                datetime(2024, 1, 2, h, tzinfo=timezone.utc) for h in range(24)
            ],
            "target": list(range(24)),
        })
        result = _compute_daily_forecast(history, test_day_data, "target")
        assert "q05" in result.columns
        assert "q50" in result.columns
        assert "q95" in result.columns
        assert len(result) == 24

    def test_no_data_leakage(self):
        """Verify test day data is NOT in the history used for quantile computation."""
        history = pl.DataFrame({
            "feeder": [1, 1],
            "timestamp_UTC": [
                datetime(2024, 1, 1, 0, tzinfo=timezone.utc),
                datetime(2024, 1, 1, 1, tzinfo=timezone.utc),
            ],
            "target": [100.0, 200.0],
        })
        test_day_data = pl.DataFrame({
            "feeder": [1, 1],
            "timestamp_UTC": [
                datetime(2024, 1, 2, 0, tzinfo=timezone.utc),
                datetime(2024, 1, 2, 1, tzinfo=timezone.utc),
            ],
            "target": [999.0, 999.0],
        })
        result = _compute_daily_forecast(history, test_day_data, "target")
        assert result["q50"][0] == pytest.approx(100.0)
        assert result["q50"][1] == pytest.approx(200.0)
