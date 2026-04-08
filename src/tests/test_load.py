from datetime import datetime, timezone
from pathlib import Path

import pytest

from src.load.load import (
    DATA_DIR,
    DATASET_END,
    DATASET_START,
    MEASUREMENT_COLUMNS,
    MEASUREMENT_RANGE_FOLDERS,
    METADATA_COLUMNS,
    WEATHER_COLUMNS,
    _classify_columns,
    _feeder_id_to_filepath,
    _validate_params,
    load_data,
)

HAS_MEASUREMENT_DATA = (DATA_DIR / "feeder_measurement_data_001-040" / "feeder_001.parquet").exists()
HAS_WEATHER_DATA = (DATA_DIR / "weather_data.parquet").exists()
HAS_METADATA = (DATA_DIR / "feeder_metadata.csv").exists()
HAS_ALL_DATA = HAS_MEASUREMENT_DATA and HAS_WEATHER_DATA and HAS_METADATA


# --- Validation tests (no data needed) ---

class TestValidation:
    def test_validate_invalid_feeders_type(self):
        with pytest.raises(ValueError, match="list of integers"):
            _validate_params("not a list", None, None, None, None, None)

    def test_validate_invalid_feeders_range(self):
        with pytest.raises(ValueError, match="Invalid feeder IDs"):
            _validate_params([0, 201], None, None, None, None, None)

    def test_validate_invalid_columns(self):
        with pytest.raises(ValueError, match="Invalid column names"):
            _validate_params(None, ["nonexistent_col"], None, None, None, None)

    def test_validate_empty_columns(self):
        with pytest.raises(ValueError, match="non-empty list"):
            _validate_params(None, [], None, None, None, None)

    def test_validate_invalid_time_period_not_tuple(self):
        with pytest.raises(ValueError, match="tuple"):
            _validate_params(None, None, "2024-01-01", None, None, None)

    def test_validate_invalid_time_period_start_after_end(self):
        with pytest.raises(ValueError, match="start must be before end"):
            _validate_params(None, None, ("2024-06-01", "2024-01-01"), None, None, None)

    def test_validate_time_period_out_of_range(self):
        with pytest.raises(ValueError, match="out of dataset range"):
            _validate_params(None, None, ("2020-01-01", "2020-02-01"), None, None, None)

    def test_validate_invalid_resolution(self):
        with pytest.raises(ValueError, match="Invalid temporal_resolution"):
            _validate_params(None, None, None, "2min", None, None)

    def test_validate_aggregation_without_resolution(self):
        with pytest.raises(ValueError, match="requires temporal_resolution"):
            _validate_params(None, None, None, None, {"active_power_kW": "mean"}, None)

    def test_validate_aggregation_invalid_column(self):
        with pytest.raises(ValueError, match="not in requested columns"):
            _validate_params(
                None, ["active_power_kW"], None, "1h",
                {"voltage_one_phase_V": "mean"}, None,
            )

    def test_validate_aggregation_invalid_method(self):
        with pytest.raises(ValueError, match="Invalid aggregation"):
            _validate_params(None, None, None, "1h", {"active_power_kW": "median"}, None)

    def test_validate_interpolation_without_resolution(self):
        with pytest.raises(ValueError, match="requires temporal_resolution"):
            _validate_params(None, None, None, None, None, {"air_temperature_C": "linear"})

    def test_validate_valid_params(self):
        _validate_params([1, 2], ["active_power_kW"], ("2024-01-01", "2024-06-01"), "1h", {"active_power_kW": "mean"}, None)


# --- Unit tests (no data needed) ---

class TestClassifyColumns:
    def test_classify_columns_all_none(self):
        result = _classify_columns(None)
        assert "measurement" in result
        assert "weather" in result
        assert "metadata" in result

    def test_classify_columns_measurement_only(self):
        result = _classify_columns(["active_power_kW", "voltage_one_phase_V"])
        assert "measurement" in result
        assert "weather" not in result
        assert "metadata" not in result
        assert "feeder" in result["measurement"]
        assert "timestamp_UTC" in result["measurement"]

    def test_classify_columns_weather_only(self):
        result = _classify_columns(["air_temperature_C"])
        assert "weather" in result
        assert "measurement" not in result
        assert "feeder" in result["weather"]
        assert "timestamp_UTC" in result["weather"]

    def test_classify_columns_metadata_only(self):
        result = _classify_columns(["housing_units_count"])
        assert "metadata" in result
        assert "measurement" not in result
        assert "feeder" in result["metadata"]
        assert "date" in result["metadata"]

    def test_classify_columns_mixed(self):
        result = _classify_columns(["active_power_kW", "air_temperature_C", "housing_units_count"])
        assert "measurement" in result
        assert "weather" in result
        assert "metadata" in result

    def test_classify_columns_feeder_only(self):
        result = _classify_columns(["feeder"])
        assert "measurement" in result


class TestFeederIdToFilepath:
    def test_feeder_1(self):
        path = _feeder_id_to_filepath(1)
        assert path == DATA_DIR / "feeder_measurement_data_001-040" / "feeder_001.parquet"

    def test_feeder_40(self):
        path = _feeder_id_to_filepath(40)
        assert path == DATA_DIR / "feeder_measurement_data_001-040" / "feeder_040.parquet"

    def test_feeder_41(self):
        path = _feeder_id_to_filepath(41)
        assert path == DATA_DIR / "feeder_measurement_data_041-080" / "feeder_041.parquet"

    def test_feeder_200(self):
        path = _feeder_id_to_filepath(200)
        assert path == DATA_DIR / "feeder_measurement_data_161-200" / "feeder_200.parquet"

    def test_feeder_100(self):
        path = _feeder_id_to_filepath(100)
        assert path == DATA_DIR / "feeder_measurement_data_081-120" / "feeder_100.parquet"


# --- Integration tests (skip if no data) ---

class TestIntegration:
    @pytest.mark.skipif(not HAS_MEASUREMENT_DATA, reason="Measurement data not available")
    def test_load_single_feeder(self):
        df = load_data(
            feeders=[1],
            columns=["active_power_kW"],
            time_period=("2024-01-01", "2024-01-02"),
        )
        assert "feeder" in df.columns
        assert "active_power_kW" in df.columns
        assert df["feeder"].unique().to_list() == [1]
        assert len(df) > 0

    @pytest.mark.skipif(not HAS_MEASUREMENT_DATA, reason="Measurement data not available")
    def test_load_with_time_period(self):
        df = load_data(
            feeders=[1],
            columns=["active_power_kW"],
            time_period=("2024-06-01", "2024-06-02"),
        )
        assert len(df) > 0
        assert df["timestamp_UTC"].min() >= datetime(2024, 6, 1, tzinfo=timezone.utc)
        assert df["timestamp_UTC"].max() <= datetime(2024, 6, 2, tzinfo=timezone.utc)

    @pytest.mark.skipif(not HAS_WEATHER_DATA, reason="Weather data not available")
    def test_load_weather_only(self):
        df = load_data(
            feeders=[1],
            columns=["air_temperature_C"],
            time_period=("2024-01-01", "2024-01-02"),
        )
        assert "air_temperature_C" in df.columns
        assert "feeder" in df.columns
        assert len(df) > 0

    @pytest.mark.skipif(not HAS_METADATA, reason="Metadata not available")
    def test_load_metadata_only(self):
        df = load_data(
            feeders=[1],
            columns=["housing_units_count"],
        )
        assert "housing_units_count" in df.columns
        assert "feeder" in df.columns
        assert len(df) > 0

    @pytest.mark.skipif(not (HAS_MEASUREMENT_DATA and HAS_WEATHER_DATA), reason="Data not available")
    def test_join_measurement_weather(self):
        df = load_data(
            feeders=[1],
            columns=["active_power_kW", "air_temperature_C"],
            time_period=("2024-01-01", "2024-01-02"),
        )
        assert "active_power_kW" in df.columns
        assert "air_temperature_C" in df.columns
        assert len(df) > 0

    @pytest.mark.skipif(not HAS_ALL_DATA, reason="Not all data available")
    def test_join_with_metadata(self):
        df = load_data(
            feeders=[1],
            columns=["active_power_kW", "housing_units_count"],
            time_period=("2024-01-01", "2024-01-02"),
        )
        assert "active_power_kW" in df.columns
        assert "housing_units_count" in df.columns
        assert len(df) > 0

# --- Tests for temporal_resolution / aggregation / interpolation ---

class TestAggregationInterpolationUnit:
    """Unit tests for _aggregate, _interpolate, _needs_aggregation, _needs_interpolation."""

    def test_needs_aggregation_true(self):
        from src.load.load import _needs_aggregation
        assert _needs_aggregation("1min", "1h") is True
        assert _needs_aggregation("1min", "15min") is True
        assert _needs_aggregation("1h", "1d") is True

    def test_needs_aggregation_false_same_resolution(self):
        from src.load.load import _needs_aggregation
        assert _needs_aggregation("1h", "1h") is False

    def test_needs_aggregation_false_finer(self):
        from src.load.load import _needs_aggregation
        assert _needs_aggregation("1h", "5min") is False

    def test_needs_interpolation_true(self):
        from src.load.load import _needs_interpolation
        assert _needs_interpolation("1h", "15min") is True
        assert _needs_interpolation("1h", "5min") is True
        assert _needs_interpolation("1h", "1min") is True

    def test_needs_interpolation_false_same_resolution(self):
        from src.load.load import _needs_interpolation
        assert _needs_interpolation("1h", "1h") is False

    def test_needs_interpolation_false_coarser(self):
        from src.load.load import _needs_interpolation
        assert _needs_interpolation("1min", "1h") is False

    def test_aggregate_mean(self):
        import polars as pl
        from src.load.load import _aggregate

        df = pl.DataFrame({
            "feeder": [1] * 60,
            "timestamp_UTC": pl.datetime_range(
                datetime(2024, 1, 1, tzinfo=timezone.utc),
                datetime(2024, 1, 1, 0, 59, tzinfo=timezone.utc),
                interval="1m",
                eager=True,
            ),
            "active_power_kW": list(range(60)),
        })
        result = _aggregate(df, "1h", {})
        assert len(result) == 1
        assert result["active_power_kW"][0] == pytest.approx(sum(range(60)) / 60)

    def test_aggregate_sum(self):
        import polars as pl
        from src.load.load import _aggregate

        df = pl.DataFrame({
            "feeder": [1] * 60,
            "timestamp_UTC": pl.datetime_range(
                datetime(2024, 1, 1, tzinfo=timezone.utc),
                datetime(2024, 1, 1, 0, 59, tzinfo=timezone.utc),
                interval="1m",
                eager=True,
            ),
            "active_power_kW": [2.0] * 60,
        })
        result = _aggregate(df, "1h", {"active_power_kW": "sum"})
        assert result["active_power_kW"][0] == pytest.approx(120.0)

    def test_aggregate_min_max(self):
        import polars as pl
        from src.load.load import _aggregate

        df = pl.DataFrame({
            "feeder": [1] * 60,
            "timestamp_UTC": pl.datetime_range(
                datetime(2024, 1, 1, tzinfo=timezone.utc),
                datetime(2024, 1, 1, 0, 59, tzinfo=timezone.utc),
                interval="1m",
                eager=True,
            ),
            "val_a": list(range(60)),
            "val_b": list(range(60)),
        })
        result = _aggregate(df, "1h", {"val_a": "min", "val_b": "max"})
        assert result["val_a"][0] == 0
        assert result["val_b"][0] == 59

    def test_aggregate_to_15min(self):
        import polars as pl
        from src.load.load import _aggregate

        df = pl.DataFrame({
            "feeder": [1] * 60,
            "timestamp_UTC": pl.datetime_range(
                datetime(2024, 1, 1, tzinfo=timezone.utc),
                datetime(2024, 1, 1, 0, 59, tzinfo=timezone.utc),
                interval="1m",
                eager=True,
            ),
            "active_power_kW": [1.0] * 60,
        })
        result = _aggregate(df, "15min", {})
        assert len(result) == 4
        assert all(v == pytest.approx(1.0) for v in result["active_power_kW"].to_list())

    def test_aggregate_multiple_feeders(self):
        import polars as pl
        from src.load.load import _aggregate

        ts = pl.datetime_range(
            datetime(2024, 1, 1, tzinfo=timezone.utc),
            datetime(2024, 1, 1, 0, 59, tzinfo=timezone.utc),
            interval="1m",
            eager=True,
        )
        df = pl.concat([
            pl.DataFrame({"feeder": [1] * 60, "timestamp_UTC": ts, "val": [10.0] * 60}),
            pl.DataFrame({"feeder": [2] * 60, "timestamp_UTC": ts, "val": [20.0] * 60}),
        ])
        result = _aggregate(df, "1h", {"val": "mean"})
        assert len(result) == 2
        f1 = result.filter(pl.col("feeder") == 1)["val"][0]
        f2 = result.filter(pl.col("feeder") == 2)["val"][0]
        assert f1 == pytest.approx(10.0)
        assert f2 == pytest.approx(20.0)

    def test_aggregate_drops_timestamp_cet_cest(self):
        import polars as pl
        from src.load.load import _aggregate

        df = pl.DataFrame({
            "feeder": [1] * 10,
            "timestamp_UTC": pl.datetime_range(
                datetime(2024, 1, 1, tzinfo=timezone.utc),
                datetime(2024, 1, 1, 0, 9, tzinfo=timezone.utc),
                interval="1m",
                eager=True,
            ),
            "timestamp_CET_CEST": ["dummy"] * 10,
            "active_power_kW": [1.0] * 10,
        })
        result = _aggregate(df, "15min", {})
        assert "timestamp_CET_CEST" not in result.columns

    def test_interpolate_linear(self):
        import polars as pl
        from src.load.load import _interpolate

        df = pl.DataFrame({
            "feeder": [1] * 3,
            "timestamp_UTC": [
                datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc),
                datetime(2024, 1, 1, 1, 0, tzinfo=timezone.utc),
                datetime(2024, 1, 1, 2, 0, tzinfo=timezone.utc),
            ],
            "air_temperature_C": [0.0, 6.0, 12.0],
        })
        result = _interpolate(df, "30min", {})
        assert len(result) == 5
        assert result["air_temperature_C"][1] == pytest.approx(3.0)
        assert result["air_temperature_C"][3] == pytest.approx(9.0)

    def test_interpolate_forward(self):
        import polars as pl
        from src.load.load import _interpolate

        df = pl.DataFrame({
            "feeder": [1] * 2,
            "timestamp_UTC": [
                datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc),
                datetime(2024, 1, 1, 1, 0, tzinfo=timezone.utc),
            ],
            "air_temperature_C": [10.0, 20.0],
        })
        result = _interpolate(df, "30min", {"air_temperature_C": "forward"})
        assert len(result) == 3
        assert result["air_temperature_C"][1] == pytest.approx(10.0)

    def test_interpolate_backward(self):
        import polars as pl
        from src.load.load import _interpolate

        df = pl.DataFrame({
            "feeder": [1] * 2,
            "timestamp_UTC": [
                datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc),
                datetime(2024, 1, 1, 1, 0, tzinfo=timezone.utc),
            ],
            "air_temperature_C": [10.0, 20.0],
        })
        result = _interpolate(df, "30min", {"air_temperature_C": "backward"})
        assert len(result) == 3
        assert result["air_temperature_C"][1] == pytest.approx(20.0)

    def test_interpolate_to_15min(self):
        import polars as pl
        from src.load.load import _interpolate

        df = pl.DataFrame({
            "feeder": [1] * 2,
            "timestamp_UTC": [
                datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc),
                datetime(2024, 1, 1, 1, 0, tzinfo=timezone.utc),
            ],
            "val": [0.0, 100.0],
        })
        result = _interpolate(df, "15min", {"val": "linear"})
        assert len(result) == 5
        assert result["val"][0] == pytest.approx(0.0)
        assert result["val"][1] == pytest.approx(25.0)
        assert result["val"][2] == pytest.approx(50.0)
        assert result["val"][3] == pytest.approx(75.0)
        assert result["val"][4] == pytest.approx(100.0)

    def test_interpolate_multiple_feeders(self):
        import polars as pl
        from src.load.load import _interpolate

        df = pl.concat([
            pl.DataFrame({
                "feeder": [1, 1],
                "timestamp_UTC": [
                    datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc),
                    datetime(2024, 1, 1, 1, 0, tzinfo=timezone.utc),
                ],
                "val": [0.0, 10.0],
            }),
            pl.DataFrame({
                "feeder": [2, 2],
                "timestamp_UTC": [
                    datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc),
                    datetime(2024, 1, 1, 1, 0, tzinfo=timezone.utc),
                ],
                "val": [100.0, 200.0],
            }),
        ])
        result = _interpolate(df, "30min", {})
        f1 = result.filter(pl.col("feeder") == 1).sort("timestamp_UTC")
        f2 = result.filter(pl.col("feeder") == 2).sort("timestamp_UTC")
        assert len(f1) == 3
        assert len(f2) == 3
        assert f1["val"][1] == pytest.approx(5.0)
        assert f2["val"][1] == pytest.approx(150.0)

    def test_interpolate_drops_timestamp_cet_cest(self):
        import polars as pl
        from src.load.load import _interpolate

        df = pl.DataFrame({
            "feeder": [1, 1],
            "timestamp_UTC": [
                datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc),
                datetime(2024, 1, 1, 1, 0, tzinfo=timezone.utc),
            ],
            "timestamp_CET_CEST": ["a", "b"],
            "val": [0.0, 10.0],
        })
        result = _interpolate(df, "30min", {})
        assert "timestamp_CET_CEST" not in result.columns

    def test_interpolate_mixed_methods_per_column(self):
        import polars as pl
        from src.load.load import _interpolate

        df = pl.DataFrame({
            "feeder": [1, 1],
            "timestamp_UTC": [
                datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc),
                datetime(2024, 1, 1, 1, 0, tzinfo=timezone.utc),
            ],
            "col_lin": [0.0, 10.0],
            "col_fwd": [0.0, 10.0],
        })
        result = _interpolate(df, "30min", {"col_lin": "linear", "col_fwd": "forward"})
        assert result["col_lin"][1] == pytest.approx(5.0)
        assert result["col_fwd"][1] == pytest.approx(0.0)


class TestAggregationInterpolationIntegration:
    """Integration tests that load real data with various temporal_resolution combos."""

    @pytest.mark.skipif(not HAS_MEASUREMENT_DATA, reason="Measurement data not available")
    def test_measurement_aggregation_to_15min(self):
        df = load_data(
            feeders=[1],
            columns=["active_power_kW"],
            time_period=("2024-01-01", "2024-01-01T01:00:00"),
            temporal_resolution="15min",
        )
        timestamps = df["timestamp_UTC"].unique().sort()
        diffs = timestamps.diff().drop_nulls()
        assert all(d.total_seconds() == 900 for d in diffs.to_list())

    @pytest.mark.skipif(not HAS_MEASUREMENT_DATA, reason="Measurement data not available")
    def test_measurement_aggregation_to_1h_mean(self):
        df = load_data(
            feeders=[1],
            columns=["active_power_kW"],
            time_period=("2024-01-01", "2024-01-01T03:00:00"),
            temporal_resolution="1h",
            aggregation_type={"active_power_kW": "mean"},
        )
        timestamps = df["timestamp_UTC"].unique().sort()
        diffs = timestamps.diff().drop_nulls()
        assert all(d.total_seconds() == 3600 for d in diffs.to_list())

    @pytest.mark.skipif(not HAS_MEASUREMENT_DATA, reason="Measurement data not available")
    def test_measurement_aggregation_to_1h_sum(self):
        df_sum = load_data(
            feeders=[1],
            columns=["active_power_kW"],
            time_period=("2024-01-01", "2024-01-01T01:00:00"),
            temporal_resolution="1h",
            aggregation_type={"active_power_kW": "sum"},
        )
        df_mean = load_data(
            feeders=[1],
            columns=["active_power_kW"],
            time_period=("2024-01-01", "2024-01-01T01:00:00"),
            temporal_resolution="1h",
            aggregation_type={"active_power_kW": "mean"},
        )
        assert df_sum["active_power_kW"][0] != df_mean["active_power_kW"][0]

    @pytest.mark.skipif(not HAS_MEASUREMENT_DATA, reason="Measurement data not available")
    def test_measurement_aggregation_default_is_mean(self):
        df_default = load_data(
            feeders=[1],
            columns=["active_power_kW"],
            time_period=("2024-01-01", "2024-01-01T01:00:00"),
            temporal_resolution="1h",
        )
        df_mean = load_data(
            feeders=[1],
            columns=["active_power_kW"],
            time_period=("2024-01-01", "2024-01-01T01:00:00"),
            temporal_resolution="1h",
            aggregation_type={"active_power_kW": "mean"},
        )
        assert df_default["active_power_kW"][0] == pytest.approx(df_mean["active_power_kW"][0])

    @pytest.mark.skipif(not HAS_MEASUREMENT_DATA, reason="Measurement data not available")
    def test_measurement_aggregation_to_1d(self):
        df = load_data(
            feeders=[1],
            columns=["active_power_kW"],
            time_period=("2024-01-01", "2024-01-03"),
            temporal_resolution="1d",
        )
        assert len(df) == 3

    @pytest.mark.skipif(not HAS_WEATHER_DATA, reason="Weather data not available")
    def test_weather_aggregation_to_1d(self):
        df = load_data(
            feeders=[1],
            columns=["air_temperature_C"],
            time_period=("2024-01-01", "2024-01-03"),
            temporal_resolution="1d",
        )
        assert len(df) == 3
        assert "air_temperature_C" in df.columns

    @pytest.mark.skipif(not HAS_WEATHER_DATA, reason="Weather data not available")
    def test_weather_interpolation_to_15min(self):
        df = load_data(
            feeders=[1],
            columns=["air_temperature_C"],
            time_period=("2024-01-01", "2024-01-01T03:00:00"),
            temporal_resolution="15min",
        )
        timestamps = df["timestamp_UTC"].unique().sort()
        diffs = timestamps.diff().drop_nulls()
        assert all(d.total_seconds() == 900 for d in diffs.to_list())

    @pytest.mark.skipif(not HAS_WEATHER_DATA, reason="Weather data not available")
    def test_weather_interpolation_forward(self):
        df = load_data(
            feeders=[1],
            columns=["air_temperature_C"],
            time_period=("2024-01-01", "2024-01-01T02:00:00"),
            temporal_resolution="30min",
            interpolation_type={"air_temperature_C": "forward"},
        )
        timestamps = df["timestamp_UTC"].unique().sort()
        diffs = timestamps.diff().drop_nulls()
        assert all(d.total_seconds() == 1800 for d in diffs.to_list())

    @pytest.mark.skipif(not (HAS_MEASUREMENT_DATA and HAS_WEATHER_DATA), reason="Data not available")
    def test_combined_measurement_agg_weather_at_native(self):
        """Aggregate measurements to 1h; weather is already at 1h so no change."""
        df = load_data(
            feeders=[1],
            columns=["active_power_kW", "air_temperature_C"],
            time_period=("2024-01-01", "2024-01-01T03:00:00"),
            temporal_resolution="1h",
        )
        assert "active_power_kW" in df.columns
        assert "air_temperature_C" in df.columns
        assert len(df) > 0

    @pytest.mark.skipif(not (HAS_MEASUREMENT_DATA and HAS_WEATHER_DATA), reason="Data not available")
    def test_combined_measurement_and_weather_aggregated_to_1d(self):
        df = load_data(
            feeders=[1],
            columns=["active_power_kW", "air_temperature_C"],
            time_period=("2024-01-01", "2024-01-03"),
            temporal_resolution="1d",
            aggregation_type={"active_power_kW": "sum", "air_temperature_C": "mean"},
        )
        assert len(df) == 3
        assert "active_power_kW" in df.columns
        assert "air_temperature_C" in df.columns

    @pytest.mark.skipif(not HAS_MEASUREMENT_DATA, reason="Measurement data not available")
    def test_measurement_at_native_1min_resolution(self):
        """Requesting 1min resolution should be a no-op for measurements (already 1min)."""
        df = load_data(
            feeders=[1],
            columns=["active_power_kW"],
            time_period=("2024-01-01", "2024-01-01T00:05:00"),
            temporal_resolution="1min",
        )
        assert len(df) == 6  # 00:00, 00:01, 00:02, 00:03, 00:04, 00:05
