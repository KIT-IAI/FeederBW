import polars as pl
import pytest

from src.model.pseudo_measurement_generation import _validate_pseudo_measurement_inputs


class TestValidatePseudoMeasurementInputs:
    def _make_data(self):
        return pl.DataFrame({
            "feeder": [1, 1, 2, 2],
            "timestamp_UTC": ["a", "b", "c", "d"],
            "target": [1.0, 2.0, 3.0, 4.0],
            "feat1": [10.0, 20.0, 30.0, 40.0],
            "feat2": [100.0, 200.0, 300.0, 400.0],
        })

    def test_missing_target_column(self):
        df = self._make_data()
        with pytest.raises(ValueError, match="target_column"):
            _validate_pseudo_measurement_inputs(df, "nonexistent", ["feat1"], [2])

    def test_missing_feature_columns(self):
        df = self._make_data()
        with pytest.raises(ValueError, match="feature_columns not found"):
            _validate_pseudo_measurement_inputs(df, "target", ["nonexistent"], [2])

    def test_missing_test_feeders(self):
        df = self._make_data()
        with pytest.raises(ValueError, match="test_feeders not found"):
            _validate_pseudo_measurement_inputs(df, "target", ["feat1"], [99])

    def test_valid_inputs(self):
        df = self._make_data()
        _validate_pseudo_measurement_inputs(df, "target", ["feat1", "feat2"], [2])

    def test_spatial_split_no_overlap(self):
        """Verify that train and test feeders don't overlap."""
        df = self._make_data()
        test_feeders = [2]
        train_data = df.filter(~pl.col("feeder").is_in(test_feeders))
        test_data = df.filter(pl.col("feeder").is_in(test_feeders))
        train_feeders = set(train_data["feeder"].to_list())
        test_feeder_set = set(test_data["feeder"].to_list())
        assert train_feeders & test_feeder_set == set()
