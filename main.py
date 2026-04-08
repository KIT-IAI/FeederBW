"""
Basic example: download, load, forecast, and pseudo-measurement generation.

Run from the project root:
    uv run python main.py
"""

from src.download.download import download_data
from src.load.load import load_data
from src.model.forecast import forecast
from src.model.pseudo_measurement_generation import pseudo_measurement_generation


def main() -> None:
    # --- 1. Download a small subset of the dataset ---
    print("Downloading feeder metadata, weather data, and feeders 001-040...")
    download_data(files=["feeder_metadata", "weather_data", "001-040"])

    # --- 2. Load measurement + weather data for a few feeders and a short period ---
    print("\nLoading data...")
    data = load_data(
        feeders=list(range(1, 41)),
        columns=[
            "feeder", "timestamp_UTC",
            "active_power_kW",
            "housing_units_count", "PV_systems_kW", "air_temperature_C"
        ],
        time_period=("2024-01-01T00:00:00", "2024-04-01T00:00:00"),
        temporal_resolution="1h",
        aggregation_type={"active_power_kW": "mean"},
    )
    print(f"Loaded {len(data)} rows, {data.width} columns for feeders 1-40.")
    print(data.head(5))

    # --- 3. Naive quantile forecast baseline ---
    print("\nRunning forecast baseline (split date: 2024-03-01)...")
    forecast_metrics = forecast(
        data=data,
        split_date="2024-03-01",
        target_column="active_power_kW",
    )
    print("Forecast metrics (per feeder + average):")
    print(forecast_metrics)
    print("Results saved to data/forecast_metrics.csv and data/forecast_figure.png")

    # --- 4. Pseudo-measurement generation across feeders ---
    print("\nRunning pseudo-measurement generation (test feeders: [36, 37, 38, 39, 40])...")
    pseudo_metrics = pseudo_measurement_generation(
        data=data,
        target_column="active_power_kW",
        feature_columns=["housing_units_count", "PV_systems_kW", "air_temperature_C"],
        test_feeders=[36, 37, 38, 39, 40],
    )
    print("Pseudo-measurement metrics (per feeder + average):")
    print(pseudo_metrics)
    print("Results saved to data/pseudo_measurement_metrics.csv and data/pseudo_measurement_figure.png")


if __name__ == "__main__":
    main()
