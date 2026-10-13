import random
from datetime import date, datetime, timedelta
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import polars as pl

from src.model._metrics import compute_metrics


def forecast(
    data: pl.DataFrame,
    split_date: str | date,
    target_column: str,
) -> pl.DataFrame:
    """Naive quantile forecast baseline using historical same-time-of-day quantiles.

    For each test day (from split_date onwards), computes 5th, 50th, and 95th
    percentile forecasts from all historically available values at the same time
    of day. History grows by one day after each forecast.

    Args:
        data: DataFrame with columns 'feeder', 'timestamp_UTC', and target_column.
        split_date: Date (inclusive) at which the test set starts.
        target_column: Column name to forecast.

    Returns:
        DataFrame with per-feeder MAE, pinball losses, and winkler score,
        plus an 'average' row across all feeders.

    Raises:
        ValueError: If required columns are missing, or train/test set is empty.
    """
    split_date = _parse_split_date(split_date)
    _validate_forecast_inputs(data, split_date, target_column)

    train = data.filter(pl.col("timestamp_UTC").cast(pl.Date) < split_date)
    test = data.filter(pl.col("timestamp_UTC").cast(pl.Date) >= split_date)

    test_days = (
        test.select(pl.col("timestamp_UTC").cast(pl.Date).alias("date"))
        .unique()
        .sort("date")["date"]
        .to_list()
    )

    history = train
    daily_results = []

    for test_day in test_days:
        test_day_data = test.filter(pl.col("timestamp_UTC").cast(pl.Date) == test_day)
        day_eval = _compute_daily_forecast(history, test_day_data, target_column)
        daily_results.append(day_eval)
        history = pl.concat([history, test_day_data])

    eval_df = pl.concat(daily_results)
    metrics = compute_metrics(eval_df, target_column)

    _save_metrics(metrics, Path("data/forecast_metrics.csv"))
    _plot_forecast(data, eval_df, target_column, test_days, Path("data/forecast_figure.png"))

    return metrics


def _parse_split_date(split_date: str | date) -> date:
    if isinstance(split_date, date):
        return split_date
    return date.fromisoformat(split_date)


def _validate_forecast_inputs(data: pl.DataFrame, split_date: date, target_column: str) -> None:
    required = {"feeder", "timestamp_UTC", target_column}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Missing columns in data: {missing}")

    train = data.filter(pl.col("timestamp_UTC").cast(pl.Date) < split_date)
    test = data.filter(pl.col("timestamp_UTC").cast(pl.Date) >= split_date)

    if train.is_empty():
        raise ValueError(f"Training set is empty for split_date={split_date}.")
    if test.is_empty():
        raise ValueError(f"Test set is empty for split_date={split_date}.")


def _compute_daily_forecast(
    history: pl.DataFrame,
    test_day_data: pl.DataFrame,
    target_column: str,
) -> pl.DataFrame:
    quantiles = history.with_columns(
        pl.col("timestamp_UTC").dt.time().alias("time_of_day")
    ).group_by(["feeder", "time_of_day"]).agg(
        pl.col(target_column).quantile(0.05, interpolation="midpoint").alias("q05"),
        pl.col(target_column).quantile(0.50, interpolation="midpoint").alias("q50"),
        pl.col(target_column).quantile(0.95, interpolation="midpoint").alias("q95"),
    )

    test_with_time = test_day_data.with_columns(
        pl.col("timestamp_UTC").dt.time().alias("time_of_day")
    )

    return test_with_time.join(quantiles, on=["feeder", "time_of_day"], how="left").drop("time_of_day")


def _save_metrics(metrics: pl.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    metrics.write_csv(path)


def _plot_forecast(
    data: pl.DataFrame,
    eval_df: pl.DataFrame,
    target_column: str,
    test_days: list[date],
    output_path: Path,
) -> None:
    random.seed(42)

    feeders = eval_df["feeder"].unique().to_list()
    pairs = [(f, d) for f in feeders for d in test_days]
    selected = sorted(random.sample(pairs, min(9, len(pairs))))

    fig, axes = plt.subplots(3, 3, figsize=(18, 12))
    axes_flat = axes.flatten()

    for idx, (feeder, test_day) in enumerate(selected):
        ax = axes_flat[idx]

        lag_start = test_day - timedelta(days=7)
        lag_data = data.filter(
            (pl.col("timestamp_UTC").cast(pl.Date) >= lag_start)
            & (pl.col("timestamp_UTC").cast(pl.Date) < test_day)
            & (pl.col("feeder") == feeder)
        ).sort("timestamp_UTC")

        forecast_data = eval_df.filter(
            (pl.col("timestamp_UTC").cast(pl.Date) == test_day)
            & (pl.col("feeder") == feeder)
        ).sort("timestamp_UTC")

        if lag_data.height > 0:
            lag_ts = lag_data["timestamp_UTC"].to_list()
            lag_vals = lag_data[target_column].to_list()
            ax.plot(lag_ts, lag_vals, color="blue", label="History")

        if forecast_data.height > 0:
            fc_ts = forecast_data["timestamp_UTC"].to_list()
            fc_q50 = forecast_data["q50"].to_list()
            fc_q05 = forecast_data["q05"].to_list()
            fc_q95 = forecast_data["q95"].to_list()
            fc_actual = forecast_data[target_column].to_list()

            ax.plot(fc_ts, fc_q50, color="orange", linestyle="--", label="Forecast (median)")
            ax.fill_between(fc_ts, fc_q05, fc_q95, color="orange", alpha=0.3, label="90% PI")
            ax.plot(fc_ts, fc_actual, color="green", label="Actual")

        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
        ax.tick_params(axis="x", rotation=45)
        ax.set_title(f"Feeder {feeder}, {test_day}")

    # NOTE(copilot): Hide unused subplots if fewer than 9 pairs were selected
    for idx in range(len(selected), 9):
        axes_flat[idx].set_visible(False)

    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower right")
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
