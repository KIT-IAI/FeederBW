import random
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import polars as pl
from sklearn.ensemble import HistGradientBoostingRegressor

from src.model._metrics import compute_metrics


def pseudo_measurement_generation(
    data: pl.DataFrame,
    target_column: str,
    feature_columns: list[str],
    test_feeders: list[int | str],
) -> pl.DataFrame:
    """Pseudo-measurement generation using HistGradientBoosting quantile regression.

    Trains three HistGradientBoosting models (for 5th, 50th, 95th percentiles)
    on training feeders (all feeders not in test_feeders), then generates
    pseudo-measurements for test feeders.

    Note:
        This baseline applies a spatial-only split (train on some feeders, test on
        others). No temporal train/test split is applied — all time steps are used
        for both training and testing feeders.

    Args:
        data: DataFrame with 'feeder', 'timestamp_UTC', target_column, and feature_columns.
        target_column: Column name for which pseudo-measurements are generated.
        feature_columns: List of column names used as model features.
            All features must be numeric (or integer-encoded categoricals).
        test_feeders: Feeder identifiers for which pseudo-measurements are generated.
            All other feeders are used for training.

    Returns:
        DataFrame with per-feeder MAE, pinball losses, and winkler score,
        plus an 'average' row across all feeders.

    Raises:
        ValueError: If target_column, feature_columns, or test_feeders are invalid.
    """
    _validate_pseudo_measurement_inputs(data, target_column, feature_columns, test_feeders)

    train_data = data.filter(~pl.col("feeder").is_in(test_feeders))
    test_data = data.filter(pl.col("feeder").is_in(test_feeders))

    if train_data.is_empty():
        raise ValueError("Training data is empty: all feeders are test feeders.")
    if test_data.is_empty():
        raise ValueError("Test data is empty: no data found for specified test_feeders.")

    temporal_features = ["hour", "day_of_week", "month"]
    train_data = _add_temporal_features(train_data)
    test_data = _add_temporal_features(test_data)
    all_features = feature_columns + temporal_features

    X_train = train_data.select(all_features).to_numpy()
    y_train = train_data[target_column].to_numpy()
    X_test = test_data.select(all_features).to_numpy()

    model_q05, model_q50, model_q95 = _train_quantile_models(X_train, y_train)

    test_data_with_preds = test_data.with_columns([
        pl.Series("q05", model_q05.predict(X_test)),
        pl.Series("q50", model_q50.predict(X_test)),
        pl.Series("q95", model_q95.predict(X_test)),
    ])

    metrics = compute_metrics(test_data_with_preds, target_column)
    _save_metrics(metrics, Path("data/pseudo_measurement_metrics.csv"))
    _plot_pseudo_measurements(
        test_data_with_preds,
        target_column,
        Path("data/pseudo_measurement_figure.png"),
    )

    return metrics


def _validate_pseudo_measurement_inputs(
    data: pl.DataFrame,
    target_column: str,
    feature_columns: list[str],
    test_feeders: list[int | str],
) -> None:
    if target_column not in data.columns:
        raise ValueError(f"target_column '{target_column}' not found in data columns.")

    missing_features = [c for c in feature_columns if c not in data.columns]
    if missing_features:
        raise ValueError(f"feature_columns not found in data: {missing_features}")

    available_feeders = set(data["feeder"].to_list())
    missing_feeders = [f for f in test_feeders if f not in available_feeders]
    if missing_feeders:
        raise ValueError(f"test_feeders not found in data: {missing_feeders}")


def _add_temporal_features(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(
        pl.col("timestamp_UTC").dt.hour().alias("hour"),
        pl.col("timestamp_UTC").dt.weekday().alias("day_of_week"),
        pl.col("timestamp_UTC").dt.month().alias("month"),
    )


def _train_quantile_models(
    X_train,
    y_train,
) -> tuple[HistGradientBoostingRegressor, HistGradientBoostingRegressor, HistGradientBoostingRegressor]:
    model_q05 = HistGradientBoostingRegressor(loss="quantile", quantile=0.05, random_state=42)
    model_q50 = HistGradientBoostingRegressor(loss="quantile", quantile=0.50, random_state=42)
    model_q95 = HistGradientBoostingRegressor(loss="quantile", quantile=0.95, random_state=42)

    model_q05.fit(X_train, y_train)
    model_q50.fit(X_train, y_train)
    model_q95.fit(X_train, y_train)

    return model_q05, model_q50, model_q95


def _save_metrics(metrics: pl.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    metrics.write_csv(path)


def _plot_pseudo_measurements(
    test_data_with_preds: pl.DataFrame,
    target_column: str,
    output_path: Path,
) -> None:
    random.seed(42)

    df = test_data_with_preds.with_columns(
        pl.col("timestamp_UTC").dt.truncate("1w").alias("week_start")
    )

    max_week_start = df["week_start"].max()
    pairs = (
        df.filter(pl.col("week_start") < max_week_start)
        .select(["feeder", "week_start"])
        .unique()
        .to_dicts()
    )
    selected = random.sample(pairs, min(9, len(pairs)))

    fig, axes = plt.subplots(3, 3, figsize=(18, 12))
    axes_flat = axes.flatten()

    for idx, pair in enumerate(selected):
        ax = axes_flat[idx]
        feeder = pair["feeder"]
        week_start = pair["week_start"]

        week_data = df.filter(
            (pl.col("feeder") == feeder) & (pl.col("week_start") == week_start)
        ).sort("timestamp_UTC")

        ts = week_data["timestamp_UTC"].to_list()
        actual = week_data[target_column].to_list()
        q05 = week_data["q05"].to_list()
        q50 = week_data["q50"].to_list()
        q95 = week_data["q95"].to_list()

        ax.fill_between(ts, q05, q95, color="orange", alpha=0.3)
        ax.plot(ts, q50, color="orange", linestyle="--", label="Pseudo-measurement")
        ax.plot(ts, actual, color="blue", label="Actual")

        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
        ax.tick_params(axis="x", rotation=45)
        ax.set_title(f"Feeder {feeder}, Week of {week_start}")

    # NOTE(copilot): Hide unused subplots if fewer than 9 pairs were selected
    for idx in range(len(selected), 9):
        axes_flat[idx].set_visible(False)

    all_handles, all_labels = [], []
    for child_ax in fig.axes:
        h, l = child_ax.get_legend_handles_labels()
        for handle, label in zip(h, l):
            if label not in all_labels:
                all_handles.append(handle)
                all_labels.append(label)
    fig.legend(all_handles, all_labels, loc="lower right")
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
