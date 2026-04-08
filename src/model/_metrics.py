import polars as pl


def compute_metrics(eval_df: pl.DataFrame, target_column: str) -> pl.DataFrame:
    """Compute per-feeder forecast metrics: MAE, pinball losses, and Winkler score.

    Args:
        eval_df: DataFrame with columns 'feeder', target_column, 'q05', 'q50', 'q95'.
        target_column: Name of the actual target column.

    Returns:
        Per-feeder metrics DataFrame with an appended 'average' row.
    """
    valid = eval_df.filter(
        pl.all_horizontal(
            pl.col("q05").is_not_null(),
            pl.col("q50").is_not_null(),
            pl.col("q95").is_not_null(),
        )
    )

    metrics = valid.group_by("feeder").agg(
        (pl.col(target_column) - pl.col("q50")).abs().mean().alias("mae"),
        pl.when(pl.col(target_column) >= pl.col("q05"))
        .then(0.05 * (pl.col(target_column) - pl.col("q05")))
        .otherwise(0.95 * (pl.col("q05") - pl.col(target_column)))
        .mean()
        .alias("pinball_q05"),
        pl.when(pl.col(target_column) >= pl.col("q50"))
        .then(0.50 * (pl.col(target_column) - pl.col("q50")))
        .otherwise(0.50 * (pl.col("q50") - pl.col(target_column)))
        .mean()
        .alias("pinball_q50"),
        pl.when(pl.col(target_column) >= pl.col("q95"))
        .then(0.95 * (pl.col(target_column) - pl.col("q95")))
        .otherwise(0.05 * (pl.col("q95") - pl.col(target_column)))
        .mean()
        .alias("pinball_q95"),
        (
            (pl.col("q95") - pl.col("q05"))
            + 20.0 * (pl.col("q05") - pl.col(target_column)).clip(lower_bound=0)
            + 20.0 * (pl.col(target_column) - pl.col("q95")).clip(lower_bound=0)
        )
        .mean()
        .alias("winkler_score"),
    ).sort("feeder")

    avg_row = metrics.select([
        pl.lit("average").alias("feeder"),
        pl.col("mae").mean(),
        pl.col("pinball_q05").mean(),
        pl.col("pinball_q50").mean(),
        pl.col("pinball_q95").mean(),
        pl.col("winkler_score").mean(),
    ])

    metrics = metrics.with_columns(pl.col("feeder").cast(pl.String))
    return pl.concat([metrics, avg_row])
