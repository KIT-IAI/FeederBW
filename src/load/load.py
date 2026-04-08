from datetime import datetime, timezone
from pathlib import Path
import warnings

import polars as pl

from src.config import DATA_DIR
FEEDER_RANGE = range(1, 201)
DATASET_START = datetime(2023, 3, 31, 22, 0, tzinfo=timezone.utc)
DATASET_END = datetime(2025, 3, 31, 22, 0, tzinfo=timezone.utc)
VALID_TEMPORAL_RESOLUTIONS = ("1min", "5min", "15min", "30min", "1h", "1d", "1w", "1mo")
VALID_AGGREGATIONS = ("mean", "sum", "min", "max")
VALID_INTERPOLATIONS = ("linear", "forward", "backward")
MEASUREMENT_RANGE_FOLDERS = ["001-040", "041-080", "081-120", "121-160", "161-200"]

MEASUREMENT_COLUMNS = {
    "feeder", "timestamp_UTC", "timestamp_CET_CEST",
    "voltage_one_phase_V",
    "active_power_kW", "reactive_power_kVar", "apparent_power_kVA",
    "l1_active_current_A", "l1_reactive_current_A", "l1_apparent_current_A", "l1_power_factor_cos_phi",
    "l2_active_current_A", "l2_reactive_current_A", "l2_apparent_current_A", "l2_power_factor_cos_phi",
    "l3_active_current_A", "l3_reactive_current_A", "l3_apparent_current_A", "l3_power_factor_cos_phi",
    "lpen_apparent_current_A",
}

WEATHER_COLUMNS = {
    "feeder", "timestamp_UTC", "timestamp_CET_CEST",
    "direct_radiation_W/m2", "diffuse_radiation_W/m2", "air_temperature_C",
    "precipitation_kg/m2", "snow_height_m", "humidity_kg/kg",
    "max_wind_gust_m/s", "meridional_wind_m/s", "zonal_wind_m/s",
}

METADATA_COLUMNS = {
    "feeder", "date",
    "housing_units_count", "industry_and_commerce_count",
    "storage_heaters_kW", "heat_pumps_kW", "electric_heaters_kW",
    "hot_water_tanks_kW", "flow-type_heaters_kW", "EV_chargers_kW",
    "public_lighting_kW", "other_consumers_kW",
    "batteries_kW", "PV_systems_kW", "chp_kW",
    "g0_general_kWh", "g1_workdays_kWh", "g2_evening_kWh",
    "g3_continuous_kWh", "g4_shop_hairdresser_kWh", "g5_bakery_kWh", "g6_weekend_kWh",
}

ALL_COLUMNS = MEASUREMENT_COLUMNS | WEATHER_COLUMNS | METADATA_COLUMNS

_RESOLUTION_ORDER = {
    "1min": 0, "5min": 1, "15min": 2, "30min": 3,
    "1h": 4, "1d": 5, "1w": 6, "1mo": 7,
}

_POLARS_DURATION = {
    "1min": "1m", "5min": "5m", "15min": "15m", "30min": "30m",
    "1h": "1h", "1d": "1d", "1w": "1w", "1mo": "1mo",
}


def _to_polars_duration(resolution: str) -> str:
    return _POLARS_DURATION.get(resolution, resolution)


def load_data(
    feeders: list[int] | None = None,
    columns: list[str] | None = None,
    time_period: tuple[str | datetime, str | datetime] | None = None,
    temporal_resolution: str | None = None,
    aggregation_type: dict[str, str] | None = None,
    interpolation_type: dict[str, str] | None = None,
) -> pl.DataFrame:
    _validate_params(feeders, columns, time_period, temporal_resolution, aggregation_type, interpolation_type)
    feeders = feeders or list(FEEDER_RANGE)
    sources = _classify_columns(columns)

    dfs: dict[str, pl.DataFrame] = {}

    if "measurement" in sources:
        meas = _load_measurements(feeders, sources["measurement"], time_period)
        if temporal_resolution and _needs_aggregation("1min", temporal_resolution):
            meas = _aggregate(meas, temporal_resolution, aggregation_type or {})
        dfs["measurement"] = meas

    if "weather" in sources:
        weather = _load_weather(feeders, sources["weather"], time_period)
        if temporal_resolution and _needs_aggregation("1h", temporal_resolution):
            weather = _aggregate(weather, temporal_resolution, aggregation_type or {})
        elif temporal_resolution and _needs_interpolation("1h", temporal_resolution):
            weather = _interpolate(weather, temporal_resolution, interpolation_type or {})
        dfs["weather"] = weather

    if "metadata" in sources:
        meta = _load_metadata(feeders, sources["metadata"])
        dfs["metadata"] = meta

    result = _join_data(dfs.get("measurement"), dfs.get("weather"), dfs.get("metadata"))

    if columns is not None:
        always_keep = ["feeder", "timestamp_UTC"]
        ordered = [c for c in columns if c in result.columns]
        for c in always_keep:
            if c not in ordered and c in result.columns:
                ordered.insert(0, c)
        result = result.select(ordered)

    return result


# --- Validation ---

def _validate_params(feeders, columns, time_period, temporal_resolution, aggregation_type, interpolation_type):
    if feeders is not None:
        if not isinstance(feeders, list) or not all(isinstance(f, int) for f in feeders):
            raise ValueError("feeders must be a list of integers.")
        invalid = [f for f in feeders if f not in FEEDER_RANGE]
        if invalid:
            raise ValueError(f"Invalid feeder IDs: {invalid}. Must be in 1-200.")

    if columns is not None:
        if not isinstance(columns, list) or not columns:
            raise ValueError("columns must be a non-empty list of column name strings.")
        invalid = [c for c in columns if c not in ALL_COLUMNS]
        if invalid:
            raise ValueError(f"Invalid column names: {invalid}.")

    if time_period is not None:
        if not isinstance(time_period, tuple) or len(time_period) != 2:
            raise ValueError("time_period must be a tuple of (start, end).")
        start, end = _parse_datetime(time_period[0]), _parse_datetime(time_period[1])
        if start >= end:
            raise ValueError("time_period start must be before end.")
        if start < DATASET_START or end > DATASET_END:
            raise ValueError(
                f"time_period out of dataset range [{DATASET_START}, {DATASET_END}]."
            )

    if temporal_resolution is not None:
        if temporal_resolution not in VALID_TEMPORAL_RESOLUTIONS:
            raise ValueError(
                f"Invalid temporal_resolution '{temporal_resolution}'. "
                f"Valid options: {VALID_TEMPORAL_RESOLUTIONS}"
            )

    requested_cols = set(columns) if columns else ALL_COLUMNS

    if aggregation_type is not None:
        if temporal_resolution is None:
            raise ValueError("aggregation_type requires temporal_resolution to be set.")
        for col, agg in aggregation_type.items():
            if col not in requested_cols:
                raise ValueError(f"aggregation_type column '{col}' not in requested columns.")
            if agg not in VALID_AGGREGATIONS:
                raise ValueError(f"Invalid aggregation '{agg}' for column '{col}'. Valid: {VALID_AGGREGATIONS}")

    if interpolation_type is not None:
        if temporal_resolution is None:
            raise ValueError("interpolation_type requires temporal_resolution to be set.")
        for col, interp in interpolation_type.items():
            if col not in requested_cols:
                raise ValueError(f"interpolation_type column '{col}' not in requested columns.")
            if interp not in VALID_INTERPOLATIONS:
                raise ValueError(f"Invalid interpolation '{interp}' for column '{col}'. Valid: {VALID_INTERPOLATIONS}")


def _parse_datetime(value: str | datetime) -> datetime:
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value
    return datetime.fromisoformat(value).replace(tzinfo=timezone.utc)


# --- Column classification ---

def _classify_columns(columns: list[str] | None) -> dict[str, list[str]]:
    if columns is None:
        return {
            "measurement": list(MEASUREMENT_COLUMNS),
            "weather": list(WEATHER_COLUMNS),
            "metadata": list(METADATA_COLUMNS),
        }

    col_set = set(columns)
    sources: dict[str, list[str]] = {}

    meas_cols = col_set & (MEASUREMENT_COLUMNS - {"feeder", "timestamp_UTC", "timestamp_CET_CEST"})
    if meas_cols:
        sources["measurement"] = list(meas_cols | {"feeder", "timestamp_UTC"})

    weather_cols = col_set & (WEATHER_COLUMNS - {"feeder", "timestamp_UTC", "timestamp_CET_CEST"})
    if weather_cols:
        sources["weather"] = list(weather_cols | {"feeder", "timestamp_UTC"})

    meta_cols = col_set & (METADATA_COLUMNS - {"feeder", "date"})
    if meta_cols:
        sources["metadata"] = list(meta_cols | {"feeder", "date"})

    # NOTE(copilot): If only join keys requested and no source-specific columns matched,
    # default to measurement source for feeder/timestamp_UTC.
    if not sources and col_set <= {"feeder", "timestamp_UTC", "timestamp_CET_CEST", "date"}:
        if "date" in col_set:
            sources["metadata"] = list((col_set & {"feeder", "date"}) | {"feeder", "date"})
        else:
            sources["measurement"] = list((col_set & {"feeder", "timestamp_UTC", "timestamp_CET_CEST"}) | {"feeder", "timestamp_UTC"})

    return sources


# --- File path resolution ---

def _feeder_id_to_filepath(feeder_id: int) -> Path:
    folder_index = (feeder_id - 1) // 40
    folder = MEASUREMENT_RANGE_FOLDERS[folder_index]
    return DATA_DIR / f"feeder_measurement_data_{folder}" / f"feeder_{feeder_id:03d}.parquet"


# --- Data loaders ---

def _load_measurements(feeders: list[int], columns: list[str], time_period) -> pl.DataFrame:
    frames = []
    for fid in feeders:
        path = _feeder_id_to_filepath(fid)
        lf = pl.scan_parquet(path, hive_partitioning=False)
        schema_names = lf.collect_schema().names()
        missing = [c for c in columns if c not in schema_names]
        if missing:
            warnings.warn(f"Columns {missing} not found in {path.name}, skipping them.")
        lf = lf.select([c for c in columns if c in schema_names])
        lf = lf.with_columns(
            pl.col("feeder").cast(pl.Int64),
            pl.col("timestamp_UTC").str.to_datetime("%Y-%m-%d %H:%M:%S").dt.replace_time_zone("UTC"),
        )
        if time_period is not None:
            start, end = _parse_datetime(time_period[0]), _parse_datetime(time_period[1])
            lf = lf.filter(
                (pl.col("timestamp_UTC") >= start) & (pl.col("timestamp_UTC") <= end)
            )
        frames.append(lf.collect())

    return pl.concat(frames)


def _load_weather(feeders: list[int], columns: list[str], time_period) -> pl.DataFrame:
    path = DATA_DIR / "weather_data.parquet"
    lf = pl.scan_parquet(path, hive_partitioning=False)
    lf = lf.select([c for c in columns if c in lf.collect_schema().names()])
    lf = lf.with_columns(
        pl.col("feeder").cast(pl.Int64),
        pl.col("timestamp_UTC").str.to_datetime("%Y-%m-%d %H:%M:%S").dt.replace_time_zone("UTC"),
    )
    lf = lf.filter(pl.col("feeder").is_in(feeders))
    if time_period is not None:
        start, end = _parse_datetime(time_period[0]), _parse_datetime(time_period[1])
        lf = lf.filter(
            (pl.col("timestamp_UTC") >= start) & (pl.col("timestamp_UTC") <= end)
        )
    return lf.collect()


def _load_metadata(feeders: list[int], columns: list[str]) -> pl.DataFrame:
    path = DATA_DIR / "feeder_metadata.csv"
    df = pl.read_csv(path, columns=[c for c in columns if c != "timestamp_UTC"])
    df = df.filter(pl.col("feeder").is_in(feeders))
    # NOTE(m): the dates start at 2023-04-01 but the first measurement timestamp is 2023-03-31 22:00:00 UTC, 
    # Hence, we align the metadata timestamps (date) to the measurement timestamps so that the join
    # based on feeder and timestamp_UTC works correctly for the first two hours of the dataset.
    df = df.with_columns(
        pl.col("feeder").cast(pl.Int64),
        pl.col("date")
        .str.to_date("%Y-%m-%d")
        .cast(pl.Datetime("us"))
        .dt.replace_time_zone("Europe/Berlin")
        .dt.convert_time_zone("UTC")
        .alias("timestamp_UTC"),
    )
    return df.sort(["feeder", "timestamp_UTC"])


# --- Resolution comparison ---

def _needs_aggregation(source_res: str, target_res: str) -> bool:
    return _RESOLUTION_ORDER.get(target_res, 0) > _RESOLUTION_ORDER.get(source_res, 0)


def _needs_interpolation(source_res: str, target_res: str) -> bool:
    return _RESOLUTION_ORDER.get(target_res, 0) < _RESOLUTION_ORDER.get(source_res, 0)


# --- Aggregation & Interpolation ---

def _aggregate(df: pl.DataFrame, target_resolution: str, aggregation_type: dict[str, str]) -> pl.DataFrame:
    df = df.sort(["feeder", "timestamp_UTC"])
    value_cols = [c for c in df.columns if c not in {"feeder", "timestamp_UTC", "timestamp_CET_CEST"}]

    agg_exprs = []
    for col in value_cols:
        agg = aggregation_type.get(col, "mean")
        agg_exprs.append(_agg_expr(col, agg))

    result = df.group_by_dynamic(
        "timestamp_UTC", every=_to_polars_duration(target_resolution), group_by="feeder"
    ).agg(agg_exprs)

    if "timestamp_CET_CEST" in result.columns:
        result = result.drop("timestamp_CET_CEST")

    return result


def _agg_expr(col: str, agg: str) -> pl.Expr:
    if agg == "mean":
        return pl.col(col).mean()
    elif agg == "sum":
        return pl.col(col).sum()
    elif agg == "min":
        return pl.col(col).min()
    elif agg == "max":
        return pl.col(col).max()
    raise ValueError(f"Unknown aggregation: {agg}")


def _interpolate(df: pl.DataFrame, target_resolution: str, interpolation_type: dict[str, str]) -> pl.DataFrame:
    df = df.sort(["feeder", "timestamp_UTC"])

    if "timestamp_CET_CEST" in df.columns:
        df = df.drop("timestamp_CET_CEST")

    df = df.upsample("timestamp_UTC", every=_to_polars_duration(target_resolution), group_by="feeder")

    value_cols = [c for c in df.columns if c not in {"feeder", "timestamp_UTC"}]
    interp_exprs = []
    for col in value_cols:
        method = interpolation_type.get(col, "linear")
        interp_exprs.append(_interp_expr(col, method).over("feeder"))

    if interp_exprs:
        df = df.with_columns(interp_exprs)

    return df


def _interp_expr(col: str, method: str) -> pl.Expr:
    if method == "linear":
        return pl.col(col).interpolate()
    elif method == "forward":
        return pl.col(col).forward_fill()
    elif method == "backward":
        return pl.col(col).backward_fill()
    raise ValueError(f"Unknown interpolation: {method}")


# --- Join ---

def _join_data(
    measurement_df: pl.DataFrame | None,
    weather_df: pl.DataFrame | None,
    metadata_df: pl.DataFrame | None,
) -> pl.DataFrame:
    present = [df for df in (measurement_df, weather_df, metadata_df) if df is not None]
    if len(present) == 1:
        return present[0]

    result = measurement_df if measurement_df is not None else weather_df

    if measurement_df is not None and weather_df is not None:
        weather_sorted = weather_df.sort(["feeder", "timestamp_UTC"])
        result_sorted = result.sort(["feeder", "timestamp_UTC"])
        # NOTE(copilot): Drop timestamp_CET_CEST from weather before join to avoid duplicate
        if "timestamp_CET_CEST" in weather_sorted.columns:
            weather_sorted = weather_sorted.drop("timestamp_CET_CEST") 
        with warnings.catch_warnings():
            # NOTE (m): warning can be ignored. 
            warnings.filterwarnings("ignore", message="Sortedness of columns", category=UserWarning)
            # NOTE(m): It would be enough here to do a regular join because the resolution of weather data 
            # and measurement data was already aligned before calling this method. However, we use a join_asof
            # here to be more robust.
            result = result_sorted.join_asof(
                weather_sorted,
                on="timestamp_UTC",
                by="feeder",
                strategy="backward",
                tolerance="1h",
            )

    if metadata_df is not None and result is not None:
        meta_sorted = metadata_df.sort(["feeder", "timestamp_UTC"])
        result_sorted = result.sort(["feeder", "timestamp_UTC"])
        if "date" in meta_sorted.columns:
            meta_sorted = meta_sorted.drop("date")
        with warnings.catch_warnings():
            # NOTE (m): warning can be ignored. 
            warnings.filterwarnings("ignore", message="Sortedness of columns", category=UserWarning)
            result = result_sorted.join_asof(
                meta_sorted,
                on="timestamp_UTC",
                by="feeder",
                strategy="backward",
            )
    elif metadata_df is not None:
        result = metadata_df

    return result
