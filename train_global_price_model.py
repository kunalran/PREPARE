from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder


BASE_COLUMNS = [
    "State",
    "District",
    "Market",
    "Commodity",
    "Date",
    "Arrival_Quantity",
    "Modal_Price",
    "latitude",
    "longitude",
]

WEATHER_PREFIXES = ("t", "tp", "ssr", "r")
WEATHER_HOURS = range(24)


def replace_inf_with_nan(frame):
    if isinstance(frame, pd.DataFrame):
        return frame.replace([np.inf, -np.inf], np.nan)
    arr = np.asarray(frame, dtype=float)
    arr = arr.copy()
    arr[~np.isfinite(arr)] = np.nan
    return arr


@dataclass(frozen=True)
class Config:
    data_dir: Path
    output_dir: Path
    horizons: list[int]
    validation_days: int
    min_series_observations: int
    dense_min_pct: float | None
    max_train_rows_per_horizon: int | None
    random_state: int


def parse_args() -> Config:
    parser = argparse.ArgumentParser(
        description="Train pooled crop price forecasting models for 1-15 day horizons."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("final_data_hourly"),
        help="Directory containing agmarknet_*_final_hourly.csv files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("models/global_histgb"),
        help="Directory where trained models and metrics will be saved.",
    )
    parser.add_argument(
        "--horizons",
        type=str,
        default="1,2,3,4,5,6,7,8,9,10,11,12,13,14,15",
        help="Comma-separated forecast horizons in days.",
    )
    parser.add_argument(
        "--validation-days",
        type=int,
        default=90,
        help="Number of trailing calendar days reserved for validation.",
    )
    parser.add_argument(
        "--min-series-observations",
        type=int,
        default=30,
        help="Drop crop-market series with fewer observed price points than this.",
    )
    parser.add_argument(
        "--dense-min-pct",
        type=float,
        default=None,
        help="Optional minimum %% non-null Modal_Price per market-year to keep.",
    )
    parser.add_argument(
        "--max-train-rows-per-horizon",
        type=int,
        default=250000,
        help="Optional cap for training rows per horizon to keep runtime bounded.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    args = parser.parse_args()
    horizons = sorted({int(x.strip()) for x in args.horizons.split(",") if x.strip()})
    return Config(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        horizons=horizons,
        validation_days=args.validation_days,
        min_series_observations=args.min_series_observations,
        dense_min_pct=args.dense_min_pct,
        max_train_rows_per_horizon=args.max_train_rows_per_horizon,
        random_state=args.random_state,
    )


def weather_columns() -> list[str]:
    return [f"{prefix}{hour:02d}" for prefix in WEATHER_PREFIXES for hour in WEATHER_HOURS]


def filter_dense_market_years(
    df: pd.DataFrame,
    target: str = "Modal_Price",
    min_pct: float = 50.0,
) -> pd.DataFrame:
    df = df.copy()
    df["_Year"] = pd.to_datetime(df["Date"]).dt.year
    group_cols = ["Commodity", "State", "District", "Market", "_Year"]
    grp = df.groupby(group_cols)[target]
    n_total = grp.transform("size")
    n_present = grp.transform(lambda x: x.notna().sum())
    pct_present = n_present / n_total * 100.0
    filtered = df[pct_present >= min_pct].copy()
    filtered.drop(columns=["_Year"], inplace=True)
    return filtered


def _causal_ratio_fill(series: pd.Series, dow: pd.Series) -> tuple[pd.Series, pd.Series]:
    # Past-only local context plus DOW ratio, inspired by the imputation script
    rolling_ctx = series.shift(1).rolling(7, min_periods=1).mean()
    overall_mean = series.shift(1).expanding(min_periods=3).mean()
    dow_mean = series.groupby(dow).transform(
        lambda s: s.shift(1).expanding(min_periods=2).mean()
    )
    dow_ratio = (dow_mean / overall_mean).replace([np.inf, -np.inf], np.nan)
    dow_ratio = dow_ratio.fillna(1.0).clip(lower=0.25, upper=4.0)

    filled = series.copy()
    fill_values = rolling_ctx * dow_ratio
    missing = filled.isna()
    filled.loc[missing] = fill_values.loc[missing]
    fallback = overall_mean.loc[missing]
    still_missing = filled.isna()
    filled.loc[still_missing] = fallback.loc[still_missing]
    return filled, dow_ratio


def load_and_engineer_crop(
    path: Path,
    horizons: list[int],
    dense_min_pct: float | None = None,
    observed_only: bool = True,
) -> pd.DataFrame:
    usecols = BASE_COLUMNS + weather_columns()
    df = pd.read_csv(path, usecols=usecols, parse_dates=["Date"])
    if dense_min_pct is not None:
        rows_before = len(df)
        df = filter_dense_market_years(df, min_pct=dense_min_pct)
        print(
            f"  Dense market-year filter at {dense_min_pct:.0f}%: "
            f"{rows_before:,} -> {len(df):,} rows"
        )
    df["series_id"] = (
        df["Commodity"].astype(str)
        + "||"
        + df["State"].astype(str)
        + "||"
        + df["District"].astype(str)
        + "||"
        + df["Market"].astype(str)
    )
    df = df.sort_values(["series_id", "Date"]).reset_index(drop=True)

    temp_cols = [f"t{hour:02d}" for hour in WEATHER_HOURS]
    rain_cols = [f"tp{hour:02d}" for hour in WEATHER_HOURS]
    solar_cols = [f"ssr{hour:02d}" for hour in WEATHER_HOURS]
    humid_cols = [f"r{hour:02d}" for hour in WEATHER_HOURS]

    df["temp_mean"] = df[temp_cols].mean(axis=1)
    df["temp_min"] = df[temp_cols].min(axis=1)
    df["temp_max"] = df[temp_cols].max(axis=1)
    df["temp_range"] = df["temp_max"] - df["temp_min"]
    df["rain_sum"] = df[rain_cols].sum(axis=1)
    df["solar_sum"] = df[solar_cols].sum(axis=1)
    df["solar_peak"] = df[solar_cols].max(axis=1)
    df["rh_mean"] = df[humid_cols].mean(axis=1)
    df["rh_min"] = df[humid_cols].min(axis=1)
    df["rh_max"] = df[humid_cols].max(axis=1)

    df["month"] = df["Date"].dt.month.astype(np.int16)
    df["day_of_week_num"] = df["Date"].dt.dayofweek.astype(np.int16)
    df["day_of_year"] = df["Date"].dt.dayofyear.astype(np.int16)
    df["week_of_year"] = df["Date"].dt.isocalendar().week.astype(np.int16)
    df["year"] = df["Date"].dt.year.astype(np.int16)
    df["is_month_start"] = df["Date"].dt.is_month_start.astype(np.int8)
    df["is_month_end"] = df["Date"].dt.is_month_end.astype(np.int8)

    day_angle = 2.0 * np.pi * df["day_of_year"] / 366.0
    week_angle = 2.0 * np.pi * df["day_of_week_num"] / 7.0
    df["day_of_year_sin"] = np.sin(day_angle)
    df["day_of_year_cos"] = np.cos(day_angle)
    df["day_of_week_sin"] = np.sin(week_angle)
    df["day_of_week_cos"] = np.cos(week_angle)

    grouped_price = df.groupby("series_id", sort=False)["Modal_Price"]
    grouped_arrival = df.groupby("series_id", sort=False)["Arrival_Quantity"]

    imputed_prices: list[pd.Series] = []
    imputed_arrivals: list[pd.Series] = []
    price_dow_ratios: list[pd.Series] = []
    arrival_dow_ratios: list[pd.Series] = []
    for _, group in df.groupby("series_id", sort=False):
        price_filled, price_ratio = _causal_ratio_fill(
            group["Modal_Price"],
            group["day_of_week_num"],
        )
        arrival_filled, arrival_ratio = _causal_ratio_fill(
            group["Arrival_Quantity"],
            group["day_of_week_num"],
        )
        imputed_prices.append(price_filled)
        imputed_arrivals.append(arrival_filled)
        arrival_dow_ratios.append(arrival_ratio)
        price_dow_ratios.append(price_ratio)

    df["Modal_Price_CausalFilled"] = pd.concat(imputed_prices).sort_index()
    df["Arrival_Quantity_CausalFilled"] = pd.concat(imputed_arrivals).sort_index()
    df["arrival_dow_ratio"] = pd.concat(arrival_dow_ratios).sort_index()
    df["price_dow_ratio"] = pd.concat(price_dow_ratios).sort_index()
    grouped_price_filled = df.groupby("series_id", sort=False)["Modal_Price_CausalFilled"]
    grouped_arrival_filled = df.groupby("series_id", sort=False)["Arrival_Quantity_CausalFilled"]
    state_day_group = df.groupby(["Commodity", "State", "Date"], sort=False)
    commodity_day_group = df.groupby(["Commodity", "Date"], sort=False)
    df["state_price_mean"] = state_day_group["Modal_Price_CausalFilled"].transform("mean")
    df["state_arrival_mean"] = state_day_group["Arrival_Quantity_CausalFilled"].transform("mean")
    df["national_price_mean"] = commodity_day_group["Modal_Price_CausalFilled"].transform("mean")
    df["national_arrival_mean"] = commodity_day_group["Arrival_Quantity_CausalFilled"].transform("mean")
    first_dates = df.groupby("series_id", sort=False)["Date"].transform("min")
    df["series_age_days"] = (df["Date"] - first_dates).dt.days.astype(np.int32)

    for lag in (1, 7, 14, 28, 56, 84, 112, 168):
        df[f"price_lag_{lag}"] = grouped_price_filled.shift(lag)
        df[f"arrival_lag_{lag}"] = grouped_arrival_filled.shift(lag)

    for window in (7, 14, 28, 56, 84, 112, 168):
        df[f"price_roll_mean_{window}"] = grouped_price_filled.transform(
            lambda s: s.shift(1).rolling(window=window, min_periods=1).mean()
        )
        df[f"price_roll_std_{window}"] = grouped_price_filled.transform(
            lambda s: s.shift(1).rolling(window=window, min_periods=2).std()
        )
        df[f"arrival_roll_mean_{window}"] = grouped_arrival_filled.transform(
            lambda s: s.shift(1).rolling(window=window, min_periods=1).mean()
        )
    df["price_roll_mean_21"] = grouped_price_filled.transform(
        lambda s: s.shift(1).rolling(window=21, min_periods=1).mean()
    )
    df["arrival_roll_mean_21"] = grouped_arrival_filled.transform(
        lambda s: s.shift(1).rolling(window=21, min_periods=1).mean()
    )
    for window in (28, 56, 84, 168):
        df[f"price_roll_min_{window}"] = grouped_price_filled.transform(
            lambda s: s.shift(1).rolling(window=window, min_periods=1).min()
        )
        df[f"price_roll_max_{window}"] = grouped_price_filled.transform(
            lambda s: s.shift(1).rolling(window=window, min_periods=1).max()
        )

    observed_mask = df["Modal_Price"].notna()
    df["price_log1p"] = np.where(observed_mask, np.log1p(df["Modal_Price"]), np.nan)
    df["price_filled_log1p"] = np.where(
        df["Modal_Price_CausalFilled"].notna(),
        np.log1p(df["Modal_Price_CausalFilled"]),
        np.nan,
    )
    df["arrival_log1p"] = np.where(
        df["Arrival_Quantity_CausalFilled"].notna(),
        np.log1p(df["Arrival_Quantity_CausalFilled"]),
        np.nan,
    )
    df["price_vs_roll7"] = df["Modal_Price"] / df["price_roll_mean_7"]
    df["price_vs_roll28"] = df["Modal_Price"] / df["price_roll_mean_28"]
    df["price_vs_roll84"] = df["Modal_Price"] / df["price_roll_mean_84"]
    df["price_vs_roll168"] = df["Modal_Price"] / df["price_roll_mean_168"]
    df["price_vs_state_mean"] = df["Modal_Price"] / df["state_price_mean"]
    df["price_vs_national_mean"] = df["Modal_Price"] / df["national_price_mean"]
    df["arrival_vs_roll28"] = (
        df["Arrival_Quantity_CausalFilled"] / df["arrival_roll_mean_28"]
    )
    df["arrival_vs_state_mean"] = (
        df["Arrival_Quantity_CausalFilled"] / df["state_arrival_mean"]
    )
    df["arrival_vs_national_mean"] = (
        df["Arrival_Quantity_CausalFilled"] / df["national_arrival_mean"]
    )
    df["price_minus_state_mean"] = df["Modal_Price"] - df["state_price_mean"]
    df["price_minus_national_mean"] = df["Modal_Price"] - df["national_price_mean"]
    df["price_trend_7_28"] = df["price_roll_mean_7"] - df["price_roll_mean_28"]
    df["price_trend_14_56"] = df["price_roll_mean_14"] - df["price_roll_mean_56"]
    df["price_trend_28_84"] = df["price_roll_mean_28"] - df["price_roll_mean_84"]
    df["price_trend_28_168"] = df["price_roll_mean_28"] - df["price_roll_mean_168"]
    df["arrival_trend_7_28"] = df["arrival_roll_mean_7"] - df["arrival_roll_mean_28"]
    df["arrival_trend_28_84"] = df["arrival_roll_mean_28"] - df["arrival_roll_mean_84"]
    df["price_range_28"] = df["price_roll_max_28"] - df["price_roll_min_28"]
    df["price_range_84"] = df["price_roll_max_84"] - df["price_roll_min_84"]
    df["price_range_168"] = df["price_roll_max_168"] - df["price_roll_min_168"]
    df["price_volatility_ratio_28"] = df["price_roll_std_28"] / df["price_roll_mean_28"]
    df["price_volatility_ratio_84"] = df["price_roll_std_84"] / df["price_roll_mean_84"]
    df["price_volatility_ratio_168"] = df["price_roll_std_168"] / df["price_roll_mean_168"]

    # Ratio features can blow up when rolling means are 0 or extremely small.
    ratio_like_cols = [
        "price_vs_roll7",
        "price_vs_roll28",
        "price_vs_roll84",
        "price_vs_roll168",
        "price_vs_state_mean",
        "price_vs_national_mean",
        "arrival_vs_roll28",
        "arrival_vs_state_mean",
        "arrival_vs_national_mean",
        "price_volatility_ratio_28",
        "price_volatility_ratio_84",
        "price_volatility_ratio_168",
    ]
    df[ratio_like_cols] = df[ratio_like_cols].replace([np.inf, -np.inf], np.nan)

    for horizon in horizons:
        df[f"target_{horizon}d"] = grouped_price.shift(-horizon)

    keep_columns = [
        "Commodity",
        "State",
        "District",
        "Market",
        "series_id",
        "Date",
        "latitude",
        "longitude",
        "Arrival_Quantity",
        "Modal_Price_CausalFilled",
        "Arrival_Quantity_CausalFilled",
        "arrival_log1p",
        "Modal_Price",
        "price_log1p",
        "price_filled_log1p",
        "temp_mean",
        "temp_min",
        "temp_max",
        "temp_range",
        "rain_sum",
        "solar_sum",
        "solar_peak",
        "rh_mean",
        "rh_min",
        "rh_max",
        "state_price_mean",
        "state_arrival_mean",
        "national_price_mean",
        "national_arrival_mean",
        "month",
        "day_of_week_num",
        "day_of_year",
        "week_of_year",
        "year",
        "series_age_days",
        "is_month_start",
        "is_month_end",
        "day_of_year_sin",
        "day_of_year_cos",
        "day_of_week_sin",
        "day_of_week_cos",
        "price_dow_ratio",
        "arrival_dow_ratio",
        "price_lag_1",
        "price_lag_7",
        "price_lag_14",
        "price_lag_28",
        "price_lag_56",
        "price_lag_84",
        "price_lag_112",
        "price_lag_168",
        "arrival_lag_1",
        "arrival_lag_7",
        "arrival_lag_14",
        "arrival_lag_28",
        "arrival_lag_56",
        "arrival_lag_84",
        "arrival_lag_112",
        "arrival_lag_168",
        "price_roll_mean_7",
        "price_roll_mean_14",
        "price_roll_mean_21",
        "price_roll_mean_28",
        "price_roll_mean_56",
        "price_roll_mean_84",
        "price_roll_mean_112",
        "price_roll_mean_168",
        "price_roll_std_7",
        "price_roll_std_14",
        "price_roll_std_28",
        "price_roll_std_56",
        "price_roll_std_84",
        "price_roll_std_112",
        "price_roll_std_168",
        "arrival_roll_mean_7",
        "arrival_roll_mean_14",
        "arrival_roll_mean_21",
        "arrival_roll_mean_28",
        "arrival_roll_mean_56",
        "arrival_roll_mean_84",
        "arrival_roll_mean_112",
        "arrival_roll_mean_168",
        "price_roll_min_28",
        "price_roll_max_28",
        "price_roll_min_56",
        "price_roll_max_56",
        "price_roll_min_84",
        "price_roll_max_84",
        "price_roll_min_168",
        "price_roll_max_168",
        "price_vs_roll7",
        "price_vs_roll28",
        "price_vs_roll84",
        "price_vs_roll168",
        "price_vs_state_mean",
        "price_vs_national_mean",
        "arrival_vs_roll28",
        "arrival_vs_state_mean",
        "arrival_vs_national_mean",
        "price_minus_state_mean",
        "price_minus_national_mean",
        "price_trend_7_28",
        "price_trend_14_56",
        "price_trend_28_84",
        "price_trend_28_168",
        "arrival_trend_7_28",
        "arrival_trend_28_84",
        "price_range_28",
        "price_range_84",
        "price_range_168",
        "price_volatility_ratio_28",
        "price_volatility_ratio_84",
        "price_volatility_ratio_168",
    ] + [f"target_{horizon}d" for horizon in horizons]

    if observed_only:
        return df.loc[observed_mask, keep_columns].copy()
    return df.loc[:, keep_columns].copy()


def load_training_frame(config: Config) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    csv_paths = sorted(config.data_dir.glob("agmarknet_*_final_hourly.csv"))
    if not csv_paths:
        raise FileNotFoundError(f"No hourly data files found in {config.data_dir}")

    for path in csv_paths:
        print(f"Loading {path.name} ...")
        frame = load_and_engineer_crop(path, config.horizons, config.dense_min_pct)
        frames.append(frame)

    data = pd.concat(frames, ignore_index=True)

    series_counts = data.groupby("series_id")["Modal_Price"].size()
    valid_series = series_counts[series_counts >= config.min_series_observations].index
    filtered = data[data["series_id"].isin(valid_series)].copy()
    print(
        f"Kept {len(valid_series):,} series with at least "
        f"{config.min_series_observations} observed prices."
    )
    print(f"Training frame rows after filtering: {len(filtered):,}")
    return filtered


def model_params_for_horizon(horizon: int) -> dict[str, float | int | bool | str]:
    if horizon >= 12:
        return {
            "loss": "absolute_error",
            "learning_rate": 0.03,
            "max_depth": 12,
            "max_iter": 900,
            "min_samples_leaf": 15,
            "l2_regularization": 0.0,
            "early_stopping": False,
            "random_state": 42,
        }
    if horizon >= 7:
        return {
            "loss": "squared_error",
            "learning_rate": 0.035,
            "max_depth": 10,
            "max_iter": 500,
            "min_samples_leaf": 20,
            "l2_regularization": 0.5,
            "early_stopping": False,
            "random_state": 42,
        }
    return {
        "loss": "squared_error",
        "learning_rate": 0.04,
        "max_depth": 8,
        "max_iter": 350,
        "min_samples_leaf": 30,
        "l2_regularization": 1.0,
        "early_stopping": False,
        "random_state": 42,
    }


def make_pipeline(
    categorical_features: list[str],
    numeric_features: list[str],
    horizon: int,
) -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "encoder",
                            OrdinalEncoder(
                                handle_unknown="use_encoded_value",
                                unknown_value=-1,
                                encoded_missing_value=-1,
                            ),
                        ),
                    ]
                ),
                categorical_features,
            ),
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("sanitize", FunctionTransformer(replace_inf_with_nan)),
                        ("imputer", SimpleImputer(strategy="median")),
                    ]
                ),
                numeric_features,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    model = HistGradientBoostingRegressor(**model_params_for_horizon(horizon))

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )


def sample_training_rows(
    train_frame: pd.DataFrame,
    max_rows: int | None,
    random_state: int,
) -> pd.DataFrame:
    if max_rows is None or len(train_frame) <= max_rows:
        return train_frame
    return train_frame.sample(n=max_rows, random_state=random_state)


def safe_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = y_true > 1e-6
    if not np.any(mask):
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0)


def safe_wape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = float(np.sum(np.abs(y_true)))
    if denom <= 1e-6:
        return float("nan")
    return float(np.sum(np.abs(y_true - y_pred)) / denom * 100.0)


def train_models(data: pd.DataFrame, config: Config) -> list[dict]:
    config.output_dir.mkdir(parents=True, exist_ok=True)

    categorical_features = ["Commodity", "State", "District", "Market", "series_id"]
    numeric_features = [
        "latitude",
        "longitude",
        "Arrival_Quantity",
        "Modal_Price_CausalFilled",
        "Arrival_Quantity_CausalFilled",
        "arrival_log1p",
        "Modal_Price",
        "price_log1p",
        "price_filled_log1p",
        "temp_mean",
        "temp_min",
        "temp_max",
        "temp_range",
        "rain_sum",
        "solar_sum",
        "solar_peak",
        "rh_mean",
        "rh_min",
        "rh_max",
        "state_price_mean",
        "state_arrival_mean",
        "national_price_mean",
        "national_arrival_mean",
        "month",
        "day_of_week_num",
        "day_of_year",
        "week_of_year",
        "is_month_start",
        "is_month_end",
        "day_of_year_sin",
        "day_of_year_cos",
        "day_of_week_sin",
        "day_of_week_cos",
        "price_dow_ratio",
        "arrival_dow_ratio",
        "price_lag_1",
        "price_lag_7",
        "price_lag_14",
        "price_lag_28",
        "price_lag_56",
        "price_lag_84",
        "price_lag_112",
        "price_lag_168",
        "arrival_lag_1",
        "arrival_lag_7",
        "arrival_lag_14",
        "arrival_lag_28",
        "arrival_lag_56",
        "arrival_lag_84",
        "arrival_lag_112",
        "arrival_lag_168",
        "price_roll_mean_7",
        "price_roll_mean_14",
        "price_roll_mean_21",
        "price_roll_mean_28",
        "price_roll_mean_56",
        "price_roll_mean_84",
        "price_roll_mean_112",
        "price_roll_mean_168",
        "price_roll_std_7",
        "price_roll_std_14",
        "price_roll_std_28",
        "price_roll_std_56",
        "price_roll_std_84",
        "price_roll_std_112",
        "price_roll_std_168",
        "arrival_roll_mean_7",
        "arrival_roll_mean_14",
        "arrival_roll_mean_21",
        "arrival_roll_mean_28",
        "arrival_roll_mean_56",
        "arrival_roll_mean_84",
        "arrival_roll_mean_112",
        "arrival_roll_mean_168",
        "price_roll_min_28",
        "price_roll_max_28",
        "price_roll_min_56",
        "price_roll_max_56",
        "price_roll_min_84",
        "price_roll_max_84",
        "price_roll_min_168",
        "price_roll_max_168",
        "price_vs_roll7",
        "price_vs_roll28",
        "price_vs_roll84",
        "price_vs_roll168",
        "price_vs_state_mean",
        "price_vs_national_mean",
        "arrival_vs_roll28",
        "arrival_vs_state_mean",
        "arrival_vs_national_mean",
        "price_minus_state_mean",
        "price_minus_national_mean",
        "price_trend_7_28",
        "price_trend_14_56",
        "price_trend_28_84",
        "price_trend_28_168",
        "arrival_trend_7_28",
        "arrival_trend_28_84",
        "price_range_28",
        "price_range_84",
        "price_range_168",
        "price_volatility_ratio_28",
        "price_volatility_ratio_84",
        "price_volatility_ratio_168",
    ]

    max_date = data["Date"].max()
    validation_start = max_date - pd.Timedelta(days=config.validation_days - 1)

    metrics: list[dict] = []
    for horizon in config.horizons:
        target_col = f"target_{horizon}d"
        horizon_frame = data[data[target_col].notna()].copy()
        if horizon_frame.empty:
            print(f"Skipping horizon {horizon}: no rows with non-null targets.")
            continue

        train_frame = horizon_frame[horizon_frame["Date"] < validation_start].copy()
        val_frame = horizon_frame[horizon_frame["Date"] >= validation_start].copy()
        if train_frame.empty or val_frame.empty:
            print(f"Skipping horizon {horizon}: train or validation split is empty.")
            continue

        train_frame = sample_training_rows(
            train_frame,
            config.max_train_rows_per_horizon,
            config.random_state + horizon,
        )

        X_train = train_frame[categorical_features + numeric_features]
        y_train = np.log1p(train_frame[target_col].to_numpy())

        X_val = val_frame[categorical_features + numeric_features]
        y_val = val_frame[target_col].to_numpy()

        pipeline = make_pipeline(categorical_features, numeric_features, horizon)
        print(
            f"Training horizon {horizon}d with {len(train_frame):,} train rows "
            f"and {len(val_frame):,} validation rows ..."
        )
        pipeline.fit(X_train, y_train)

        pred_log = pipeline.predict(X_val)
        preds = np.expm1(pred_log)
        preds = np.clip(preds, a_min=0.0, a_max=None)

        horizon_metrics = {
            "horizon_days": horizon,
            "train_rows": int(len(train_frame)),
            "validation_rows": int(len(val_frame)),
            "validation_start": validation_start.strftime("%Y-%m-%d"),
            "validation_end": max_date.strftime("%Y-%m-%d"),
            "mae": float(mean_absolute_error(y_val, preds)),
            "rmse": float(np.sqrt(mean_squared_error(y_val, preds))),
            "r2": float(r2_score(y_val, preds)),
            "mape_pct": safe_mape(y_val, preds),
            "wape_pct": safe_wape(y_val, preds),
        }
        metrics.append(horizon_metrics)

        artifact = {
            "model": pipeline,
            "categorical_features": categorical_features,
            "numeric_features": numeric_features,
            "horizon_days": horizon,
            "validation_start": validation_start.strftime("%Y-%m-%d"),
            "validation_end": max_date.strftime("%Y-%m-%d"),
        }
        model_path = config.output_dir / f"global_price_model_{horizon}d.joblib"
        joblib.dump(artifact, model_path)
        print(
            f"Saved {model_path} | "
            f"MAE={horizon_metrics['mae']:.2f} RMSE={horizon_metrics['rmse']:.2f} "
            f"WAPE={horizon_metrics['wape_pct']:.2f}%"
        )

    return metrics


def main() -> None:
    config = parse_args()
    data = load_training_frame(config)
    metrics = train_models(data, config)

    summary = {
        "data_dir": str(config.data_dir),
        "output_dir": str(config.output_dir),
        "horizons": config.horizons,
        "validation_days": config.validation_days,
        "min_series_observations": config.min_series_observations,
        "dense_min_pct": config.dense_min_pct,
        "max_train_rows_per_horizon": config.max_train_rows_per_horizon,
        "metrics": metrics,
    }
    summary_path = config.output_dir / "metrics.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote metrics summary to {summary_path}")


if __name__ == "__main__":
    main()
