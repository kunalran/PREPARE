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
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OrdinalEncoder


@dataclass(frozen=True)
class Config:
    data_path: Path
    output_dir: Path
    horizons: list[int]
    validation_days: int
    min_series_observations: int
    max_train_rows_per_horizon: int | None
    random_state: int
    anchor_col: str
    blend_step: float


def parse_args() -> Config:
    parser = argparse.ArgumentParser(
        description=(
            "Train a tomato-specific daily forecasting model using reference "
            "DOW-ratio imputation and a blended residual formulation."
        )
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("final_data/agmarknet_tomato_data_final.csv"),
        help="Path to the tomato daily CSV in final_data.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("models/per_crop_histgb_targeted_v2"),
        help="Directory where models and metrics will be saved.",
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
        help="Drop mandi series with fewer observed tomato prices than this.",
    )
    parser.add_argument(
        "--max-train-rows-per-horizon",
        type=int,
        default=250000,
        help="Optional cap for training rows per horizon.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--anchor-col",
        choices=("price_roll_mean_7", "price_roll_mean_14", "price_roll_mean_28", "price_roll_mean_56", "price_roll_mean_84", "price_roll_mean_168", "Modal_Price_DOWFilled"),
        default="price_roll_mean_28",
        help="Residual anchor to learn corrections around.",
    )
    parser.add_argument(
        "--blend-step",
        type=float,
        default=0.05,
        help="Grid step for validation blend alpha search.",
    )
    args = parser.parse_args()
    horizons = sorted({int(x.strip()) for x in args.horizons.split(",") if x.strip()})
    return Config(
        data_path=args.data_path,
        output_dir=args.output_dir,
        horizons=horizons,
        validation_days=args.validation_days,
        min_series_observations=args.min_series_observations,
        max_train_rows_per_horizon=args.max_train_rows_per_horizon,
        random_state=args.random_state,
        anchor_col=args.anchor_col,
        blend_step=args.blend_step,
    )


def replace_inf_with_nan(frame):
    if isinstance(frame, pd.DataFrame):
        return frame.replace([np.inf, -np.inf], np.nan)
    arr = np.asarray(frame, dtype=float)
    arr = arr.copy()
    arr[~np.isfinite(arr)] = np.nan
    return arr


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


def sample_training_rows(
    train_frame: pd.DataFrame,
    max_rows: int | None,
    random_state: int,
) -> pd.DataFrame:
    if max_rows is None or len(train_frame) <= max_rows:
        return train_frame
    return train_frame.sample(n=max_rows, random_state=random_state)


def model_params_for_horizon(horizon: int) -> dict[str, float | int | bool | str]:
    if horizon >= 12:
        return {
            "loss": "absolute_error",
            "learning_rate": 0.03,
            "max_depth": 10,
            "max_iter": 700,
            "min_samples_leaf": 15,
            "l2_regularization": 0.1,
            "early_stopping": False,
            "random_state": 42,
        }
    if horizon >= 7:
        return {
            "loss": "squared_error",
            "learning_rate": 0.035,
            "max_depth": 8,
            "max_iter": 500,
            "min_samples_leaf": 20,
            "l2_regularization": 0.3,
            "early_stopping": False,
            "random_state": 42,
        }
    return {
        "loss": "squared_error",
        "learning_rate": 0.04,
        "max_depth": 8,
        "max_iter": 350,
        "min_samples_leaf": 25,
        "l2_regularization": 0.5,
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
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", HistGradientBoostingRegressor(**model_params_for_horizon(horizon))),
        ]
    )


def impute_dow_ratio(df: pd.DataFrame, col: str) -> pd.Series:
    """
    Reference DOW-ratio imputation from dataimputationref/apply_dow_ratio_imputation.py.

    This is intentionally centered because the user asked for the tomato
    imputation strategy from the DOW-ratio script rather than the causal
    forecasting-time approximation used elsewhere in the repo.
    """
    result = df[col].copy()
    df_work = df[["State", "Market", "Date", col]].copy()
    df_work["_DayOfWeek"] = pd.to_datetime(df_work["Date"]).dt.dayofweek

    for _, idx in df_work.groupby(["State", "Market"]).groups.items():
        grp = df_work.loc[idx].sort_values("Date")
        series = grp[col]
        dow = grp["_DayOfWeek"]

        overall_mean = series.mean()
        if pd.isna(overall_mean) or overall_mean == 0:
            continue

        dow_ratios = (series.groupby(dow).mean() / overall_mean).fillna(1.0)
        rolling_ctx = series.rolling(7, center=True, min_periods=1).mean()

        filled = series.copy()
        for ix in series.index[series.isna()]:
            ctx = rolling_ctx.loc[ix]
            if pd.notna(ctx):
                filled.loc[ix] = ctx * dow_ratios.get(dow.loc[ix], 1.0)

        result.loc[grp.index] = filled.values

    return result


def feature_columns() -> tuple[list[str], list[str]]:
    categorical = ["State", "District", "Market", "series_id"]
    numeric = [
        "latitude",
        "longitude",
        "t",
        "tp",
        "ssr",
        "r",
        "Arrival_Quantity",
        "Arrival_Quantity_DOWFilled",
        "Modal_Price_DOWFilled",
        "modal_price_missing",
        "arrival_missing",
        "state_price_mean",
        "national_price_mean",
        "state_arrival_mean",
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
        "price_lag_1",
        "price_lag_7",
        "price_lag_14",
        "price_lag_21",
        "price_lag_28",
        "price_lag_56",
        "price_lag_84",
        "price_lag_112",
        "price_lag_168",
        "arrival_lag_1",
        "arrival_lag_7",
        "arrival_lag_14",
        "arrival_lag_21",
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
        "arrival_roll_mean_7",
        "arrival_roll_mean_14",
        "arrival_roll_mean_21",
        "arrival_roll_mean_28",
        "arrival_roll_mean_56",
        "arrival_roll_mean_84",
        "arrival_roll_mean_112",
        "arrival_roll_mean_168",
        "price_roll_std_7",
        "price_roll_std_14",
        "price_roll_std_28",
        "price_roll_std_56",
        "price_roll_std_84",
        "price_roll_std_168",
        "price_vs_roll14",
        "price_vs_roll28",
        "price_vs_state_mean",
        "price_vs_national_mean",
        "price_minus_state_mean",
        "price_minus_national_mean",
        "price_trend_7_28",
        "price_trend_14_56",
        "arrival_trend_7_28",
        "arrival_trend_28_84",
    ]
    return categorical, numeric


def filter_series(frame: pd.DataFrame, min_series_observations: int) -> pd.DataFrame:
    counts = frame.groupby("series_id")["Modal_Price"].count()
    valid_series = counts[counts >= min_series_observations].index
    filtered = frame[frame["series_id"].isin(valid_series)].copy()
    print(
        f"Kept {len(valid_series):,} tomato series with at least "
        f"{min_series_observations} observed prices."
    )
    print(f"Rows after series filter: {len(filtered):,}")
    return filtered


def ratio_feature(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return (numerator / denominator).replace([np.inf, -np.inf], np.nan)


def load_and_engineer_tomato(path: Path, horizons: list[int]) -> pd.DataFrame:
    usecols = [
        "State",
        "District",
        "Market",
        "Commodity",
        "Date",
        "Arrival_Quantity",
        "Modal_Price",
        "latitude",
        "longitude",
        "t",
        "tp",
        "ssr",
        "r",
    ]
    df = pd.read_csv(path, usecols=usecols, parse_dates=["Date"], low_memory=False)
    df["series_id"] = (
        df["State"].astype(str)
        + "||"
        + df["District"].astype(str)
        + "||"
        + df["Market"].astype(str)
    )
    df = df.sort_values(["series_id", "Date"]).reset_index(drop=True)

    print("Applying reference DOW-ratio imputation for tomato price and arrival ...")
    df["Modal_Price_DOWFilled"] = impute_dow_ratio(df, "Modal_Price")
    df["Arrival_Quantity_DOWFilled"] = impute_dow_ratio(df, "Arrival_Quantity")
    df["modal_price_missing"] = df["Modal_Price"].isna().astype(np.int8)
    df["arrival_missing"] = df["Arrival_Quantity"].isna().astype(np.int8)

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

    grouped_price = df.groupby("series_id", sort=False)["Modal_Price_DOWFilled"]
    grouped_arrival = df.groupby("series_id", sort=False)["Arrival_Quantity_DOWFilled"]
    grouped_observed_price = df.groupby("series_id", sort=False)["Modal_Price"]

    state_day = df.groupby(["State", "Date"], sort=False)
    national_day = df.groupby(["Date"], sort=False)
    df["state_price_mean"] = state_day["Modal_Price_DOWFilled"].transform("mean")
    df["national_price_mean"] = national_day["Modal_Price_DOWFilled"].transform("mean")
    df["state_arrival_mean"] = state_day["Arrival_Quantity_DOWFilled"].transform("mean")
    df["national_arrival_mean"] = national_day["Arrival_Quantity_DOWFilled"].transform("mean")

    first_dates = df.groupby("series_id", sort=False)["Date"].transform("min")
    df["series_age_days"] = (df["Date"] - first_dates).dt.days.astype(np.int32)

    for lag in (1, 7, 14, 21, 28, 56, 84, 112, 168):
        df[f"price_lag_{lag}"] = grouped_price.shift(lag)
        df[f"arrival_lag_{lag}"] = grouped_arrival.shift(lag)

    for window in (7, 14, 21, 28, 56, 84, 112, 168):
        df[f"price_roll_mean_{window}"] = grouped_price.transform(
            lambda s: s.shift(1).rolling(window=window, min_periods=1).mean()
        )
        df[f"arrival_roll_mean_{window}"] = grouped_arrival.transform(
            lambda s: s.shift(1).rolling(window=window, min_periods=1).mean()
        )

    for window in (7, 14, 28, 56, 84, 168):
        df[f"price_roll_std_{window}"] = grouped_price.transform(
            lambda s: s.shift(1).rolling(window=window, min_periods=2).std()
        )

    df["price_vs_roll14"] = ratio_feature(
        df["Modal_Price_DOWFilled"], df["price_roll_mean_14"]
    )
    df["price_vs_roll28"] = ratio_feature(
        df["Modal_Price_DOWFilled"], df["price_roll_mean_28"]
    )
    df["price_vs_state_mean"] = ratio_feature(
        df["Modal_Price_DOWFilled"], df["state_price_mean"]
    )
    df["price_vs_national_mean"] = ratio_feature(
        df["Modal_Price_DOWFilled"], df["national_price_mean"]
    )
    df["price_minus_state_mean"] = (
        df["Modal_Price_DOWFilled"] - df["state_price_mean"]
    )
    df["price_minus_national_mean"] = (
        df["Modal_Price_DOWFilled"] - df["national_price_mean"]
    )
    df["price_trend_7_28"] = df["price_roll_mean_7"] - df["price_roll_mean_28"]
    df["price_trend_14_56"] = df["price_roll_mean_14"] - df["price_roll_mean_56"]
    df["arrival_trend_7_28"] = df["arrival_roll_mean_7"] - df["arrival_roll_mean_28"]
    df["arrival_trend_28_84"] = (
        df["arrival_roll_mean_28"] - df["arrival_roll_mean_84"]
    )

    for horizon in horizons:
        df[f"target_{horizon}d"] = grouped_observed_price.shift(-horizon)

    keep_columns = [
        "State",
        "District",
        "Market",
        "Commodity",
        "series_id",
        "Date",
        "Arrival_Quantity",
        "Modal_Price",
        "latitude",
        "longitude",
        "t",
        "tp",
        "ssr",
        "r",
        "Modal_Price_DOWFilled",
        "Arrival_Quantity_DOWFilled",
        "modal_price_missing",
        "arrival_missing",
        "month",
        "day_of_week_num",
        "day_of_year",
        "week_of_year",
        "year",
        "is_month_start",
        "is_month_end",
        "day_of_year_sin",
        "day_of_year_cos",
        "day_of_week_sin",
        "day_of_week_cos",
        "series_age_days",
        "state_price_mean",
        "national_price_mean",
        "state_arrival_mean",
        "national_arrival_mean",
        "price_lag_1",
        "price_lag_7",
        "price_lag_14",
        "price_lag_21",
        "price_lag_28",
        "price_lag_56",
        "price_lag_84",
        "price_lag_112",
        "price_lag_168",
        "arrival_lag_1",
        "arrival_lag_7",
        "arrival_lag_14",
        "arrival_lag_21",
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
        "arrival_roll_mean_7",
        "arrival_roll_mean_14",
        "arrival_roll_mean_21",
        "arrival_roll_mean_28",
        "arrival_roll_mean_56",
        "arrival_roll_mean_84",
        "arrival_roll_mean_112",
        "arrival_roll_mean_168",
        "price_roll_std_7",
        "price_roll_std_14",
        "price_roll_std_28",
        "price_roll_std_56",
        "price_roll_std_84",
        "price_roll_std_168",
        "price_vs_roll14",
        "price_vs_roll28",
        "price_vs_state_mean",
        "price_vs_national_mean",
        "price_minus_state_mean",
        "price_minus_national_mean",
        "price_trend_7_28",
        "price_trend_14_56",
        "arrival_trend_7_28",
        "arrival_trend_28_84",
    ] + [f"target_{horizon}d" for horizon in horizons]
    return df.loc[:, keep_columns].copy()


def tune_blend(
    y_true: np.ndarray,
    anchor_preds: np.ndarray,
    model_preds: np.ndarray,
    blend_step: float,
) -> tuple[float, np.ndarray, float]:
    best_alpha = 0.0
    best_preds = anchor_preds
    best_r2 = r2_score(y_true, anchor_preds)
    alphas = np.arange(0.0, 1.0 + blend_step / 2.0, blend_step)
    for alpha in alphas:
        preds = alpha * model_preds + (1.0 - alpha) * anchor_preds
        score = r2_score(y_true, preds)
        if score > best_r2:
            best_r2 = score
            best_alpha = float(alpha)
            best_preds = preds
    return best_alpha, best_preds, best_r2


def train_models(data: pd.DataFrame, config: Config) -> list[dict]:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    categorical_features, numeric_features = feature_columns()

    max_date = data["Date"].max()
    validation_start = max_date - pd.Timedelta(days=config.validation_days - 1)
    metrics: list[dict] = []

    for horizon in config.horizons:
        target_col = f"target_{horizon}d"
        horizon_frame = data[data[target_col].notna()].copy()
        if horizon_frame.empty:
            print(f"Skipping tomato {horizon}d: no rows with non-null targets.")
            continue

        train_frame = horizon_frame[horizon_frame["Date"] < validation_start].copy()
        val_frame = horizon_frame[horizon_frame["Date"] >= validation_start].copy()
        if train_frame.empty or val_frame.empty:
            print(f"Skipping tomato {horizon}d: train or validation split is empty.")
            continue

        train_frame = sample_training_rows(
            train_frame,
            config.max_train_rows_per_horizon,
            config.random_state + horizon,
        )

        train_frame = train_frame.copy()
        val_frame = val_frame.copy()
        train_frame["_anchor"] = train_frame[config.anchor_col].fillna(
            train_frame["Modal_Price_DOWFilled"]
        )
        val_frame["_anchor"] = val_frame[config.anchor_col].fillna(
            val_frame["Modal_Price_DOWFilled"]
        )
        train_frame = train_frame[train_frame["_anchor"].notna()].copy()
        val_frame = val_frame[val_frame["_anchor"].notna()].copy()
        if train_frame.empty or val_frame.empty:
            print(f"Skipping tomato {horizon}d: no rows with a usable anchor.")
            continue

        anchor_train_log = np.log1p(
            np.clip(train_frame["_anchor"].to_numpy(), a_min=0.0, a_max=None)
        )
        anchor_val_log = np.log1p(
            np.clip(val_frame["_anchor"].to_numpy(), a_min=0.0, a_max=None)
        )

        X_train = train_frame[categorical_features + numeric_features]
        X_val = val_frame[categorical_features + numeric_features]
        y_train = np.log1p(train_frame[target_col].to_numpy()) - anchor_train_log
        y_val = val_frame[target_col].to_numpy()

        model = make_pipeline(categorical_features, numeric_features, horizon)
        print(
            f"Training tomato {horizon}d around {config.anchor_col} with "
            f"{len(train_frame):,} train rows and {len(val_frame):,} validation rows ..."
        )
        model.fit(X_train, y_train)

        pred_residual = model.predict(X_val)
        model_only_preds = np.expm1(pred_residual + anchor_val_log)
        model_only_preds = np.clip(model_only_preds, a_min=0.0, a_max=None)
        anchor_preds = np.clip(val_frame["_anchor"].to_numpy(), a_min=0.0, a_max=None)

        best_alpha, blended_preds, best_r2 = tune_blend(
            y_val,
            anchor_preds,
            model_only_preds,
            config.blend_step,
        )

        row = {
            "crop": "tomato",
            "horizon_days": horizon,
            "anchor_col": config.anchor_col,
            "best_alpha": best_alpha,
            "train_rows": int(len(train_frame)),
            "validation_rows": int(len(val_frame)),
            "validation_start": validation_start.strftime("%Y-%m-%d"),
            "validation_end": max_date.strftime("%Y-%m-%d"),
            "anchor_r2": float(r2_score(y_val, anchor_preds)),
            "model_only_r2": float(r2_score(y_val, model_only_preds)),
            "r2": float(best_r2),
            "mae": float(mean_absolute_error(y_val, blended_preds)),
            "rmse": float(np.sqrt(mean_squared_error(y_val, blended_preds))),
            "mape_pct": safe_mape(y_val, blended_preds),
            "wape_pct": safe_wape(y_val, blended_preds),
        }
        metrics.append(row)

        artifact = {
            "model": model,
            "categorical_features": categorical_features,
            "numeric_features": numeric_features,
            "crop": "tomato",
            "horizon_days": horizon,
            "anchor_col": config.anchor_col,
            "best_alpha": best_alpha,
            "validation_start": row["validation_start"],
            "validation_end": row["validation_end"],
            "metrics": row,
        }
        model_path = config.output_dir / f"tomato_model_{horizon}d.joblib"
        joblib.dump(artifact, model_path)
        print(
            f"  tomato {horizon}d: anchor_r2={row['anchor_r2']:.4f}, "
            f"model_r2={row['model_only_r2']:.4f}, blend_r2={row['r2']:.4f}, "
            f"alpha={best_alpha:.2f}"
        )

    return metrics


def main() -> None:
    config = parse_args()
    if not config.data_path.exists():
        raise FileNotFoundError(f"Tomato data file not found: {config.data_path}")

    print(f"Loading tomato daily data from {config.data_path} ...")
    data = load_and_engineer_tomato(config.data_path, config.horizons)
    data = filter_series(data, config.min_series_observations)
    metrics = train_models(data, config)

    summary = {
        "data_path": str(config.data_path),
        "anchor_col": config.anchor_col,
        "blend_step": config.blend_step,
        "metrics": metrics,
    }
    summary_path = config.output_dir / "tomato_training_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"Saved tomato summary to {summary_path}")


if __name__ == "__main__":
    main()
