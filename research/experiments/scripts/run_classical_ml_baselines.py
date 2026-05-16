from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, LinearRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from prepare_experiments import (  # type: ignore
    ensure_venv,
    experiment_config,
    load_crop_frames,
    metric_row,
    sample_training_rows,
    save_results,
    set_local_runtime_dirs,
    simple_numeric_features,
    split_train_val,
)


def build_models(random_state: int) -> dict[str, Pipeline]:
    common = [
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]
    return {
        "linear_regression": Pipeline(common + [("model", LinearRegression())]),
        "ridge_alpha_10": Pipeline(common + [("model", Ridge(alpha=10.0, random_state=random_state))]),
        "elasticnet_a001_l05": Pipeline(
            common
            + [
                (
                    "model",
                    ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=5000, random_state=random_state),
                )
            ]
        ),
    }


def run_classical_ml() -> Path:
    ensure_venv()
    cfg = experiment_config()
    set_local_runtime_dirs(cfg.output_root)
    frames = load_crop_frames(cfg)
    features = simple_numeric_features()
    horizons = (1, 15)
    rows: list[dict[str, object]] = []

    for crop, frame in sorted(frames.items()):
        for horizon in horizons:
            train_frame, val_frame, target_col, validation_start, validation_end = split_train_val(
                frame,
                horizon,
                cfg.validation_days,
            )
            if train_frame.empty or val_frame.empty:
                continue
            train_frame = sample_training_rows(train_frame, cfg.max_train_rows_simple, cfg.random_state + horizon)
            x_train = train_frame[features]
            y_train = train_frame[target_col].to_numpy(dtype=float)
            x_val = val_frame[features]
            y_val = val_frame[target_col].to_numpy(dtype=float)

            for model_name, pipeline in build_models(cfg.random_state + horizon).items():
                pipeline.fit(x_train, y_train)
                preds = np.clip(pipeline.predict(x_val), a_min=0.0, a_max=None)
                rows.append(
                    metric_row(
                        experiment_family="classical_ml",
                        experiment_name=model_name,
                        crop=crop,
                        horizon=horizon,
                        validation_rows=len(val_frame),
                        validation_start=validation_start,
                        validation_end=validation_end,
                        y_true=y_val,
                        preds=preds,
                        extra={"train_rows": int(len(train_frame)), "feature_count": len(features)},
                    )
                )

    return save_results(cfg, "classical_ml_metrics", rows)


if __name__ == "__main__":
    output = run_classical_ml()
    print(f"Completed classical ML sweep: {output}")
