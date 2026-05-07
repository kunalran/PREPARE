from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor


class FeatureBaselineRegressor:
    def __init__(
        self,
        feature_name: str,
        *,
        clip_min: float | None = 0.0,
    ) -> None:
        self.feature_name = feature_name
        self.clip_min = clip_min
        self.fallback_value_: float = 0.0

    def fit(self, X: pd.DataFrame, y: pd.Series | np.ndarray | None = None) -> "FeatureBaselineRegressor":
        feature = pd.to_numeric(X[self.feature_name], errors="coerce")
        finite = feature[np.isfinite(feature)]
        if len(finite) > 0:
            self.fallback_value_ = float(np.median(finite))
        elif y is not None:
            target = pd.to_numeric(pd.Series(y), errors="coerce")
            finite_target = target[np.isfinite(target)]
            self.fallback_value_ = float(np.median(finite_target)) if len(finite_target) > 0 else 0.0
        else:
            self.fallback_value_ = 0.0
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        values = pd.to_numeric(X[self.feature_name], errors="coerce").to_numpy(dtype=float, copy=True)
        missing = ~np.isfinite(values)
        if missing.any():
            values[missing] = self.fallback_value_
        if self.clip_min is not None:
            values = np.clip(values, a_min=self.clip_min, a_max=None)
        return values


class NumericSubsetHistGBRegressor:
    def __init__(
        self,
        feature_names: list[str],
        *,
        learning_rate: float = 0.03,
        max_depth: int = 6,
        max_iter: int = 400,
        min_samples_leaf: int = 20,
        l2_regularization: float = 0.3,
        clip_min: float | None = 0.0,
    ) -> None:
        self.feature_names = feature_names
        self.clip_min = clip_min
        self.medians_: dict[str, float] = {}
        self.model = HistGradientBoostingRegressor(
            loss="squared_error",
            learning_rate=learning_rate,
            max_depth=max_depth,
            max_iter=max_iter,
            min_samples_leaf=min_samples_leaf,
            l2_regularization=l2_regularization,
            early_stopping=False,
            random_state=42,
        )

    def _prepare(self, X: pd.DataFrame) -> pd.DataFrame:
        frame = X[self.feature_names].apply(pd.to_numeric, errors="coerce")
        frame = frame.replace([np.inf, -np.inf], np.nan)
        if self.medians_:
            frame = frame.fillna(self.medians_)
        return frame

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series | np.ndarray,
    ) -> "NumericSubsetHistGBRegressor":
        frame = self._prepare(X)
        self.medians_ = frame.median().to_dict()
        frame = frame.fillna(self.medians_)
        self.model.fit(frame, np.asarray(y, dtype=float))
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        frame = self._prepare(X)
        values = self.model.predict(frame)
        if self.clip_min is not None:
            values = np.clip(values, a_min=self.clip_min, a_max=None)
        return values
