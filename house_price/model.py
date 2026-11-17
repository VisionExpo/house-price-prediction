"""Model training and inference helpers based on the completed notebook."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "extracted" / "Data" / "data.csv"
TARGET = "SalePrice"
INPUT_FEATURES = (
    "GrLivArea",
    "LotArea",
    "OverallQual",
    "YearBuilt",
    "BedroomAbvGr",
    "FullBath",
    "GarageCars",
    "Neighborhood",
)

VALIDATION_METRICS = {
    "Linear Regression": {"mae": 20466.13, "rmse": 31294.99, "r2": 0.8723},
    "Random Forest": {"mae": 17350.76, "rmse": 28376.65, "r2": 0.8950},
    "Gradient Boosting": {"mae": 16946.24, "rmse": 27667.91, "r2": 0.9002},
}

# Rounded directly from the notebook's fitted Gradient Boosting feature-importance table.
FEATURE_IMPORTANCE = [
    {"feature": "OverallQual", "importance": 0.51},
    {"feature": "GrLivArea", "importance": 0.15},
    {"feature": "GarageCars", "importance": 0.05},
    {"feature": "TotalBsmtSF", "importance": 0.04},
]


def available_neighborhoods() -> list[str]:
    """Return the neighborhoods present in the project's Ames dataset."""
    if not DATA_PATH.exists():
        return []
    values = pd.read_csv(DATA_PATH, usecols=["Neighborhood"])["Neighborhood"]
    return sorted(values.dropna().astype(str).unique().tolist())


class HousePriceModel:
    """Notebook-aligned pipeline and training-split defaults."""

    def __init__(
        self,
        pipeline: Pipeline,
        defaults: dict[str, Any],
        feature_columns: list[str],
        neighborhoods: list[str],
    ) -> None:
        self.pipeline = pipeline
        self.defaults = defaults
        self.feature_columns = feature_columns
        self.neighborhoods = neighborhoods

    def predict(self, supplied_features: dict[str, Any]) -> float:
        row = self.defaults.copy()
        row.update(supplied_features)
        frame = pd.DataFrame([row], columns=self.feature_columns)
        return float(self.pipeline.predict(frame)[0])


@lru_cache(maxsize=1)
def get_model() -> HousePriceModel:
    """Fit the notebook's model once, in memory, when the first prediction is requested."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Ames dataset not found at {DATA_PATH}. See README.md for setup instructions."
        )

    frame = pd.read_csv(DATA_PATH)
    target = frame[TARGET]
    features = frame.drop(columns=[TARGET, "Id"])
    x_train, _, y_train, _ = train_test_split(
        features,
        target,
        test_size=0.20,
        random_state=42,
    )

    numeric_features = x_train.select_dtypes(include=np.number).columns.tolist()
    categorical_features = x_train.select_dtypes(
        include=["object", "category", "str"]
    ).columns.tolist()

    numeric_transformer = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median"))]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "model",
                GradientBoostingRegressor(
                    n_estimators=200,
                    learning_rate=0.05,
                    max_depth=3,
                    loss="squared_error",
                    random_state=42,
                ),
            ),
        ]
    )
    pipeline.fit(x_train, y_train)

    defaults = x_train.median(numeric_only=True).to_dict()
    categorical_defaults = x_train[categorical_features].mode(dropna=True).iloc[0]
    defaults.update(categorical_defaults.to_dict())

    return HousePriceModel(
        pipeline=pipeline,
        defaults=defaults,
        feature_columns=features.columns.tolist(),
        neighborhoods=sorted(
            features["Neighborhood"].dropna().astype(str).unique().tolist()
        ),
    )
