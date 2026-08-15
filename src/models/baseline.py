"""Классический бейзлайн: XGBoost-регрессия на объединённых признаках
аптамера и белка — планка качества для нейросети."""
import numpy as np
from xgboost import XGBRegressor


def train_baseline(X_train, y_train, params: dict, seed: int) -> XGBRegressor:
    model = XGBRegressor(
        objective="reg:squarederror",
        random_state=seed,
        n_jobs=0,
        **params,
    )
    model.fit(X_train, y_train)
    return model


def predict_baseline(model: XGBRegressor, X) -> np.ndarray:
    return model.predict(X)
