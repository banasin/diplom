import numpy as np
from src.models.baseline import train_baseline, predict_baseline


def test_baseline_overfits_small():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 8))
    y = X[:, 0] * 2 - X[:, 1]                   # простая линейная зависимость
    params = {"n_estimators": 200, "max_depth": 4, "learning_rate": 0.1}
    model = train_baseline(X, y, params, seed=42)
    pred = predict_baseline(model, X)
    rmse = np.sqrt(np.mean((pred - y) ** 2))
    assert rmse < 0.5                           # должен хорошо запомнить train
