import numpy as np
import pandas as pd
from src.run_part1 import evaluate_split


def test_evaluate_split_returns_metrics():
    rng = np.random.default_rng(0)
    n = 60
    X_apt = rng.normal(size=(n, 6)).astype("float32")
    X_prot = rng.normal(size=(n, 5)).astype("float32")
    y = (X_apt[:, 0] + X_prot[:, 0]).astype("float32")
    is_test = np.array([i % 3 == 0 for i in range(n)])
    from src.config import load_config
    cfg = load_config("configs/part1.yaml")
    cfg.nn["max_epochs"] = 30
    rows = evaluate_split(X_apt, X_prot, y, is_test, "smoke", cfg)
    models = {r["model"] for r in rows}
    assert models == {"xgboost", "twotower"}
    for r in rows:
        assert set(r) >= {"model", "split", "rmse", "pearson", "n_train", "n_test"}
