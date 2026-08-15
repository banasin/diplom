import numpy as np
from src.evaluation.metrics import regression_metrics


def test_perfect_prediction():
    y = np.array([-7.0, -8.0, -9.0, -6.0])
    m = regression_metrics(y, y)
    assert abs(m["rmse"]) < 1e-9
    assert abs(m["mae"]) < 1e-9
    assert abs(m["pearson"] - 1.0) < 1e-9
    assert abs(m["spearman"] - 1.0) < 1e-9
    assert abs(m["r2"] - 1.0) < 1e-9


def test_known_rmse():
    y = np.array([0.0, 0.0])
    p = np.array([1.0, -1.0])
    assert abs(regression_metrics(y, p)["rmse"] - 1.0) < 1e-9
