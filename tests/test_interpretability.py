import numpy as np
from xgboost import XGBRegressor
from src.interpretability import shap_feature_importance


def test_shap_importance_shape_and_group():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 4))
    y = X[:, 0] * 2                                  # важен только признак 0
    model = XGBRegressor(n_estimators=30, max_depth=3).fit(X, y)
    names = ["AAAA", "structure_frac_paired", "esm_0", "other_x"]
    df = shap_feature_importance(model, X, names, top_k=4)
    assert list(df.columns) == ["feature", "mean_abs_shap", "group"]
    assert len(df) == 4
    assert df["feature"].iloc[0] == "AAAA"          # самый важный
    groups = dict(zip(df["feature"], df["group"]))
    assert groups["AAAA"] == "kmer"
    assert groups["structure_frac_paired"] == "structure"
    assert groups["esm_0"] == "esm"
    assert groups["other_x"] == "other"
