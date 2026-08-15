"""Механистическая интерпретация вклада признаков через TreeSHAP (Часть 4).
Значения SHAP вычисляются встроенным в XGBoost `pred_contribs` (алгоритм TreeSHAP,
эквивалент shap.TreeExplainer) — надёжно и без проблем совместимости версий."""
import numpy as np
import pandas as pd
import xgboost as xgb

_ALPHABET = set("ACGT")


def _group_of(name: str) -> str:
    if name.startswith("structure_"):
        return "structure"
    if name.startswith("esm_"):
        return "esm"
    if name and set(name) <= _ALPHABET:
        return "kmer"
    return "other"


def shap_feature_importance(model, X, feature_names, top_k: int = 30) -> pd.DataFrame:
    booster = model.get_booster()
    contribs = booster.predict(xgb.DMatrix(np.asarray(X)), pred_contribs=True)
    # contribs: (n, n_features + 1); последний столбец — свободный член (bias)
    mean_abs = np.abs(contribs[:, :-1]).mean(axis=0)
    df = pd.DataFrame({"feature": list(feature_names),
                       "mean_abs_shap": mean_abs.astype(float)})
    df["group"] = df["feature"].map(_group_of)
    return (df.sort_values("mean_abs_shap", ascending=False)
            .head(top_k).reset_index(drop=True))
