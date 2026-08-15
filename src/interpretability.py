"""Механистическая интерпретация вклада признаков через SHAP (TreeExplainer)
для XGBoost-модели аффинности (Часть 4)."""
import numpy as np
import pandas as pd
import shap
import warnings
import re
import contextlib

_ALPHABET = set("ACGT")


def _group_of(name: str) -> str:
    if name.startswith("structure_"):
        return "structure"
    if name.startswith("esm_"):
        return "esm"
    if name and set(name) <= _ALPHABET:
        return "kmer"
    return "other"


@contextlib.contextmanager
def _patched_float_for_xgb():
    """Context manager to patch float() for XGBoost 2.0+ base_score format parsing."""
    import builtins
    original_float = builtins.float

    def patched_float(x):
        if isinstance(x, str) and "[" in x and "]" in x:
            # Extract numeric value from list format like '[-1.4357099E-1]'
            match = re.search(r'[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?', x)
            if match:
                return original_float(match.group())
        return original_float(x)

    builtins.float = patched_float
    try:
        yield
    finally:
        builtins.float = original_float


def shap_feature_importance(model, X, feature_names, top_k: int = 30) -> pd.DataFrame:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        try:
            # Use patched float only during TreeExplainer creation
            with _patched_float_for_xgb():
                explainer = shap.TreeExplainer(model)
            # Call shap_values without patch to avoid numpy isinstance() issues
            sv = np.asarray(explainer.shap_values(X, check_additivity=False))
        except (ValueError, TypeError, AttributeError) as e:
            # Fallback to KernelExplainer if TreeExplainer fails
            if ("base_score" in str(e) or "could not convert" in str(e)):
                explainer = shap.KernelExplainer(
                    model.predict,
                    shap.sample(X, min(100, len(X)))
                )
                sv = np.asarray(explainer.shap_values(X))
            else:
                raise

    mean_abs = np.abs(sv).mean(axis=0)
    df = pd.DataFrame({"feature": list(feature_names),
                       "mean_abs_shap": mean_abs.astype(float)})
    df["group"] = df["feature"].map(_group_of)
    return (df.sort_values("mean_abs_shap", ascending=False)
            .head(top_k).reset_index(drop=True))
