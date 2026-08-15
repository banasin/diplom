"""per-target AUC: для каждой мишени — ранжирование аптамеров по силе связывания;
общая площадка для сопоставления с per-target IAP MNA/N-граммных моделей (H8)."""
import numpy as np
from sklearn.metrics import roc_auc_score


def per_target_auc(strength, is_binder) -> float:
    s = np.asarray(strength, dtype=float)
    y = np.asarray(is_binder).astype(int)
    if 0 < y.sum() < len(y):
        return float(roc_auc_score(y, s))
    return float("nan")
