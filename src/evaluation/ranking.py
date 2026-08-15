"""Ранжирующие метрики для оценки retrieval истинных мишеней аптамера из
предсказанного профиля силы связывания (Часть 3, H5)."""
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def ranking_metrics(strength, is_positive, ks=(1, 5, 10)) -> dict:
    s = np.asarray(strength, dtype=float)
    y = np.asarray(is_positive).astype(int)
    out: dict = {}
    if 0 < y.sum() < len(y):
        out["roc_auc"] = float(roc_auc_score(y, s))
        out["average_precision"] = float(average_precision_score(y, s))
    else:
        out["roc_auc"] = float("nan")
        out["average_precision"] = float("nan")

    order = np.argsort(-s)                    # по убыванию силы связывания
    ranked = y[order]
    for k in ks:
        out[f"precision@{k}"] = (float(ranked[:k].mean())
                                 if k <= len(y) else float("nan"))
    pos_ranks = np.where(ranked == 1)[0]
    out["best_true_rank"] = int(pos_ranks[0] + 1) if len(pos_ranks) else -1
    return out
