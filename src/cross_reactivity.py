"""Профили аптамер×панель и оценка кросс-реактивности как эмерджентного свойства
парной модели аффинности (Часть 3)."""
import numpy as np
from scipy.stats import mannwhitneyu


def predict_profiles(pair_predict_fn, apt_feats, panel_emb) -> np.ndarray:
    """Матрица предсказанных log-Kd (N аптамеров × P мишеней панели). Строит
    декартово произведение признаков и предсказывает батчем."""
    apt_feats = np.asarray(apt_feats, dtype=np.float32)
    panel_emb = np.asarray(panel_emb, dtype=np.float32)
    n, p = len(apt_feats), len(panel_emb)
    if n == 0 or p == 0:
        return np.zeros((n, p), dtype=np.float32)
    Xa = np.repeat(apt_feats, p, axis=0)          # (N*P, ka)
    Xp = np.tile(panel_emb, (n, 1))               # (N*P, D)
    preds = np.asarray(pair_predict_fn(Xa, Xp), dtype=np.float32)
    return preds.reshape(n, p)


def cross_reactivity_score(profile, binding_log_kd: float) -> dict:
    profile = np.asarray(profile, dtype=float)
    n_binders = int((profile <= binding_log_kd).sum())    # предсказанные связывания
    return {"n_binders": n_binders, "spread": float(profile.std())}


def emergence_test(scores_multi, scores_single) -> dict:
    a = np.asarray(scores_multi, dtype=float)
    b = np.asarray(scores_single, dtype=float)
    u, p = mannwhitneyu(a, b, alternative="greater")
    n1, n2 = len(a), len(b)
    rank_biserial = 2.0 * u / (n1 * n2) - 1.0     # эффект: >0 => мульти сильнее
    return {"u": float(u), "p_value": float(p),
            "rank_biserial": float(rank_biserial),
            "median_multi": float(np.median(a)),
            "median_single": float(np.median(b))}
