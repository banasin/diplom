"""Оркестратор Части 1: разбиение → признаки → бейзлайн и сеть → метрики H1/H2.

Запуск из корня репозитория:
    .venv/Scripts/python.exe -m src.run_part1
"""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.config import Config, load_config
from src.data.split import (build_pairs, assign_clusters, make_splits,
                            assert_no_cluster_leakage, check_split_balance)
from src.features.assemble import build_feature_matrix
from src.features.protein_seqs import load_sequences
from src.features.protein import embed_sequences
from src.models.baseline import train_baseline, predict_baseline
from src.models.twotower import train_twotower, predict_twotower
from src.evaluation.metrics import regression_metrics

METRICS_DIR = Path("results/metrics")
FIG_DIR = Path("results/figures")


def evaluate_split(X_apt, X_prot, y, is_test, split_name, cfg: Config):
    """Обучить обе модели на train-части и оценить на test-части одного split."""
    tr, te = ~is_test, is_test
    Xa_tr, Xa_te = X_apt[tr], X_apt[te]
    Xp_tr, Xp_te = X_prot[tr], X_prot[te]
    y_tr, y_te = y[tr], y[te]
    rows = []

    # --- XGBoost на конкатенации признаков ---
    Xc_tr = np.hstack([Xa_tr, Xp_tr]); Xc_te = np.hstack([Xa_te, Xp_te])
    xgb = train_baseline(Xc_tr, y_tr, cfg.xgb, cfg.seed)
    p_xgb = predict_baseline(xgb, Xc_te)
    rows.append({"model": "xgboost", "split": split_name,
                 **regression_metrics(y_te, p_xgb),
                 "n_train": int(tr.sum()), "n_test": int(te.sum())})
    _scatter(y_te, p_xgb, f"xgboost_{split_name}")

    # --- Two-tower сеть ---
    net, stats = train_twotower(Xa_tr, Xp_tr, y_tr, cfg.nn, cfg.seed)
    p_net = predict_twotower(net, Xa_te, Xp_te, stats)
    rows.append({"model": "twotower", "split": split_name,
                 **regression_metrics(y_te, p_net),
                 "n_train": int(tr.sum()), "n_test": int(te.sum())})
    _scatter(y_te, p_net, f"twotower_{split_name}")
    return rows


def _scatter(y_true, y_pred, name: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(4, 4))
    plt.scatter(y_true, y_pred, s=10, alpha=0.5)
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    plt.plot(lims, lims, "r--", lw=1)
    plt.xlabel("истинный log_Kd"); plt.ylabel("предсказанный log_Kd")
    plt.title(name); plt.tight_layout()
    plt.savefig(FIG_DIR / f"pred_vs_true_{name}.png", dpi=120)
    plt.close()


def run(cfg: Config, data_path: str = "data/dataset_protein.parquet") -> pd.DataFrame:
    df = pd.read_parquet(data_path)
    pairs = build_pairs(df)
    pairs = assign_clusters(pairs, cfg.identity_threshold)
    pairs = make_splits(pairs, cfg.test_size, cfg.seed)
    assert_no_cluster_leakage(pairs)

    # последовательности и ESM-2 эмбеддинги уникальных мишеней (по UniProt ID)
    accs = sorted(pairs["target_key"].dropna().unique().tolist())
    seqs = load_sequences(accs, "data/uniprot_seqs.json")
    emb = embed_sequences(seqs, cfg.esm_model, "data/esm2_emb.npz")

    # признаки через общий сборщик (Task 7); пары без эмбеддинга мишени отброшены,
    # kept сохраняет колонки split_cluster/split_random для оценки ниже
    X_apt, X_prot, y, kept = build_feature_matrix(pairs, emb, cfg.kmer_k)
    n_dropped = len(pairs) - len(kept)
    print(f"Пар всего: {len(pairs)}; с эмбеддингом мишени: {len(kept)}; "
          f"отброшено без UniProt-эмбеддинга: {n_dropped}")

    check_split_balance(kept["split_cluster"], cfg.test_size, "cluster")

    all_rows = []
    for split_col, name in [("split_cluster", "cluster"), ("split_random", "random")]:
        is_test = (kept[split_col] == "test").to_numpy()
        all_rows += evaluate_split(X_apt, X_prot, y, is_test, name, cfg)

    metrics = pd.DataFrame(all_rows)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(METRICS_DIR / "part1_metrics.csv", index=False,
                   encoding="utf-8")
    return metrics


def main() -> None:
    cfg = load_config("configs/part1.yaml")
    metrics = run(cfg)
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
