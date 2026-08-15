"""Оркестратор Части 3: кросс-реактивность как эмерджентное свойство парной
модели. Обучает XGBoost и two-tower на train, строит профили test-аптамеров по
панели белков, считает retrieval (H5) и эмерджентную кросс-реактивность (H6).

Запуск из корня: .venv/Scripts/python.exe -m src.run_part3
"""
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import Part3Config, load_part3_config
from src.data.split import (build_pairs, assign_clusters, make_splits,
                            assert_no_cluster_leakage, check_split_balance)
from src.data.spectrum import build_target_panel, known_targets_by_aptamer
from src.features.aptamer import aptamer_features
from src.features.protein_seqs import load_sequences
from src.features.protein import embed_sequences
from src.models.baseline import train_baseline, predict_baseline
from src.models.twotower import train_twotower, predict_twotower
from src.cross_reactivity import (predict_profiles, cross_reactivity_score,
                                  emergence_test)
from src.evaluation.ranking import ranking_metrics

METRICS_DIR = Path("results/metrics")


def evaluate_retrieval(profiles, test_aptamers, panel, known, model_name,
                       precision_ks) -> list:
    """H5: для каждого test-аптамера ранжирование панели по силе связывания."""
    panel_arr = np.array(panel)
    rows = []
    for i, apt in enumerate(test_aptamers):
        pos_keys = known.get(apt, set())
        is_pos = np.isin(panel_arr, list(pos_keys))
        if is_pos.sum() == 0:
            continue
        strength = -profiles[i]                    # сила = −log_Kd
        m = ranking_metrics(strength, is_pos, ks=tuple(precision_ks))
        rows.append({"model": model_name, "aptamer": apt,
                     "n_known": int(is_pos.sum()), **m})
    return rows


def _xgb_predict_fn(model):
    return lambda Xa, Xp: predict_baseline(model, np.hstack([Xa, Xp]))


def _twotower_predict_fn(model, stats):
    return lambda Xa, Xp: predict_twotower(model, Xa, Xp, stats)


def run_part3(cfg: Part3Config):
    df = pd.read_parquet("data/dataset_protein.parquet")
    pairs = build_pairs(df, keep_qualified=True)
    pairs = assign_clusters(pairs, cfg.identity_threshold)
    pairs = make_splits(pairs, cfg.test_size, cfg.seed)
    assert_no_cluster_leakage(pairs)

    accs = sorted(pairs["target_key"].dropna().unique())
    seqs = load_sequences(accs, "data/uniprot_seqs.json")
    prot_emb = embed_sequences(seqs, cfg.esm_model, "data/esm2_emb.npz")

    panel = build_target_panel(pairs, prot_emb)
    known = known_targets_by_aptamer(pairs)
    panel_emb = np.stack([prot_emb[t] for t in panel]).astype(np.float32)

    # --- обучающие признаки (только пары с эмбеддингом мишени) ---
    train_pairs = pairs[(pairs.split_cluster == "train")
                        & (pairs.target_key.isin(prot_emb))].reset_index(drop=True)
    check_split_balance(pairs[pairs.target_key.isin(prot_emb)]["split_cluster"],
                        cfg.test_size, "cluster")
    Xa_tr = np.stack([aptamer_features(r.aptamer_seq, r.aptamer_type, cfg.kmer_k)
                      for r in train_pairs.itertuples()]).astype(np.float32)
    Xp_tr = np.stack([prot_emb[t] for t in train_pairs.target_key]).astype(np.float32)
    y_tr = train_pairs["log_Kd"].to_numpy(np.float32)

    xgb = train_baseline(np.hstack([Xa_tr, Xp_tr]), y_tr, cfg.xgb, cfg.seed)
    net, stats = train_twotower(Xa_tr, Xp_tr, y_tr, cfg.nn, cfg.seed)

    # --- test-аптамеры (уникальные, с ≥1 известной мишенью в панели) ---
    panel_set = set(panel)
    test_rows = pairs[pairs.split_cluster == "test"]
    test_aptamers = [a for a in dict.fromkeys(
        zip(test_rows.aptamer_seq, test_rows.aptamer_type))
        if known.get(a[0], set()) & panel_set]
    apt_seqs = [a[0] for a in test_aptamers]
    apt_types = [a[1] for a in test_aptamers]
    test_feats = np.stack([aptamer_features(s, t, cfg.kmer_k)
                           for s, t in zip(apt_seqs, apt_types)]).astype(np.float32)

    ranking_rows, cross_rows = [], []
    for model_name, pred_fn in [("xgboost", _xgb_predict_fn(xgb)),
                                ("twotower", _twotower_predict_fn(net, stats))]:
        profiles = predict_profiles(pred_fn, test_feats, panel_emb)   # (N, P)
        ranking_rows += evaluate_retrieval(profiles, apt_seqs, panel, known,
                                           model_name, cfg.precision_ks)
        # H6: оценки кросс-реактивности + тест эмерджентности
        scores = [cross_reactivity_score(profiles[i], cfg.binding_log_kd)["n_binders"]
                  for i in range(len(apt_seqs))]
        n_known = [len(known.get(s, set()) & panel_set) for s in apt_seqs]
        multi = [sc for sc, nk in zip(scores, n_known) if nk >= 2]
        single = [sc for sc, nk in zip(scores, n_known) if nk == 1]
        rec = {"model": model_name, "n_multi": len(multi), "n_single": len(single),
               "mean_score_multi": float(np.mean(multi)) if multi else float("nan"),
               "mean_score_single": float(np.mean(single)) if single else float("nan")}
        if len(multi) >= 1 and len(single) >= 1:
            rec.update(emergence_test(multi, single))
        cross_rows.append(rec)

    ranking = pd.DataFrame(ranking_rows)
    cross = pd.DataFrame(cross_rows)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    ranking.to_csv(METRICS_DIR / "part3_ranking.csv", index=False, encoding="utf-8")
    cross.to_csv(METRICS_DIR / "part3_crossreact.csv", index=False, encoding="utf-8")
    return ranking, cross


def main() -> None:
    cfg = load_part3_config("configs/part3.yaml")
    ranking, cross = run_part3(cfg)
    print("=== H5 (retrieval), сводка по моделям ===")
    print(ranking.groupby("model")[["roc_auc", "average_precision",
          "best_true_rank"]].mean().to_string())
    print("\n=== H6 (кросс-реактивность) ===")
    print(cross.to_string(index=False))


if __name__ == "__main__":
    main()
