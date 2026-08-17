"""Оркестратор Части 5: абляция 2×2 {регрессия|ранжирование} × {k-меры|DNABERT}
на задаче спектра. Регрессия и модель — из twotower.py; ранжирование — из
ranking_train.py. Строит профили test-аптамеров по панели белков, считает
per-aptamer ROC-AUC и per-target AUC, сравнивает ячейки, baseline и OLMPASS IAP.

Запуск: .venv/Scripts/python.exe -m src.run_part5
"""
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import Part5Config, load_part5_config
from src.data.split import (build_pairs, assign_clusters, make_splits,
                            assert_no_cluster_leakage, check_split_balance)
from src.data.spectrum import build_target_panel, known_targets_by_aptamer
from src.data.olmpass import load_olmpass_iap
from src.features.aptamer import aptamer_features
from src.features.aptamer_lm import embed_aptamers
from src.features.protein_seqs import load_sequences
from src.features.protein import embed_sequences
from src.models.twotower import train_twotower, predict_twotower
from src.models.ranking_train import train_ranking, predict_scores
from src.evaluation.ranking import ranking_metrics
from src.evaluation.per_target import per_target_auc

METRICS_DIR = Path("results/metrics")


def evaluate_cell(strength_fn, apt_feat, test_aptamers, panel, panel_emb, known,
                  precision_ks=(1, 5, 10)):
    """Профили test-аптамеров по панели; средние per-aptamer ROC-AUC и per-target
    AUC. strength_fn(X_apt, X_prot) -> сила связывания (больше = сильнее)."""
    panel_arr = np.array(panel)
    P = np.stack([panel_emb[t] for t in panel]).astype(np.float32)
    profiles = {}
    for a in test_aptamers:
        Xa = np.repeat(apt_feat[a][None, :], len(panel), axis=0).astype(np.float32)
        profiles[a] = np.asarray(strength_fn(Xa, P), dtype=float)
    aucs = []
    for a in test_aptamers:
        is_pos = np.isin(panel_arr, list(known.get(a, set())))
        if is_pos.sum() == 0:
            continue
        m = ranking_metrics(profiles[a], is_pos, ks=tuple(precision_ks))
        if not np.isnan(m["roc_auc"]):
            aucs.append(m["roc_auc"])
    pt = []
    for j, tgt in enumerate(panel):
        strength = np.array([profiles[a][j] for a in test_aptamers])
        is_binder = np.array([tgt in known.get(a, set()) for a in test_aptamers])
        v = per_target_auc(strength, is_binder)
        if not np.isnan(v):
            pt.append(v)
    return (float(np.mean(aucs)) if aucs else float("nan"),
            float(np.mean(pt)) if pt else float("nan"))


def _apt_matrix(keys, feat):
    return np.stack([feat[k] for k in keys]).astype(np.float32)


def run_part5(cfg: Part5Config) -> pd.DataFrame:
    df = pd.read_parquet("data/dataset_protein.parquet")
    pairs = build_pairs(df, keep_qualified=True)
    pairs = assign_clusters(pairs, cfg.identity_threshold)
    pairs = make_splits(pairs, cfg.test_size, cfg.seed)
    assert_no_cluster_leakage(pairs)

    accs = sorted(pairs["target_key"].dropna().unique())
    seqs = load_sequences(accs, "data/uniprot_seqs.json")
    prot_emb = embed_sequences(seqs, cfg.esm_model, "data/esm2_emb.npz")

    kept = pairs[pairs.target_key.isin(prot_emb)].reset_index(drop=True)
    check_split_balance(kept["split_cluster"], cfg.test_size, "cluster")
    panel = build_target_panel(kept, prot_emb)
    known = known_targets_by_aptamer(kept)

    uniq = list(dict.fromkeys(zip(kept.aptamer_seq, kept.aptamer_type)))
    kmer_feat = {s: aptamer_features(s, t, cfg.kmer_k) for s, t in uniq}
    nt_raw = embed_aptamers({s: s for s, _ in uniq}, cfg.nt_model,
                            "data/nt_apt_emb.npz")
    nt_feat = {s: nt_raw[s] for s, _ in uniq}
    encodings = {"kmer": kmer_feat, "nt": nt_feat}

    train = kept[kept.split_cluster == "train"].reset_index(drop=True)
    test_aptamers = [s for s in dict.fromkeys(
        kept[kept.split_cluster == "test"].aptamer_seq) if known.get(s, set())]

    rank_cfg = {**cfg.nn, "n_negatives": cfg.n_negatives,
                "ranking_loss": cfg.ranking_loss}
    rows = []
    for enc_name, feat in encodings.items():
        # --- регрессия (twotower.py) ---
        Xa = _apt_matrix(train.aptamer_seq, feat)
        Xp = np.stack([prot_emb[t] for t in train.target_key]).astype(np.float32)
        y = train["log_Kd"].to_numpy(np.float32)
        reg_model, reg_stats = train_twotower(Xa, Xp, y, cfg.nn, cfg.seed)
        reg_strength = (lambda A, Pp, m=reg_model, st=reg_stats:
                        -predict_twotower(m, A, Pp, st))          # сила = −log-Kd
        roc, pt = evaluate_cell(reg_strength, feat, test_aptamers, panel,
                                prot_emb, known)
        rows.append({"encoding": enc_name, "objective": "regression",
                     "mean_roc_auc": roc, "mean_per_target_auc": pt})

        # --- ранжирование (ranking_train.py) ---
        train_targets = list(dict.fromkeys(train.target_key))
        tgt_index = {t: i for i, t in enumerate(train_targets)}
        neg_pool = np.stack([prot_emb[t] for t in train_targets]).astype(np.float32)
        pos_apt = _apt_matrix(train.aptamer_seq, feat)
        pos_prot = np.stack([prot_emb[t] for t in train.target_key]).astype(np.float32)
        allowed = np.ones((len(train), len(train_targets)), dtype=bool)
        for r, s in enumerate(train.aptamer_seq):
            for t in known.get(s, set()):
                if t in tgt_index:
                    allowed[r, tgt_index[t]] = False             # известные — не негативы
        rank_model = train_ranking(pos_apt, pos_prot, neg_pool, allowed,
                                   rank_cfg, cfg.seed)
        rank_strength = lambda A, Pp, m=rank_model: predict_scores(m, A, Pp)
        roc, pt = evaluate_cell(rank_strength, feat, test_aptamers, panel,
                                prot_emb, known)
        rows.append({"encoding": enc_name, "objective": "ranking",
                     "mean_roc_auc": roc, "mean_per_target_auc": pt})

    spectrum = pd.DataFrame(rows)

    best = spectrum.sort_values("mean_per_target_auc", ascending=False).iloc[0]
    olm = load_olmpass_iap(cfg.best_models_dir)
    olm_agg = (olm.groupby("target_identifier")["iap"].mean().reset_index()
               .rename(columns={"iap": "olmpass_iap"}))
    vs = pd.DataFrame([{"best_cell": f"{best.encoding}+{best.objective}",
                        "our_mean_per_target_auc": best.mean_per_target_auc,
                        "olmpass_mean_iap": olm_agg["olmpass_iap"].mean(),
                        "n_olmpass_targets": len(olm_agg)}])

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    spectrum.to_csv(METRICS_DIR / "part5_spectrum.csv", index=False, encoding="utf-8")
    vs.to_csv(METRICS_DIR / "part5_vs_olmpass.csv", index=False, encoding="utf-8")
    return spectrum


def main() -> None:
    cfg = load_part5_config("configs/part5.yaml")
    spectrum = run_part5(cfg)
    print("=== Абляция 2x2 (задача спектра) ===")
    print(spectrum.round(3).to_string(index=False))
    vs = pd.read_csv(METRICS_DIR / "part5_vs_olmpass.csv")
    print("\n=== Лучшая ячейка vs OLMPASS ===")
    print(vs.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
