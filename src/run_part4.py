"""Оркестратор Части 4: структурный уровень признаков (H7), SHAP-интерпретация и
сопоставление с MNA/N-граммными моделями по per-target качеству (H8).

Запуск из корня: .venv/Scripts/python.exe -m src.run_part4
"""
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import Part4Config, load_part4_config
from src.data.split import (build_pairs, assign_clusters, make_splits,
                            assert_no_cluster_leakage, check_split_balance)
from src.data.olmpass import load_olmpass_iap
from src.features.aptamer import aptamer_features, KMER_ALPHABET
from src.features.structure import structure_features, STRUCT_FEATURE_NAMES
from src.features.protein_seqs import load_sequences
from src.features.protein import embed_sequences
from src.models.baseline import train_baseline, predict_baseline
from src.interpretability import shap_feature_importance
from src.evaluation.per_target import per_target_auc
from src.evaluation.metrics import regression_metrics

METRICS_DIR = Path("results/metrics")


def _kmer_names(k: int) -> list:
    return ["".join(p) for p in product(KMER_ALPHABET, repeat=k)]


def build_aptamer_matrix(pairs, k: int, with_structure: bool):
    """Матрица аптамерных признаков и имена. aptamer_features = [k-меры|is_rna|len];
    при with_structure добавляются структурные признаки (Nussinov)."""
    names = _kmer_names(k) + ["is_rna", "seq_len"]
    if with_structure:
        names = names + [f"structure_{s}" for s in STRUCT_FEATURE_NAMES]
    rows = []
    for r in pairs.itertuples():
        f = aptamer_features(r.aptamer_seq, r.aptamer_type, k)
        if with_structure:
            f = np.concatenate([f, structure_features(r.aptamer_seq)])
        rows.append(f)
    return np.stack(rows).astype(np.float32), names


def run_part4(cfg: Part4Config) -> dict:
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
    tr = (kept.split_cluster == "train").to_numpy()
    te = ~tr
    Xp = np.stack([prot_emb[t] for t in kept.target_key]).astype(np.float32)
    y = kept["log_Kd"].to_numpy(np.float32)

    esm_names = [f"esm_{i}" for i in range(Xp.shape[1])]

    # --- H7: абляция структуры ---
    h7_rows = []
    models = {}
    for with_struct in (False, True):
        Xa, apt_names = build_aptamer_matrix(kept, cfg.kmer_k, with_struct)
        X = np.hstack([Xa, Xp])
        feat_names = apt_names + esm_names
        model = train_baseline(X[tr], y[tr], cfg.xgb, cfg.seed)
        pred = predict_baseline(model, X[te])
        tag = "kmer+structure+esm" if with_struct else "kmer+esm"
        h7_rows.append({"features": tag, **regression_metrics(y[te], pred),
                        "n_train": int(tr.sum()), "n_test": int(te.sum())})
        models[with_struct] = (model, X, feat_names)

    # --- SHAP по расширенной модели ---
    model_s, X_s, names_s = models[True]
    shap_all = shap_feature_importance(model_s, X_s[tr], names_s, top_k=len(names_s))
    group_df = (shap_all.groupby("group")["mean_abs_shap"].sum()
                .reset_index().sort_values("mean_abs_shap", ascending=False))
    shap_df = shap_all.head(cfg.shap_top_k).reset_index(drop=True)

    # --- H8: per-target AUC (наш, на test) vs OLMPASS IAP ---
    #     Для каждой белковой мишени: сила = −предсказанный log-Kd test-аптамеров;
    #     позитивы = test-аптамеры с измеренной парой к этой мишени.
    test_kept = kept[te].reset_index(drop=True)
    Xa_te, _ = build_aptamer_matrix(test_kept, cfg.kmer_k, True)
    test_apt_seqs = test_kept["aptamer_seq"].tolist()
    test_apt_types = test_kept["aptamer_type"].tolist()
    uniq_apt = list(dict.fromkeys(zip(test_apt_seqs, test_apt_types)))
    apt_feat = {a: build_aptamer_matrix(
        pd.DataFrame({"aptamer_seq": [a[0]], "aptamer_type": [a[1]]}),
        cfg.kmer_k, True)[0][0] for a in uniq_apt}
    known_pairs = set(zip(test_kept["aptamer_seq"], test_kept["target_key"]))
    per_target_rows = []
    for tgt in sorted(test_kept["target_key"].unique()):
        strengths, binders = [], []
        for a in uniq_apt:
            feat = np.concatenate([apt_feat[a], prot_emb[tgt]]).astype(np.float32)
            logkd = predict_baseline(model_s, feat[None, :])[0]
            strengths.append(-logkd)                       # сила = −log-Kd
            binders.append((a[0], tgt) in known_pairs)
        auc = per_target_auc(np.array(strengths), np.array(binders))
        if not np.isnan(auc):
            per_target_rows.append({"target_identifier": tgt, "our_auc": auc,
                                    "n_binders": int(sum(binders))})
    our_pt = pd.DataFrame(per_target_rows)

    olmpass = load_olmpass_iap(cfg.best_models_dir)
    # OLMPASS IAP по мишени (усреднение по DNA/RNA×MNA/Ngram×порогам)
    olm_agg = (olmpass.groupby("target_identifier")["iap"].mean()
               .reset_index().rename(columns={"iap": "olmpass_iap"}))
    h8 = our_pt.merge(olm_agg, on="target_identifier", how="inner")

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(h7_rows).to_csv(METRICS_DIR / "part4_h7.csv", index=False,
                                 encoding="utf-8")
    shap_df.to_csv(METRICS_DIR / "part4_shap.csv", index=False, encoding="utf-8")
    h8.to_csv(METRICS_DIR / "part4_h8.csv", index=False, encoding="utf-8")
    return {"h7": pd.DataFrame(h7_rows), "shap": shap_df, "group": group_df, "h8": h8}


def main() -> None:
    cfg = load_part4_config("configs/part4.yaml")
    out = run_part4(cfg)
    print("=== H7 (структура улучшает аффинность?) ===")
    print(out["h7"].to_string(index=False))
    print("\n=== Вклад по группам признаков (SHAP) ===")
    print(out["group"].to_string(index=False))
    print("\n=== Топ признаков (SHAP) ===")
    print(out["shap"].head(15).to_string(index=False))
    print("\n=== H8: наш per-target AUC vs OLMPASS IAP (общие мишени) ===")
    h8 = out["h8"]
    if len(h8):
        print(f"мишеней: {len(h8)}; средний our_AUC={h8.our_auc.mean():.3f}; "
              f"средний OLMPASS_IAP={h8.olmpass_iap.mean():.3f}")
    else:
        print("нет общих мишеней с валидной AUC")


if __name__ == "__main__":
    main()
