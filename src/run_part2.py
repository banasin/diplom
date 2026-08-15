"""Оркестратор Части 2: объединённые пары → сплит → признаки (ESM-2 + ChemBERTa)
→ единая модель с тремя слияниями (H4) + отдельные по-типные модели (H3) +
XGBoost по типу; метрики по типам и общие.

Запуск из корня: .venv/Scripts/python.exe -m src.run_part2
"""
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import Part2Config, load_part2_config
from src.data.split import (build_unified_pairs, assign_clusters, make_splits,
                            assert_no_cluster_leakage, check_split_balance)
from src.features.assemble import build_unified_features
from src.features.protein_seqs import load_sequences
from src.features.protein import embed_sequences, embed_protein_tokens
from src.features.molecule_smiles import load_smiles
from src.features.molecule import embed_molecules
from src.models.unified import train_unified, predict_unified
from src.evaluation.metrics import regression_metrics

METRICS_DIR = Path("results/metrics")


def _metrics_rows(y_true, y_pred, ttype, model, fusion, split):
    rows = []
    for name, mask in [("protein", ttype == 0), ("molecule", ttype == 1),
                       ("overall", np.ones(len(ttype), bool))]:
        if mask.sum() >= 2:
            rows.append({"model": model, "fusion": fusion, "split": split,
                         "target_type": name, **regression_metrics(
                             y_true[mask], y_pred[mask]),
                         "n_test": int(mask.sum())})
    return rows


def evaluate_unified(uf, train_mask, test_mask, fusion, cfg: Part2Config):
    """Обучить единую модель (данное слияние) на train, оценить по типам на test."""
    cfg_nn = dict(cfg.nn)
    cfg_nn["shared_dim"] = cfg.shared_dim
    tr_uf = _subset(uf, train_mask)
    model, stats = train_unified(tr_uf, cfg_nn, fusion, cfg.seed)
    test_idx = np.where(test_mask)[0]
    pred = predict_unified(model, uf, test_idx, stats)
    return _metrics_rows(uf.y[test_idx], pred, uf.target_type[test_idx],
                         "unified", fusion, "cluster")


def _subset(uf, mask):
    """Подвыборка UnifiedFeatures по булевой маске строк (таблицы эмбеддингов
    переиспользуются целиком; указатели остаются валидными)."""
    from src.features.assemble import UnifiedFeatures
    idx = np.where(mask)[0]
    return UnifiedFeatures(
        apt_feats=uf.apt_feats[idx], apt_seqs=[uf.apt_seqs[i] for i in idx],
        target_type=uf.target_type[idx], prot_ptr=uf.prot_ptr[idx],
        mol_ptr=uf.mol_ptr[idx], prot_pooled=uf.prot_pooled,
        prot_tokens=uf.prot_tokens, mol_pooled=uf.mol_pooled,
        mol_tokens=uf.mol_tokens, y=uf.y[idx], kept=uf.kept.iloc[idx])


def run_part2(cfg: Part2Config) -> pd.DataFrame:
    prot_df = pd.read_parquet("data/dataset_protein.parquet")
    mol_df = pd.read_parquet("data/dataset_smallmol.parquet")
    pairs = build_unified_pairs(prot_df, mol_df)
    pairs = assign_clusters(pairs, cfg.identity_threshold)
    pairs = make_splits(pairs, cfg.test_size, cfg.seed)
    assert_no_cluster_leakage(pairs)

    # --- признаки мишеней ---
    prot_keys = sorted(pairs[pairs.target_type == "protein"]["target_key"]
                       .dropna().unique())
    seqs = load_sequences(prot_keys, "data/uniprot_seqs.json")
    prot_pooled = embed_sequences(seqs, cfg.esm_model, "data/esm2_emb.npz")
    prot_tokens = embed_protein_tokens(seqs, cfg.esm_model,
                                       "data/esm2_tokens.npz", cfg.m_tokens)
    mol_rows = pairs[pairs.target_type == "molecule"][["target_key"]].drop_duplicates()
    mol_df_keys = mol_df.set_index("target_key")["target_cid"].to_dict()
    keys_cids = {k: mol_df_keys.get(k) for k in mol_rows["target_key"]}
    smiles = load_smiles(keys_cids, "data/pubchem_smiles.json")
    mol_pooled, mol_tokens = embed_molecules(smiles, cfg.chem_model,
                                             "data/chemberta_emb.npz", cfg.m_tokens)

    uf = build_unified_features(pairs, prot_pooled, prot_tokens,
                                mol_pooled, mol_tokens, cfg.kmer_k)
    print(f"Пар всего: {len(pairs)}; с эмбеддингом: {len(uf.kept)}; "
          f"отброшено: {len(pairs) - len(uf.kept)}")
    check_split_balance(uf.kept["split_cluster"], cfg.test_size, "cluster")

    tr = (uf.kept["split_cluster"] == "train").to_numpy()
    te = ~tr
    all_rows = []
    for fusion in cfg.fusions:
        all_rows += evaluate_unified(uf, tr, te, fusion, cfg)

    metrics = pd.DataFrame(all_rows)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(METRICS_DIR / "part2_metrics.csv", index=False, encoding="utf-8")
    return metrics


def main() -> None:
    cfg = load_part2_config("configs/part2.yaml")
    metrics = run_part2(cfg)
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
