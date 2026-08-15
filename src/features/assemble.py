"""Сборка числовых матриц признаков из таблицы пар, k-меров аптамера и
эмбеддингов белка. Пары без эмбеддинга мишени отбрасываются."""
import numpy as np
import pandas as pd
from dataclasses import dataclass

from src.features.aptamer import aptamer_features


@dataclass
class UnifiedFeatures:
    apt_feats: np.ndarray
    apt_seqs: list
    target_type: np.ndarray
    prot_ptr: np.ndarray
    mol_ptr: np.ndarray
    prot_pooled: np.ndarray
    prot_tokens: np.ndarray
    mol_pooled: np.ndarray
    mol_tokens: np.ndarray
    y: np.ndarray
    kept: "pd.DataFrame"


def build_unified_features(pairs, prot_pooled, prot_tokens,
                           mol_pooled, mol_tokens, k) -> UnifiedFeatures:
    def has_emb(row):
        return (row.target_key in prot_pooled) if row.target_type == "protein" \
            else (row.target_key in mol_pooled)

    kept = pairs[[has_emb(r) for r in pairs.itertuples()]].reset_index(drop=True)

    apt_feats = np.stack([aptamer_features(r.aptamer_seq, r.aptamer_type, k)
                          for r in kept.itertuples()]).astype(np.float32) \
        if len(kept) else np.empty((0, 4 ** k + 2), np.float32)
    apt_seqs = kept["aptamer_seq"].tolist()

    prot_keys = [k_ for k_ in dict.fromkeys(
        kept[kept.target_type == "protein"]["target_key"])]
    mol_keys = [k_ for k_ in dict.fromkeys(
        kept[kept.target_type == "molecule"]["target_key"])]
    prot_index = {key: i for i, key in enumerate(prot_keys)}
    mol_index = {key: i for i, key in enumerate(mol_keys)}

    ttype = np.array([0 if t == "protein" else 1
                      for t in kept["target_type"]], dtype=np.int64)
    prot_ptr = np.array([prot_index.get(key, -1) if t == "protein" else -1
                         for key, t in zip(kept["target_key"], kept["target_type"])],
                        dtype=np.int64)
    mol_ptr = np.array([mol_index.get(key, -1) if t == "molecule" else -1
                        for key, t in zip(kept["target_key"], kept["target_type"])],
                       dtype=np.int64)

    def stack(keys, table):
        return np.stack([table[key] for key in keys]).astype(np.float32) \
            if keys else np.zeros((0, 0), np.float32)

    return UnifiedFeatures(
        apt_feats=apt_feats, apt_seqs=apt_seqs, target_type=ttype,
        prot_ptr=prot_ptr, mol_ptr=mol_ptr,
        prot_pooled=stack(prot_keys, prot_pooled),
        prot_tokens=stack(prot_keys, prot_tokens),
        mol_pooled=stack(mol_keys, mol_pooled),
        mol_tokens=stack(mol_keys, mol_tokens),
        y=kept["log_Kd"].to_numpy(np.float32), kept=kept,
    )


def build_feature_matrix(pairs: pd.DataFrame, protein_emb: dict[str, np.ndarray],
                         k: int):
    kept = pairs[pairs["target_key"].isin(protein_emb)].reset_index(drop=True)
    X_apt = np.stack([
        aptamer_features(r.aptamer_seq, r.aptamer_type, k)
        for r in kept.itertuples()
    ]) if len(kept) else np.empty((0, 4 ** k + 2), dtype=np.float32)
    X_prot = np.stack([protein_emb[key] for key in kept["target_key"]]) \
        if len(kept) else np.empty((0, 0), dtype=np.float32)
    y = kept["log_Kd"].to_numpy(dtype=np.float32)
    return X_apt.astype(np.float32), X_prot.astype(np.float32), y, kept
