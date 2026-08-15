"""Сборка числовых матриц признаков из таблицы пар, k-меров аптамера и
эмбеддингов белка. Пары без эмбеддинга мишени отбрасываются."""
import numpy as np
import pandas as pd

from src.features.aptamer import aptamer_features


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
