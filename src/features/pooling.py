"""Chunk-pooling: усреднение матрицы (L, H) в фиксированное число m «токенов».
Общий для молекул (ChemBERTa) и белков (ESM-2)."""
import numpy as np


def chunk_pool(mat: np.ndarray, m: int) -> np.ndarray:
    """(L, H) -> (m, H): усреднение по m примерно равным чанкам вдоль L.
    Пустой вход → нулевая матрица (m, H)."""
    if len(mat) == 0:
        H = mat.shape[1] if mat.ndim == 2 else 0
        return np.zeros((m, H), dtype=np.float32)
    idx = np.array_split(np.arange(len(mat)), m)
    return np.stack([mat[c].mean(0) if len(c) else mat.mean(0)
                     for c in idx]).astype(np.float32)
