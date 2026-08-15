"""Кодирование последовательности аптамера в числовой вектор (k-меры + флаги).

Согласовано с N-граммной моделью: частоты k-меров над алфавитом ACGT с
нормализацией U→T (РНК и ДНК кодируются в одном алфавите; химию различает
отдельный флаг is_rna)."""
from itertools import product
import numpy as np

KMER_ALPHABET = "ACGT"


def normalize_seq(seq: str) -> str:
    return seq.strip().upper().replace("U", "T")


def _kmer_index(k: int) -> dict[str, int]:
    return {"".join(p): i for i, p in enumerate(product(KMER_ALPHABET, repeat=k))}


def kmer_vector(seq: str, k: int) -> np.ndarray:
    seq = normalize_seq(seq)
    idx = _kmer_index(k)
    vec = np.zeros(len(idx), dtype=np.float32)
    for i in range(len(seq) - k + 1):
        j = idx.get(seq[i:i + k])
        if j is not None:                 # пропускаем k-меры с посторонними символами
            vec[j] += 1.0
    total = vec.sum()
    if total > 0:
        vec /= total
    return vec


def aptamer_features(seq: str, aptamer_type: str, k: int) -> np.ndarray:
    kv = kmer_vector(seq, k)
    is_rna = np.float32(1.0 if str(aptamer_type).upper() == "RNA" else 0.0)
    seq_len = np.float32(len(normalize_seq(seq)))
    return np.concatenate([kv, [is_rna, seq_len]]).astype(np.float32)
