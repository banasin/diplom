"""Задача 2: таблица уникальных пар с медианой log-Kd и разбиения train/test.

Кластерное разбиение группирует пары по кластеру последовательности аптамера
(без утечки гомологии); случайное — контроль для проверки гипотезы H2."""
import numpy as np
import pandas as pd

from src.data.cluster import greedy_cluster

PAIR_KEY = ["aptamer_seq", "aptamer_type", "target_key"]


def build_pairs(df: pd.DataFrame) -> pd.DataFrame:
    df = df[~df["qualified_any"].astype(bool)].copy()
    pairs = (
        df.groupby(PAIR_KEY, dropna=False)["log_Kd"]
        .median()
        .reset_index()
    )
    return pairs


def assign_clusters(pairs: pd.DataFrame, threshold: float) -> pd.DataFrame:
    pairs = pairs.copy()
    uniq = pairs["aptamer_seq"].drop_duplicates().tolist()
    labels = greedy_cluster(uniq, threshold)
    seq2cluster = dict(zip(uniq, labels))
    pairs["cluster_id"] = pairs["aptamer_seq"].map(seq2cluster)
    return pairs


def _assign_by_groups(group_ids: np.ndarray, test_size: float,
                      seed: int) -> np.ndarray:
    """Разложить группы по train/test так, чтобы суммарная доля строк в test
    была близка к test_size; группа целиком на одной стороне."""
    rng = np.random.default_rng(seed)
    groups, counts = np.unique(group_ids, return_counts=True)
    perm = rng.permutation(len(groups))
    groups, counts = groups[perm], counts[perm]
    target = test_size * counts.sum()
    in_test, acc = set(), 0
    for g, c in zip(groups, counts):
        if acc < target:
            in_test.add(g)
            acc += c
    return np.where(np.isin(group_ids, list(in_test)), "test", "train")


def make_splits(pairs: pd.DataFrame, test_size: float, seed: int) -> pd.DataFrame:
    pairs = pairs.copy()
    pairs["split_cluster"] = _assign_by_groups(
        pairs["cluster_id"].to_numpy(), test_size, seed)
    # случайное разбиение: каждая пара — своя группа
    pairs["split_random"] = _assign_by_groups(
        np.arange(len(pairs)), test_size, seed)
    return pairs


def assert_no_cluster_leakage(pairs: pd.DataFrame) -> None:
    sides = pairs.groupby("cluster_id")["split_cluster"].nunique()
    leaked = sides[sides > 1].index.tolist()
    assert not leaked, f"Утечка: кластеры в обоих сплитах: {leaked}"
