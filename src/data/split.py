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
    """Разложить группы по train/test так, чтобы доля строк в test была близка
    к test_size, а КРУПНЫЕ группы попадали в train (test набирается из мелких
    групп). Группа целиком на одной стороне — без утечки гомологии."""
    rng = np.random.default_rng(seed)
    groups, counts = np.unique(group_ids, return_counts=True)
    # перемешиваем для детерминированного разрешения равных размеров,
    # затем сортируем по убыванию размера (стабильно)
    perm = rng.permutation(len(groups))
    groups, counts = groups[perm], counts[perm]
    order = np.argsort(-counts, kind="stable")
    groups, counts = groups[order], counts[order]
    target = test_size * counts.sum()
    in_test, test_acc = [], 0
    for g, c in zip(groups, counts):
        if test_acc + c <= target:      # добавляем только пока не переполним долю
            in_test.append(g)
            test_acc += c
    return np.where(np.isin(group_ids, in_test), "test", "train")


def make_splits(pairs: pd.DataFrame, test_size: float, seed: int) -> pd.DataFrame:
    pairs = pairs.copy()
    pairs["split_cluster"] = _assign_by_groups(
        pairs["cluster_id"].to_numpy(), test_size, seed)
    # случайное разбиение: каждая пара — своя группа
    pairs["split_random"] = _assign_by_groups(
        np.arange(len(pairs)), test_size, seed)
    return pairs


def check_split_balance(split_labels, test_size: float, name: str = "cluster",
                        tol: float = 2.0) -> None:
    """Проверить, что доля test близка к test_size; иначе разбиение вырождено
    (обычно из-за слишком крупных кластеров — стоит повысить identity_threshold)."""
    frac = float(np.mean(np.asarray(split_labels) == "test"))
    lo, hi = test_size / tol, min(1.0, test_size * tol)
    if not (lo <= frac <= hi):
        raise ValueError(
            f"Разбиение '{name}' вырождено: доля test={frac:.3f} вне "
            f"[{lo:.3f}, {hi:.3f}]. Вероятно, кластеры слишком крупные — "
            f"повысьте identity_threshold.")


def assert_no_cluster_leakage(pairs: pd.DataFrame) -> None:
    sides = pairs.groupby("cluster_id")["split_cluster"].nunique()
    leaked = sides[sides > 1].index.tolist()
    if leaked:
        raise ValueError(f"Утечка: кластеры в обоих сплитах: {leaked}")
