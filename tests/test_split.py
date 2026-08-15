import numpy as np
import pandas as pd
import pytest
from src.data.split import (
    build_pairs, assign_clusters, make_splits, assert_no_cluster_leakage,
    check_split_balance,
)


def _raw():
    # две записи одной пары (медиана), одна оговорённая (дропается), одна обычная
    return pd.DataFrame({
        "aptamer_seq": ["ACGTACGTAA", "ACGTACGTAA", "TTTTGGGGCC", "ACGTACGTAA"],
        "aptamer_type": ["DNA", "DNA", "DNA", "DNA"],
        "target_key":   ["P1", "P1", "P1", "P2"],
        "log_Kd":       [-8.0, -6.0, -7.0, -9.0],
        "qualified_any": [False, False, True, False],
    })


def test_build_pairs_drops_qualified_and_medians():
    pairs = build_pairs(_raw())
    # (ACGTACGTAA,P1) медиана(-8,-6)=-7 ; (ACGTACGTAA,P2)=-9 ; qualified-строка ушла
    assert len(pairs) == 2
    row = pairs[(pairs.target_key == "P1")].iloc[0]
    assert row.log_Kd == -7.0


def test_assign_clusters_merges_similar_sequences():
    pairs = pd.DataFrame({
        "aptamer_seq": ["ACGTACGTAA", "ACGTACGTAT"],
        "aptamer_type": ["DNA", "DNA"], "target_key": ["P1", "P2"],
        "log_Kd": [-7.0, -8.0],
    })
    out = assign_clusters(pairs, threshold=0.8)
    assert out.cluster_id.nunique() == 1


def test_make_splits_no_leakage():
    pairs = build_pairs(_raw())
    pairs = assign_clusters(pairs, threshold=0.8)
    pairs = make_splits(pairs, test_size=0.5, seed=42)
    assert set(pairs.split_cluster) <= {"train", "test"}
    assert set(pairs.split_random) <= {"train", "test"}
    assert_no_cluster_leakage(pairs)          # не должно бросить


def test_make_splits_large_cluster_goes_to_train():
    # большой кластер (8 пар) должен уйти в train; test набирается из мелких
    pairs = pd.DataFrame({
        "cluster_id": [0] * 8 + [1, 2],
        "log_Kd": [-7.0] * 10,
    })
    out = make_splits(pairs, test_size=0.2, seed=42)
    assert set(out[out.cluster_id == 0]["split_cluster"]) == {"train"}
    assert (out["split_cluster"] == "test").sum() == 2   # два мелких кластера
    assert_no_cluster_leakage(out)


def test_leakage_detector_catches_leak():
    pairs = pd.DataFrame({
        "cluster_id": [0, 0], "split_cluster": ["train", "test"],
    })
    with pytest.raises(ValueError):
        assert_no_cluster_leakage(pairs)


def test_check_split_balance_ok():
    labels = np.array(["train"] * 80 + ["test"] * 20)
    check_split_balance(labels, 0.2)   # доля 0.2 в норме — не бросает


def test_check_split_balance_flags_degenerate():
    labels = np.array(["test"] * 99 + ["train"])   # почти всё в test — вырождено
    with pytest.raises(ValueError):
        check_split_balance(labels, 0.2)
