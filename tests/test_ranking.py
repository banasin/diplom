import numpy as np
from src.evaluation.ranking import ranking_metrics


def test_perfect_ranking():
    # позитивы имеют наибольшую силу
    strength = np.array([9.0, 8.0, 1.0, 0.5])
    is_pos = np.array([True, True, False, False])
    m = ranking_metrics(strength, is_pos, ks=(1, 2))
    assert abs(m["roc_auc"] - 1.0) < 1e-9
    assert m["precision@1"] == 1.0
    assert m["precision@2"] == 1.0
    assert m["best_true_rank"] == 1


def test_worst_ranking():
    strength = np.array([1.0, 0.5, 9.0, 8.0])
    is_pos = np.array([True, True, False, False])
    m = ranking_metrics(strength, is_pos, ks=(1,))
    assert abs(m["roc_auc"] - 0.0) < 1e-9
    assert m["precision@1"] == 0.0
    assert m["best_true_rank"] == 3         # лучшая истинная — третья по силе


def test_no_positives_returns_nan():
    m = ranking_metrics(np.array([1.0, 2.0]), np.array([False, False]))
    assert np.isnan(m["roc_auc"])
