import numpy as np
from src.evaluation.per_target import per_target_auc


def test_per_target_auc_perfect():
    strength = np.array([9.0, 8.0, 1.0, 0.5])
    is_binder = np.array([True, True, False, False])
    assert abs(per_target_auc(strength, is_binder) - 1.0) < 1e-9


def test_per_target_auc_degenerate_nan():
    assert np.isnan(per_target_auc(np.array([1.0, 2.0]),
                                   np.array([True, True])))
