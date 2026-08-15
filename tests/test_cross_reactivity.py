import numpy as np
from src.cross_reactivity import (predict_profiles, cross_reactivity_score,
                                  emergence_test)


def test_predict_profiles_shape_and_order():
    apt = np.array([[1.0], [2.0]], np.float32)     # N=2, ka=1
    panel = np.array([[10.0], [20.0], [30.0]], np.float32)  # P=3, D=1
    # pair_predict_fn: сумма первых признаков — детерминированная проверка формы/порядка
    def fn(Xa, Xp):
        return Xa[:, 0] + Xp[:, 0]
    prof = predict_profiles(fn, apt, panel)
    assert prof.shape == (2, 3)
    assert np.allclose(prof[0], [11.0, 21.0, 31.0])   # аптамер 0 × панель
    assert np.allclose(prof[1], [12.0, 22.0, 32.0])


def test_cross_reactivity_score_counts_binders():
    profile = np.array([-8.0, -7.5, -6.0, -5.0])       # порог -7 -> 2 связывания
    sc = cross_reactivity_score(profile, binding_log_kd=-7.0)
    assert sc["n_binders"] == 2
    assert sc["spread"] > 0


def test_emergence_test_detects_shift():
    multi = [5, 6, 7, 8, 6]
    single = [1, 2, 1, 0, 2]
    res = emergence_test(multi, single)
    assert res["p_value"] < 0.05
    assert res["rank_biserial"] > 0                    # мульти > специфичные
    assert res["median_multi"] > res["median_single"]
