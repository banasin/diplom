import numpy as np
from src.run_part5 import evaluate_cell


def test_evaluate_cell_returns_two_means():
    panel = ["P1", "P2", "P3"]
    panel_emb = {"P1": np.zeros(3, "float32"), "P2": np.ones(3, "float32"),
                 "P3": np.ones(3, "float32") * 2}
    test_aptamers = ["A1", "A2"]
    apt_feat = {"A1": np.array([1.0], "float32"), "A2": np.array([2.0], "float32")}
    known = {"A1": {"P1"}, "A2": {"P2"}}
    def strength_fn(Xa, Xp):
        return Xa[:, 0] + Xp[:, 0]
    roc, pt = evaluate_cell(strength_fn, apt_feat, test_aptamers, panel,
                            panel_emb, known)
    assert np.isnan(roc) or 0.0 <= roc <= 1.0
    assert np.isnan(pt) or 0.0 <= pt <= 1.0
