import numpy as np
from src.run_part3 import evaluate_retrieval


def test_evaluate_retrieval_rows():
    panel = ["P1", "P2", "P3"]
    test_aptamers = ["A1", "A2"]
    known = {"A1": {"P1"}, "A2": {"P2", "P3"}}
    # профили: сила = -log_Kd; строим log_Kd так, что известные — сильнейшие
    profiles = np.array([[-9.0, -5.0, -4.0],      # A1: P1 сильнее всего (верно)
                         [-5.0, -9.0, -8.0]])     # A2: P2,P3 сильнее (верно)
    rows = evaluate_retrieval(profiles, test_aptamers, panel, known,
                              "xgboost", precision_ks=(1,))
    assert len(rows) == 2
    for r in rows:
        assert set(r) >= {"model", "aptamer", "roc_auc", "best_true_rank",
                          "n_known"}
    # у A1 идеальный ретрив: истинная мишень — ранг 1
    a1 = [r for r in rows if r["aptamer"] == "A1"][0]
    assert a1["best_true_rank"] == 1
