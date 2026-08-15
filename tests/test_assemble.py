import numpy as np
import pandas as pd
from src.features.assemble import build_feature_matrix


def test_build_matrix_drops_missing_protein_and_aligns():
    pairs = pd.DataFrame({
        "aptamer_seq": ["ACGTACGT", "TTTTGGGG"],
        "aptamer_type": ["DNA", "RNA"],
        "target_key": ["P1", "P_missing"],
        "log_Kd": [-7.0, -8.0],
    })
    emb = {"P1": np.ones(4, dtype=np.float32)}
    X_apt, X_prot, y, kept = build_feature_matrix(pairs, emb, k=2)
    assert len(kept) == 1                      # пара с отсутствующим белком ушла
    assert X_apt.shape == (1, 16 + 2)
    assert X_prot.shape == (1, 4)
    assert y.tolist() == [-7.0]
