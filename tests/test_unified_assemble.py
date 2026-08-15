import numpy as np
import pandas as pd
from src.features.assemble import build_unified_features


def test_unified_features_routing_and_drop():
    pairs = pd.DataFrame({
        "aptamer_seq": ["ACGTACGT", "TTTTGGGG", "ACGTACGT"],
        "aptamer_type": ["DNA", "RNA", "DNA"],
        "target_key": ["P1", "CID1", "P_missing"],
        "target_type": ["protein", "molecule", "protein"],
        "log_Kd": [-7.0, -8.0, -9.0],
    })
    prot_pooled = {"P1": np.ones(4, np.float32)}
    prot_tokens = {"P1": np.ones((2, 4), np.float32)}
    mol_pooled = {"CID1": np.ones(3, np.float32) * 2}
    mol_tokens = {"CID1": np.ones((2, 3), np.float32) * 2}
    uf = build_unified_features(pairs, prot_pooled, prot_tokens,
                                mol_pooled, mol_tokens, k=2)
    assert len(uf.kept) == 2                       # P_missing отброшен
    assert uf.apt_feats.shape == (2, 16 + 2)
    assert uf.prot_pooled.shape == (1, 4) and uf.mol_pooled.shape == (1, 3)
    assert uf.prot_tokens.shape == (1, 2, 4) and uf.mol_tokens.shape == (1, 2, 3)
    # маршрутизация: строка 0 — protein, строка 1 — molecule
    t0 = uf.target_type[0]
    assert set(uf.target_type.tolist()) == {0, 1}
    # у protein-строки валиден prot_ptr, у molecule — mol_ptr
    prot_row = int(np.where(uf.target_type == 0)[0][0])
    mol_row = int(np.where(uf.target_type == 1)[0][0])
    assert uf.prot_ptr[prot_row] >= 0 and uf.mol_ptr[mol_row] >= 0
    assert uf.y.shape == (2,)
