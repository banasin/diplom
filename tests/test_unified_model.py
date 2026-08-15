import numpy as np
import pandas as pd
from src.features.assemble import build_unified_features
from src.models.unified import train_unified, predict_unified

CFG = {"aptamer_hidden": 32, "head_hidden": 16, "bilinear_out": 16,
       "attn_heads": 2, "aptamer_tokens": 4, "dropout": 0.0, "lr": 0.01,
       "batch_size": 16, "max_epochs": 300, "patience": 300, "val_fraction": 0.25}


def _uf(n=24):
    rng = np.random.default_rng(0)
    seqs = ["ACGTACGT", "TTTTGGGG"]
    rows = []
    for i in range(n):
        is_prot = i % 2 == 0
        rows.append({"aptamer_seq": seqs[i % 2], "aptamer_type": "DNA",
                     "target_key": "P1" if is_prot else "M1",
                     "target_type": "protein" if is_prot else "molecule",
                     "log_Kd": -7.0 + (0.5 if is_prot else -0.5)})
    pairs = pd.DataFrame(rows)
    pp = {"P1": rng.normal(size=5).astype("float32")}
    pt = {"P1": rng.normal(size=(4, 5)).astype("float32")}
    mp = {"M1": rng.normal(size=3).astype("float32")}
    mt = {"M1": rng.normal(size=(4, 3)).astype("float32")}
    return build_unified_features(pairs, pp, pt, mp, mt, k=2)


def test_unified_learns_mixed_types():
    uf = _uf()
    for fusion in ["concat", "bilinear", "cross_attention"]:
        model, stats = train_unified(uf, CFG, fusion, seed=42)
        idx = np.arange(len(uf.y))
        pred = predict_unified(model, uf, idx, stats)
        assert pred.shape == (len(uf.y),)
        # два типа имеют разный средний log_Kd — модель должна их различать
        p_prot = pred[uf.target_type == 0].mean()
        p_mol = pred[uf.target_type == 1].mean()
        assert p_prot > p_mol
