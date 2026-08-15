import numpy as np
import pandas as pd
from src.features.assemble import build_unified_features
from src.run_part2 import evaluate_unified


def _uf(n=40):
    rng = np.random.default_rng(0)
    rows = []
    for i in range(n):
        is_prot = i % 2 == 0
        rows.append({"aptamer_seq": "ACGTACGT" if i % 2 else "TTTTGGGG",
                     "aptamer_type": "DNA",
                     "target_key": "P1" if is_prot else "M1",
                     "target_type": "protein" if is_prot else "molecule",
                     "log_Kd": -7.0 + rng.normal() * 0.1})
    pairs = pd.DataFrame(rows)
    pp = {"P1": rng.normal(size=5).astype("float32")}
    pt = {"P1": rng.normal(size=(4, 5)).astype("float32")}
    mp = {"M1": rng.normal(size=3).astype("float32")}
    mt = {"M1": rng.normal(size=(4, 3)).astype("float32")}
    return build_unified_features(pairs, pp, pt, mp, mt, k=2)


def test_evaluate_unified_returns_rows():
    uf = _uf()
    from src.config import load_part2_config
    cfg = load_part2_config("configs/part2.yaml")
    cfg.nn["max_epochs"] = 20
    cfg.nn["shared_dim"] = cfg.shared_dim
    mask = np.array([i % 5 != 0 for i in range(len(uf.y))])
    rows = evaluate_unified(uf, mask, ~mask, "concat", cfg)
    types = {r["target_type"] for r in rows}
    assert types == {"protein", "molecule", "overall"}
    for r in rows:
        assert set(r) >= {"model", "fusion", "split", "target_type", "rmse", "n_test"}
