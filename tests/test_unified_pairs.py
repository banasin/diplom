import pandas as pd
from src.data.split import build_pairs, build_unified_pairs


def _df(target, qual):
    return pd.DataFrame({
        "aptamer_seq": ["ACGTACGTAA", "ACGTACGTAA"],
        "aptamer_type": ["DNA", "DNA"],
        "target_key": [target, target],
        "log_Kd": [-8.0, -6.0],
        "qualified_any": [qual, qual],
    })


def test_build_pairs_keep_qualified():
    df = _df("P1", True)
    dropped = build_pairs(df)                      # по умолчанию отбрасывает
    kept = build_pairs(df, keep_qualified=True)    # оставляет
    assert len(dropped) == 0
    assert len(kept) == 1 and kept.iloc[0]["log_Kd"] == -7.0


def test_build_unified_pairs_tags_type():
    prot = _df("P1", False)
    mol = _df("CID 1", True)
    uni = build_unified_pairs(prot, mol)
    assert set(uni["target_type"]) == {"protein", "molecule"}
    assert len(uni) == 2
    assert set(uni.columns) >= {"aptamer_seq", "aptamer_type", "target_key",
                                "log_Kd", "target_type"}
