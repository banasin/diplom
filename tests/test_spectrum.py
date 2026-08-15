import numpy as np
import pandas as pd
from src.data.spectrum import build_target_panel, known_targets_by_aptamer


def _pairs():
    return pd.DataFrame({
        "aptamer_seq": ["A1", "A1", "A2"],
        "target_key": ["P1", "P2", "P1"],
    })


def test_build_target_panel_filters_by_emb():
    emb = {"P1": np.zeros(4, np.float32)}         # P2 без эмбеддинга
    panel = build_target_panel(_pairs(), emb)
    assert panel == ["P1"]


def test_known_targets_by_aptamer():
    known = known_targets_by_aptamer(_pairs())
    assert known["A1"] == {"P1", "P2"}
    assert known["A2"] == {"P1"}
