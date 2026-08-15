import numpy as np
from src.features.protein import embed_sequences


def test_embed_shape_and_determinism(tmp_path):
    seqs = {"A": "MKTVAAA", "B": "GGGGSSS"}
    cache = tmp_path / "emb.npz"
    # для скорости теста используем крошечную ESM-2 (8M, 320-мерная)
    out = embed_sequences(seqs, "esm2_t6_8M_UR50D", str(cache), batch_size=2)
    assert set(out) == {"A", "B"}
    assert out["A"].shape == (320,)
    # детерминизм: повторный вызов (уже из кэша) даёт тот же вектор
    out2 = embed_sequences(seqs, "esm2_t6_8M_UR50D", str(cache), batch_size=2)
    assert np.allclose(out["A"], out2["A"])
