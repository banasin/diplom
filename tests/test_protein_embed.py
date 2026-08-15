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


def test_embed_partial_cache_hit(tmp_path):
    cache = tmp_path / "emb.npz"
    first = embed_sequences({"A": "MKTVAAA"}, "esm2_t6_8M_UR50D", str(cache), batch_size=2)
    # второй вызов: A уже в кэше, B новый -> читаем кэш и дописываем
    second = embed_sequences({"A": "MKTVAAA", "B": "GGGGSSS"},
                             "esm2_t6_8M_UR50D", str(cache), batch_size=2)
    assert set(second) == {"A", "B"}
    assert np.allclose(second["A"], first["A"])   # ранее закэшированный вектор не изменился
    assert second["B"].shape == (320,)
