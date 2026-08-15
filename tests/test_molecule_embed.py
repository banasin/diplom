import numpy as np
from src.features.pooling import chunk_pool
from src.features.molecule import embed_molecules

M = "DeepChem/ChemBERTa-77M-MLM"


def test_chunk_pool_shape_and_empty():
    mat = np.arange(20, dtype=np.float32).reshape(10, 2)
    assert chunk_pool(mat, 4).shape == (4, 2)
    assert chunk_pool(np.zeros((0, 2), np.float32), 4).shape == (4, 2)


def test_embed_molecules_shapes_and_cache(tmp_path):
    smiles = {"a": "CCO", "b": "c1ccccc1"}
    cache = tmp_path / "mol.npz"
    pooled, tokens = embed_molecules(smiles, M, str(cache), m_tokens=16, batch_size=2)
    assert set(pooled) == {"a", "b"} and set(tokens) == {"a", "b"}
    H = pooled["a"].shape[0]
    assert pooled["a"].shape == (H,)
    assert tokens["a"].shape == (16, H)
    # детерминизм из кэша
    pooled2, tokens2 = embed_molecules(smiles, M, str(cache), m_tokens=16)
    assert np.allclose(pooled["a"], pooled2["a"])
    assert np.allclose(tokens["a"], tokens2["a"])
