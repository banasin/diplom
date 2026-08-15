import numpy as np
from src.features.protein import embed_protein_tokens


def test_protein_tokens_shape(tmp_path):
    seqs = {"A": "MKTVAAAA", "B": "GGGGSSSS"}
    cache = tmp_path / "ptok.npz"
    toks = embed_protein_tokens(seqs, "esm2_t6_8M_UR50D", str(cache),
                                m_tokens=16, batch_size=2)
    assert set(toks) == {"A", "B"}
    assert toks["A"].shape == (16, 320)
    toks2 = embed_protein_tokens(seqs, "esm2_t6_8M_UR50D", str(cache), m_tokens=16)
    assert np.allclose(toks["A"], toks2["A"])
