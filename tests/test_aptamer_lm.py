import numpy as np
from src.features.aptamer_lm import embed_aptamers, _kmerize

M = "zhihan1996/DNA_bert_6"


def test_kmerize():
    assert _kmerize("ACGTACG") == "ACGTAC CGTACG"       # 6-меры, скользящее окно
    assert _kmerize("ACGUACG") == _kmerize("ACGTACG")   # U→T


def test_embed_aptamers_shape_and_cache(tmp_path):
    seqs = {"ACGTACGTAA": "ACGTACGTAA", "TTTTGGGGCC": "TTTTGGGGCC"}
    cache = tmp_path / "apt.npz"
    out = embed_aptamers(seqs, M, str(cache), batch_size=2)
    assert set(out) == set(seqs)
    H = out["ACGTACGTAA"].shape[0]
    assert out["ACGTACGTAA"].shape == (H,)
    out2 = embed_aptamers(seqs, M, str(cache))            # из кэша
    assert np.allclose(out["ACGTACGTAA"], out2["ACGTACGTAA"])


def test_u_equals_t(tmp_path):
    a = embed_aptamers({"x": "ACGUACGUAC"}, M, str(tmp_path / "a.npz"))
    b = embed_aptamers({"x": "ACGTACGTAC"}, M, str(tmp_path / "b.npz"))
    assert np.allclose(a["x"], b["x"])
