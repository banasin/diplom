import numpy as np
from src.features.aptamer import normalize_seq, kmer_vector, aptamer_features


def test_normalize_u_to_t():
    assert normalize_seq("acgu") == "ACGT"


def test_kmer_vector_counts_1mer():
    v = kmer_vector("AACG", k=1)          # A:2 C:1 G:1 T:0 -> частоты
    # порядок алфавита ACGT
    assert np.allclose(v, [0.5, 0.25, 0.25, 0.0])


def test_kmer_vector_u_equals_t():
    assert np.allclose(kmer_vector("ACGU", 2), kmer_vector("ACGT", 2))


def test_kmer_vector_length():
    assert kmer_vector("ACGTACGT", k=3).shape == (64,)


def test_aptamer_features_shape_and_flags():
    f = aptamer_features("ACGT", "RNA", k=2)
    assert f.shape == (16 + 2,)
    assert f[-2] == 1.0          # is_rna
    assert f[-1] == 4.0          # длина
    f_dna = aptamer_features("ACGT", "DNA", k=2)
    assert f_dna[-2] == 0.0
