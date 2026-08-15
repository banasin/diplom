from src.data.cluster import identity, greedy_cluster


def test_identity_equal():
    assert identity("ACGTACGT", "ACGTACGT") == 1.0


def test_identity_one_mismatch():
    assert abs(identity("ACGTACGT", "ACGTACGA") - 7 / 8) < 1e-9


def test_identity_u_equals_t():
    assert identity("ACGU", "ACGT") == 1.0


def test_greedy_cluster_groups_similar():
    seqs = ["ACGTACGTAA", "ACGTACGTAT", "TTTTGGGGCC"]  # 0 и 1 близки, 2 иной
    labels = greedy_cluster(seqs, threshold=0.8)
    assert labels[0] == labels[1]
    assert labels[2] != labels[0]
    assert len(set(labels)) == 2


def test_greedy_cluster_all_distinct():
    seqs = ["AAAAAAAAAA", "CCCCCCCCCC", "GGGGGGGGGG"]
    labels = greedy_cluster(seqs, threshold=0.8)
    assert len(set(labels)) == 3
