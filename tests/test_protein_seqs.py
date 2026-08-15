import json
from src.features.protein_seqs import parse_fasta, load_sequences


def test_parse_fasta_joins_lines():
    text = ">sp|P12345|X\nMKTV\nAAAA\n"
    assert parse_fasta(text) == "MKTVAAAA"


def test_load_sequences_uses_cache(tmp_path):
    cache = tmp_path / "seqs.json"
    cache.write_text(json.dumps({"P1": "MKTV"}), encoding="utf-8")
    # P1 есть в кэше -> сеть не нужна; просим только P1
    out = load_sequences(["P1"], str(cache))
    assert out == {"P1": "MKTV"}
