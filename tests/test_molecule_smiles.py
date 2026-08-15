import json
from src.features.molecule_smiles import parse_cid, load_smiles


def test_parse_cid():
    assert parse_cid("CID 12345") == 12345
    assert parse_cid("CID 6450878") == 6450878
    assert parse_cid(None) is None
    assert parse_cid("не CID") is None


def test_load_smiles_uses_cache(tmp_path):
    cache = tmp_path / "smiles.json"
    cache.write_text(json.dumps({"CID 1": "CCO"}), encoding="utf-8")
    out = load_smiles({"CID 1": "CID 1"}, str(cache))
    assert out == {"CID 1": "CCO"}
