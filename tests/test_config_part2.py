from src.config import load_part2_config


def test_load_part2_config_defaults():
    cfg = load_part2_config("configs/part2.yaml")
    assert cfg.seed == 42
    assert cfg.chem_model == "DeepChem/ChemBERTa-77M-MLM"
    assert cfg.m_tokens == 16
    assert cfg.shared_dim == 128
    assert cfg.fusions == ["concat", "bilinear", "cross_attention"]
    assert isinstance(cfg.nn, dict) and isinstance(cfg.xgb, dict)
