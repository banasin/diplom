from src.config import load_config


def test_load_config_defaults():
    cfg = load_config("configs/part1.yaml")
    assert cfg.seed == 42
    assert cfg.identity_threshold == 0.95
    assert cfg.test_size == 0.2
    assert cfg.kmer_k == 4
    assert cfg.esm_model == "esm2_t33_650M_UR50D"
    assert isinstance(cfg.xgb, dict) and isinstance(cfg.nn, dict)
