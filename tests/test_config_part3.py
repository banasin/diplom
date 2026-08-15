from src.config import load_part3_config


def test_load_part3_config_defaults():
    cfg = load_part3_config("configs/part3.yaml")
    assert cfg.seed == 42
    assert cfg.identity_threshold == 0.95
    assert cfg.binding_log_kd == -7.0
    assert cfg.precision_ks == [1, 5, 10]
    assert isinstance(cfg.nn, dict) and isinstance(cfg.xgb, dict)
    # nn содержит ключи, нужные two-tower Части 1
    for key in ["aptamer_hidden", "protein_hidden", "embed_dim", "head_hidden",
                "dropout", "lr", "batch_size", "max_epochs", "patience",
                "val_fraction"]:
        assert key in cfg.nn
