from src.config import load_part4_config


def test_load_part4_config_defaults():
    cfg = load_part4_config("configs/part4.yaml")
    assert cfg.seed == 42
    assert cfg.identity_threshold == 0.95
    assert cfg.best_models_dir == "best_models"
    assert cfg.shap_top_k == 30
    assert cfg.binding_log_kd == -7.0
    assert isinstance(cfg.xgb, dict)


def test_shap_importable():
    import shap  # noqa: F401
