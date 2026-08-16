from src.config import load_part5_config


def test_load_part5_config_defaults():
    cfg = load_part5_config("configs/part5.yaml")
    assert cfg.seed == 42
    assert cfg.n_negatives == 5
    assert cfg.ranking_loss == "bpr"
    assert cfg.nt_model.startswith("InstaDeepAI/")
    for key in ["aptamer_hidden", "protein_hidden", "embed_dim", "head_hidden",
                "dropout", "lr", "batch_size", "max_epochs", "patience",
                "val_fraction", "margin"]:
        assert key in cfg.nn


def test_einops_importable():
    import einops  # noqa: F401
