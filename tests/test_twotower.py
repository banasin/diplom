import numpy as np
from src.models.twotower import train_twotower, predict_twotower

CFG = {"aptamer_hidden": 32, "protein_hidden": 32, "embed_dim": 16,
       "head_hidden": 16, "dropout": 0.0, "lr": 0.01, "batch_size": 16,
       "max_epochs": 300, "patience": 300, "val_fraction": 0.25}


def test_twotower_forward_and_learns():
    rng = np.random.default_rng(0)
    X_apt = rng.normal(size=(32, 6)).astype("float32")
    X_prot = rng.normal(size=(32, 5)).astype("float32")
    y = (X_apt[:, 0] + X_prot[:, 0]).astype("float32")
    model, stats = train_twotower(X_apt, X_prot, y, CFG, seed=42)
    pred = predict_twotower(model, X_apt, X_prot, stats)
    assert pred.shape == (32,)
    corr = np.corrcoef(pred, y)[0, 1]
    assert corr > 0.8                           # сеть уловила зависимость
