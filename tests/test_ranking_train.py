import numpy as np
from src.models.twotower import train_twotower
from src.models.ranking_train import train_ranking, predict_scores

CFG = {"aptamer_hidden": 32, "protein_hidden": 32, "embed_dim": 16,
       "head_hidden": 16, "dropout": 0.0, "lr": 0.01, "batch_size": 8,
       "max_epochs": 300, "patience": 300, "val_fraction": 0.25, "margin": 1.0,
       "n_negatives": 2, "ranking_loss": "bpr"}


def test_predict_scores_shape_on_regression_model():
    rng = np.random.default_rng(0)
    Xa = rng.normal(size=(30, 6)).astype("float32")
    Xp = rng.normal(size=(30, 5)).astype("float32")
    y = (Xa[:, 0] + Xp[:, 0]).astype("float32")
    model, _ = train_twotower(Xa, Xp, y, CFG, seed=42)
    s = predict_scores(model, Xa, Xp)
    assert s.shape == (30,)


def test_ranking_positive_scored_higher():
    # N аптамеров, N мишеней; апт i связывает мишень i (позитив), негатив — любая
    # чужая мишень. После рангового обучения большинство аптамеров ставят свою
    # мишень выше средней чужой (модель усвоила ранжирование).
    rng = np.random.default_rng(0)
    N, da, dp = 12, 4, 4
    apt = rng.normal(size=(N, da)).astype("float32")
    tgt = rng.normal(size=(N, dp)).astype("float32")
    code = rng.normal(size=N).astype("float32")
    apt[:, 0] = code                                           # апт i и мишень i делят
    tgt[:, 0] = code                                           # «код» → обучаемая связь
    allowed = ~np.eye(N, dtype=bool)                           # негатив = не своя мишень
    cfg = dict(CFG, val_fraction=0.2)
    model = train_ranking(apt, tgt, tgt, allowed, cfg, seed=42)
    # в агрегате: скор к своим мишеням выше, чем к сдвинутым чужим
    shift = (np.arange(N) + 1) % N
    s_own = predict_scores(model, apt, tgt)                    # (апт_i, своя мишень_i)
    s_neg = predict_scores(model, apt, tgt[shift])            # (апт_i, чужая мишень)
    assert s_own.mean() > s_neg.mean()
