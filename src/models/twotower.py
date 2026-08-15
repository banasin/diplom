"""Базовая two-tower архитектура: отдельные башни аптамера и белка, слияние
конкатенацией, MLP-голова → log_Kd. Без билинейных/attention-слияний (Часть 2)."""
import numpy as np
import torch
from torch import nn


def _seed_everything(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class _Tower(nn.Module):
    def __init__(self, in_dim: int, hidden: int, out_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, out_dim), nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


class TwoTower(nn.Module):
    def __init__(self, apt_dim: int, prot_dim: int, cfg: dict):
        super().__init__()
        e = cfg["embed_dim"]
        self.apt = _Tower(apt_dim, cfg["aptamer_hidden"], e, cfg["dropout"])
        self.prot = _Tower(prot_dim, cfg["protein_hidden"], e, cfg["dropout"])
        self.head = nn.Sequential(
            nn.Linear(2 * e, cfg["head_hidden"]), nn.ReLU(),
            nn.Dropout(cfg["dropout"]), nn.Linear(cfg["head_hidden"], 1),
        )

    def forward(self, x_apt, x_prot):
        z = torch.cat([self.apt(x_apt), self.prot(x_prot)], dim=1)
        return self.head(z).squeeze(1)


def _to_tensor(a, device):
    return torch.tensor(np.asarray(a, dtype=np.float32), device=device)


def train_twotower(X_apt, X_prot, y, cfg: dict, seed: int):
    _seed_everything(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    y = np.asarray(y, dtype=np.float32)
    y_mean, y_std = float(y.mean()), float(y.std() + 1e-8)
    y_norm = (y - y_mean) / y_std

    n = len(y)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_val = max(1, int(cfg["val_fraction"] * n))
    val_idx, tr_idx = idx[:n_val], idx[n_val:]

    Xa = _to_tensor(X_apt, device); Xp = _to_tensor(X_prot, device)
    yt = _to_tensor(y_norm, device)

    model = TwoTower(X_apt.shape[1], X_prot.shape[1], cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    loss_fn = nn.MSELoss()

    best_val, best_state, bad = float("inf"), None, 0
    bs = cfg["batch_size"]
    for _ in range(cfg["max_epochs"]):
        model.train()
        perm = rng.permutation(len(tr_idx))
        for i in range(0, len(tr_idx), bs):
            b = tr_idx[perm[i:i + bs]]
            opt.zero_grad()
            out = model(Xa[b], Xp[b])
            loss = loss_fn(out, yt[b])
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            v = loss_fn(model(Xa[val_idx], Xp[val_idx]), yt[val_idx]).item()
        if v < best_val - 1e-5:
            best_val, best_state, bad = v, {k: t.clone() for k, t in
                                            model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= cfg["patience"]:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, {"y_mean": y_mean, "y_std": y_std}


def predict_twotower(model: TwoTower, X_apt, X_prot, stats: dict) -> np.ndarray:
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        out = model(_to_tensor(X_apt, device), _to_tensor(X_prot, device))
    return out.cpu().numpy() * stats["y_std"] + stats["y_mean"]
