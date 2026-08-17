"""Ранговое обучение two-tower скор-модели (Часть 5): для каждой позитивной пары
(аптамер, мишень) сэмплируем K негативных мишеней и оптимизируем
score(поз) > score(нег). Модель — переиспускаемый TwoTower из twotower.py."""
import numpy as np
import torch
import torch.nn.functional as F

from src.models.twotower import TwoTower, _seed_everything, _to_tensor


def predict_scores(model, X_apt, X_prot) -> np.ndarray:
    """Сырой скор TwoTower.forward (сила связывания; больше = сильнее)."""
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        out = model(_to_tensor(X_apt, device), _to_tensor(X_prot, device))
    return out.cpu().numpy()


def train_ranking(pos_apt, pos_prot, neg_pool, allowed, cfg: dict, seed: int):
    _seed_everything(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    A = _to_tensor(pos_apt, device)        # (P, da)
    Ppos = _to_tensor(pos_prot, device)    # (P, dp)
    Pool = _to_tensor(neg_pool, device)    # (M, dp)
    P = A.shape[0]
    K = cfg["n_negatives"]
    allowed_idx = [np.where(row)[0] for row in np.asarray(allowed, dtype=bool)]

    rng = np.random.default_rng(seed)
    idx = rng.permutation(P)
    n_val = max(1, int(cfg["val_fraction"] * P))
    val, tr = idx[:n_val], idx[n_val:]

    model = TwoTower(pos_apt.shape[1], neg_pool.shape[1], cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    margin = cfg.get("margin", 1.0)
    use_bpr = cfg.get("ranking_loss", "bpr") == "bpr"
    bs = cfg["batch_size"]

    def sample_neg(rows):
        out = np.empty((len(rows), K), dtype=np.int64)
        for r, p in enumerate(rows):
            pool = allowed_idx[p]
            out[r] = (rng.choice(pool, size=K, replace=len(pool) < K)
                      if len(pool) else np.zeros(K, dtype=np.int64))
        return out

    def batch_loss(rows):
        neg = sample_neg(rows)                                 # (B, K)
        a = A[rows]                                            # (B, da)
        s_pos = model(a, Ppos[rows])                           # (B,)
        a_rep = a.unsqueeze(1).expand(-1, K, -1).reshape(-1, a.shape[1])
        n_rep = Pool[torch.as_tensor(neg.reshape(-1), device=device)]
        s_neg = model(a_rep, n_rep).reshape(len(rows), K)
        diff = s_pos.unsqueeze(1) - s_neg                      # (B, K)
        if use_bpr:
            return -F.logsigmoid(diff).mean()
        return F.relu(margin - diff).mean()

    best, best_state, bad = float("inf"), None, 0
    for _ in range(cfg["max_epochs"]):
        model.train()
        perm = rng.permutation(len(tr))
        for i in range(0, len(tr), bs):
            rows = tr[perm[i:i + bs]]
            opt.zero_grad()
            loss = batch_loss(rows)
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            v = batch_loss(val).item()
        if v < best - 1e-5:
            best, best_state, bad = v, {k: t.clone() for k, t in
                                        model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= cfg["patience"]:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model
