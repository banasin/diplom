"""Единая модель предсказания log-Kd для разнородных мишеней (белки+молекулы) в
общем пространстве. Общая башня аптамера; тип-специфичная проекция мишени в
общую размерность D + эмбеддинг типа; сменный модуль слияния."""
import numpy as np
import torch
from torch import nn

from src.models.fusion import ConcatFusion, BilinearFusion, CrossAttentionFusion

ALPHABET = "ACGT"
_AIDX = {c: i for i, c in enumerate(ALPHABET)}


def _seed_everything(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _onehot(seqs: list, max_len: int) -> np.ndarray:
    """(N, max_len, 4) one-hot по ACGT (U→T), с нормализацией и падом нулями."""
    out = np.zeros((len(seqs), max_len, 4), dtype=np.float32)
    for i, s in enumerate(seqs):
        s = s.upper().replace("U", "T")
        for j, ch in enumerate(s[:max_len]):
            if ch in _AIDX:
                out[i, j, _AIDX[ch]] = 1.0
    return out


class UnifiedModel(nn.Module):
    def __init__(self, apt_dim: int, prot_dim: int, mol_dim: int,
                 cfg: dict, fusion: str):
        super().__init__()
        D = cfg["shared_dim"] if "shared_dim" in cfg else 128
        self.fusion_name = fusion
        # башня аптамера (вектор) для concat/bilinear
        self.apt_tower = nn.Sequential(
            nn.Linear(apt_dim, cfg["aptamer_hidden"]), nn.ReLU(),
            nn.Dropout(cfg["dropout"]), nn.Linear(cfg["aptamer_hidden"], D), nn.ReLU())
        # токенная ветвь аптамера (для cross-attention): conv по one-hot
        self.apt_conv = nn.Sequential(
            nn.Conv1d(4, D, kernel_size=5, padding=2), nn.ReLU())
        self.apt_tokens = cfg["aptamer_tokens"]
        # тип-специфичные проекции мишени в общую размерность D
        self.proj_prot = nn.Linear(prot_dim, D)
        self.proj_mol = nn.Linear(mol_dim, D)
        self.type_emb = nn.Embedding(2, D)
        self.D = D
        # модуль слияния
        if fusion == "concat":
            self.fuse = ConcatFusion(D, D, cfg["head_hidden"])
        elif fusion == "bilinear":
            self.fuse = BilinearFusion(D, D, cfg["bilinear_out"])
        elif fusion == "cross_attention":
            self.fuse = CrossAttentionFusion(D, cfg["attn_heads"], cfg["head_hidden"])
        else:
            raise ValueError(f"неизвестное слияние: {fusion}")
        head_in = cfg["head_hidden"] if fusion != "bilinear" else cfg["bilinear_out"]
        self.head = nn.Sequential(
            nn.Linear(head_in, cfg["head_hidden"]), nn.ReLU(),
            nn.Dropout(cfg["dropout"]), nn.Linear(cfg["head_hidden"], 1))
        self._tables = None

    def set_target_tables(self, prot_pooled, prot_tokens, mol_pooled, mol_tokens,
                          device):
        def t(a):
            return torch.tensor(np.asarray(a, dtype=np.float32), device=device)
        self._tables = dict(pp=t(prot_pooled), pt=t(prot_tokens),
                            mp=t(mol_pooled), mt=t(mol_tokens))

    def _target_pooled(self, ttype, prot_ptr, mol_ptr):
        B = ttype.shape[0]
        out = torch.zeros(B, self.D, device=ttype.device)
        pm = ttype == 0
        mm = ttype == 1
        if pm.any():
            out[pm] = self.proj_prot(self._tables["pp"][prot_ptr[pm]])
        if mm.any():
            out[mm] = self.proj_mol(self._tables["mp"][mol_ptr[mm]])
        return out + self.type_emb(ttype)

    def _target_tokens(self, ttype, prot_ptr, mol_ptr):
        B = ttype.shape[0]
        M = self._tables["pt"].shape[1] if self._tables["pt"].numel() else \
            self._tables["mt"].shape[1]
        out = torch.zeros(B, M, self.D, device=ttype.device)
        pm = ttype == 0
        mm = ttype == 1
        if pm.any():
            out[pm] = self.proj_prot(self._tables["pt"][prot_ptr[pm]])
        if mm.any():
            out[mm] = self.proj_mol(self._tables["mt"][mol_ptr[mm]])
        out = out + self.type_emb(ttype).unsqueeze(1)
        return out

    def forward(self, apt_feats, apt_onehot, ttype, prot_ptr, mol_ptr):
        if self.fusion_name == "cross_attention":
            conv = self.apt_conv(apt_onehot.transpose(1, 2))          # (B, D, L)
            a_tok = torch.nn.functional.adaptive_avg_pool1d(
                conv, self.apt_tokens).transpose(1, 2)                # (B, Ta, D)
            t_tok = self._target_tokens(ttype, prot_ptr, mol_ptr)
            fused = self.fuse(a_tok, t_tok)
        else:
            a = self.apt_tower(apt_feats)
            t = self._target_pooled(ttype, prot_ptr, mol_ptr)
            fused = self.fuse(a, t)
        return self.head(fused).squeeze(1)


def _to(a, device, dtype=torch.float32):
    return torch.tensor(np.asarray(a), dtype=dtype, device=device)


def train_unified(uf, cfg: dict, fusion: str, seed: int):
    _seed_everything(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = dict(cfg)
    cfg.setdefault("shared_dim", uf_shared_dim(cfg))
    y = uf.y.astype(np.float32)
    y_mean, y_std = float(y.mean()), float(y.std() + 1e-8)
    y_norm = (y - y_mean) / y_std

    max_len = max((len(s) for s in uf.apt_seqs), default=1)
    onehot = _onehot(uf.apt_seqs, max_len)

    prot_dim = uf.prot_pooled.shape[1] if uf.prot_pooled.size else 1
    mol_dim = uf.mol_pooled.shape[1] if uf.mol_pooled.size else 1
    model = UnifiedModel(uf.apt_feats.shape[1], prot_dim, mol_dim, cfg, fusion).to(device)
    model.set_target_tables(uf.prot_pooled, uf.prot_tokens,
                            uf.mol_pooled, uf.mol_tokens, device)

    apt = _to(uf.apt_feats, device)
    oh = _to(onehot, device)
    ttype = _to(uf.target_type, device, torch.long)
    pptr = _to(uf.prot_ptr, device, torch.long)
    mptr = _to(uf.mol_ptr, device, torch.long)
    yt = _to(y_norm, device)

    n = len(y)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_val = max(1, int(cfg["val_fraction"] * n))
    val_idx, tr_idx = idx[:n_val], idx[n_val:]

    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    loss_fn = nn.MSELoss()
    best, best_state, bad = float("inf"), None, 0
    bs = cfg["batch_size"]
    for _ in range(cfg["max_epochs"]):
        model.train()
        perm = rng.permutation(len(tr_idx))
        for i in range(0, len(tr_idx), bs):
            b = tr_idx[perm[i:i + bs]]
            opt.zero_grad()
            out = model(apt[b], oh[b], ttype[b], pptr[b], mptr[b])
            loss = loss_fn(out, yt[b])
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            v = loss_fn(model(apt[val_idx], oh[val_idx], ttype[val_idx],
                              pptr[val_idx], mptr[val_idx]), yt[val_idx]).item()
        if v < best - 1e-5:
            best, best_state, bad = v, {k: t.clone() for k, t in
                                        model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= cfg["patience"]:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, {"y_mean": y_mean, "y_std": y_std,
                   "max_len": max_len}


def uf_shared_dim(cfg: dict) -> int:
    return cfg.get("shared_dim", 128)


def predict_unified(model, uf, idx, stats) -> np.ndarray:
    device = next(model.parameters()).device
    onehot = _onehot([uf.apt_seqs[i] for i in idx], stats["max_len"])
    model.eval()
    with torch.no_grad():
        out = model(_to(uf.apt_feats[idx], device), _to(onehot, device),
                    _to(uf.target_type[idx], device, torch.long),
                    _to(uf.prot_ptr[idx], device, torch.long),
                    _to(uf.mol_ptr[idx], device, torch.long))
    return out.cpu().numpy() * stats["y_std"] + stats["y_mean"]
