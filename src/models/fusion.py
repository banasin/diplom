"""Механизмы слияния представлений аптамера и мишени (сравнение — H4)."""
import torch
from torch import nn


class ConcatFusion(nn.Module):
    """Конкатенация двух векторов → линейная проекция."""
    def __init__(self, a_dim: int, t_dim: int, out_dim: int):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(a_dim + t_dim, out_dim), nn.ReLU())

    def forward(self, a, t):
        return self.proj(torch.cat([a, t], dim=1))


class BilinearFusion(nn.Module):
    """Билинейное взаимодействие аптамер×мишень."""
    def __init__(self, a_dim: int, t_dim: int, out_dim: int):
        super().__init__()
        self.bilinear = nn.Bilinear(a_dim, t_dim, out_dim)
        self.act = nn.ReLU()

    def forward(self, a, t):
        return self.act(self.bilinear(a, t))


class CrossAttentionFusion(nn.Module):
    """Multihead cross-attention: токены аптамера как query, токены мишени как
    key/value; выход усредняется по токенам аптамера."""
    def __init__(self, dim: int, heads: int, out_dim: int):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.proj = nn.Sequential(nn.Linear(dim, out_dim), nn.ReLU())

    def forward(self, a_tok, t_tok):
        attended, _ = self.attn(a_tok, t_tok, t_tok)     # (B, Ta, dim)
        return self.proj(attended.mean(dim=1))
