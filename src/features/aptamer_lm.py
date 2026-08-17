"""Эмбеддинги аптамеров предобученной нуклеотидной моделью DNABERT: k-мерная
токенизация последовательности (скользящее окно), mean-pooling представлений,
кэш .npz (вне git). U≡T (Часть 5).

Примечание: Nucleotide Transformer несовместим с установленным transformers 5.15
(remote-код обращается к отсутствующему config.rope_theta), поэтому в качестве
предобученного нуклеотидного энкодера используется DNABERT — он грузится как
обычный BERT через общий загрузчик с конвертацией .bin→safetensors."""
import os
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

from src.features.molecule import _load_model  # загрузчик с safetensors-fallback

DNABERT_K = 6


def _norm(s: str) -> str:
    return s.strip().upper().replace("U", "T")


def _kmerize(s: str, k: int = DNABERT_K) -> str:
    """Последовательность → строка перекрывающихся k-меров (формат входа DNABERT)."""
    s = _norm(s)
    if len(s) < k:
        return s
    return " ".join(s[i:i + k] for i in range(len(s) - k + 1))


def _atomic_savez(cache_file: Path, **arrays) -> None:
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache_file.with_name(cache_file.name + ".tmp")
    with open(tmp, "wb") as f:
        np.savez(f, **arrays)
    os.replace(tmp, cache_file)


def embed_aptamers(seqs: dict, model_name: str, cache_path: str,
                   batch_size: int = 8) -> dict:
    cache_file = Path(cache_path)
    cached: dict = {}
    if cache_file.exists():
        with np.load(cache_file, allow_pickle=False) as npz:
            cached = {k: np.array(npz[k]) for k in npz.files}

    todo = {k: _kmerize(v) for k, v in seqs.items() if k not in cached}
    if todo:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        tok = AutoTokenizer.from_pretrained(model_name)
        model = _load_model(model_name).eval().to(device)
        items = list(todo.items())
        for i in range(0, len(items), batch_size):
            chunk = items[i:i + batch_size]
            enc = tok([v for _, v in chunk], return_tensors="pt", padding=True,
                      truncation=True, max_length=512).to(device)
            with torch.no_grad():
                hs = model(**enc).last_hidden_state              # (B, L, H)
            mask = enc["attention_mask"].unsqueeze(-1).to(hs.dtype)
            pooled = (hs * mask).sum(1) / mask.sum(1).clamp(min=1.0)
            for j, (k, _) in enumerate(chunk):
                cached[k] = pooled[j].cpu().numpy().astype(np.float32)
        _atomic_savez(cache_file, **cached)

    return {k: cached[k] for k in seqs if k in cached}
