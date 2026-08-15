"""ChemBERTa эмбеддинги малых молекул: mean-pooling (одиночный вектор) и
chunk-pooling в M «токенов» (для cross-attention). Кэш .npz (вне git)."""
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from src.features.pooling import chunk_pool


def _atomic_savez(cache_file: Path, **arrays) -> None:
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache_file.with_name(cache_file.name + ".tmp")
    with open(tmp, "wb") as f:
        np.savez(f, **arrays)
    os.replace(tmp, cache_file)


def _load_model(model_name: str):
    """AutoModel.from_pretrained, с обходом окружения: некоторые чекпойнты
    (напр. ChemBERTa-77M) распространяются только как pickle-`.bin`, а
    transformers >= 4.5x отказывается грузить pickle-веса без torch >= 2.6
    (CVE-2025-32434). Если это тот случай — конвертируем чекпойнт в
    safetensors локально (во временную копию снапшота) и грузим уже его;
    сама модель и веса при этом не меняются."""
    try:
        return AutoModel.from_pretrained(model_name)
    except ValueError as err:
        if "torch.load" not in str(err):
            raise
        return _load_model_via_local_safetensors(model_name)


def _load_model_via_local_safetensors(model_name: str):
    from huggingface_hub import snapshot_download
    from safetensors.torch import save_file

    snapshot = Path(snapshot_download(model_name))
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp = Path(tmp_dir)
        weight_files = [f for f in snapshot.iterdir()
                        if f.suffix in (".bin", ".pt", ".pth")
                        and f.stem.startswith(("pytorch_model", "model"))]
        for f in snapshot.iterdir():
            if f not in weight_files:
                shutil.copy(f, tmp / f.name)
        state_dict: dict = {}
        for wf in weight_files:
            sd = torch.load(wf, map_location="cpu", weights_only=True)
            state_dict.update({k: v.clone().contiguous() for k, v in sd.items()})
        save_file(state_dict, str(tmp / "model.safetensors"))
        return AutoModel.from_pretrained(str(tmp))


def embed_molecules(smiles: dict, model_name: str, cache_path: str,
                    m_tokens: int, batch_size: int = 16):
    cache_file = Path(cache_path)
    pooled: dict = {}
    tokens: dict = {}
    if cache_file.exists():
        with np.load(cache_file, allow_pickle=False) as npz:
            for k in npz.files:
                if k.startswith("p::"):
                    pooled[k[3:]] = np.array(npz[k])
                elif k.startswith("t::"):
                    tokens[k[3:]] = np.array(npz[k])

    todo = {k: v for k, v in smiles.items() if k not in pooled}
    if todo:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        tok = AutoTokenizer.from_pretrained(model_name)
        model = _load_model(model_name).eval().to(device)
        items = list(todo.items())
        for i in range(0, len(items), batch_size):
            chunk = items[i:i + batch_size]
            enc = tok([v for _, v in chunk], return_tensors="pt",
                      padding=True, truncation=True, max_length=256).to(device)
            with torch.no_grad():
                out = model(**enc).last_hidden_state          # (B, L, H)
            mask = enc["attention_mask"].bool()
            for j, (k, _) in enumerate(chunk):
                valid = out[j][mask[j]].cpu().numpy().astype(np.float32)  # (Lv, H)
                pooled[k] = valid.mean(0)
                tokens[k] = chunk_pool(valid, m_tokens)
        arrays = {f"p::{k}": v for k, v in pooled.items()}
        arrays.update({f"t::{k}": v for k, v in tokens.items()})
        _atomic_savez(cache_file, **arrays)

    return ({k: pooled[k] for k in smiles if k in pooled},
            {k: tokens[k] for k in smiles if k in tokens})
