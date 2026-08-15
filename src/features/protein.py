"""ESM-2 эмбеддинги белков: mean-pooling представлений последнего слоя.

Эмбеддинг каждого белка считается один раз и кэшируется (.npz, вне git)."""
from pathlib import Path

import numpy as np
import torch
import esm

_MODEL_LAYERS = {
    "esm2_t6_8M_UR50D": 6,
    "esm2_t12_35M_UR50D": 12,
    "esm2_t30_150M_UR50D": 30,
    "esm2_t33_650M_UR50D": 33,
}
MAX_RESIDUES = 1022


def _load_model(model_name: str):
    model, alphabet = getattr(esm.pretrained, model_name)()
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    return model, alphabet, alphabet.get_batch_converter(), device


def embed_sequences(seqs: dict[str, str], model_name: str, cache_path: str,
                    batch_size: int = 8) -> dict[str, np.ndarray]:
    cache_file = Path(cache_path)
    cached: dict[str, np.ndarray] = {}
    if cache_file.exists():
        npz = np.load(cache_file, allow_pickle=False)
        cached = {k: npz[k] for k in npz.files}

    todo = {k: v[:MAX_RESIDUES] for k, v in seqs.items() if k not in cached}
    if todo:
        model, _, batch_converter, device = _load_model(model_name)
        layer = _MODEL_LAYERS[model_name]
        items = list(todo.items())
        for i in range(0, len(items), batch_size):
            chunk = items[i:i + batch_size]
            _, _, tokens = batch_converter([(k, v) for k, v in chunk])
            tokens = tokens.to(device)
            with torch.no_grad():
                out = model(tokens, repr_layers=[layer], return_contacts=False)
            reps = out["representations"][layer]           # (B, L, D)
            for j, (k, v) in enumerate(chunk):
                # исключаем BOS(0) и EOS(len+1); усредняем по реальным остаткам
                emb = reps[j, 1:len(v) + 1].mean(0).cpu().numpy().astype(np.float32)
                cached[k] = emb
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache_file, **cached)

    return {k: cached[k] for k in seqs if k in cached}
