"""Парсинг выходных файлов MNA/N-граммных SAR-моделей (OLMPASS/MultiPASS):
per-target IAP из `*_targets.csv`. Сами модели проприетарны и не запускаются;
используются только их выходы (Часть 4, H8)."""
from pathlib import Path

import pandas as pd


def parse_targets_csv(path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=";", decimal=",", encoding="utf-8-sig")
    df["IAP"] = pd.to_numeric(
        df["IAP"].astype(str).str.replace(",", ".", regex=False), errors="coerce")
    return df


def load_olmpass_iap(best_models_dir) -> pd.DataFrame:
    rows = []
    for thr_dir in sorted(Path(best_models_dir).glob("*nM")):
        for f in sorted(thr_dir.glob("*_targets.csv")):
            parts = f.stem.split("_")                    # DNA_MNA_22_targets
            aptamer_type, descriptor = parts[0], parts[1]
            try:
                df = parse_targets_csv(f)
            except pd.errors.EmptyDataError:
                continue                              # пустой/битый файл модели — пропускаем
            for r in df.itertuples():
                rows.append({
                    "threshold": thr_dir.name,
                    "aptamer_type": aptamer_type,
                    "descriptor": descriptor,
                    "target_identifier": getattr(r, "Identifier"),
                    "iap": getattr(r, "IAP"),
                    "n_actives": getattr(r, "N"),
                })
    return pd.DataFrame(rows)
