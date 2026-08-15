"""Загрузка параметров эксперимента Части 1 из YAML в датакласс."""
from dataclasses import dataclass
from pathlib import Path
import yaml


@dataclass
class Config:
    seed: int
    identity_threshold: float
    test_size: float
    kmer_k: int
    esm_model: str
    xgb: dict
    nn: dict


def load_config(path: str = "configs/part1.yaml") -> Config:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return Config(
        seed=int(data["seed"]),
        identity_threshold=float(data["identity_threshold"]),
        test_size=float(data["test_size"]),
        kmer_k=int(data["kmer_k"]),
        esm_model=str(data["esm_model"]),
        xgb=dict(data["xgb"]),
        nn=dict(data["nn"]),
    )
