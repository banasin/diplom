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


@dataclass
class Part2Config:
    seed: int
    identity_threshold: float
    test_size: float
    kmer_k: int
    esm_model: str
    chem_model: str
    m_tokens: int
    shared_dim: int
    fusions: list
    nn: dict
    xgb: dict


def load_part2_config(path: str = "configs/part2.yaml") -> Part2Config:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return Part2Config(
        seed=int(data["seed"]),
        identity_threshold=float(data["identity_threshold"]),
        test_size=float(data["test_size"]),
        kmer_k=int(data["kmer_k"]),
        esm_model=str(data["esm_model"]),
        chem_model=str(data["chem_model"]),
        m_tokens=int(data["m_tokens"]),
        shared_dim=int(data["shared_dim"]),
        fusions=list(data["fusions"]),
        nn=dict(data["nn"]),
        xgb=dict(data["xgb"]),
    )


@dataclass
class Part3Config:
    seed: int
    identity_threshold: float
    test_size: float
    kmer_k: int
    esm_model: str
    binding_log_kd: float
    precision_ks: list
    nn: dict
    xgb: dict


def load_part3_config(path: str = "configs/part3.yaml") -> Part3Config:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return Part3Config(
        seed=int(data["seed"]),
        identity_threshold=float(data["identity_threshold"]),
        test_size=float(data["test_size"]),
        kmer_k=int(data["kmer_k"]),
        esm_model=str(data["esm_model"]),
        binding_log_kd=float(data["binding_log_kd"]),
        precision_ks=list(data["precision_ks"]),
        nn=dict(data["nn"]),
        xgb=dict(data["xgb"]),
    )
