def test_core_imports():
    import numpy, pandas, sklearn, scipy, yaml  # noqa: F401
    import xgboost  # noqa: F401


def test_torch_cuda_available():
    import torch
    assert torch.cuda.is_available(), "CUDA недоступна — проверь установку torch"


def test_esm_import():
    import esm  # noqa: F401
    assert hasattr(esm.pretrained, "esm2_t33_650M_UR50D")
