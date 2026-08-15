import torch
from src.models.fusion import ConcatFusion, BilinearFusion, CrossAttentionFusion


def test_concat_fusion_shape():
    f = ConcatFusion(6, 5, 8)
    out = f(torch.randn(4, 6), torch.randn(4, 5))
    assert out.shape == (4, 8)


def test_bilinear_fusion_shape():
    f = BilinearFusion(6, 5, 8)
    out = f(torch.randn(4, 6), torch.randn(4, 5))
    assert out.shape == (4, 8)


def test_cross_attention_fusion_shape():
    f = CrossAttentionFusion(dim=8, heads=2, out_dim=8)
    out = f(torch.randn(4, 3, 8), torch.randn(4, 5, 8))
    assert out.shape == (4, 8)
