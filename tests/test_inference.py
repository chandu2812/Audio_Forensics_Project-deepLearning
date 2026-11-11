"""Unit tests for inference (smoke tests, no trained weights required)."""
import torch
import pytest
from src.models.cnn_model import AudioCNN
from src.models.transformer_model import AudioTransformer


def test_cnn_forward_shape():
    model = AudioCNN()
    x = torch.randn(2, 1, 128, 256)  # batch=2, channel=1, n_mels=128, frames=256
    out = model(x)
    assert out.shape == (2, 2)


def test_transformer_forward_shape():
    model = AudioTransformer(n_mels=128, target_frames=256)
    x = torch.randn(2, 1, 128, 256)
    out = model(x)
    assert out.shape == (2, 2)


def test_cnn_output_is_finite():
    model = AudioCNN()
    x = torch.randn(1, 1, 128, 256)
    out = model(x)
    assert torch.isfinite(out).all()