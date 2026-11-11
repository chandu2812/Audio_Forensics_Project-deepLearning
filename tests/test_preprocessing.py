"""Unit tests for preprocessing modules."""
import numpy as np
import pytest
from src.preprocessing.normalize_audio import normalize_peak, normalize_audio
from src.preprocessing.clean_audio import remove_dc_offset
from src.preprocessing.generate_spectrogram import pad_or_truncate, generate_mel_spectrogram


def test_normalize_peak():
    audio = np.array([0.1, -0.5, 0.3])
    normalized = normalize_peak(audio, target_peak=1.0)
    assert np.isclose(np.max(np.abs(normalized)), 1.0)


def test_remove_dc_offset():
    audio = np.array([1.0, 2.0, 3.0])
    result = remove_dc_offset(audio)
    assert np.isclose(np.mean(result), 0.0)


def test_pad_or_truncate_padding():
    spec = np.random.randn(128, 100)
    result = pad_or_truncate(spec, target_frames=256)
    assert result.shape == (128, 256)


def test_pad_or_truncate_truncation():
    spec = np.random.randn(128, 300)
    result = pad_or_truncate(spec, target_frames=256)
    assert result.shape == (128, 256)


def test_generate_mel_spectrogram_shape():
    audio = np.random.randn(16000)  # 1 second at 16 kHz
    mel = generate_mel_spectrogram(audio, sample_rate=16000, n_mels=128)
    assert mel.shape[0] == 128
    assert mel.shape[1] > 0