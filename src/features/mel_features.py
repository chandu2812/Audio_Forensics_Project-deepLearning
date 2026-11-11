"""Mel feature extraction pipeline."""
import numpy as np
from src.preprocessing.load_audio import load_audio
from src.preprocessing.clean_audio import clean_audio_pipeline
from src.preprocessing.normalize_audio import normalize_audio
from src.preprocessing.generate_spectrogram import (
    generate_mel_spectrogram,
    pad_or_truncate,
)


def extract_features(
    filepath: str,
    sample_rate: int = 16000,
    n_mels: int = 128,
    target_frames: int = 256,
    normalize_method: str = "peak",
) -> np.ndarray:
    """
    End-to-end feature extraction: load → clean → normalize → Mel spectrogram.

    Returns:
        Log-Mel spectrogram of shape (n_mels, target_frames).
    """
    audio, sr = load_audio(filepath, sample_rate=sample_rate)
    audio = clean_audio_pipeline(audio)
    audio = normalize_audio(audio, method=normalize_method)
    mel = generate_mel_spectrogram(audio, sample_rate=sr, n_mels=n_mels)
    mel = pad_or_truncate(mel, target_frames=target_frames)
    return mel