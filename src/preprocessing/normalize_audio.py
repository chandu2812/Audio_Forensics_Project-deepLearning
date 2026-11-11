"""Audio normalization utilities."""
import numpy as np


def normalize_peak(audio: np.ndarray, target_peak: float = 0.95) -> np.ndarray:
    """Normalize audio so the peak amplitude equals target_peak."""
    peak = np.max(np.abs(audio))
    if peak > 0:
        return audio * (target_peak / peak)
    return audio


def normalize_rms(audio: np.ndarray, target_rms: float = 0.1) -> np.ndarray:
    """Normalize audio by RMS energy."""
    rms = np.sqrt(np.mean(audio ** 2))
    if rms > 0:
        return audio * (target_rms / rms)
    return audio


def normalize_audio(audio: np.ndarray, method: str = "peak") -> np.ndarray:
    """Apply chosen normalization method."""
    if method == "peak":
        return normalize_peak(audio)
    elif method == "rms":
        return normalize_rms(audio)
    else:
        raise ValueError(f"Unknown normalization method: {method}")