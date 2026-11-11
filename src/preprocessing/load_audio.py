"""Audio loading utilities."""
import librosa
import numpy as np
from typing import Tuple, Optional


def load_audio(
    filepath: str,
    sample_rate: int = 16000,
    mono: bool = True,
    duration: Optional[float] = None,
) -> Tuple[np.ndarray, int]:
    """
    Load an audio file and resample to target sample rate.

    Args:
        filepath: Path to audio file.
        sample_rate: Target sample rate (default 16000 Hz for ASVspoof).
        mono: Convert to mono if True.
        duration: Optional max duration in seconds.

    Returns:
        (audio_array, sample_rate)
    """
    audio, sr = librosa.load(
        filepath,
        sr=sample_rate,
        mono=mono,
        duration=duration,
    )
    return audio, sr