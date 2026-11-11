"""Audio cleaning: trim silence, remove DC offset."""
import numpy as np
import librosa


def trim_silence(audio: np.ndarray, top_db: float = 20.0) -> np.ndarray:
    """Trim leading/trailing silence."""
    return librosa.effects.trim(audio, top_db=top_db)[0]


def remove_dc_offset(audio: np.ndarray) -> np.ndarray:
    """Remove DC offset by subtracting the mean."""
    return audio - np.mean(audio)


def clean_audio_pipeline(audio: np.ndarray) -> np.ndarray:
    """Full cleaning pipeline."""
    audio = remove_dc_offset(audio)
    audio = trim_silence(audio)
    return audio