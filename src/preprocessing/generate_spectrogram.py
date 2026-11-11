"""Generate Mel spectrograms from audio using Librosa."""
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from pathlib import Path


def generate_mel_spectrogram(
    audio: np.ndarray,
    sample_rate: int = 16000,
    n_mels: int = 128,
    n_fft: int = 1024,
    hop_length: int = 512,
) -> np.ndarray:
    """
    Generate a log-Mel spectrogram.

    Returns:
        2D numpy array of shape (n_mels, time_frames) in dB.
    """
    mel_spec = librosa.feature.melspectrogram(
        y=audio,
        sr=sample_rate,
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length,
    )
    log_mel = librosa.power_to_db(mel_spec, ref=np.max)
    return log_mel


def save_spectrogram_image(
    mel_spec: np.ndarray,
    output_path: Path,
    sample_rate: int = 16000,
    hop_length: int = 512,
    title: str = "Mel Spectrogram",
):
    """Save a Mel spectrogram as an image (PNG)."""
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(
        mel_spec,
        sr=sample_rate,
        hop_length=hop_length,
        x_axis="time",
        y_axis="mel",
        cmap="viridis",
    )
    plt.colorbar(format="%+2.0f dB")
    plt.title(title)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()


def pad_or_truncate(spec: np.ndarray, target_frames: int = 256) -> np.ndarray:
    """Pad or truncate spectrogram to a fixed number of time frames."""
    n_mels, frames = spec.shape
    if frames < target_frames:
        pad_width = target_frames - frames
        spec = np.pad(spec, ((0, 0), (0, pad_width)), mode="constant", constant_values=-80.0)
    elif frames > target_frames:
        spec = spec[:, :target_frames]
    return spec