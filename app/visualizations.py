"""Visualization helpers for the Streamlit app."""
import matplotlib.pyplot as plt
import librosa.display
import numpy as np


def plot_waveform(audio: np.ndarray, sr: int) -> plt.Figure:
    """Create a waveform plot."""
    fig, ax = plt.subplots(figsize=(10, 2))
    librosa.display.waveshow(audio, sr=sr, ax=ax, color="steelblue")
    ax.set_title("Waveform")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    plt.tight_layout()
    return fig


def plot_spectrogram_image(mel: np.ndarray, sr: int = 16000) -> plt.Figure:
    """Create a Mel spectrogram plot."""
    fig, ax = plt.subplots(figsize=(10, 4))
    img = librosa.display.specshow(
        mel, sr=sr, hop_length=512,
        x_axis="time", y_axis="mel", cmap="viridis", ax=ax,
    )
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    ax.set_title("Mel Spectrogram")
    plt.tight_layout()
    return fig