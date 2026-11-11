"""Backend logic for the Streamlit app."""
import librosa
import numpy as np
from src.models.model_inference import DeepfakeDetector
from src.features.mel_features import extract_features
from src.utils.paths import ensure_directories

def get_audio_metadata(filepath: str) -> dict:
    """Extract basic metadata from the audio file."""
    y, sr = librosa.load(filepath, sr=None)
    duration = librosa.get_duration(y=y, sr=sr)
    return {
        "duration_sec": round(duration, 2),
        "sample_rate": sr,
        "samples": len(y)
    }

def analyze_audio(filepath: str, model_choice: str = "cnn") -> dict:
    """Analyze audio and return predictions (supports single or dual model)."""
    ensure_directories()
    
    metadata = get_audio_metadata(filepath)
    
    # Generate waveform & spectrogram data
    audio, sr = librosa.load(filepath, sr=16000, mono=True)
    mel = extract_features(filepath)

    results = {}
    
    if model_choice in ["cnn", "compare"]:
        detector_cnn = DeepfakeDetector(model_type="cnn")
        results["cnn"] = detector_cnn.predict(filepath)
        
    if model_choice in ["transformer", "compare"]:
        detector_trans = DeepfakeDetector(model_type="transformer")
        results["transformer"] = detector_trans.predict(filepath)

    return {
        "predictions": results,
        "metadata": metadata,
        "audio": audio,
        "sample_rate": sr,
        "mel_spectrogram": mel,
    }