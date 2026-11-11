"""Inference wrapper for trained models."""
import torch
import numpy as np
from src.features.mel_features import extract_features
from src.models.cnn_model import AudioCNN
from src.models.transformer_model import AudioTransformer


class DeepfakeDetector:
    """Unified inference interface for CNN or Transformer models."""

    def __init__(self, model_type: str = "cnn", device: str = "cpu"):
        self.device = torch.device(device)
        self.model_type = model_type
        self.model = self._load_model()
        self.model.eval()

    def _load_model(self):
        if self.model_type == "cnn":
            model = AudioCNN()
        elif self.model_type == "transformer":
            model = AudioTransformer()
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

        # Load checkpoint if available
        from src.utils.paths import CNN_BEST_MODEL, TRANSFORMER_BEST_MODEL
        ckpt_path = CNN_BEST_MODEL if self.model_type == "cnn" else TRANSFORMER_BEST_MODEL
        if ckpt_path.exists():
            state = torch.load(ckpt_path, map_location=self.device)
            model.load_state_dict(state)
            print(f"[inference] Loaded weights from {ckpt_path}")
        else:
            print(f"[inference] WARNING: No checkpoint at {ckpt_path}. Using random weights.")

        return model.to(self.device)

    @torch.no_grad()
    def predict(self, filepath: str) -> dict:
        """
        Predict whether an audio file is real or synthetic.

        Returns:
            dict with keys: label, confidence, probabilities
        """
        mel = extract_features(filepath)
        tensor = torch.tensor(mel, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        tensor = tensor.to(self.device)

        logits = self.model(tensor)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        pred = int(np.argmax(probs))

        return {
            "label": "Synthetic" if pred == 1 else "Real",
            "confidence": float(probs[pred]),
            "probabilities": {"real": float(probs[0]), "synthetic": float(probs[1])},
        }