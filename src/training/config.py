"""Training configuration."""
from dataclasses import dataclass, field


def _detect_device() -> str:
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class TrainingConfig:
    sample_rate: int = 16000
    n_mels: int = 128
    target_frames: int = 256
    batch_size: int = 64          # RTX 4050 has 6 GB VRAM
    num_workers: int = 4

    model_type: str = "cnn"
    n_classes: int = 2

    epochs: int = 30
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    patience: int = 7

    train_csv: str = "data/processed/train.csv"
    val_csv:   str = "data/processed/validation.csv"
    test_csv:  str = "data/processed/test.csv"
    audio_root: str = "."         # project root; CSV paths are project-root-relative
    checkpoint_dir: str = "models"
    output_dir: str = "outputs"

    device: str = field(default_factory=_detect_device)