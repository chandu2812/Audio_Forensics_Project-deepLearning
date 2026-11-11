"""PyTorch Dataset and DataLoader for audio deepfake detection."""
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from pathlib import Path
from src.features.mel_features import extract_features
from src.utils.paths import PROJECT_ROOT

class AudioDeepfakeDataset(Dataset):
    """Dataset that loads audio, extracts Mel spectrograms, and returns tensors."""

    def __init__(
        self,
        csv_path: str,
        audio_root: str,
        sample_rate: int = 16000,
        n_mels: int = 128,
        target_frames: int = 256,
        transform=None,
    ):
        self.df = pd.read_csv(csv_path)
        self.audio_root = Path(audio_root)
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.target_frames = target_frames
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        fp = Path(row["file_path"])
        candidate = fp if fp.is_absolute() else Path(self.audio_root) / fp
        if not candidate.exists() and not fp.is_absolute() and fp.exists():
            candidate = fp

        if not candidate.exists():
            raise FileNotFoundError(f"Audio file not found: {candidate}")

        label = int(row["label"])

        # --- Check for precomputed .npy cache ---
        try:
            # Recreate the path relative to the project root
            rel = candidate.resolve().relative_to(PROJECT_ROOT.resolve())
            npy_path = (PROJECT_ROOT / "data" / "processed" / "spectrograms" / rel).with_suffix(".npy")
        except ValueError:
            npy_path = None

        if npy_path is not None and npy_path.exists():
            mel = np.load(npy_path)
        else:
            # Fallback to live extraction if cache is missing
            mel = extract_features(str(candidate))

        mel = np.expand_dims(mel, axis=0)
        tensor = torch.tensor(mel, dtype=torch.float32)

        if self.transform:
            tensor = self.transform(tensor)

        return tensor, label


def create_dataloaders(
    train_csv: str,
    val_csv: str,
    test_csv: str,
    audio_root: str,
    batch_size: int = 64,    # Increased for RTX 4050
    num_workers: int = 4,    # Optimized for Windows/Laptop CPUs
    sample_rate: int = 16000,
    n_mels: int = 128,
    target_frames: int = 256,
):
    """Create train, validation, and test DataLoaders."""
    train_ds = AudioDeepfakeDataset(
        train_csv, audio_root, sample_rate=sample_rate,
        n_mels=n_mels, target_frames=target_frames
    )
    val_ds = AudioDeepfakeDataset(
        val_csv, audio_root, sample_rate=sample_rate,
        n_mels=n_mels, target_frames=target_frames
    )
    test_ds = AudioDeepfakeDataset(
        test_csv, audio_root, sample_rate=sample_rate,
        n_mels=n_mels, target_frames=target_frames
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    return train_loader, val_loader, test_loader