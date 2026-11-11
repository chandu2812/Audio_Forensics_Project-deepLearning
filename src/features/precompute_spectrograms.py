"""
Precompute Mel spectrograms for all audio in the CSVs.
Saves .npy files to data/processed/spectrograms/.
"""
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from src.features.mel_features import extract_features

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SPECT_DIR = PROJECT_ROOT / "data" / "processed" / "spectrograms"

def cache_split(csv_path: Path, name: str):
    df = pd.read_csv(csv_path)
    print(f"[{name}] {len(df)} files to cache")

    for _, row in tqdm(df.iterrows(), total=len(df), desc=name):
        src = Path(row["file_path"])
        if not src.is_absolute():
            src = PROJECT_ROOT / src
        if not src.exists():
            continue

        # Save as .npy in a mirrored folder structure
        rel = src.relative_to(PROJECT_ROOT)
        out = SPECT_DIR / rel.with_suffix(".npy")
        out.parent.mkdir(parents=True, exist_ok=True)

        if out.exists():
            continue

        try:
            mel = extract_features(str(src))
            np.save(out, mel.astype(np.float32))
        except Exception as e:
            print(f"Error {src}: {e}")

if __name__ == "__main__":
    SPECT_DIR.mkdir(parents=True, exist_ok=True)
    for split, path in [
        ("train", PROJECT_ROOT / "data/processed/train.csv"),
        ("val",   PROJECT_ROOT / "data/processed/validation.csv"),
        ("test",  PROJECT_ROOT / "data/processed/test.csv"),
    ]:
        if path.exists() and path.stat().st_size > 0:
            cache_split(path, split)
    print("✅ Spectrogram cache complete.")