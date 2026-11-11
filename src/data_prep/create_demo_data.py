"""
Generate a small synthetic dataset for immediate pipeline testing.
NOT suitable for real training — just to verify code works end-to-end.
"""
import numpy as np
import soundfile as sf
import pandas as pd
from pathlib import Path

SR = 16000
DURATION = 2.0


def generate_audio(label: str) -> np.ndarray:
    t = np.linspace(0, DURATION, int(SR * DURATION), endpoint=False)
    if label == "real":
        # speech-like: multi-harmonic + slow amplitude envelope
        sig = (0.5 * np.sin(2 * np.pi * 150 * t)
               + 0.3 * np.sin(2 * np.pi * 300 * t)
               + 0.2 * np.sin(2 * np.pi * 450 * t))
        env = 0.5 + 0.5 * np.sin(2 * np.pi * 3 * t)
    else:
        # synth-like: single tone + quantization artifacts
        sig = np.sin(2 * np.pi * 180 * t)
        sig = np.round(sig * 8) / 8.0
        env = 0.5 + 0.5 * np.sin(2 * np.pi * 5 * t)
    sig = sig * env + 0.02 * np.random.randn(len(t))
    sig = sig / (np.max(np.abs(sig)) + 1e-8) * 0.9
    return sig.astype(np.float32)


def main():
    root = Path("data/processed")
    audio_root = root / "audio"
    rows = []

    for label_name, label_val, count in [("real", 0, 50), ("fake", 1, 50)]:
        d = audio_root / label_name
        d.mkdir(parents=True, exist_ok=True)
        for i in range(count):
            sig = generate_audio(label_name)
            fname = f"{label_name}_{i:04d}.wav"
            sf.write(d / fname, sig, SR)
            rows.append({"file_path": str(d / fname), "label": label_val})

    df = pd.DataFrame(rows).sample(frac=1, random_state=42).reset_index(drop=True)
    n = len(df)
    n_train = int(0.7 * n)
    n_val = int(0.15 * n)
    df.iloc[:n_train].to_csv(root / "train.csv", index=False)
    df.iloc[n_train:n_train + n_val].to_csv(root / "validation.csv", index=False)
    df.iloc[n_train + n_val:].to_csv(root / "test.csv", index=False)
    print(f"✅ Demo dataset ready: {n_train} train / {n_val} val / {n - n_train - n_val} test")


if __name__ == "__main__":
    main()