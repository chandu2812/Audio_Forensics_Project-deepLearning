"""
Prepare train/val/test CSVs from ASVspoof 2019 LA + WaveFake.
CSV format: file_path (project-root-relative), label (0=real/bonafide, 1=fake/spoof)
"""
import pandas as pd
from pathlib import Path
from src.utils.paths import RAW_DIR, PROCESSED_DIR, ensure_directories

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _rel(p: Path) -> str:
    try:
        return str(p.resolve().relative_to(PROJECT_ROOT.resolve()))
    except ValueError:
        return str(p.resolve())


def parse_protocol(protocol_path: Path) -> pd.DataFrame:
    """ASVspoof CM protocol: SPEAKER FILE_ID - SYSTEM KEY"""
    rows = []
    with open(protocol_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            file_id, key = parts[1], parts[-1].lower()
            rows.append({"file_id": file_id, "label": 0 if key == "bonafide" else 1})
    return pd.DataFrame(rows)


def prepare_asvspoof() -> dict:
    raw = RAW_DIR / "ASVspoof"
    proto_dir = raw / "ASVspoof2019_LA_cm_protocols"
    splits = {
        "train": (raw / "ASVspoof2019_LA_train" / "flac",
                  proto_dir / "ASVspoof2019.LA.cm.train.trn.txt"),
        "dev":   (raw / "ASVspoof2019_LA_dev"   / "flac",
                  proto_dir / "ASVspoof2019.LA.cm.dev.trl.txt"),
    }
    out = {}
    for split, (audio_dir, proto) in splits.items():
        if not proto.exists() or not audio_dir.exists():
            print(f"⚠️  Skipping ASVspoof {split}: missing files")
            print(f"    Looked for protocol: {proto}")
            print(f"    Looked for audio dir: {audio_dir}")
            continue
        df = parse_protocol(proto)
        df["file_path"] = df["file_id"].apply(
            lambda fid: _rel(audio_dir / f"{fid}.flac")
        )
        missing = df["file_path"].apply(lambda p: not (PROJECT_ROOT / p).exists()).sum()
        if missing:
            print(f"⚠️  ASVspoof {split}: {missing} files missing on disk (check extraction)")
        out[split] = df[["file_path", "label"]]
        print(f"✅ ASVspoof {split}: {len(df)} samples")
    return out


def prepare_wavefake() -> pd.DataFrame:
    raw = RAW_DIR / "WaveFake"
    rows = []
    for folder, label in [("real", 0), ("fake", 1)]:
        d = raw / folder
        if not d.exists():
            continue
        for wav in d.rglob("*.wav"):
            rows.append({"file_path": _rel(wav), "label": label})
    df = pd.DataFrame(rows)
    if not df.empty:
        print(f"✅ WaveFake: {len(df)} samples")
    return df


def build_splits(asv: dict, wf: pd.DataFrame, val_ratio=0.1, test_ratio=0.1):
    frames = []
    if "train" in asv: frames.append(asv["train"])
    if "dev"   in asv: frames.append(asv["dev"])
    if not wf.empty:   frames.append(wf)
    if not frames:
        raise RuntimeError("No data prepared — check dataset extraction")

    combined = (pd.concat(frames, ignore_index=True)
                  .sample(frac=1, random_state=42)
                  .reset_index(drop=True))

    n = len(combined)
    n_test = int(n * test_ratio)
    n_val = int(n * val_ratio)
    n_train = n - n_test - n_val

    ensure_directories()
    combined.iloc[:n_train].to_csv(PROCESSED_DIR / "train.csv", index=False)
    combined.iloc[n_train:n_train + n_val].to_csv(PROCESSED_DIR / "validation.csv", index=False)
    combined.iloc[n_train + n_val:].to_csv(PROCESSED_DIR / "test.csv", index=False)
    print(f"✅ Splits → train={n_train}  val={n_val}  test={n_test}")


if __name__ == "__main__":
    print("Preparing datasets...")
    asv = prepare_asvspoof()
    wf  = prepare_wavefake()
    build_splits(asv, wf)