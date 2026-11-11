"""Training script for the Transformer model."""
import torch
import torch.nn as nn
from torch.optim import AdamW
from tqdm import tqdm
from src.training.config import TrainingConfig
from src.features.dataset_loader import create_dataloaders
from src.models.transformer_model import AudioTransformer
from src.utils.paths import TRANSFORMER_BEST_MODEL, ensure_directories
from src.utils.reproducibility import set_seed


def train_transformer(config: TrainingConfig = TrainingConfig()):
    set_seed(42)
    ensure_directories()

    device = torch.device(config.device)
    print(f"[train_transformer] Using device: {device}")

    train_loader, val_loader, _ = create_dataloaders(
        config.train_csv, config.val_csv, config.test_csv,
        config.audio_root, batch_size=config.batch_size,
        num_workers=config.num_workers, sample_rate=config.sample_rate,
        n_mels=config.n_mels, target_frames=config.target_frames,
    )

    model = AudioTransformer(
        n_mels=config.n_mels,
        target_frames=config.target_frames,
        n_classes=config.n_classes,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(1, config.epochs + 1):
        model.train()
        train_loss = 0.0
        for mel, labels in tqdm(train_loader, desc=f"Epoch {epoch}/{config.epochs} [Train]"):
            mel, labels = mel.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(mel)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for mel, labels in tqdm(val_loader, desc=f"Epoch {epoch}/{config.epochs} [Val]"):
                mel, labels = mel.to(device), labels.to(device)
                logits = model(mel)
                loss = criterion(logits, labels)
                val_loss += loss.item()
                preds = logits.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        avg_val_loss = val_loss / len(val_loader)
        val_acc = correct / total
        print(f"Epoch {epoch}: train_loss={avg_train_loss:.4f}, "
              f"val_loss={avg_val_loss:.4f}, val_acc={val_acc:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), TRANSFORMER_BEST_MODEL)
            print(f"  → Best model saved to {TRANSFORMER_BEST_MODEL}")
        else:
            patience_counter += 1
            if patience_counter >= config.patience:
                print(f"[train_transformer] Early stopping at epoch {epoch}")
                break

    print("[train_transformer] Training complete.")
    return model


if __name__ == "__main__":
    train_transformer()