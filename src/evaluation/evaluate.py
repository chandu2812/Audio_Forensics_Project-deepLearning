"""Evaluate a trained model on the test set."""
import torch
import numpy as np
from tqdm import tqdm
from src.evaluation.plot_results import plot_confusion_matrix, plot_roc_curve
from src.training.config import TrainingConfig
from src.features.dataset_loader import create_dataloaders
from src.models.cnn_model import AudioCNN
from src.models.transformer_model import AudioTransformer
from src.evaluation.metrics import compute_all_metrics
from src.utils.paths import CNN_BEST_MODEL, TRANSFORMER_BEST_MODEL


@torch.no_grad()
def evaluate_model(model_type: str = "cnn", config: TrainingConfig = TrainingConfig()):
    device = torch.device(config.device)

    _, _, test_loader = create_dataloaders(
        config.train_csv, config.val_csv, config.test_csv,
        config.audio_root, batch_size=config.batch_size,
        num_workers=config.num_workers, sample_rate=config.sample_rate,
        n_mels=config.n_mels, target_frames=config.target_frames,
    )

    if model_type == "cnn":
        model = AudioCNN(n_mels=config.n_mels).to(device)
        ckpt = CNN_BEST_MODEL
    elif model_type == "transformer":
        model = AudioTransformer(
            n_mels=config.n_mels, target_frames=config.target_frames
        ).to(device)
        ckpt = TRANSFORMER_BEST_MODEL
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()

    all_labels = []
    all_preds = []
    all_probs = []

    for mel, labels in tqdm(test_loader, desc="Evaluating"):
        mel, labels = mel.to(device), labels.to(device)
        logits = model(mel)
        probs = torch.softmax(logits, dim=1)
        preds = logits.argmax(dim=1)

        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())
        all_probs.extend(probs[:, 1].cpu().numpy())

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    y_probs = np.array(all_probs)

    metrics = compute_all_metrics(y_true, y_pred, y_probs)
    print(f"\n=== {model_type.upper()} Evaluation Results ===")
    for name, value in metrics.items():
        print(f"  {name:12s}: {value:.4f}")
        
    plot_confusion_matrix(y_true, y_pred, model_name=model_type)
    plot_roc_curve(y_true, y_probs, model_name=model_type)

    return metrics, y_true, y_pred, y_probs


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="cnn", choices=["cnn", "transformer"])
    args = parser.parse_args()
    evaluate_model(args.model)