"""Main entry point: orchestrate the full pipeline."""
import argparse
from src.utils.paths import ensure_directories
from src.utils.reproducibility import set_seed
from src.training.config import TrainingConfig


def main():
    parser = argparse.ArgumentParser(description="Audio Forensics Pipeline")
    parser.add_argument(
        "command",
        choices=["train-cnn", "train-transformer", "evaluate-cnn", "evaluate-transformer", "setup"],
        help="Command to run",
    )
    parser.add_argument("--epochs", type=int, default=None, help="Override training epochs")
    args = parser.parse_args()

    set_seed(42)
    ensure_directories()

    config = TrainingConfig()
    if args.epochs is not None:
        config.epochs = args.epochs

    if args.command == "setup":
        print("✅ Project directories verified.")
        return

    if args.command == "train-cnn":
        from src.training.train_cnn import train_cnn
        train_cnn(config)

    elif args.command == "train-transformer":
        from src.training.train_transformer import train_transformer
        train_transformer(config)

    elif args.command == "evaluate-cnn":
        from src.evaluation.evaluate import evaluate_model
        evaluate_model("cnn", config)

    elif args.command == "evaluate-transformer":
        from src.evaluation.evaluate import evaluate_model
        evaluate_model("transformer", config)


if __name__ == "__main__":
    main()