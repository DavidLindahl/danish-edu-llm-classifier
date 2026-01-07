"""Main entry point for the Danish Educational Content Classifier.

This module provides a unified CLI interface for training, evaluation, and inference.
"""

import argparse
import os
import sys
from pathlib import Path

import yaml

# Add src to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))

from src.training.train import main as train_main, load_config


def load_config_file(config_path):
    """Load configuration from YAML file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def train_command(args):
    """Handle training command."""
    print("=" * 70)
    print("Starting Training Pipeline")
    print("=" * 70)

    # Load configuration
    if args.config:
        config_path = args.config
    else:
        # Default config path
        default_config = Path(__file__).parent / "src" / "training" / "config" / "base.yaml"
        config_path = str(default_config)

    try:
        config = load_config_file(config_path)
        print(f"✓ Loaded config from: {config_path}")
    except Exception as e:
        print(f"✗ Error loading config: {e}")
        sys.exit(1)

    # Override config with CLI arguments if provided
    if args.model_name:
        config["model_name"] = args.model_name
    if args.num_danish_samples is not None:
        config["num_danish_samples"] = args.num_danish_samples
    if args.num_english_samples is not None:
        config["num_english_samples"] = args.num_english_samples
    if args.learning_rate:
        config["learning_rate"] = float(args.learning_rate)
    if args.epochs:
        config["num_train_epochs"] = args.epochs
    if args.batch_size:
        config["per_device_train_batch_size"] = args.batch_size

    # Extract parameters
    val_split = config.get("val_split", 0.1)
    model_name = config["model_name"]
    model_dir = config.get("model_dir", "models")
    num_danish_samples = config.get("num_danish_samples", 0)
    num_english_samples = config.get("num_english_samples", 5000)
    learning_rate = float(config.get("learning_rate", 3e-4))
    num_train_epochs = config.get("num_train_epochs", 5)
    per_device_train_batch_size = config.get("per_device_train_batch_size", 16)
    per_device_eval_batch_size = config.get("per_device_eval_batch_size", 32)
    evaluation_strategy = config.get("evaluation_strategy", "steps")
    save_strategy = config.get("save_strategy", "steps")
    eval_steps = config.get("eval_steps", 50)

    # Run training
    try:
        trainer, metrics = train_main(
            val_split=val_split,
            model_name=model_name,
            model_dir=model_dir,
            num_danish_samples=num_danish_samples,
            num_english_samples=num_english_samples,
            learning_rate=learning_rate,
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=per_device_train_batch_size,
            per_device_eval_batch_size=per_device_eval_batch_size,
            evaluation_strategy=evaluation_strategy,
            eval_steps=eval_steps,
            save_strategy=save_strategy,
            config=config,
        )
        print("\n" + "=" * 70)
        print("Training completed successfully!")
        print("=" * 70)
        return trainer, metrics
    except Exception as e:
        print(f"\n✗ Training failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


def eval_command(args):
    """Handle evaluation command."""
    print("=" * 70)
    print("Starting Evaluation Pipeline")
    print("=" * 70)

    if not args.model_path:
        print("✗ Error: --model-path is required for evaluation")
        sys.exit(1)

    if not os.path.exists(args.model_path):
        print(f"✗ Error: Model path does not exist: {args.model_path}")
        sys.exit(1)

    # Import evaluation function
    try:
        from src.evaluation.test import evaluate_single_model
    except ImportError as e:
        print(f"✗ Error importing evaluation module: {e}")
        sys.exit(1)

    test_data_path = args.test_data or "src/annotation/test_final.csv"
    if not os.path.exists(test_data_path):
        print(f"✗ Error: Test data file not found: {test_data_path}")
        sys.exit(1)

    device = args.device or ("cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu")
    batch_size = args.batch_size or 32

    try:
        result_df, metrics = evaluate_single_model(
            model_path=args.model_path,
            test_data_path=test_data_path,
            device_str=device,
            batch_size=batch_size,
        )

        # Save results if output path provided
        if args.output:
            output_dir = os.path.dirname(args.output) or "."
            os.makedirs(output_dir, exist_ok=True)
            result_df.to_csv(args.output, index=False)
            print(f"\n✓ Results saved to: {args.output}")

        print("\n" + "=" * 70)
        print("Evaluation completed successfully!")
        print("=" * 70)
        return result_df, metrics
    except Exception as e:
        print(f"\n✗ Evaluation failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


def main():
    """Main entry point with subcommands."""
    parser = argparse.ArgumentParser(
        description="Danish Educational Content Classifier - ML Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Train command
    train_parser = subparsers.add_parser("train", help="Train a model")
    train_parser.add_argument(
        "--config",
        type=str,
        help="Path to YAML config file (default: src/training/config/base.yaml)",
    )
    train_parser.add_argument("--model-name", type=str, help="Override model name from config")
    train_parser.add_argument("--num-danish-samples", type=int, help="Override number of Danish samples")
    train_parser.add_argument("--num-english-samples", type=int, help="Override number of English samples")
    train_parser.add_argument("--learning-rate", type=float, help="Override learning rate")
    train_parser.add_argument("--epochs", type=int, help="Override number of training epochs")
    train_parser.add_argument("--batch-size", type=int, help="Override training batch size")

    # Eval command
    eval_parser = subparsers.add_parser("eval", help="Evaluate a trained model")
    eval_parser.add_argument("--model-path", type=str, required=True, help="Path to trained model")
    eval_parser.add_argument("--test-data", type=str, help="Path to test data CSV (default: src/annotation/test_final.csv)")
    eval_parser.add_argument("--output", type=str, help="Path to save evaluation results CSV")
    eval_parser.add_argument("--device", type=str, choices=["cpu", "cuda", "mps"], help="Device to use for evaluation")
    eval_parser.add_argument("--batch-size", type=int, help="Batch size for evaluation (default: 32)")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    # Route to appropriate command handler
    if args.command == "train":
        train_command(args)
    elif args.command == "eval":
        eval_command(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
