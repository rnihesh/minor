#!/usr/bin/env python3
"""
Speech Emotion Recognition - Main Entry Point

Usage:
    python main.py train              # Train the model
    python main.py evaluate           # Evaluate trained model
    python main.py predict --audio <path>  # Predict emotion from audio
"""

import argparse
import sys
import os

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def train_command(args):
    """Run training pipeline."""
    from src.train import train_model
    from src.evaluate import plot_training_history, evaluate_model

    model, history, (X_test, y_test) = train_model(
        batch_size=args.batch_size,
        epochs=args.epochs
    )

    # Plot training history
    plot_training_history(history)

    # Evaluate on test set
    print("\nEvaluating on test set...")
    evaluate_model(model, X_test, y_test)


def evaluate_command(args):
    """Run evaluation pipeline."""
    from src.evaluate import run_evaluation

    run_evaluation(model_path=args.model)


def predict_command(args):
    """Run prediction on audio file."""
    from src.predict import predict_emotion, format_prediction

    if not args.audio:
        print("Error: --audio argument is required")
        sys.exit(1)

    result = predict_emotion(args.audio, model_path=args.model)
    print(format_prediction(result))


def verify_gpu():
    """Verify TensorFlow GPU (Metal) availability."""
    import tensorflow as tf

    print("TensorFlow version:", tf.__version__)
    print("\nPhysical devices:")
    for device in tf.config.list_physical_devices():
        print(f"  {device.device_type}: {device.name}")

    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"\nMetal GPU acceleration: ENABLED ({len(gpus)} GPU(s))")
    else:
        print("\nMetal GPU acceleration: NOT AVAILABLE (using CPU)")


def main():
    parser = argparse.ArgumentParser(
        description='Speech Emotion Recognition using CNN-LSTM',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python main.py train
    python main.py train --epochs 50 --batch-size 16
    python main.py evaluate
    python main.py evaluate --model models/emotion_cnn_lstm_best.keras
    python main.py predict --audio archive/Actor_01/03-01-01-01-01-01-01.wav
    python main.py verify-gpu
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Train command
    train_parser = subparsers.add_parser('train', help='Train the model')
    train_parser.add_argument(
        '--epochs', type=int, default=100,
        help='Maximum number of training epochs (default: 100)'
    )
    train_parser.add_argument(
        '--batch-size', type=int, default=32,
        help='Training batch size (default: 32)'
    )

    # Evaluate command
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate trained model')
    eval_parser.add_argument(
        '--model', type=str, default=None,
        help='Path to saved model (uses best model if not specified)'
    )

    # Predict command
    pred_parser = subparsers.add_parser('predict', help='Predict emotion from audio')
    pred_parser.add_argument(
        '--audio', type=str, required=True,
        help='Path to audio file'
    )
    pred_parser.add_argument(
        '--model', type=str, default=None,
        help='Path to saved model (uses best model if not specified)'
    )

    # Verify GPU command
    subparsers.add_parser('verify-gpu', help='Verify Metal GPU availability')

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    if args.command == 'train':
        train_command(args)
    elif args.command == 'evaluate':
        evaluate_command(args)
    elif args.command == 'predict':
        predict_command(args)
    elif args.command == 'verify-gpu':
        verify_gpu()
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
