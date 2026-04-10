#!/usr/bin/env python3
"""
Speech Emotion Recognition - Main Entry Point

Usage:
    python main.py train                     # Train model(s)
    python main.py evaluate                  # Evaluate trained model
    python main.py predict --audio <path>  # Predict emotion from audio
    python main.py benchmark                # Benchmark against top-6 papers
    python main.py download-datasets        # Download open dataset pack
"""

import argparse
import sys
import os

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def train_command(args):
    """Run training pipeline."""
    from src.train import train_model

    datasets = [name.strip() for name in args.datasets.split(",") if name.strip()]
    run_summary = train_model(
        batch_size=args.batch_size,
        epochs=args.epochs,
        datasets=datasets,
        protocol=args.protocol,
        model_variant=args.model_variant,
        feature_bundle=args.feature_bundle,
        use_focal_loss=args.use_focal_loss,
        use_augmentation=not args.no_augmentation,
        class_weighting=not args.no_class_weight,
    )
    print("\nTraining complete. Run summary:")
    print(f"  Run ID: {run_summary['run_id']}")
    print(f"  Results available for protocols: {list(run_summary['results'].keys())}")


def evaluate_command(args):
    """Run evaluation pipeline."""
    from src.evaluate import run_evaluation

    datasets = [name.strip() for name in args.datasets.split(",") if name.strip()] if args.datasets else None
    run_evaluation(model_path=args.model, datasets=datasets, protocol=args.protocol)


def benchmark_command(args):
    """Run benchmark report generation."""
    from src.benchmark import run_benchmark

    datasets = [name.strip() for name in args.datasets.split(",") if name.strip()] if args.datasets else None
    result = run_benchmark(
        papers=args.papers,
        protocol=args.protocol,
        run_id=args.run_id,
        datasets=datasets,
        train_if_missing=args.train_if_missing,
        model_variant=args.model_variant,
        feature_bundle=args.feature_bundle,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )
    print(f"Benchmark markdown: {result['report_path']}")
    print(f"Benchmark payload:  {result['payload_path']}")


def download_datasets_command(args):
    """Download dataset packs."""
    from src.download_datasets import download_datasets

    report = download_datasets(pack=args.pack, root=args.root)
    print(f"Dataset root: {report['root']}")
    for dataset_id, result in report["results"].items():
        print(f"\n[{dataset_id}] status={result['status']}")
        print(f"  path: {result['path']}")
        if result["status"] != "downloaded":
            print(f"  manual: {result.get('manual_instructions', 'n/a')}")
            errors = result.get("errors", [])
            if errors:
                print(f"  errors: {errors[-1]}")


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
    python main.py train --epochs 50 --batch-size 16 --protocol dual --model-variant attention
    python main.py train --datasets ravdess,crema_d,tess,savee --feature-bundle robust
    python main.py evaluate
    python main.py evaluate --model models/emotion_cnn_lstm_best.keras
    python main.py benchmark --papers first6 --protocol dual
    python main.py download-datasets --pack open4
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
    train_parser.add_argument(
        '--datasets', type=str, default='ravdess,crema_d,tess,savee',
        help='Comma-separated dataset IDs (default: ravdess,crema_d,tess,savee)'
    )
    train_parser.add_argument(
        '--protocol', type=str, default='dual', choices=['random', 'speaker', 'dual'],
        help='Evaluation protocol strategy (default: dual)'
    )
    train_parser.add_argument(
        '--model-variant', type=str, default='attention', choices=['baseline', 'attention', 'lightweight'],
        help='Model architecture variant (default: attention)'
    )
    train_parser.add_argument(
        '--feature-bundle', type=str, default='robust', choices=['mfcc', 'robust'],
        help='Feature bundle to extract (default: robust)'
    )
    train_parser.add_argument(
        '--use-focal-loss', action='store_true',
        help='Enable focal loss'
    )
    train_parser.add_argument(
        '--no-augmentation', action='store_true',
        help='Disable waveform/spec augmentations'
    )
    train_parser.add_argument(
        '--no-class-weight', action='store_true',
        help='Disable class weighting'
    )

    # Evaluate command
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate trained model')
    eval_parser.add_argument(
        '--model', type=str, default=None,
        help='Path to saved model (uses best model if not specified)'
    )
    eval_parser.add_argument(
        '--datasets', type=str, default=None,
        help='Optional comma-separated dataset IDs for multi-dataset evaluation'
    )
    eval_parser.add_argument(
        '--protocol', type=str, default='random', choices=['random', 'speaker', 'dual'],
        help='Split protocol for evaluation when --datasets is provided'
    )

    # Benchmark command
    benchmark_parser = subparsers.add_parser('benchmark', help='Generate benchmark report')
    benchmark_parser.add_argument(
        '--papers', type=str, default='first6',
        help='Benchmark paper set identifier (default: first6)'
    )
    benchmark_parser.add_argument(
        '--protocol', type=str, default='dual', choices=['random', 'speaker', 'dual'],
        help='Protocol selector for benchmark context'
    )
    benchmark_parser.add_argument(
        '--run-id', type=str, default=None,
        help='Existing run_id in outputs/runs to benchmark'
    )
    benchmark_parser.add_argument(
        '--datasets', type=str, default='ravdess,crema_d,tess,savee',
        help='Datasets used if auto-training is required'
    )
    benchmark_parser.add_argument(
        '--train-if-missing', action='store_true',
        help='Auto-train if no run summary exists'
    )
    benchmark_parser.add_argument(
        '--model-variant', type=str, default='attention', choices=['baseline', 'attention', 'lightweight'],
        help='Model variant for auto-training mode'
    )
    benchmark_parser.add_argument(
        '--feature-bundle', type=str, default='robust', choices=['mfcc', 'robust'],
        help='Feature bundle for auto-training mode'
    )
    benchmark_parser.add_argument(
        '--epochs', type=int, default=15,
        help='Epochs for auto-training mode'
    )
    benchmark_parser.add_argument(
        '--batch-size', type=int, default=32,
        help='Batch size for auto-training mode'
    )

    # Download datasets command
    download_parser = subparsers.add_parser('download-datasets', help='Download open dataset packs')
    download_parser.add_argument(
        '--pack', type=str, default='open4',
        help='Dataset pack id (default: open4)'
    )
    download_parser.add_argument(
        '--root', type=str, default=None,
        help='Target root directory for downloaded datasets'
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
    elif args.command == 'benchmark':
        benchmark_command(args)
    elif args.command == 'download-datasets':
        download_datasets_command(args)
    elif args.command == 'predict':
        predict_command(args)
    elif args.command == 'verify-gpu':
        verify_gpu()
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
