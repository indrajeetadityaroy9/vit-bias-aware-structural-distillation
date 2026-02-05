"""
CLI entry point for the research framework.

Usage:
    python -m src train config.yaml
    python -m src train-distill config.yaml
    python -m src train-ss-distill config.yaml
    python -m src evaluate config.yaml checkpoint.pth
    python -m src analyze config.yaml checkpoint.pth [--metrics all] [--output-dir path]
    torchrun --nproc_per_node=N -m src train config.yaml
"""

import argparse

from src.training.runner import train, evaluate, analyze


def main():
    parser = argparse.ArgumentParser(description='H100 Training CLI')
    subparsers = parser.add_subparsers(dest='command')

    # Train
    train_p = subparsers.add_parser('train')
    train_p.add_argument('config', type=str)

    # Distill
    distill_p = subparsers.add_parser('train-distill')
    distill_p.add_argument('config', type=str)

    # SS Distill
    ss_p = subparsers.add_parser('train-ss-distill')
    ss_p.add_argument('config', type=str)

    # Evaluate
    eval_p = subparsers.add_parser('evaluate')
    eval_p.add_argument('config', type=str)
    eval_p.add_argument('checkpoint', type=str)

    # Analyze
    analyze_p = subparsers.add_parser('analyze')
    analyze_p.add_argument('config', type=str)
    analyze_p.add_argument('checkpoint', type=str)
    analyze_p.add_argument('--metrics', type=str, default='all')
    analyze_p.add_argument('--output-dir', type=str, default=None)

    args = parser.parse_args()

    if args.command == 'train':
        train(args.config, 'standard')
    elif args.command == 'train-distill':
        train(args.config, 'distill')
    elif args.command == 'train-ss-distill':
        train(args.config, 'ss_distill')
    elif args.command == 'evaluate':
        evaluate(args.config, args.checkpoint)
    elif args.command == 'analyze':
        analyze(args.config, args.checkpoint, args.metrics, args.output_dir)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
