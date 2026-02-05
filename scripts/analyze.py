#!/usr/bin/env python
"""Thin entry point for analytics. See: python -m src analyze --help"""
import sys
from src.training.runner import analyze

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python scripts/analyze.py <config.yaml> <checkpoint.pth> [--metrics all]")
        sys.exit(1)
    config_path = sys.argv[1]
    checkpoint_path = sys.argv[2]
    metrics = 'all'
    output_dir = None
    for i, arg in enumerate(sys.argv[3:], 3):
        if arg == '--metrics' and i + 1 < len(sys.argv):
            metrics = sys.argv[i + 1]
        elif arg == '--output-dir' and i + 1 < len(sys.argv):
            output_dir = sys.argv[i + 1]
    analyze(config_path, checkpoint_path, metrics, output_dir)
