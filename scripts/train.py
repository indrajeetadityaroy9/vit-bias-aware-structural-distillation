#!/usr/bin/env python
"""Thin entry point for training. See: python -m src train --help"""
import sys
from src.training.runner import train

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python scripts/train.py <config.yaml> [standard|distill|ss_distill]")
        sys.exit(1)
    config_path = sys.argv[1]
    mode = sys.argv[2] if len(sys.argv) > 2 else 'standard'
    train(config_path, mode)
