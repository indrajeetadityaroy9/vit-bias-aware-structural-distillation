#!/usr/bin/env python
"""Thin entry point for evaluation. See: python -m src evaluate --help"""
import sys
from src.training.runner import evaluate

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python scripts/eval.py <config.yaml> <checkpoint.pth>")
        sys.exit(1)
    evaluate(sys.argv[1], sys.argv[2])
