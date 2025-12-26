#!/bin/bash

set -e

echo "Running Quick Smoke Test"
echo ""
echo "This will:"
echo "  - Train MNIST for 2 epochs"
echo "  - Use 1 GPU"
echo "  - Run evaluation"
echo ""

cd "$(dirname "$0")/.."

python main.py train configs/mnist_test_config.yaml --num-gpus 1

if [ $? -eq 0 ]; then
    echo ""
    echo "Smoke Test Passed!"
    echo ""
else
    echo ""
    echo "Smoke Test Failed"
    exit 1
fi
