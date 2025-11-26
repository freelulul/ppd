#!/bin/bash
# Environment setup script for P-PD experiments
#
# This script installs SGLang and its dependencies for P/D disaggregation experiments.
#
# Usage:
#   ./setup_environment.sh
#
# Prerequisites:
#   - CUDA 12.x
#   - Python 3.10+
#   - pip

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PPD_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

echo "=========================================="
echo "P-PD Environment Setup"
echo "=========================================="
echo ""
echo "PPD Root: $PPD_ROOT"
echo ""

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $PYTHON_VERSION"

# Check CUDA
if command -v nvcc &> /dev/null; then
    CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $5}' | sed 's/,//')
    echo "CUDA version: $CUDA_VERSION"
else
    echo "WARNING: nvcc not found in PATH"
fi

# Check GPU
if command -v nvidia-smi &> /dev/null; then
    GPU_COUNT=$(nvidia-smi --query-gpu=count --format=csv,noheader | head -1)
    echo "GPUs found: $GPU_COUNT"
    nvidia-smi --query-gpu=index,name,memory.total --format=csv
else
    echo "ERROR: nvidia-smi not found"
    exit 1
fi

echo ""
echo "=========================================="
echo "Installing SGLang"
echo "=========================================="

# Method 1: Install from source (recommended for development)
echo ""
echo "Installing SGLang from source..."
cd "$PPD_ROOT"

# Install in editable mode
pip install -e "python[all]" --find-links https://flashinfer.ai/whl/cu124/torch2.5/flashinfer-python

# Install sgl-router
echo ""
echo "Installing sgl-router..."
cd "$PPD_ROOT/sgl-router"
pip install -e ./bindings/python

# Install additional dependencies
echo ""
echo "Installing additional dependencies..."
pip install transformers requests aiohttp uvicorn fastapi orjson

# Verify installation
echo ""
echo "=========================================="
echo "Verifying Installation"
echo "=========================================="

python3 -c "
import sglang
print(f'SGLang imported successfully')

try:
    from sglang.srt.disaggregation.utils import DisaggregationMode
    print('Disaggregation module available')
except ImportError as e:
    print(f'Disaggregation module not available: {e}')

try:
    import sglang_router
    print('sglang_router imported successfully')
except ImportError as e:
    print(f'sglang_router not available: {e}')
"

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Run the quick test:"
echo "   cd $PPD_ROOT/ppd/scripts"
echo "   ./run_quick_test.sh meta-llama/Llama-3.1-8B-Instruct"
echo ""
echo "2. Or start servers manually:"
echo "   python3 scripts/start_pd_servers.py --model <model_path> --keep-running"
echo ""
echo "3. Then run validation:"
echo "   python3 scripts/validate_setup.py"
