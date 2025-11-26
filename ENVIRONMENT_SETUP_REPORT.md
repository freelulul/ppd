# P-PD Disaggregation Environment Setup Report

## Overview

This report documents the complete environment setup process for running Prefill-Decode (P/D) disaggregation experiments using SGLang. The setup enables research on the append-prefill trade-off analysis where the cache rate `r = L/(L+m)` determines optimal routing between P-machine and D-machine.

**Date**: 2025-11-26
**Hardware**: 4x NVIDIA H100 80GB HBM3 GPUs with NVLink (NV18 interconnect)

## Environment Summary

| Component | Version/Details |
|-----------|----------------|
| OS | Ubuntu 22.04 (Linux 5.4.0-176-generic) |
| Python | 3.10 |
| PyTorch | 2.9.1 |
| CUDA | 12.2 |
| SGLang | 0.5.5.post3 (from source) |
| sgl-kernel | 0.3.18.post1 |
| FlashInfer | 0.2.7.post1 |
| Mooncake | 0.3.7.post2 |
| Model | meta-llama/Llama-3.1-8B-Instruct |

## Step-by-Step Setup Instructions

### 1. Clone and Install SGLang from Source

```bash
# Clone the repository (already done in /workspace/ppd)
cd /workspace/ppd

# Install SGLang in development mode
pip install -e ./python

# Verify installation
python -c "import sglang; print(f'SGLang version: {sglang.__version__}')"
```

### 2. Fix sgl-kernel Compatibility

If you encounter `undefined symbol` errors with sgl-kernel:

```bash
# Upgrade sgl-kernel to match PyTorch version
pip install sgl-kernel --upgrade --no-cache-dir

# Verify the upgrade
pip show sgl-kernel
```

### 3. Install Mooncake Transfer Engine

```bash
# Install mooncake for KV cache transfer
pip install mooncake-transfer-engine --upgrade

# For environments without CUDA (development):
# pip install mooncake-transfer-engine-non-cuda --upgrade

# Install required system dependencies (if not present)
apt-get update
apt-get install -y libibverbs1 rdma-core
```

### 4. Download Model

```bash
# Login to HuggingFace
python -c "
from huggingface_hub import login
login(token='YOUR_HF_TOKEN')
"

# Download model to /dev/shm for faster access (recommended for testing)
python -c "
from huggingface_hub import snapshot_download
path = snapshot_download(
    repo_id='meta-llama/Llama-3.1-8B-Instruct',
    local_dir='/dev/shm/models/Llama-3.1-8B-Instruct',
    ignore_patterns=['*.bin', '*.pth'],
)
print(f'Model downloaded to: {path}')
"
```

### 5. Configure TCP Transport (for environments without InfiniBand RDMA)

A modification was made to `/workspace/ppd/python/sglang/srt/disaggregation/mooncake/transfer_engine.py` to support TCP transport:

```python
# Added environment variable support for TCP transport
elif get_bool_env_var("SGLANG_MOONCAKE_USE_TCP", "false"):
    logger.info("Using TCP transport for Mooncake (no RDMA hardware)")
    ret_value = self.engine.initialize(
        hostname,
        "P2PHANDSHAKE",
        "tcp",
        device_name if device_name is not None else "",
    )
```

**Note**: For environments with InfiniBand RDMA, this modification is not needed.

## Running P/D Disaggregation Tests

### Quick Start

```bash
cd /workspace/ppd/ppd/scripts

# Run simple P/D test with TCP transport (no RDMA required)
python simple_pd_test.py \
    --model /dev/shm/models/Llama-3.1-8B-Instruct \
    --prefill-gpu 0 \
    --decode-gpu 1 \
    --use-tcp

# Keep servers running after test
python simple_pd_test.py \
    --model /dev/shm/models/Llama-3.1-8B-Instruct \
    --prefill-gpu 0 \
    --decode-gpu 1 \
    --use-tcp \
    --keep-running
```

### Manual Server Launch

**Prefill Server (GPU 0)**:
```bash
CUDA_VISIBLE_DEVICES=0 SGLANG_MOONCAKE_USE_TCP=true python -m sglang.launch_server \
    --model-path /dev/shm/models/Llama-3.1-8B-Instruct \
    --host 127.0.0.1 \
    --port 30100 \
    --disaggregation-mode prefill \
    --disaggregation-transfer-backend mooncake \
    --disaggregation-bootstrap-port 9000 \
    --tp 1 \
    --trust-remote-code \
    --mem-fraction-static 0.85
```

**Decode Server (GPU 1)**:
```bash
CUDA_VISIBLE_DEVICES=1 SGLANG_MOONCAKE_USE_TCP=true python -m sglang.launch_server \
    --model-path /dev/shm/models/Llama-3.1-8B-Instruct \
    --host 127.0.0.1 \
    --port 30200 \
    --disaggregation-mode decode \
    --disaggregation-transfer-backend mooncake \
    --tp 1 \
    --trust-remote-code \
    --mem-fraction-static 0.85
```

**Router (Mini Load Balancer)**:
```bash
python -m sglang_router.launch_router \
    --pd-disaggregation \
    --mini-lb \
    --host 127.0.0.1 \
    --port 30000 \
    --prefill http://127.0.0.1:30100 9000 \
    --decode http://127.0.0.1:30200
```

## Key Files Created

```
/workspace/ppd/
├── ppd/
│   ├── configs/
│   │   ├── __init__.py
│   │   └── default_config.py      # Configuration dataclasses
│   ├── scripts/
│   │   ├── __init__.py
│   │   ├── simple_pd_test.py      # Simple P/D test script
│   │   ├── start_pd_servers.py    # Server launch utilities
│   │   └── validate_setup.py      # Environment validation
│   ├── experiments/
│   │   ├── __init__.py
│   │   └── multi_turn_append_test.py  # Append-prefill experiments
│   ├── __init__.py
│   └── README.md
└── ENVIRONMENT_SETUP_REPORT.md    # This file
```

## Troubleshooting

### Issue 1: sgl-kernel Undefined Symbol Error
```
undefined symbol: _ZNK3c106SymInt6sym_neERKS0_
```
**Solution**: Upgrade sgl-kernel: `pip install sgl-kernel --upgrade --no-cache-dir`

### Issue 2: Mooncake Import Error - libibverbs.so.1
```
ImportError: libibverbs.so.1: cannot open shared object file
```
**Solution**: Install RDMA libraries: `apt-get install -y libibverbs1 rdma-core`

### Issue 3: Mooncake RDMA Initialization Failure
```
No RDMA devices found
```
**Solution**: Set `SGLANG_MOONCAKE_USE_TCP=true` environment variable

### Issue 4: CUDA Invalid Device Ordinal
```
torch.AcceleratorError: CUDA error: invalid device ordinal
```
**Solution**: Do NOT use `--base-gpu-id` when using `CUDA_VISIBLE_DEVICES`. The device appears as GPU 0 internally.

### Issue 5: APT Repository Errors
**Solution**: Fix sources list:
```bash
sed -i 's|mirror.serverion.com|archive.ubuntu.com|g' /etc/apt/sources.list
apt-get update
```

## Hardware Topology

The 4x H100 GPUs are connected via NVLink with full mesh connectivity:

```
GPU0	GPU1	GPU2	GPU3
GPU0	 X 	NV18	NV18	NV18
GPU1	NV18	 X 	NV18	NV18
GPU2	NV18	NV18	 X 	NV18
GPU3	NV18	NV18	NV18	 X
```

This high-bandwidth interconnect (NV18 = 18 NVLink lanes) enables efficient KV cache transfer between GPUs.

## Next Steps for Append-Prefill Trade-off Analysis

1. **Run baseline measurements**: Use `multi_turn_append_test.py` to measure latency for different (L, m) combinations
2. **Vary cache rates**: Test r = 0.1, 0.3, 0.5, 0.7, 0.9 with fixed total tokens
3. **Collect metrics**: TTFT, TPOT, end-to-end latency, KV transfer time
4. **Analyze crossover point**: Find the cache rate threshold where D-machine becomes more efficient than P-machine for append-prefill

## References

- SGLang Documentation: https://sgl-project.github.io/
- Mooncake: https://github.com/kvcache-ai/Mooncake
- P/D Disaggregation: https://sgl-project.github.io/references/disaggregation.html
