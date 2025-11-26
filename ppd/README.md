# P-PD: Prefill-Append Prefill-Decode Experiment Framework

This framework enables systematic study of the append-prefill trade-off in multi-turn LLM inference conversations with P/D (Prefill-Decode) disaggregation.

## Core Hypothesis

In traditional P/D disaggregation, all prefill operations go to compute-intensive machines (P-machines) and all decode operations go to memory-intensive machines (D-machines). However, in multi-turn conversations, the "append-prefill" phase (processing new user input) has varying characteristics:

- **High cache rate** (small append `m`, large context `L`): The operation is more **decode-like** - small amount of new computation on top of existing KV cache. Transferring KV cache back to P-machine wastes bandwidth.

- **Low cache rate** (large append `m`, small context `L`): The operation is more **prefill-like** - significant parallel computation needed. Better to route to P-machine.

**Cache Rate**: `r = L / (L + m)` where `L` is context length and `m` is new append length.

## Directory Structure

```
ppd/
├── configs/                 # Configuration classes
│   ├── __init__.py
│   └── default_config.py   # Default experiment configuration
├── scripts/                 # Utility scripts
│   ├── start_pd_servers.py # Start P/D disaggregated servers
│   ├── validate_setup.py   # Validate environment setup
│   └── run_quick_test.sh   # Quick test script
├── experiments/             # Experiment implementations
│   └── multi_turn_append_test.py  # Main append-prefill experiment
├── results/                 # Output directory for results
└── utils/                   # Utility functions
```

## Quick Start

### Prerequisites

1. **Hardware**: At least 2 GPUs (1 for prefill, 1 for decode)
2. **Software**:
   - SGLang installed with disaggregation support
   - Mooncake transfer backend (or nixl/fake for testing)

### Step 1: Start P/D Servers

Using the quick test script (recommended for first-time setup):

```bash
cd ppd/scripts
./run_quick_test.sh meta-llama/Llama-3.1-8B-Instruct
```

Or manually:

```bash
# Start Prefill Server (GPU 0)
CUDA_VISIBLE_DEVICES=0 python3 -m sglang.launch_server \
    --model-path meta-llama/Llama-3.1-8B-Instruct \
    --host 127.0.0.1 --port 30100 \
    --disaggregation-mode prefill \
    --disaggregation-transfer-backend mooncake \
    --disaggregation-bootstrap-port 9000 \
    --tp 1

# Start Decode Server (GPU 1)
CUDA_VISIBLE_DEVICES=1 python3 -m sglang.launch_server \
    --model-path meta-llama/Llama-3.1-8B-Instruct \
    --host 127.0.0.1 --port 30200 \
    --disaggregation-mode decode \
    --disaggregation-transfer-backend mooncake \
    --base-gpu-id 1 \
    --tp 1

# Start Router
python3 -m sglang_router.launch_router \
    --pd-disaggregation --mini-lb \
    --host 127.0.0.1 --port 30000 \
    --prefill http://127.0.0.1:30100 9000 \
    --decode http://127.0.0.1:30200
```

### Step 2: Validate Setup

```bash
python3 scripts/validate_setup.py --router-url http://127.0.0.1:30000
```

### Step 3: Run Experiments

**Quick validation (minimal test cases):**
```bash
python3 experiments/multi_turn_append_test.py \
    --router-url http://127.0.0.1:30000 \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --validate-only
```

**Minimal experiment:**
```bash
python3 experiments/multi_turn_append_test.py \
    --router-url http://127.0.0.1:30000 \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --minimal \
    --output-dir results
```

**Full experiment:**
```bash
python3 experiments/multi_turn_append_test.py \
    --router-url http://127.0.0.1:30000 \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --repetitions 5 \
    --output-dir results
```

## Test Case Design

The framework sweeps across:

| Parameter | Values | Description |
|-----------|--------|-------------|
| `m` (append length) | 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048 | New tokens in user prompt |
| `L` (context length) | 512, 1024, 2048, 4096, 8192, 16384 | Existing conversation history |

This generates test cases covering cache rates from ~0.02 (very prefill-like) to ~0.99 (very decode-like).

## Expected Results

Based on the P-PD hypothesis, we expect to observe:

1. **High cache rate regime** (r > 0.8):
   - End-to-end latency should be dominated by decode-like characteristics
   - Routing to D-machine should be optimal
   - KV transfer overhead significant if sent to P-machine

2. **Low cache rate regime** (r < 0.3):
   - Performance should be compute-bound
   - Routing to P-machine should be optimal
   - Benefit from parallel compute kernels

3. **Transition region** (0.3 < r < 0.8):
   - Performance depends on hardware characteristics
   - This is where intelligent routing provides the most value

## Configuration

Edit `configs/default_config.py` to customize:

- Hardware configuration (GPU IDs, TP size)
- Model configuration (path, context length)
- Server configuration (ports, transfer backend)
- Experiment parameters (sweep ranges, repetitions)

## Output Format

Results are saved as JSON files in the output directory:

```json
{
  "config": {...},
  "start_time": "2024-...",
  "end_time": "2024-...",
  "summary": {
    "total_tests": 72,
    "by_cache_rate": {
      "low": {"mean_e2e": 150.2, ...},
      "high": {"mean_e2e": 45.3, ...}
    }
  },
  "results": [
    {
      "test_case": {"context_length": 512, "append_length": 8, "cache_rate": 0.984},
      "metrics": {"e2e_latency_ms": 42.5, "tpot_ms": 5.3, ...}
    },
    ...
  ]
}
```

## Next Steps (Research Direction)

1. **Phase 1 (Current)**: Validate that append-prefill shows different characteristics based on cache rate

2. **Phase 2**: Build a cost model to predict optimal routing:
   - Input: L, m, hardware specs (B_net, B_mem, C_flops), queue lengths
   - Output: Route to P-machine or D-machine

3. **Phase 3**: Implement AppendRouter in SGLang:
   - Online parameter estimation
   - Adaptive threshold with hysteresis
   - Integration with existing scheduler

4. **Phase 4**: Evaluation at scale:
   - Multi-GPU clusters
   - Real workload traces (ShareGPT, etc.)
   - End-to-end system comparison
