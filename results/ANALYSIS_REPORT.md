# Append-Prefill Trade-off Analysis Report

## Executive Summary

This report presents the findings from comprehensive testing of append-prefill characteristics in a P/D (Prefill/Decode) disaggregated LLM inference system. The key finding is:

**TTFT (Time To First Token) remains approximately constant (~19-24ms) regardless of append length (m), indicating that append-prefill in SGLang behaves more like DECODE operations than PREFILL operations for the tested input sizes.**

## Experimental Setup

- **Hardware**: NVIDIA H100 80GB HBM3 GPU (single GPU mode for baseline)
- **Model**: Llama-3.1-8B-Instruct
- **Framework**: SGLang 0.5.5.post3
- **Test Method**: Streaming API for accurate TTFT measurement

## Key Experiments

### Experiment 1: Fixed Total Input (1024 tokens), Vary Cache Rate

| L (Context) | m (Append) | Cache Rate (r) | TTFT (ms) |
|-------------|------------|----------------|-----------|
| 102 | 922 | 0.10 | 24.0 |
| 204 | 820 | 0.20 | 21.7 |
| 307 | 717 | 0.30 | 23.0 |
| 512 | 512 | 0.50 | 20.9 |
| 716 | 308 | 0.70 | 20.4 |
| 922 | 102 | 0.90 | 20.9 |

**Finding**: TTFT varies only 14% (20.4-24.0ms) despite cache rate ranging from 0.1 to 0.9.

### Experiment 2: Fixed Cache Rate (0.5), Vary Scale

| Total Input | L | m | TTFT (ms) | ms/token |
|-------------|---|---|-----------|----------|
| 256 | 128 | 128 | 20.7 | 0.0809 |
| 512 | 256 | 256 | 20.8 | 0.0405 |
| 1024 | 512 | 512 | 19.6 | 0.0191 |
| 2048 | 1024 | 1024 | 23.7 | 0.0116 |

**Finding**: When input increases 8x, TTFT only increases 1.1x. **Scaling factor = 0.14** (far from 1.0 which would indicate linear prefill-like scaling).

### Experiment 3: Fixed Context (L=512), Vary Append (CRITICAL TEST)

| m (Append) | Total Input | Cache Rate | TTFT (ms) |
|------------|-------------|------------|-----------|
| 32 | 544 | 0.94 | 20.9 |
| 64 | 576 | 0.89 | 19.5 |
| 128 | 640 | 0.80 | 19.3 |
| 256 | 768 | 0.67 | 19.5 |
| 512 | 1024 | 0.50 | 19.6 |
| 1024 | 1536 | 0.33 | 21.3 |

**Finding**: When m increases 32x (from 32 to 1024), TTFT only changes 1.02x (essentially constant).

**CONCLUSION: Append-prefill is DECODE-LIKE, not PREFILL-LIKE**

### Experiment 4: Fixed Append (m=128), Vary Context

| L (Context) | Total Input | Cache Rate | TTFT (ms) |
|-------------|-------------|------------|-----------|
| 64 | 192 | 0.33 | 19.6 |
| 256 | 384 | 0.67 | 19.1 |
| 512 | 640 | 0.80 | 19.2 |
| 1024 | 1152 | 0.89 | 19.6 |
| 2048 | 2176 | 0.94 | 24.7 |

**Finding**: When context increases 32x, TTFT only changes 1.26x. **Scaling factor = 0.11**.

## Analysis by Cache Rate Ranges

| Cache Rate Range | Avg TTFT (ms) | % of Total Latency | Samples |
|-----------------|---------------|-------------------|---------|
| Low (r < 0.3) | 22.9 | 9.9% | 2 |
| Mid (0.3 ≤ r < 0.7) | 20.8 | 9.1% | 15 |
| High (r ≥ 0.7) | 20.6 | 9.0% | 8 |

**Finding**: TTFT is remarkably consistent across all cache rate ranges (~20-23ms).

## Theoretical Implications

### Why is Append-Prefill Decode-Like?

1. **KV Cache Efficiency**: With prefix caching, the KV cache for context L is already computed. Only m new tokens need attention computation.

2. **Attention Pattern**: For append tokens, attention is against (L+m) keys but only for m query positions. This is more memory-bound (reading L cached KV) than compute-bound.

3. **Batch Size Effect**: Single request processing doesn't fully utilize GPU compute. TTFT is dominated by kernel launch overhead and memory access rather than FLOPs.

### Comparison: Prefill vs Decode vs Append-Prefill

| Operation | Characteristic | Time Scaling | Bottleneck |
|-----------|---------------|--------------|------------|
| Prefill | Process N new tokens | O(N²) | Compute |
| Decode | Generate 1 token | O(context) | Memory |
| Append-Prefill | Process m tokens with L cached | ~O(1) to O(m) | Memory/Overhead |

## Routing Decision Framework for P/D Disaggregation

Based on these findings, the routing decision should consider:

### Current Model (Before This Analysis)
```
Route to P-machine if: r < threshold (e.g., 0.5)
Rationale: More new tokens = more prefill work
```

### Revised Model (After This Analysis)
```
For single requests with prefix caching:
- TTFT is relatively constant regardless of m
- Main cost is KV cache transfer for L tokens
- Therefore: Minimize transfers by keeping requests on D-machine when possible

Route to P-machine only if:
1. Batch of requests can amortize transfer cost
2. Very large m (>2048) where compute becomes significant
3. P-machine has spare capacity
```

### Quantitative Threshold

Given TTFT ≈ 20ms constant and KV transfer cost:
- KV cache size per token ≈ 2 * n_layers * hidden_dim * 2 bytes (bf16)
- For Llama-8B: ~1MB per 1024 tokens
- Transfer at 25GB/s (PCIe) → ~0.04ms per 1024 tokens

**Transfer is cheap relative to fixed TTFT overhead.** The decision should focus on:
- Batching efficiency
- Queue lengths on P vs D machines
- SLO requirements

## Conclusions

1. **Append-prefill behaves like decode operations** for the tested input sizes (up to 2K tokens).

2. **Cache rate (r) has minimal impact on TTFT** in single-request scenarios.

3. **Routing decisions should prioritize**:
   - Minimizing KV transfer overhead (favor keeping on D-machine)
   - Batching similar requests on P-machine
   - Queue load balancing

4. **Future work needed**:
   - Test with larger input sizes (>8K tokens) where compute may dominate
   - Test with batched requests to see if behavior changes
   - Measure actual KV transfer times in P/D disaggregated setup

## Files Generated

- `/workspace/ppd/results/ttft_analysis.json` - Raw TTFT measurements
- `/workspace/ppd/results/detailed_analysis.json` - Detailed latency breakdown
- `/workspace/ppd/results/quick_test.json` - Initial quick test results
- `/workspace/ppd/ppd/experiments/streaming_ttft_analysis.py` - Main analysis script
- `/workspace/ppd/ppd/experiments/append_prefill_detailed_analysis.py` - Detailed analysis script

---
*Report generated: 2025-11-26*
*Hardware: NVIDIA H100 80GB*
*Model: Llama-3.1-8B-Instruct*
