# Comprehensive Append-Prefill Profiling Analysis Report

## Executive Summary

This report presents the results of a **comprehensive Grid Search experiment** (168 configurations) profiling append-prefill behavior on Llama-3.1-8B-Instruct with H100 GPU. The key innovation is **covering both overhead-bound and compute-bound regimes** by varying batch size from 1 to 64.

### Key Findings

| Finding | Value | Implication |
|---------|-------|-------------|
| **Scaling with m** | O(m^0.11) to O(m^0.37) | Sub-linear, not O(m) prefill-like |
| **Regime Transition** | BS ≈ 16-32 | Below: overhead-bound; Above: compute-bound |
| **LOCAL Routing** | 22% of configs | High cache rate + small BS prefer local |
| **Severe Interference** | 7 configs >5s | BS=64 with large L×m combinations |
| **Peak Throughput** | 767K tok/s | BS=64, L=16384, m=128, r=0.99 |

---

## 1. Experimental Setup

### 1.1 Configuration Space

```
Model:          Llama-3.1-8B-Instruct
GPU:            H100 80GB
Framework:      SGLang

Grid Search Parameters:
  Batch Size (BS):     [1, 4, 8, 16, 32, 64]
  Context Length (L):  [1024, 4096, 8192, 16384]
  Append Length (m):   [32, 128, 512, 1024, 2048, 4096, 8192]

Total Configurations: 6 × 4 × 7 = 168
Runs per Config:      3 (results averaged)
```

### 1.2 Metrics Collected

| Category | Metrics |
|----------|---------|
| **Performance** | Latency (TTFT), Throughput (tok/s), ms/append_token |
| **Features** | Cache Rate r = L/(L+m) |
| **Interference** | GPU active time, Interference score |
| **Transfer Cost** | KV cache size, Transfer time (PCIe/IB) |
| **Routing** | LOCAL vs REMOTE decision |

## 2. Regime Analysis: Overhead-Bound vs Compute-Bound

Previous experiments (BS=1) showed constant latency regardless of m, leading to the **incorrect conclusion** that append-prefill is "decode-like". This was because BS=1 workloads are **overhead-bound** on H100.

### 2.2 Results by Batch Size

| BS | L=1024 Ratio | L=4096 Ratio | Regime Classification |
|:--:|:---:|:---:|:---|
| 1 | 1.8x | 1.8x | **OVERHEAD-BOUND** |
| 4 | 3.0x | 2.5x | TRANSITION |
| 8 | 3.3x | 3.0x | TRANSITION |
| 16 | 5.7x | 2.6x | TRANSITION → COMPUTE |
| 32 | 10.7x | 3.4x | **COMPUTE-BOUND** |
| 64 | 19.4x | 8.0x | **COMPUTE-BOUND** |

*Ratio = Latency(m=8192) / Latency(m=32)*

### 2.3 Detailed Regime Map

```
                    Append Length (m)
                32      512     2K      8K
           ┌────────┬────────┬────────┬────────┐
    BS=1   │ OVHD   │ OVHD   │ OVHD   │ OVHD   │ ← Latency ~65ms constant
           ├────────┼────────┼────────┼────────┤
    BS=4   │ OVHD   │ TRANS  │ TRANS  │ TRANS  │
           ├────────┼────────┼────────┼────────┤
    BS=8   │ TRANS  │ TRANS  │ TRANS  │ TRANS  │
           ├────────┼────────┼────────┼────────┤
    BS=16  │ TRANS  │ TRANS  │ COMP   │ COMP   │ ← Transition zone
           ├────────┼────────┼────────┼────────┤
    BS=32  │ COMP   │ COMP   │ COMP   │ COMP   │
           ├────────┼────────┼────────┼────────┤
    BS=64  │ COMP   │ COMP   │ COMP   │ COMP   │ ← Latency scales 19x
           └────────┴────────┴────────┴────────┘

    OVHD = Overhead-bound (system latency dominates)
    TRANS = Transition zone
    COMP = Compute-bound (GPU compute dominates)
```

## 3. Latency vs Append Length Analysis

### 3.1 The Key Test: How Does Latency Scale with m?

**Fixed Context L = 4096, Varying Append m**

| BS | m=32 | m=512 | m=2048 | m=8192 | Scaling |
|:--:|:----:|:-----:|:------:|:------:|:-------:|
| 1 | 68ms | 68ms | 72ms | 123ms | O(m^0.11) |
| 8 | 129ms | 142ms | 168ms | 384ms | O(m^0.20) |
| 32 | 736ms | 341ms | 572ms | 2493ms | O(m^0.22) |
| 64 | 941ms | 594ms | 1071ms | 7496ms | O(m^0.37) |

### 3.2 Critical Insight: Sub-Linear Scaling

**Why is scaling O(m^0.2-0.4) instead of O(m^1.0)?**

1. **FlashAttention Optimization**: Reduces attention complexity
2. **Amortization Effect**: Fixed costs (model loading, KV cache access) amortized over more tokens
3. **Memory Bandwidth**: Not purely compute-limited

### 3.3 ms/token Analysis

| BS | m=32 | m=512 | m=2048 | m=8192 |
|:--:|:----:|:-----:|:------:|:------:|
| 1 | 2.13 | 0.13 | 0.035 | 0.015 |
| 8 | 4.02 | 0.28 | 0.082 | 0.047 |
| 32 | 23.0 | 0.67 | 0.28 | 0.30 |
| 64 | 29.4 | 1.16 | 0.52 | 0.92 |

**Key Observation**: ms/token **decreases** as m increases (up to a point), showing batching efficiency.

## 4. Routing Decision Analysis

### 4.1 Overall Statistics

```
Total Configurations:    168
Route LOCAL:             37 (22.0%)
Route REMOTE:           131 (78.0%)
```

### 4.2 When is LOCAL Routing Preferred?

LOCAL routing is chosen when: **Compute Time < KV Transfer Time**

| Factor | LOCAL Preferred | REMOTE Preferred |
|--------|-----------------|------------------|
| Batch Size | BS ≤ 8 | BS ≥ 16 |
| Cache Rate | r > 0.8 | r < 0.5 |
| Context Length | L ≥ 4096 | L ≤ 1024 |
| Append Length | Any | Large m (>4K) |

### 4.3 Routing Boundary Examples

```
BS=1, L=4096:  LOCAL for m ≤ 2048, REMOTE for m ≥ 4096
BS=4, L=8192:  LOCAL for m ≤ 4096, REMOTE for m = 8192
BS=8, L=16384: LOCAL for m ≤ 2048, REMOTE for m ≥ 4096
BS≥16:         Always REMOTE (compute dominates)
```

---

## 5. Interference Analysis

### 5.1 Latency Distribution

| Latency Range | Count | % | Impact |
|---------------|-------|---|--------|
| < 100ms | 23 | 13.7% | Minimal interference |
| 100-500ms | 88 | 52.4% | Acceptable |
| 500ms-2s | 42 | 25.0% | Noticeable to users |
| 2-5s | 8 | 4.8% | Significant degradation |
| > 5s | 7 | 4.2% | **Severe - Route to P-machine** |

### 5.2 Worst-Case Configurations

| Rank | BS | L | m | Latency | Recommendation |
|:----:|:--:|:----:|:----:|:-------:|----------------|
| 1 | 64 | 16384 | 8192 | **21.6s** | MUST route to P |
| 2 | 64 | 16384 | 4096 | 11.5s | MUST route to P |
| 3 | 64 | 8192 | 8192 | 8.1s | MUST route to P |
| 4 | 32 | 16384 | 8192 | 7.7s | MUST route to P |
| 5 | 64 | 4096 | 8192 | 7.5s | MUST route to P |

### 5.3 Interference Score

For D-machine scheduling, any append-prefill with latency > 500ms has interference_score > 0.5, meaning it will noticeably impact concurrent decode users.

---

## 6. Key Conclusions

### 6.1 Append-Prefill is NOT Decode-Like

| Aspect | Decode | Append-Prefill | Evidence |
|--------|--------|----------------|----------|
| Scaling | O(1) | O(m^0.2-0.4) | Latency increases with m |
| GPU Util | Low | Can be 100% | BS=64 saturates H100 |
| Batching | Limited | Effective | ms/token decreases |

### 6.2 Routing Strategy

```
                          ┌─────────────────┐
                          │ Append-Prefill  │
                          │   Request       │
                          └────────┬────────┘
                                   │
                    ┌──────────────┴──────────────┐
                    │      Check Parameters       │
                    └──────────────┬──────────────┘
                                   │
              ┌────────────────────┼────────────────────┐
              │                    │                    │
        ┌─────▼─────┐        ┌─────▼─────┐        ┌─────▼─────┐
        │ BS ≤ 8    │        │ BS = 16   │        │ BS ≥ 32   │
        │ r > 0.8   │        │ Transition│        │ Any r     │
        └─────┬─────┘        └─────┬─────┘        └─────┬─────┘
              │                    │                    │
        ┌─────▼─────┐        ┌─────▼─────┐        ┌─────▼─────┐
        │   LOCAL   │        │  Compare  │        │  REMOTE   │
        │ D-machine │        │ T_compute │        │ P-machine │
        └───────────┘        │vs T_xfer  │        └───────────┘
                             └───────────┘
```

### 6.3 Quantitative Routing Thresholds

**For H100 + Llama-8B + InfiniBand (25.6 GB/s):**

```python
def should_route_to_p_machine(bs, L, m):
    r = L / (L + m)

    # Always route large batches with big appends
    if bs >= 32 and m >= 4096:
        return True

    # Always keep small overhead-bound workloads local
    if bs <= 4 and r > 0.8:
        return False

    # Transition zone: compare costs
    kv_transfer_ms = (L * 2 * 4096 * 32 * 8 * 128 * 2) / (25.6e9) * 1000
    # Use measured latency from profiling table
    compute_ms = lookup_latency(bs, L, m)

    return compute_ms > kv_transfer_ms
```

---

## 7. Appendix: Sample Data Points

### High Throughput Configurations

| BS | L | m | r | Throughput |
|:--:|:----:|:----:|:----:|:----------:|
| 64 | 16384 | 128 | 0.99 | 767,849 |
| 64 | 16384 | 512 | 0.97 | 745,493 |
| 32 | 16384 | 512 | 0.97 | 725,106 |
| 64 | 16384 | 1024 | 0.94 | 721,948 |
| 32 | 16384 | 128 | 0.99 | 720,445 |

### Low Latency Configurations (< 100ms)

| BS | L | m | r | Latency |
|:--:|:----:|:----:|:----:|:-------:|
| 1 | 1024 | 32 | 0.97 | 64.6ms |
| 1 | 1024 | 128 | 0.89 | 65.4ms |
| 1 | 1024 | 512 | 0.67 | 65.7ms |
| 1 | 4096 | 32 | 0.99 | 68.1ms |
| 1 | 4096 | 128 | 0.97 | 67.9ms |
