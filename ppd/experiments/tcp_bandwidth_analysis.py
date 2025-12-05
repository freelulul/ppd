#!/usr/bin/env python3
"""
TCP Bandwidth Analysis for P-P-D Routing
=========================================
Analyzes how network bandwidth affects P-P-D routing decisions.

Key insight from P-P-D:
- Both P and D machines keep their own KV cache
- Each round, they only transfer the "new" KV they computed:
  * D → P: Transfer L tokens of KV cache (the cached context)
  * P does pure prefill on m new tokens

Cost comparison:
- LOCAL:  T_append_prefill(L, m) on D-machine (measured in profiling)
- REMOTE: T_transfer(L) + T_prefill(m) via network to P-machine

This script:
1. Uses profiling data for LOCAL cost and prefill estimates
2. Recalculates REMOTE cost with variable TCP bandwidth
3. Generates Performance vs Bandwidth analysis
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass


# Model constants (from append_prefill_profiling.py)
MODEL_CONFIG = {
    "num_layers": 32,
    "num_kv_heads": 8,
    "head_dim": 128,
    "dtype_bytes": 2,  # BF16/FP16
}

def calculate_kv_cache_size_mb(context_length: int) -> float:
    """Calculate KV cache size in MB for given context length."""
    bytes_per_token = (
        2 *  # K and V
        MODEL_CONFIG["num_layers"] *
        MODEL_CONFIG["num_kv_heads"] *
        MODEL_CONFIG["head_dim"] *
        MODEL_CONFIG["dtype_bytes"]
    )  # = 131072 bytes = 128 KB per token
    total_bytes = context_length * bytes_per_token
    return total_bytes / (1024 * 1024)


def calculate_transfer_time_ms(size_mb: float, bandwidth_gbps: float) -> float:
    """Calculate transfer time in ms given size and bandwidth."""
    size_gb = size_mb / 1024
    bandwidth_gbytes_per_sec = bandwidth_gbps / 8  # Gbps to GB/s
    return (size_gb / bandwidth_gbytes_per_sec) * 1000


@dataclass
class RoutingAnalysis:
    """Analysis result for one bandwidth level."""
    bandwidth_gbps: float
    total_configs: int
    local_count: int
    remote_count: int
    local_pct: float
    baseline_throughput: float  # All REMOTE
    adaptive_throughput: float  # Best of LOCAL/REMOTE
    improvement_pct: float


def analyze_routing_at_bandwidth(
    profiling_data: List[Dict],
    bandwidth_gbps: float
) -> RoutingAnalysis:
    """
    Analyze routing decisions at a given TCP bandwidth.

    For each profiling record:
    - LOCAL cost: record['latency_ms'] (measured append-prefill on D)
    - REMOTE cost: T_transfer(L) + T_prefill(m)
      * Transfer L tokens of KV cache at given bandwidth
      * Prefill m tokens on P-machine (use record's prefill estimate)
    """
    local_count = 0
    remote_count = 0
    baseline_total_time = 0.0
    adaptive_total_time = 0.0

    for record in profiling_data:
        L = record['context_length']
        m = record['append_length']

        # LOCAL cost: append-prefill on D-machine (from profiling)
        local_cost = record['latency_ms']

        # REMOTE cost: transfer KV(L) + prefill(m)
        # Transfer L tokens of KV cache
        kv_size_mb = calculate_kv_cache_size_mb(L)
        transfer_time = calculate_transfer_time_ms(kv_size_mb, bandwidth_gbps)

        # Prefill m tokens on P-machine
        # Use the estimate from profiling (already calculated correctly)
        prefill_time = record.get('prefill_estimate_ms', m * 0.01)  # fallback: 0.01 ms/token

        remote_cost = transfer_time + prefill_time

        # Routing decision
        if local_cost <= remote_cost:
            local_count += 1
            adaptive_time = local_cost
        else:
            remote_count += 1
            adaptive_time = remote_cost

        # Baseline: always REMOTE (traditional PD disaggregation)
        baseline_total_time += remote_cost
        adaptive_total_time += adaptive_time

    total_configs = len(profiling_data)

    # Throughput = requests / total_time
    baseline_throughput = (total_configs * 1000) / baseline_total_time if baseline_total_time > 0 else 0
    adaptive_throughput = (total_configs * 1000) / adaptive_total_time if adaptive_total_time > 0 else 0

    improvement_pct = ((adaptive_throughput - baseline_throughput) / baseline_throughput * 100
                       if baseline_throughput > 0 else 0)

    return RoutingAnalysis(
        bandwidth_gbps=bandwidth_gbps,
        total_configs=total_configs,
        local_count=local_count,
        remote_count=remote_count,
        local_pct=local_count / total_configs * 100,
        baseline_throughput=baseline_throughput,
        adaptive_throughput=adaptive_throughput,
        improvement_pct=improvement_pct
    )


def generate_analysis_plots(
    results: List[RoutingAnalysis],
    output_dir: Path
):
    """Generate Performance vs Bandwidth plots."""

    bandwidths = [r.bandwidth_gbps for r in results]
    baseline_throughputs = [r.baseline_throughput for r in results]
    adaptive_throughputs = [r.adaptive_throughput for r in results]
    improvements = [r.improvement_pct for r in results]
    local_pcts = [r.local_pct for r in results]

    # Figure 1: Performance vs Bandwidth (main result)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Throughput comparison
    ax1.semilogx(bandwidths, baseline_throughputs, 'b-o', label='Baseline PD (all REMOTE)', linewidth=2, markersize=8)
    ax1.semilogx(bandwidths, adaptive_throughputs, 'g-^', label='Adaptive P-P-D', linewidth=2, markersize=8)
    ax1.set_xlabel('Network Bandwidth (Gbps)', fontsize=12)
    ax1.set_ylabel('Throughput (requests/sec)', fontsize=12)
    ax1.set_title('Throughput vs Network Bandwidth', fontsize=14)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)

    # Right: Improvement percentage
    ax2.semilogx(bandwidths, improvements, 'r-s', linewidth=2, markersize=8)
    ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax2.fill_between(bandwidths, 0, improvements, alpha=0.3, color='red')
    ax2.set_xlabel('Network Bandwidth (Gbps)', fontsize=12)
    ax2.set_ylabel('Improvement over Baseline (%)', fontsize=12)
    ax2.set_title('Adaptive P-P-D Improvement', fontsize=14)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'real_performance_vs_bandwidth.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Figure 2: Routing decisions
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Left: LOCAL percentage
    ax1.semilogx(bandwidths, local_pcts, 'm-o', linewidth=2, markersize=8)
    ax1.axhline(y=50, color='gray', linestyle='--', alpha=0.5, label='50% threshold')
    ax1.set_xlabel('Network Bandwidth (Gbps)', fontsize=12)
    ax1.set_ylabel('Requests Kept LOCAL (%)', fontsize=12)
    ax1.set_title('Routing Decisions vs Bandwidth', fontsize=14)
    ax1.set_ylim(0, 105)
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Right: Improvement vs Local percentage
    scatter = ax2.scatter(local_pcts, improvements, c=bandwidths, cmap='viridis', s=100, norm=plt.matplotlib.colors.LogNorm())
    for i, bw in enumerate(bandwidths):
        ax2.annotate(f'{bw}G', (local_pcts[i], improvements[i]), textcoords="offset points",
                    xytext=(5, 5), fontsize=9)
    ax2.set_xlabel('Requests Kept LOCAL (%)', fontsize=12)
    ax2.set_ylabel('Improvement over Baseline (%)', fontsize=12)
    ax2.set_title('Improvement vs Local Routing', fontsize=14)
    ax2.grid(True, alpha=0.3)
    cb = plt.colorbar(scatter, ax=ax2)
    cb.set_label('Bandwidth (Gbps)')

    plt.tight_layout()
    plt.savefig(output_dir / 'real_routing_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved: {output_dir / 'real_performance_vs_bandwidth.png'}")
    print(f"Saved: {output_dir / 'real_routing_analysis.png'}")


def main():
    # Paths
    profiling_path = Path("/workspace/ppd/ppd/results/profiling_results.json")
    output_dir = Path("/workspace/ppd/ppd/results/bandwidth_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("TCP Bandwidth Analysis for P-P-D Routing")
    print("=" * 70)

    # Load profiling data
    print("\nLoading profiling data...")
    with open(profiling_path) as f:
        profiling_data = json.load(f)
    print(f"  Loaded {len(profiling_data)} profiling configurations")

    # Bandwidth levels to analyze (TCP range: 0.1 Gbps to 100 Gbps)
    bandwidth_levels = [0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 25.0, 50.0, 100.0]

    # Analyze at each bandwidth level
    print("\nAnalyzing routing decisions...")
    print(f"\n{'Bandwidth':<12}{'LOCAL %':<12}{'Baseline':<15}{'Adaptive':<15}{'Improvement':<12}")
    print("-" * 66)

    results = []
    for bw in bandwidth_levels:
        result = analyze_routing_at_bandwidth(profiling_data, bw)
        results.append(result)

        print(f"{bw:<12.1f}{result.local_pct:<12.1f}"
              f"{result.baseline_throughput:<15.2f}"
              f"{result.adaptive_throughput:<15.2f}"
              f"{result.improvement_pct:+.1f}%")

    # Generate plots
    print("\nGenerating plots...")
    generate_analysis_plots(results, output_dir)

    # Save results
    results_data = [
        {
            'bandwidth_gbps': r.bandwidth_gbps,
            'local_pct': round(r.local_pct, 2),
            'baseline_throughput': round(r.baseline_throughput, 4),
            'adaptive_throughput': round(r.adaptive_throughput, 4),
            'improvement_pct': round(r.improvement_pct, 2)
        }
        for r in results
    ]

    with open(output_dir / 'analysis_results.json', 'w') as f:
        json.dump(results_data, f, indent=2)
    print(f"Saved: {output_dir / 'analysis_results.json'}")

    # Key insights
    print("\n" + "=" * 70)
    print("KEY INSIGHTS")
    print("=" * 70)

    # Find crossover points
    for r in results:
        if r.local_pct < 50:
            print(f"\n- LOCAL < 50% when bandwidth > {r.bandwidth_gbps} Gbps")
            break

    max_improvement = max(r.improvement_pct for r in results)
    max_bw = next(r.bandwidth_gbps for r in results if r.improvement_pct == max_improvement)
    print(f"- Maximum improvement: {max_improvement:.1f}% at {max_bw} Gbps")

    # Practical insights
    for bw in [1.0, 10.0, 25.0]:
        r = next((r for r in results if r.bandwidth_gbps == bw), None)
        if r:
            print(f"- At {bw} Gbps: {r.local_pct:.0f}% LOCAL, {r.improvement_pct:+.1f}% improvement")


if __name__ == "__main__":
    main()
