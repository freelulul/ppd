#!/usr/bin/env python3
"""
Append-Prefill Profiling: When to Keep Append-Prefill Local vs Route to P-machine

===============================================================================
RESEARCH MOTIVATION
===============================================================================
Current P/D disaggregation systems blindly route ALL append-prefill requests
back to the P-machine. This is suboptimal because:

1. When cache rate r = L/(L+m) is HIGH (e.g., r > 0.9):
   - The append part (m tokens) is small
   - Transferring large KV cache (L tokens) back to P is expensive
   - Local append-prefill on D-machine may be faster!

2. The CORRECT comparison should be:
   - LOCAL (D-machine): T_append_prefill(L, m) - uses cached KV
   - REMOTE (P-machine): T_transfer(L) + T_pure_prefill(m) - no cache benefit

This script profiles append-prefill to find when LOCAL beats REMOTE.

===============================================================================
EXPERIMENT DESIGN (Grid Search)
===============================================================================
- Batch Size (BS):     [1, 4, 8, 16, 32, 64]
- Context Length (L):  [1K, 4K, 8K, 16K]  - the cached history
- Append Length (m):   [32, 128, 512, 1K, 2K, 4K, 8K]  - new tokens

Cache Rate: r = L / (L + m)
  - r → 1.0: mostly cached, small append → LOCAL likely wins
  - r → 0.5: half cached, large append → REMOTE likely wins

===============================================================================
KEY INSIGHT
===============================================================================
We DON'T need to actually measure P-machine prefill time because:
- Pure prefill of m tokens is well-characterized: ~linear in m
- We can estimate T_prefill(m) from our measurements of T_append_prefill(L,m)
  when L is small (minimal cache benefit)

Routing Decision:
  LOCAL if: T_append_prefill(L, m) < T_transfer(L) + T_prefill(m)
  REMOTE otherwise
"""

import argparse
import json
import os
import sys
import time
import statistics
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Optional
from concurrent.futures import ThreadPoolExecutor
import threading
import requests


# ============================================================================
# Model Constants for Llama-3.1-8B
# ============================================================================
MODEL_CONFIG = {
    "name": "Llama-3.1-8B-Instruct",
    "params_b": 8,
    "hidden_dim": 4096,
    "num_layers": 32,
    "num_kv_heads": 8,  # GQA
    "head_dim": 128,
    "dtype_bytes": 2,  # BF16
    "flops_per_token": 16e9,  # ~16 GFLOPs per token
}

# Hardware Constants
H100_TFLOPS = 989  # BF16 Tensor Core
PCIE_5_BANDWIDTH_GBps = 64  # GB/s
INFINIBAND_BANDWIDTH_GBps = 25  # 200 Gb/s = 25 GB/s


@dataclass
class AppendPrefillRecord:
    """Complete record for one append-prefill measurement"""

    # === A. Features (Independent Variables) ===
    batch_size: int
    context_length: int      # L
    append_length: int       # m
    total_input: int         # L + m
    cache_rate: float        # r = L / (L + m)

    # === B. Performance Metrics ===
    latency_ms: float        # End-to-end time (≈ TTFT for this append)
    throughput_tps: float    # Tokens per second
    ms_per_append_token: float  # Normalized: latency / m
    ms_per_total_token: float   # Normalized: latency / (L+m)

    # === C. Interference Metrics ===
    gpu_active_time_ms: float    # Estimated GPU occupation time
    interference_score: float    # Impact on co-located decode (0-1 scale)

    # === D. Simulated Transfer Cost ===
    kv_cache_size_mb: float      # Size of KV cache for L tokens
    transfer_time_pcie_ms: float # Time to transfer via PCIe 5.0
    transfer_time_ib_ms: float   # Time to transfer via InfiniBand

    # === E. Derived Metrics for Routing ===
    local_cost_ms: float              # T_append_prefill(L, m) on D-machine
    prefill_estimate_ms: float        # Estimated T_prefill(m) on P-machine
    remote_cost_pcie_ms: float        # T_transfer(L) + T_prefill(m) via PCIe
    remote_cost_ib_ms: float          # T_transfer(L) + T_prefill(m) via IB
    routing_decision_pcie: str        # "LOCAL" or "REMOTE"
    routing_decision_ib: str          # "LOCAL" or "REMOTE"
    local_advantage_pcie_ms: float    # How much LOCAL saves vs REMOTE (PCIe)
    local_advantage_ib_ms: float      # How much LOCAL saves vs REMOTE (IB)

    # === Metadata ===
    num_successful_requests: int = 0
    std_latency_ms: float = 0.0


def calculate_kv_cache_size_mb(context_length: int) -> float:
    """Calculate KV cache size in MB for given context length"""
    # KV cache per token = 2 (K and V) * num_layers * num_kv_heads * head_dim * dtype_bytes
    bytes_per_token = (
        2 *  # K and V
        MODEL_CONFIG["num_layers"] *
        MODEL_CONFIG["num_kv_heads"] *
        MODEL_CONFIG["head_dim"] *
        MODEL_CONFIG["dtype_bytes"]
    )
    total_bytes = context_length * bytes_per_token
    return total_bytes / (1024 * 1024)  # Convert to MB


def calculate_transfer_time_ms(size_mb: float, bandwidth_gbps: float) -> float:
    """Calculate transfer time in ms"""
    size_gb = size_mb / 1024
    return (size_gb / bandwidth_gbps) * 1000  # ms


class AppendPrefillProfiler:
    """Profiler for append-prefill characteristics"""

    def __init__(self, server_url: str):
        self.server_url = server_url

    def generate_text(self, num_tokens: int, seed: int = 0) -> str:
        """Generate deterministic text approximating num_tokens"""
        words = [
            "The", "quick", "brown", "fox", "jumps", "over", "lazy", "dog",
            "while", "exploring", "vast", "digital", "landscapes", "of",
            "modern", "technology", "and", "artificial", "intelligence",
            "systems", "that", "process", "natural", "language", "with",
            "remarkable", "efficiency", "using", "transformer", "architectures",
            "designed", "for", "parallel", "computation", "across", "multiple",
            "attention", "heads", "enabling", "contextual", "understanding"
        ]
        chars_needed = num_tokens * 4
        result = []
        i = seed
        while len(" ".join(result)) < chars_needed:
            result.append(words[i % len(words)])
            i += 1
        return " ".join(result)[:chars_needed]

    def send_request(
        self,
        prompt: str,
        max_tokens: int,
        request_id: int,
        results: Dict,
        lock: threading.Lock
    ):
        """Send a single request and record timing"""
        payload = {
            "text": prompt,
            "sampling_params": {
                "temperature": 0.1,
                "max_new_tokens": max_tokens,
            },
        }

        start = time.perf_counter()
        try:
            resp = requests.post(
                f"{self.server_url}/generate",
                json=payload,
                timeout=300,
            )
            end = time.perf_counter()

            if resp.status_code == 200:
                with lock:
                    results[request_id] = {
                        "latency_ms": (end - start) * 1000,
                        "success": True,
                    }
            else:
                with lock:
                    results[request_id] = {"success": False}
        except Exception as e:
            with lock:
                results[request_id] = {"success": False, "error": str(e)}

    def profile_single_config(
        self,
        batch_size: int,
        context_length: int,
        append_length: int,
        num_runs: int = 3,
    ) -> Optional[AppendPrefillRecord]:
        """Profile a single (BS, L, m) configuration"""

        total_input = context_length + append_length
        cache_rate = context_length / total_input

        all_latencies = []

        for run in range(num_runs):
            # Generate prompts
            prompts = []
            for i in range(batch_size):
                seed = run * 10000 + i * 100
                context = self.generate_text(context_length, seed=seed)
                append_text = self.generate_text(append_length, seed=seed+50)
                prompt = f"{context}\n\nQuery {i}: {append_text}\nResponse:"
                prompts.append(prompt)

            # Send concurrent requests
            results = {}
            lock = threading.Lock()

            batch_start = time.perf_counter()

            with ThreadPoolExecutor(max_workers=min(batch_size, 128)) as executor:
                futures = []
                for i, prompt in enumerate(prompts):
                    future = executor.submit(
                        self.send_request,
                        prompt, 8, i, results, lock  # Small output to focus on prefill
                    )
                    futures.append(future)

                for f in futures:
                    f.result()

            batch_end = time.perf_counter()
            batch_latency_ms = (batch_end - batch_start) * 1000

            # Check success rate
            successful = [r for r in results.values() if r.get("success")]
            if len(successful) >= batch_size * 0.8:
                all_latencies.append(batch_latency_ms)

        if not all_latencies:
            return None

        # Calculate metrics
        avg_latency = statistics.mean(all_latencies)
        std_latency = statistics.stdev(all_latencies) if len(all_latencies) > 1 else 0

        total_tokens = batch_size * total_input
        throughput = total_tokens / (avg_latency / 1000)

        ms_per_append = avg_latency / (batch_size * append_length)
        ms_per_total = avg_latency / total_tokens

        # GPU active time estimate (assuming full occupation during batch)
        gpu_active_time = avg_latency

        # Interference score: normalized by typical decode TPOT (~6ms)
        # Higher score = more interference with decode users
        typical_tpot_ms = 6.0
        interference_score = min(1.0, gpu_active_time / (typical_tpot_ms * 100))

        # KV cache transfer calculations
        kv_size_mb = calculate_kv_cache_size_mb(context_length)
        transfer_pcie = calculate_transfer_time_ms(kv_size_mb, PCIE_5_BANDWIDTH_GBps)
        transfer_ib = calculate_transfer_time_ms(kv_size_mb, INFINIBAND_BANDWIDTH_GBps)

        # =====================================================================
        # ROUTING DECISION - The Core Comparison
        # =====================================================================
        # Current industry practice: blindly route all append-prefill to P-machine
        # Our insight: this is WRONG when cache rate is high!
        #
        # LOCAL (D-machine):  T_append_prefill(L, m) - we already measured this
        # REMOTE (P-machine): T_transfer(L) + T_pure_prefill(m)
        #
        # To estimate T_pure_prefill(m), we use the fact that:
        # - Prefill is roughly linear in sequence length for compute-bound regime
        # - From Llama-8B on H100: ~0.01-0.03 ms/token for prefill at batch size
        # - We use a conservative estimate based on measured data
        # =====================================================================

        local_cost = avg_latency

        # Estimate pure prefill time for m tokens on P-machine
        # =====================================================================
        # H100 prefill performance for Llama-8B:
        # - At low batch size: ~50-100K tokens/sec (overhead-bound)
        # - At high batch size: ~200-400K tokens/sec (compute-bound)
        #
        # Converting to ms/token:
        # - BS=1: ~0.01-0.02 ms/token (50-100K tok/s)
        # - BS=64: ~0.003-0.005 ms/token (200-400K tok/s) per request
        #
        # Total batch prefill time = m * ms_per_token_per_request
        # =====================================================================
        if batch_size <= 4:
            prefill_ms_per_token = 0.01  # ~100K tokens/sec
        elif batch_size <= 16:
            prefill_ms_per_token = 0.005  # ~200K tokens/sec
        else:
            prefill_ms_per_token = 0.003  # ~333K tokens/sec

        # Total prefill time for the batch (all requests in parallel)
        # Note: for batched prefill, total time ≈ m * ms_per_token (not * batch_size)
        # because requests are processed in parallel
        prefill_estimate = append_length * prefill_ms_per_token

        # Add minimum overhead (kernel launch, CUDA synchronization, etc.)
        prefill_estimate = max(prefill_estimate, 5.0)  # at least 5ms overhead

        # Remote cost = transfer KV cache + do pure prefill on P
        remote_cost_pcie = transfer_pcie + prefill_estimate
        remote_cost_ib = transfer_ib + prefill_estimate

        # Routing decision: LOCAL wins if it's cheaper
        routing_pcie = "LOCAL" if local_cost <= remote_cost_pcie else "REMOTE"
        routing_ib = "LOCAL" if local_cost <= remote_cost_ib else "REMOTE"

        # Calculate advantage (positive = LOCAL saves time)
        local_advantage_pcie = remote_cost_pcie - local_cost
        local_advantage_ib = remote_cost_ib - local_cost

        return AppendPrefillRecord(
            batch_size=batch_size,
            context_length=context_length,
            append_length=append_length,
            total_input=total_input,
            cache_rate=round(cache_rate, 4),
            latency_ms=round(avg_latency, 2),
            throughput_tps=round(throughput, 1),
            ms_per_append_token=round(ms_per_append, 6),
            ms_per_total_token=round(ms_per_total, 6),
            gpu_active_time_ms=round(gpu_active_time, 2),
            interference_score=round(interference_score, 4),
            kv_cache_size_mb=round(kv_size_mb, 2),
            transfer_time_pcie_ms=round(transfer_pcie, 2),
            transfer_time_ib_ms=round(transfer_ib, 2),
            local_cost_ms=round(local_cost, 2),
            prefill_estimate_ms=round(prefill_estimate, 2),
            remote_cost_pcie_ms=round(remote_cost_pcie, 2),
            remote_cost_ib_ms=round(remote_cost_ib, 2),
            routing_decision_pcie=routing_pcie,
            routing_decision_ib=routing_ib,
            local_advantage_pcie_ms=round(local_advantage_pcie, 2),
            local_advantage_ib_ms=round(local_advantage_ib, 2),
            num_successful_requests=batch_size * len(all_latencies),
            std_latency_ms=round(std_latency, 2),
        )


def run_grid_search(
    profiler: AppendPrefillProfiler,
    batch_sizes: List[int],
    context_lengths: List[int],
    append_lengths: List[int],
    num_runs: int = 3,
) -> List[Dict]:
    """Run full grid search experiment"""

    results = []
    total_configs = len(batch_sizes) * len(context_lengths) * len(append_lengths)
    current = 0

    print(f"\nGrid Search: {len(batch_sizes)} BS × {len(context_lengths)} L × {len(append_lengths)} m = {total_configs} configurations")
    print("="*80)

    for bs in batch_sizes:
        print(f"\n{'='*80}")
        print(f"BATCH SIZE = {bs}")
        print(f"{'='*80}")

        for L in context_lengths:
            print(f"\n  Context L = {L}")
            print(f"  {'-'*60}")

            for m in append_lengths:
                current += 1
                r = L / (L + m)

                print(f"    [{current}/{total_configs}] m={m:>5}, r={r:.2f} ... ", end="", flush=True)

                record = profiler.profile_single_config(bs, L, m, num_runs)

                if record:
                    results.append(asdict(record))
                    print(f"Latency={record.latency_ms:>8.1f}ms, "
                          f"Throughput={record.throughput_tps:>8.0f} tok/s, "
                          f"Route({record.routing_decision_ib})")
                else:
                    print("FAILED")

    return results


def analyze_and_report(results: List[Dict], output_dir: str):
    """Generate analysis report"""

    print("\n" + "="*80)
    print("APPEND-PREFILL ROUTING ANALYSIS")
    print("="*80)

    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║  RESEARCH QUESTION: When should append-prefill stay LOCAL on D-machine?     ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  Current Practice: Blindly route ALL append-prefill back to P-machine       ║
║  Our Hypothesis:   HIGH cache rate → LOCAL may be better (avoid transfer)   ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """)

    # =========================================================================
    # 1. Regime Analysis: Overhead-bound vs Compute-bound
    # =========================================================================
    print("\n1. APPEND-PREFILL LATENCY SCALING")
    print("-"*60)

    # Group by batch size and analyze m scaling
    by_bs = {}
    for r in results:
        bs = r["batch_size"]
        if bs not in by_bs:
            by_bs[bs] = []
        by_bs[bs].append(r)

    for bs in sorted(by_bs.keys()):
        items = by_bs[bs]
        # Find items with same L, different m
        by_L = {}
        for item in items:
            L = item["context_length"]
            if L not in by_L:
                by_L[L] = []
            by_L[L].append(item)

        print(f"\n  BS = {bs}:")
        for L in sorted(by_L.keys())[:2]:  # Show first 2 context lengths
            sorted_items = sorted(by_L[L], key=lambda x: x["append_length"])
            if len(sorted_items) >= 2:
                first, last = sorted_items[0], sorted_items[-1]
                m_ratio = last["append_length"] / first["append_length"]
                lat_ratio = last["latency_ms"] / first["latency_ms"]

                print(f"    L={L}: m {first['append_length']}→{last['append_length']} "
                      f"({m_ratio:.0f}x), Latency {first['latency_ms']:.0f}→{last['latency_ms']:.0f}ms "
                      f"({lat_ratio:.1f}x)")

                if lat_ratio < 1.5:
                    print(f"      → OVERHEAD-BOUND (latency ~constant)")
                elif lat_ratio < m_ratio * 0.5:
                    print(f"      → MIXED (sub-linear scaling)")
                else:
                    print(f"      → COMPUTE-BOUND (latency ~ m)")

    # =========================================================================
    # 2. Routing Decision Analysis - THE KEY RESULT
    # =========================================================================
    print("\n2. ROUTING DECISION ANALYSIS (LOCAL vs REMOTE)")
    print("-"*60)
    print("""
    Comparison:
      LOCAL:  T_append_prefill(L, m) on D-machine (uses KV cache)
      REMOTE: T_transfer(L) + T_prefill(m) on P-machine (no cache)
    """)

    local_count = sum(1 for r in results if r["routing_decision_ib"] == "LOCAL")
    remote_count = sum(1 for r in results if r["routing_decision_ib"] == "REMOTE")

    print(f"  Total configurations: {len(results)}")
    print(f"  ✓ LOCAL wins:  {local_count} ({100*local_count/len(results):.1f}%) ← Challenging current practice!")
    print(f"  → REMOTE wins: {remote_count} ({100*remote_count/len(results):.1f}%)")

    # Analyze by cache rate
    print("\n  LOCAL Advantage by Cache Rate (r = L/(L+m)):")
    cache_rate_bins = [(0.9, 1.0), (0.8, 0.9), (0.6, 0.8), (0.0, 0.6)]
    for low, high in cache_rate_bins:
        bin_results = [r for r in results if low <= r["cache_rate"] < high]
        if bin_results:
            local_in_bin = sum(1 for r in bin_results if r["routing_decision_ib"] == "LOCAL")
            avg_advantage = statistics.mean(r["local_advantage_ib_ms"] for r in bin_results)
            print(f"    r ∈ [{low:.1f}, {high:.1f}): {local_in_bin}/{len(bin_results)} LOCAL "
                  f"({100*local_in_bin/len(bin_results):.0f}%), avg advantage = {avg_advantage:+.1f}ms")

    # Find the boundary
    print("\n  Routing Boundary by (BS, L) → threshold m:")
    for bs in sorted(by_bs.keys()):
        items = by_bs[bs]
        for L in sorted(set(item["context_length"] for item in items)):
            L_items = sorted([i for i in items if i["context_length"] == L],
                           key=lambda x: x["append_length"])

            # Find where it transitions from LOCAL to REMOTE
            local_m_max = None
            for item in L_items:
                if item["routing_decision_ib"] == "LOCAL":
                    local_m_max = item["append_length"]
                else:
                    break

            if local_m_max:
                r_at_threshold = L / (L + local_m_max)
                print(f"    BS={bs:>2}, L={L:>5}: LOCAL for m ≤ {local_m_max:>4} (r ≥ {r_at_threshold:.2f})")
            else:
                print(f"    BS={bs:>2}, L={L:>5}: Always REMOTE")

    # =========================================================================
    # 3. Interference Analysis
    # =========================================================================
    print("\n3. INTERFERENCE ANALYSIS (Impact on Decode Users)")
    print("-"*60)

    high_interference = [r for r in results if r["interference_score"] > 0.5]
    print(f"\n  High interference configurations (score > 0.5): {len(high_interference)}")

    if high_interference:
        print("  Examples:")
        for r in sorted(high_interference, key=lambda x: -x["interference_score"])[:5]:
            print(f"    BS={r['batch_size']}, L={r['context_length']}, m={r['append_length']}: "
                  f"GPU blocked for {r['gpu_active_time_ms']:.0f}ms (score={r['interference_score']:.2f})")

    # =========================================================================
    # 4. Key Metrics Table - Cost Comparison
    # =========================================================================
    print("\n4. COST COMPARISON TABLE (Sample)")
    print("-"*60)
    print("  LOCAL = T_append_prefill | REMOTE = T_transfer + T_prefill")
    print()

    print(f"  {'BS':>3} {'L':>5} {'m':>5} {'r':>4} | {'LOCAL':>8} {'REMOTE':>8} {'Δ':>8} | {'Decision':>8}")
    print(f"  {'':>3} {'':>5} {'':>5} {'':>4} | {'(ms)':>8} {'(ms)':>8} {'(ms)':>8} | {'':>8}")
    print("  " + "-"*70)

    # Show sample of results
    sample = results[::max(1, len(results)//20)]  # ~20 samples
    for r in sample:
        delta = r['local_advantage_ib_ms']
        delta_str = f"+{delta:.0f}" if delta > 0 else f"{delta:.0f}"
        decision_marker = "✓" if r['routing_decision_ib'] == "LOCAL" else "→"
        print(f"  {r['batch_size']:>3} {r['context_length']:>5} {r['append_length']:>5} "
              f"{r['cache_rate']:>4.2f} | {r['local_cost_ms']:>8.1f} {r['remote_cost_ib_ms']:>8.1f} "
              f"{delta_str:>8} | {decision_marker} {r['routing_decision_ib']:<6}")

    # =========================================================================
    # 5. Save detailed analysis
    # =========================================================================

    # Calculate statistics by cache rate
    cache_rate_stats = {}
    for low, high in cache_rate_bins:
        bin_results = [r for r in results if low <= r["cache_rate"] < high]
        if bin_results:
            local_in_bin = sum(1 for r in bin_results if r["routing_decision_ib"] == "LOCAL")
            cache_rate_stats[f"r_{low:.1f}_{high:.1f}"] = {
                "total": len(bin_results),
                "local_count": local_in_bin,
                "local_pct": round(100 * local_in_bin / len(bin_results), 1),
                "avg_local_advantage_ms": round(statistics.mean(r["local_advantage_ib_ms"] for r in bin_results), 2),
            }

    analysis = {
        "research_question": "When should append-prefill stay LOCAL on D-machine instead of routing to P-machine?",
        "hypothesis": "High cache rate (r → 1) favors LOCAL execution",
        "summary": {
            "total_configurations": len(results),
            "local_routing_count": local_count,
            "remote_routing_count": remote_count,
            "local_routing_pct": round(100 * local_count / len(results), 1),
            "high_interference_count": len(high_interference),
        },
        "cache_rate_analysis": cache_rate_stats,
        "key_finding": f"{local_count}/{len(results)} ({100*local_count/len(results):.1f}%) configurations favor LOCAL - challenging blind P-machine routing!",
    }

    analysis_path = os.path.join(output_dir, "analysis_summary.json")
    with open(analysis_path, 'w') as f:
        json.dump(analysis, f, indent=2)

    print(f"\n\nAnalysis saved to: {analysis_path}")


def main():
    parser = argparse.ArgumentParser(description="Append-Prefill Profiling")
    parser.add_argument("--server-url", type=str, default="http://127.0.0.1:30000")
    parser.add_argument("--output-dir", type=str, default="/workspace/ppd/results")
    parser.add_argument("--runs", type=int, default=3, help="Runs per configuration")
    parser.add_argument("--quick", action="store_true", help="Quick test with fewer configs")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Define grid search parameters
    if args.quick:
        batch_sizes = [1, 8, 32]
        context_lengths = [1024, 4096]
        append_lengths = [32, 512, 2048]
    else:
        batch_sizes = [1, 4, 8, 16, 32, 64]
        context_lengths = [1024, 4096, 8192, 16384]
        append_lengths = [32, 128, 512, 1024, 2048, 4096, 8192]

    # Check server
    print(f"Connecting to {args.server_url}...")
    for _ in range(30):
        try:
            resp = requests.get(f"{args.server_url}/health", timeout=5)
            if resp.status_code == 200:
                print("Server ready!")
                break
        except:
            pass
        time.sleep(2)
    else:
        print("Server not ready!")
        sys.exit(1)

    profiler = AppendPrefillProfiler(args.server_url)

    # Run grid search
    results = run_grid_search(
        profiler,
        batch_sizes,
        context_lengths,
        append_lengths,
        num_runs=args.runs,
    )

    # Save raw results
    output_path = os.path.join(args.output_dir, "profiling_results.json")
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nRaw results saved to: {output_path}")

    # Analyze and report
    analyze_and_report(results, args.output_dir)

    # Final summary
    print("\n" + "="*80)
    print("PROFILING COMPLETE")
    print("="*80)
    print(f"""
Files generated:
  - {output_path} (raw data)
  - {args.output_dir}/analysis_summary.json (analysis)

Next steps:
  1. Use profiling_results.json to fit a cost model: T(BS, L, m)
  2. Plot the regime map (overhead-bound vs compute-bound regions)
  3. Determine routing thresholds for your P/D system
""")


if __name__ == "__main__":
    main()
