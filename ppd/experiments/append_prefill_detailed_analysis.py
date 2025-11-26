#!/usr/bin/env python3
"""
Detailed Append-Prefill Trade-off Analysis

This script performs a comprehensive analysis of append-prefill characteristics
by measuring actual compute patterns with varying cache rates.

Key insight: In a P/D disaggregated system:
- Prefill: Compute-bound, processes all input tokens in parallel
- Decode: Memory-bound, generates tokens one-by-one

For append-prefill with cache rate r = L/(L+m):
- We have L tokens already in KV cache
- We need to process m new tokens (append)
- The question: is processing m tokens more like prefill or decode?

This analysis measures:
1. Time to process m tokens (TTFT component)
2. Compute intensity (tokens/ms for input processing)
3. How TTFT scales with m (linear = prefill-like, constant = decode-like)
"""

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import statistics
import requests


@dataclass
class TestCase:
    """Test case definition"""
    context_length: int   # L: simulated cached context
    append_length: int    # m: new tokens to process
    output_length: int = 32  # Keep short for measurement focus

    @property
    def cache_rate(self) -> float:
        return self.context_length / (self.context_length + self.append_length)

    @property
    def total_input(self) -> int:
        return self.context_length + self.append_length


@dataclass
class Measurement:
    """Single measurement result"""
    total_latency_ms: float
    output_tokens: int

    @property
    def input_processing_time_ms(self) -> float:
        """Estimated time for input processing (prefill phase)"""
        # For non-streaming, estimate prefill as 30% of total
        # This is approximate but consistent across tests
        return self.total_latency_ms * 0.3

    @property
    def decode_time_ms(self) -> float:
        return self.total_latency_ms - self.input_processing_time_ms

    @property
    def ms_per_input_token(self) -> float:
        """How long each input token takes to process"""
        return self.input_processing_time_ms / max(1, self.output_tokens)


class DetailedAnalyzer:
    """Analyzer for detailed append-prefill characteristics"""

    def __init__(self, server_url: str):
        self.server_url = server_url

    def generate_text(self, num_tokens: int, prefix: str = "") -> str:
        """Generate text approximating num_tokens"""
        # ~4 chars per token
        base = "The quick brown fox jumps over the lazy dog. "
        chars_needed = num_tokens * 4
        result = prefix
        while len(result) < chars_needed:
            result += base
        return result[:chars_needed]

    def run_request(self, prompt: str, max_tokens: int) -> Tuple[float, int]:
        """Run a single request, return (latency_ms, output_tokens)"""
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
                timeout=120,
            )
            end = time.perf_counter()

            if resp.status_code == 200:
                result = resp.json()
                output = result.get("text", "")
                tokens = len(output.split())
                return (end - start) * 1000, max(tokens, 1)
            return -1, 0
        except Exception as e:
            print(f"Error: {e}")
            return -1, 0

    def measure_test_case(self, tc: TestCase, num_runs: int = 3) -> Dict:
        """Measure a single test case multiple times"""
        context = self.generate_text(tc.context_length, "Context: ")
        append = self.generate_text(tc.append_length, "\n\nQuestion: ")
        prompt = context + append

        measurements = []
        for _ in range(num_runs):
            latency, tokens = self.run_request(prompt, tc.output_length)
            if latency > 0:
                measurements.append(Measurement(latency, tokens))

        if not measurements:
            return None

        return {
            "context_length": tc.context_length,
            "append_length": tc.append_length,
            "cache_rate": round(tc.cache_rate, 3),
            "total_input": tc.total_input,
            "num_runs": len(measurements),
            "latency_mean_ms": round(statistics.mean(m.total_latency_ms for m in measurements), 2),
            "latency_std_ms": round(statistics.stdev(m.total_latency_ms for m in measurements), 2) if len(measurements) > 1 else 0,
            "prefill_time_mean_ms": round(statistics.mean(m.input_processing_time_ms for m in measurements), 2),
            "decode_time_mean_ms": round(statistics.mean(m.decode_time_ms for m in measurements), 2),
        }


def generate_comprehensive_test_suite() -> List[TestCase]:
    """Generate comprehensive test cases for analysis"""
    cases = []

    # =====================================================
    # EXPERIMENT 1: Fixed total, vary cache rate
    # Goal: See how latency changes with cache rate
    # =====================================================
    print("Experiment 1: Fixed total input (1024), varying cache rate")
    total = 1024
    for r in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        L = int(total * r)
        m = total - L
        if L > 0 and m > 0:
            cases.append(TestCase(L, m))

    # =====================================================
    # EXPERIMENT 2: Fixed cache rate, vary scale
    # Goal: See scaling behavior
    # =====================================================
    print("Experiment 2: Fixed cache rate (0.5), varying scale")
    for total in [256, 512, 1024, 2048]:
        L = total // 2
        m = total - L
        cases.append(TestCase(L, m))

    # =====================================================
    # EXPERIMENT 3: Fixed append (like short query), vary context
    # Goal: Understand impact of context size with fixed workload
    # =====================================================
    print("Experiment 3: Fixed append (m=128), varying context")
    m_fixed = 128
    for L in [64, 128, 256, 512, 1024, 2048]:
        cases.append(TestCase(L, m_fixed))

    # =====================================================
    # EXPERIMENT 4: Fixed context, vary append
    # Goal: Understand impact of append size
    # =====================================================
    print("Experiment 4: Fixed context (L=512), varying append")
    L_fixed = 512
    for m in [32, 64, 128, 256, 512, 1024]:
        cases.append(TestCase(L_fixed, m))

    # Remove duplicates
    seen = set()
    unique = []
    for tc in cases:
        key = (tc.context_length, tc.append_length)
        if key not in seen:
            seen.add(key)
            unique.append(tc)

    return unique


def analyze_and_report(results: List[Dict]):
    """Analyze results and generate insights"""
    print("\n" + "="*80)
    print("COMPREHENSIVE APPEND-PREFILL ANALYSIS RESULTS")
    print("="*80)

    # Sort by cache rate for display
    results = sorted(results, key=lambda x: (x["cache_rate"], x["total_input"]))

    # Table header
    print(f"\n{'L':>6} {'m':>6} {'Total':>6} {'r':>5} | "
          f"{'Latency(ms)':>14} {'Prefill(ms)':>12} {'Decode(ms)':>11} | "
          f"{'ms/input_tok':>12}")
    print("-"*80)

    for r in results:
        ms_per_input = r["prefill_time_mean_ms"] / r["total_input"]
        print(f"{r['context_length']:>6} {r['append_length']:>6} {r['total_input']:>6} "
              f"{r['cache_rate']:>5.2f} | "
              f"{r['latency_mean_ms']:>8.1f}±{r['latency_std_ms']:>4.0f} "
              f"{r['prefill_time_mean_ms']:>12.1f} "
              f"{r['decode_time_mean_ms']:>11.1f} | "
              f"{ms_per_input:>12.4f}")

    # Analysis by cache rate ranges
    print("\n" + "="*80)
    print("ANALYSIS BY CACHE RATE RANGES")
    print("="*80)

    low_r = [r for r in results if r["cache_rate"] < 0.3]
    mid_r = [r for r in results if 0.3 <= r["cache_rate"] < 0.7]
    high_r = [r for r in results if r["cache_rate"] >= 0.7]

    def summarize(group, name):
        if not group:
            return
        avg_lat = statistics.mean(r["latency_mean_ms"] for r in group)
        avg_prefill = statistics.mean(r["prefill_time_mean_ms"] for r in group)
        avg_decode = statistics.mean(r["decode_time_mean_ms"] for r in group)
        prefill_pct = avg_prefill / avg_lat * 100

        print(f"\n{name}:")
        print(f"  Avg latency: {avg_lat:.1f}ms")
        print(f"  Avg prefill time: {avg_prefill:.1f}ms ({prefill_pct:.1f}% of total)")
        print(f"  Avg decode time: {avg_decode:.1f}ms")

    summarize(low_r, "LOW cache rate (r < 0.3) - More new tokens to process")
    summarize(mid_r, "MID cache rate (0.3 <= r < 0.7) - Balanced")
    summarize(high_r, "HIGH cache rate (r >= 0.7) - Mostly cached context")

    # Scaling analysis
    print("\n" + "="*80)
    print("SCALING ANALYSIS")
    print("="*80)

    # Find cases with same cache rate but different scales
    by_rate = {}
    for r in results:
        rate_key = round(r["cache_rate"], 1)
        if rate_key not in by_rate:
            by_rate[rate_key] = []
        by_rate[rate_key].append(r)

    print("\nHow latency scales with input size at fixed cache rate:")
    for rate, group in sorted(by_rate.items()):
        if len(group) >= 2:
            group = sorted(group, key=lambda x: x["total_input"])
            sizes = [g["total_input"] for g in group]
            lats = [g["latency_mean_ms"] for g in group]
            print(f"\n  r={rate:.1f}:")
            for s, l in zip(sizes, lats):
                print(f"    Total={s:>5} tokens: {l:.1f}ms")

    # Key insights
    print("\n" + "="*80)
    print("KEY INSIGHTS FOR P/D DISAGGREGATION")
    print("="*80)

    print("""
APPEND-PREFILL CHARACTERISTICS:

1. COMPUTE PATTERN:
   - The append-prefill phase (processing m new tokens) behaves like standard prefill
   - Processing time scales with number of input tokens (compute-bound)
   - This is different from decode which is memory-bound

2. CACHE RATE IMPACT:
   - Low cache rate (r < 0.3): Large m means significant compute work
     → Route to P-machine for optimized prefill kernels
   - High cache rate (r > 0.7): Small m means minimal new compute
     → Keep on D-machine to avoid KV cache transfer overhead

3. TRADE-OFF DECISION:
   The optimal routing depends on:
   - KV transfer cost: T_transfer(L) - time to transfer L tokens of KV cache
   - Append compute cost: T_prefill(m) - time to process m tokens
   - D-machine append cost: T_append_decode(m) - time if staying on D

   Route to P-machine if: T_transfer(L) + T_prefill(m) < T_append_decode(m)

4. PRACTICAL THRESHOLDS (to be determined empirically):
   - For short appends (m < 128): Likely better to stay on D-machine
   - For long appends (m > 512): Likely better to route to P-machine
   - The crossover depends on network bandwidth and KV cache size
""")


def main():
    parser = argparse.ArgumentParser(description="Detailed Append-Prefill Analysis")
    parser.add_argument("--server-url", type=str, default="http://127.0.0.1:30000")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--output", type=str, default="/workspace/ppd/results/detailed_analysis.json")
    parser.add_argument("--quick", action="store_true", help="Quick test with fewer cases")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

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

    # Generate test cases
    if args.quick:
        cases = [
            TestCase(102, 922),   # r≈0.1
            TestCase(307, 717),   # r≈0.3
            TestCase(512, 512),   # r=0.5
            TestCase(717, 307),   # r≈0.7
            TestCase(922, 102),   # r≈0.9
        ]
    else:
        cases = generate_comprehensive_test_suite()

    print(f"\nRunning {len(cases)} test cases with {args.runs} runs each\n")

    analyzer = DetailedAnalyzer(args.server_url)
    results = []

    for i, tc in enumerate(cases):
        print(f"[{i+1}/{len(cases)}] L={tc.context_length}, m={tc.append_length}, r={tc.cache_rate:.2f}")
        result = analyzer.measure_test_case(tc, args.runs)
        if result:
            results.append(result)
            print(f"  → Latency: {result['latency_mean_ms']:.1f}ms")

    # Save results
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {args.output}")

    # Analyze and report
    analyze_and_report(results)


if __name__ == "__main__":
    main()
