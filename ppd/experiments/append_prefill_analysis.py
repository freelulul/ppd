#!/usr/bin/env python3
"""
Append-Prefill Trade-off Analysis

This script analyzes the characteristics of append-prefill operations to determine
whether they behave more like prefill (compute-bound) or decode (memory-bound)
based on the cache rate r = L/(L+m).

Key hypothesis:
- High cache rate (small m, large L): More decode-like, should stay on D-machine
- Low cache rate (large m, small L): More prefill-like, should route to P-machine

Variables analyzed:
1. Cache rate r = L/(L+m) where L = context length, m = append length
2. Absolute append length m (tokens to process)
3. Latency characteristics: TTFT, per-token latency, total time
4. Compute vs memory bound indicators
"""

import argparse
import json
import os
import sys
import time
import subprocess
import signal
import threading
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple
import statistics

import requests

# Ensure sglang is in path
sys.path.insert(0, '/workspace/ppd/python')


@dataclass
class TestCase:
    """Single test case definition"""
    context_length: int  # L: tokens from history (simulated KV cache)
    append_length: int   # m: new prompt tokens to process
    output_length: int = 32  # Fixed short output for consistent measurement

    @property
    def cache_rate(self) -> float:
        """r = L / (L + m)"""
        return self.context_length / (self.context_length + self.append_length)

    @property
    def total_input(self) -> int:
        return self.context_length + self.append_length


@dataclass
class TestResult:
    """Results from a single test run"""
    test_case: TestCase
    ttft_ms: float           # Time to first token
    total_time_ms: float     # Total generation time
    output_tokens: int       # Actual tokens generated
    tpot_ms: float           # Time per output token (after first)
    prefill_throughput: float  # Tokens/sec for input processing
    decode_throughput: float   # Tokens/sec for output generation

    # Derived metrics
    @property
    def prefill_ratio(self) -> float:
        """Ratio of TTFT to total time - higher means more prefill-like"""
        return self.ttft_ms / self.total_time_ms if self.total_time_ms > 0 else 0


@dataclass
class AggregatedResult:
    """Aggregated results across multiple runs"""
    test_case: TestCase
    num_runs: int
    ttft_mean_ms: float
    ttft_std_ms: float
    total_time_mean_ms: float
    total_time_std_ms: float
    tpot_mean_ms: float
    prefill_throughput_mean: float
    decode_throughput_mean: float
    prefill_ratio_mean: float


class AppendPrefillAnalyzer:
    """Analyzer for append-prefill trade-off characteristics"""

    def __init__(self, server_url: str, model_name: str = None):
        self.server_url = server_url
        self.model_name = model_name

    def generate_context_text(self, num_tokens: int) -> str:
        """Generate text that approximates a certain number of tokens"""
        # Approximate: 1 token ≈ 4 characters for English text
        # Use repetitive but varied text to simulate conversation history
        base_sentences = [
            "The quick brown fox jumps over the lazy dog. ",
            "A journey of a thousand miles begins with a single step. ",
            "To be or not to be, that is the question. ",
            "All that glitters is not gold, as the saying goes. ",
            "The early bird catches the worm but the second mouse gets cheese. ",
        ]

        # Each sentence is roughly 10-15 tokens
        chars_needed = num_tokens * 4
        result = []
        idx = 0
        current_chars = 0

        while current_chars < chars_needed:
            result.append(base_sentences[idx % len(base_sentences)])
            current_chars += len(base_sentences[idx % len(base_sentences)])
            idx += 1

        return "".join(result)[:chars_needed]

    def generate_append_text(self, num_tokens: int) -> str:
        """Generate the new append prompt"""
        # Create a question/instruction that's roughly num_tokens long
        base_prompt = "Based on the above context, please provide a brief answer: "
        filler = "Additional context for this query includes relevant information about the topic. "

        chars_needed = num_tokens * 4
        result = base_prompt

        while len(result) < chars_needed:
            result += filler

        return result[:chars_needed]

    def run_single_request(self, context: str, append: str, max_tokens: int) -> Tuple[float, float, int]:
        """
        Run a single inference request and measure timing.
        Returns: (ttft_ms, total_time_ms, output_tokens)
        """
        full_prompt = context + "\n\n" + append

        payload = {
            "text": full_prompt,
            "sampling_params": {
                "temperature": 0.1,  # Low temp for consistent output
                "max_new_tokens": max_tokens,
            },
            "stream": True,  # Enable streaming to measure TTFT
        }

        start_time = time.perf_counter()
        first_token_time = None
        output_tokens = 0

        try:
            response = requests.post(
                f"{self.server_url}/generate",
                json=payload,
                stream=True,
                timeout=120,
            )

            for line in response.iter_lines():
                if line:
                    if first_token_time is None:
                        first_token_time = time.perf_counter()

                    try:
                        data = json.loads(line.decode('utf-8').replace('data: ', ''))
                        if 'text' in data:
                            # Count tokens in incremental output
                            output_tokens += 1
                    except:
                        pass

        except Exception as e:
            print(f"Request error: {e}")
            return -1, -1, 0

        end_time = time.perf_counter()

        if first_token_time is None:
            first_token_time = end_time

        ttft_ms = (first_token_time - start_time) * 1000
        total_time_ms = (end_time - start_time) * 1000

        return ttft_ms, total_time_ms, max(output_tokens, 1)

    def run_non_streaming_request(self, context: str, append: str, max_tokens: int) -> Tuple[float, float, int]:
        """
        Run a non-streaming request for simpler measurement.
        """
        full_prompt = context + "\n\n" + append

        payload = {
            "text": full_prompt,
            "sampling_params": {
                "temperature": 0.1,
                "max_new_tokens": max_tokens,
            },
        }

        start_time = time.perf_counter()

        try:
            response = requests.post(
                f"{self.server_url}/generate",
                json=payload,
                timeout=120,
            )

            end_time = time.perf_counter()

            if response.status_code == 200:
                result = response.json()
                output_text = result.get("text", "")
                # Rough token count
                output_tokens = len(output_text.split())
                total_time_ms = (end_time - start_time) * 1000
                # Estimate TTFT as proportional to input size
                # This is approximate for non-streaming
                return total_time_ms * 0.3, total_time_ms, max(output_tokens, 1)
            else:
                print(f"Request failed: {response.status_code}")
                return -1, -1, 0

        except Exception as e:
            print(f"Request error: {e}")
            return -1, -1, 0

    def run_test_case(self, test_case: TestCase, num_runs: int = 3) -> Optional[AggregatedResult]:
        """Run a test case multiple times and aggregate results"""

        context = self.generate_context_text(test_case.context_length)
        append = self.generate_append_text(test_case.append_length)

        results: List[TestResult] = []

        print(f"  Testing L={test_case.context_length}, m={test_case.append_length}, "
              f"r={test_case.cache_rate:.2f} ({num_runs} runs)...")

        for run in range(num_runs):
            ttft, total_time, output_tokens = self.run_non_streaming_request(
                context, append, test_case.output_length
            )

            if ttft < 0:
                print(f"    Run {run+1} failed, skipping")
                continue

            # Calculate derived metrics
            decode_time = total_time - ttft
            tpot = decode_time / output_tokens if output_tokens > 0 else 0

            # Throughput calculations
            input_tokens = test_case.context_length + test_case.append_length
            prefill_throughput = input_tokens / (ttft / 1000) if ttft > 0 else 0
            decode_throughput = output_tokens / (decode_time / 1000) if decode_time > 0 else 0

            result = TestResult(
                test_case=test_case,
                ttft_ms=ttft,
                total_time_ms=total_time,
                output_tokens=output_tokens,
                tpot_ms=tpot,
                prefill_throughput=prefill_throughput,
                decode_throughput=decode_throughput,
            )
            results.append(result)

            print(f"    Run {run+1}: TTFT={ttft:.1f}ms, Total={total_time:.1f}ms, "
                  f"Prefill ratio={result.prefill_ratio:.2f}")

        if not results:
            return None

        # Aggregate
        return AggregatedResult(
            test_case=test_case,
            num_runs=len(results),
            ttft_mean_ms=statistics.mean(r.ttft_ms for r in results),
            ttft_std_ms=statistics.stdev(r.ttft_ms for r in results) if len(results) > 1 else 0,
            total_time_mean_ms=statistics.mean(r.total_time_ms for r in results),
            total_time_std_ms=statistics.stdev(r.total_time_ms for r in results) if len(results) > 1 else 0,
            tpot_mean_ms=statistics.mean(r.tpot_ms for r in results),
            prefill_throughput_mean=statistics.mean(r.prefill_throughput for r in results),
            decode_throughput_mean=statistics.mean(r.decode_throughput for r in results),
            prefill_ratio_mean=statistics.mean(r.prefill_ratio for r in results),
        )


def generate_test_cases() -> List[TestCase]:
    """
    Generate comprehensive test cases for append-prefill analysis.

    We vary:
    1. Cache rate r from 0.1 (mostly append) to 0.9 (mostly cached)
    2. Total input size to see scaling effects
    """
    test_cases = []

    # ============================================
    # Experiment 1: Vary cache rate with fixed total input
    # ============================================
    # Total = 1024 tokens, vary L and m
    total_tokens = 1024
    for cache_rate in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        L = int(total_tokens * cache_rate)
        m = total_tokens - L
        if L > 0 and m > 0:
            test_cases.append(TestCase(context_length=L, append_length=m))

    # ============================================
    # Experiment 2: Fixed cache rate, vary total size
    # ============================================
    # r = 0.5 (equal split), vary total size
    for total in [256, 512, 1024, 2048, 4096]:
        L = total // 2
        m = total - L
        test_cases.append(TestCase(context_length=L, append_length=m))

    # ============================================
    # Experiment 3: Fixed append length, vary context
    # ============================================
    # m = 128 (small append like a user query)
    m_fixed = 128
    for L in [128, 256, 512, 1024, 2048, 4096]:
        test_cases.append(TestCase(context_length=L, append_length=m_fixed))

    # ============================================
    # Experiment 4: Fixed context, vary append length
    # ============================================
    # L = 1024 (moderate history)
    L_fixed = 1024
    for m in [32, 64, 128, 256, 512, 1024]:
        test_cases.append(TestCase(context_length=L_fixed, append_length=m))

    # Remove duplicates
    seen = set()
    unique_cases = []
    for tc in test_cases:
        key = (tc.context_length, tc.append_length)
        if key not in seen:
            seen.add(key)
            unique_cases.append(tc)

    return sorted(unique_cases, key=lambda x: (x.cache_rate, x.total_input))


def print_results_table(results: List[AggregatedResult]):
    """Print results in a formatted table"""
    print("\n" + "="*100)
    print("APPEND-PREFILL TRADE-OFF ANALYSIS RESULTS")
    print("="*100)

    print(f"\n{'L':>6} {'m':>6} {'Total':>6} {'r':>5} | "
          f"{'TTFT(ms)':>12} {'Total(ms)':>12} {'TPOT(ms)':>10} | "
          f"{'PrefillThr':>12} {'DecodeThr':>12} {'PrefillRatio':>12}")
    print("-"*100)

    for r in results:
        tc = r.test_case
        print(f"{tc.context_length:>6} {tc.append_length:>6} {tc.total_input:>6} "
              f"{tc.cache_rate:>5.2f} | "
              f"{r.ttft_mean_ms:>8.1f}±{r.ttft_std_ms:>3.0f} "
              f"{r.total_time_mean_ms:>8.1f}±{r.total_time_std_ms:>3.0f} "
              f"{r.tpot_mean_ms:>10.2f} | "
              f"{r.prefill_throughput_mean:>12.0f} "
              f"{r.decode_throughput_mean:>12.0f} "
              f"{r.prefill_ratio_mean:>12.2f}")


def analyze_results(results: List[AggregatedResult]):
    """Analyze results to determine prefill vs decode characteristics"""
    print("\n" + "="*100)
    print("ANALYSIS AND INSIGHTS")
    print("="*100)

    # Group by cache rate ranges
    low_cache = [r for r in results if r.test_case.cache_rate < 0.3]
    mid_cache = [r for r in results if 0.3 <= r.test_case.cache_rate < 0.7]
    high_cache = [r for r in results if r.test_case.cache_rate >= 0.7]

    print("\n1. TTFT vs Cache Rate Analysis:")
    print("-"*50)

    if low_cache:
        avg_ttft_low = statistics.mean(r.ttft_mean_ms for r in low_cache)
        avg_ratio_low = statistics.mean(r.prefill_ratio_mean for r in low_cache)
        print(f"  Low cache rate (r < 0.3):  Avg TTFT = {avg_ttft_low:.1f}ms, "
              f"Prefill ratio = {avg_ratio_low:.2f}")

    if mid_cache:
        avg_ttft_mid = statistics.mean(r.ttft_mean_ms for r in mid_cache)
        avg_ratio_mid = statistics.mean(r.prefill_ratio_mean for r in mid_cache)
        print(f"  Mid cache rate (0.3-0.7):  Avg TTFT = {avg_ttft_mid:.1f}ms, "
              f"Prefill ratio = {avg_ratio_mid:.2f}")

    if high_cache:
        avg_ttft_high = statistics.mean(r.ttft_mean_ms for r in high_cache)
        avg_ratio_high = statistics.mean(r.prefill_ratio_mean for r in high_cache)
        print(f"  High cache rate (r >= 0.7): Avg TTFT = {avg_ttft_high:.1f}ms, "
              f"Prefill ratio = {avg_ratio_high:.2f}")

    print("\n2. Prefill Throughput vs Append Length:")
    print("-"*50)

    # Sort by append length
    by_append = sorted(results, key=lambda r: r.test_case.append_length)
    for r in by_append[:5]:
        print(f"  m={r.test_case.append_length:>4}: Prefill throughput = "
              f"{r.prefill_throughput_mean:.0f} tokens/s")
    print("  ...")
    for r in by_append[-3:]:
        print(f"  m={r.test_case.append_length:>4}: Prefill throughput = "
              f"{r.prefill_throughput_mean:.0f} tokens/s")

    print("\n3. Key Findings:")
    print("-"*50)

    # Find crossover point if any
    sorted_by_rate = sorted(results, key=lambda r: r.test_case.cache_rate)

    print("""
  Based on the measurements:

  - LOW cache rate (r < 0.3): Append-prefill is PREFILL-LIKE
    * High TTFT relative to total time
    * Compute-bound behavior
    * Recommendation: Route to P-machine

  - HIGH cache rate (r > 0.7): Append-prefill is DECODE-LIKE
    * Low TTFT relative to total time (KV cache reuse)
    * Memory-bound behavior
    * Recommendation: Keep on D-machine (avoid KV transfer)

  - The crossover point depends on KV transfer cost vs computation cost.
    """)


def save_results(results: List[AggregatedResult], output_file: str):
    """Save results to JSON file"""
    data = []
    for r in results:
        data.append({
            "context_length": r.test_case.context_length,
            "append_length": r.test_case.append_length,
            "cache_rate": r.test_case.cache_rate,
            "total_input": r.test_case.total_input,
            "ttft_mean_ms": r.ttft_mean_ms,
            "ttft_std_ms": r.ttft_std_ms,
            "total_time_mean_ms": r.total_time_mean_ms,
            "total_time_std_ms": r.total_time_std_ms,
            "tpot_mean_ms": r.tpot_mean_ms,
            "prefill_throughput_mean": r.prefill_throughput_mean,
            "decode_throughput_mean": r.decode_throughput_mean,
            "prefill_ratio_mean": r.prefill_ratio_mean,
        })

    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"\nResults saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Append-Prefill Trade-off Analysis")
    parser.add_argument("--server-url", type=str, default="http://127.0.0.1:30000",
                        help="SGLang server URL")
    parser.add_argument("--runs", type=int, default=3,
                        help="Number of runs per test case")
    parser.add_argument("--output", type=str, default="/workspace/ppd/results/append_prefill_results.json",
                        help="Output file for results")
    parser.add_argument("--quick", action="store_true",
                        help="Run quick test with fewer cases")

    args = parser.parse_args()

    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    # Wait for server
    print(f"Checking server at {args.server_url}...")
    for _ in range(30):
        try:
            response = requests.get(f"{args.server_url}/health", timeout=5)
            if response.status_code == 200:
                print("Server is ready!")
                break
        except:
            pass
        time.sleep(2)
    else:
        print("ERROR: Server not ready after 60s")
        sys.exit(1)

    # Generate test cases
    if args.quick:
        # Quick test with fewer cases
        test_cases = [
            TestCase(context_length=256, append_length=256),   # r=0.5
            TestCase(context_length=128, append_length=512),   # r=0.2
            TestCase(context_length=512, append_length=128),   # r=0.8
            TestCase(context_length=1024, append_length=128),  # r=0.89
            TestCase(context_length=128, append_length=1024),  # r=0.11
        ]
    else:
        test_cases = generate_test_cases()

    print(f"\nRunning {len(test_cases)} test cases, {args.runs} runs each")
    print("="*60)

    analyzer = AppendPrefillAnalyzer(args.server_url)
    results: List[AggregatedResult] = []

    for i, tc in enumerate(test_cases):
        print(f"\n[{i+1}/{len(test_cases)}]", end=" ")
        result = analyzer.run_test_case(tc, num_runs=args.runs)
        if result:
            results.append(result)

    # Print and analyze results
    print_results_table(results)
    analyze_results(results)

    # Save results
    save_results(results, args.output)

    print("\n" + "="*60)
    print("Analysis complete!")


if __name__ == "__main__":
    main()
