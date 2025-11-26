#!/usr/bin/env python3
"""
Streaming TTFT Analysis for Append-Prefill Trade-off

This script measures actual TTFT (Time To First Token) using streaming API,
which gives us the true prefill time. This is crucial for understanding
append-prefill characteristics.

Key insight:
- TTFT = Time from request submission to first token generation
- This captures the actual prefill/input processing time
- If append-prefill is "prefill-like": TTFT should scale with m (append length)
- If append-prefill is "decode-like": TTFT should be more constant regardless of m
"""

import argparse
import json
import os
import sys
import time
import statistics
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple, Optional
import requests


@dataclass
class TTFTMeasurement:
    """Single TTFT measurement"""
    context_length: int     # L: simulated cached context
    append_length: int      # m: new tokens to process
    total_input: int        # L + m
    cache_rate: float       # r = L / (L + m)
    ttft_ms: float         # Time to first token (actual prefill time)
    total_time_ms: float   # End-to-end latency
    output_tokens: int     # Number of output tokens generated

    @property
    def decode_time_ms(self) -> float:
        """Time spent in decode phase"""
        return self.total_time_ms - self.ttft_ms

    @property
    def tpot_ms(self) -> float:
        """Time per output token during decode"""
        if self.output_tokens <= 1:
            return 0
        return self.decode_time_ms / (self.output_tokens - 1)

    @property
    def ms_per_input_token(self) -> float:
        """Milliseconds per input token during prefill"""
        return self.ttft_ms / self.total_input


class StreamingTTFTAnalyzer:
    """Analyzer that uses streaming to measure actual TTFT"""

    def __init__(self, server_url: str):
        self.server_url = server_url

    def generate_text(self, num_tokens: int, prefix: str = "") -> str:
        """Generate text approximating num_tokens (~4 chars per token)"""
        base = "The quick brown fox jumps over the lazy dog and runs through the forest while the sun sets beautifully over the mountains. "
        chars_needed = num_tokens * 4
        result = prefix
        while len(result) < chars_needed:
            result += base
        return result[:chars_needed]

    def measure_streaming_ttft(
        self,
        context_length: int,
        append_length: int,
        max_output_tokens: int = 32,
        timeout: int = 120
    ) -> Optional[TTFTMeasurement]:
        """
        Measure TTFT using streaming API.

        Args:
            context_length: L - simulated cached context tokens
            append_length: m - new tokens to append/process
            max_output_tokens: Maximum output tokens to generate
            timeout: Request timeout in seconds

        Returns:
            TTFTMeasurement or None if failed
        """
        # Generate prompt
        context = self.generate_text(context_length, "Context: ")
        append_text = self.generate_text(append_length, "\n\nBased on the context above, please answer briefly: ")
        prompt = context + append_text

        payload = {
            "text": prompt,
            "sampling_params": {
                "temperature": 0.1,
                "max_new_tokens": max_output_tokens,
            },
            "stream": True,
        }

        start_time = time.perf_counter()
        ttft = None
        output_text = ""
        token_count = 0

        try:
            with requests.post(
                f"{self.server_url}/generate",
                json=payload,
                stream=True,
                timeout=timeout,
            ) as response:
                if response.status_code != 200:
                    return None

                for line in response.iter_lines():
                    if line:
                        current_time = time.perf_counter()

                        # First token received - this is TTFT
                        if ttft is None:
                            ttft = (current_time - start_time) * 1000

                        # Parse the streaming response
                        try:
                            line_str = line.decode('utf-8')
                            if line_str.startswith('data: '):
                                data = json.loads(line_str[6:])
                                if 'text' in data:
                                    new_text = data['text']
                                    if new_text != output_text:
                                        output_text = new_text
                                        token_count += 1
                        except (json.JSONDecodeError, UnicodeDecodeError):
                            pass

            end_time = time.perf_counter()
            total_time = (end_time - start_time) * 1000

            if ttft is None:
                return None

            total_input = context_length + append_length
            cache_rate = context_length / total_input

            return TTFTMeasurement(
                context_length=context_length,
                append_length=append_length,
                total_input=total_input,
                cache_rate=cache_rate,
                ttft_ms=ttft,
                total_time_ms=total_time,
                output_tokens=max(token_count, 1),
            )

        except Exception as e:
            print(f"  Error: {e}")
            return None

    def measure_non_streaming_ttft(
        self,
        context_length: int,
        append_length: int,
        max_output_tokens: int = 32,
        timeout: int = 120
    ) -> Optional[TTFTMeasurement]:
        """
        Fallback: Use non-streaming with vLLM-style metrics if available.
        """
        context = self.generate_text(context_length, "Context: ")
        append_text = self.generate_text(append_length, "\n\nBased on the context above, please answer: ")
        prompt = context + append_text

        payload = {
            "text": prompt,
            "sampling_params": {
                "temperature": 0.1,
                "max_new_tokens": max_output_tokens,
            },
            "return_logprob": False,
        }

        start_time = time.perf_counter()

        try:
            response = requests.post(
                f"{self.server_url}/generate",
                json=payload,
                timeout=timeout,
            )

            end_time = time.perf_counter()
            total_time = (end_time - start_time) * 1000

            if response.status_code != 200:
                return None

            result = response.json()
            output = result.get("text", "")
            output_tokens = len(output.split())

            # Try to extract actual timing from response metadata
            meta = result.get("meta_info", {})
            ttft = meta.get("prefill_time_ms") or meta.get("ttft_ms")

            # If no actual TTFT in response, estimate based on total time
            # Using heuristic: prefill is proportional to input tokens vs total work
            if ttft is None:
                total_input = context_length + append_length
                # Prefill is compute-bound: ~O(n^2) for attention
                # Decode is memory-bound: ~O(n) per token
                # Rough model: prefill_fraction = input^2 / (input^2 + output * input)
                input_work = total_input * total_input
                decode_work = output_tokens * total_input
                total_work = input_work + decode_work
                estimated_prefill_fraction = input_work / total_work if total_work > 0 else 0.3
                ttft = total_time * estimated_prefill_fraction

            total_input = context_length + append_length
            cache_rate = context_length / total_input

            return TTFTMeasurement(
                context_length=context_length,
                append_length=append_length,
                total_input=total_input,
                cache_rate=cache_rate,
                ttft_ms=ttft,
                total_time_ms=total_time,
                output_tokens=max(output_tokens, 1),
            )

        except Exception as e:
            print(f"  Error: {e}")
            return None


def run_experiment(
    analyzer: StreamingTTFTAnalyzer,
    name: str,
    test_cases: List[Tuple[int, int]],  # List of (L, m) tuples
    num_runs: int = 3,
    use_streaming: bool = True,
) -> List[Dict]:
    """Run an experiment with multiple test cases"""
    print(f"\n{'='*60}")
    print(f"EXPERIMENT: {name}")
    print(f"{'='*60}")

    results = []

    for i, (L, m) in enumerate(test_cases):
        r = L / (L + m)
        print(f"\n[{i+1}/{len(test_cases)}] L={L}, m={m}, r={r:.2f}")

        measurements = []
        for run in range(num_runs):
            if use_streaming:
                meas = analyzer.measure_streaming_ttft(L, m)
            else:
                meas = analyzer.measure_non_streaming_ttft(L, m)

            if meas:
                measurements.append(meas)
                print(f"  Run {run+1}: TTFT={meas.ttft_ms:.1f}ms, Total={meas.total_time_ms:.1f}ms")

        if measurements:
            avg_result = {
                "context_length": L,
                "append_length": m,
                "total_input": L + m,
                "cache_rate": round(r, 3),
                "experiment": name,
                "num_runs": len(measurements),
                "ttft_mean_ms": round(statistics.mean(m.ttft_ms for m in measurements), 2),
                "ttft_std_ms": round(statistics.stdev(m.ttft_ms for m in measurements), 2) if len(measurements) > 1 else 0,
                "total_time_mean_ms": round(statistics.mean(m.total_time_ms for m in measurements), 2),
                "decode_time_mean_ms": round(statistics.mean(m.decode_time_ms for m in measurements), 2),
                "ms_per_input_token": round(statistics.mean(m.ms_per_input_token for m in measurements), 4),
                "tpot_mean_ms": round(statistics.mean(m.tpot_ms for m in measurements), 2),
            }
            results.append(avg_result)
            print(f"  AVG: TTFT={avg_result['ttft_mean_ms']:.1f}ms ± {avg_result['ttft_std_ms']:.1f}")

    return results


def analyze_prefill_vs_decode_characteristics(results: List[Dict]):
    """
    Analyze whether append-prefill behavior is more prefill-like or decode-like.

    Prefill characteristics:
    - Time scales with input length (compute-bound)
    - High throughput (tokens/ms)
    - Parallelizable

    Decode characteristics:
    - Time scales with output length (memory-bound)
    - Lower throughput
    - Sequential
    """
    print("\n" + "="*80)
    print("APPEND-PREFILL CHARACTERISTICS ANALYSIS")
    print("="*80)

    # Group by experiment type
    by_experiment = {}
    for r in results:
        exp = r.get("experiment", "unknown")
        if exp not in by_experiment:
            by_experiment[exp] = []
        by_experiment[exp].append(r)

    # Analyze TTFT scaling with input size (key test for prefill-like behavior)
    print("\n1. TTFT SCALING ANALYSIS (Prefill-like = TTFT scales with input)")
    print("-"*60)

    # Find experiments with varying total input at fixed cache rate
    for exp_name, exp_results in by_experiment.items():
        if "scale" in exp_name.lower() or "fixed" in exp_name.lower():
            sorted_results = sorted(exp_results, key=lambda x: x["total_input"])
            if len(sorted_results) >= 2:
                print(f"\n{exp_name}:")
                for r in sorted_results:
                    print(f"  Input={r['total_input']:>5}, TTFT={r['ttft_mean_ms']:.1f}ms, "
                          f"ms/token={r['ms_per_input_token']:.4f}")

                # Calculate scaling factor
                if len(sorted_results) >= 2:
                    first, last = sorted_results[0], sorted_results[-1]
                    input_ratio = last["total_input"] / first["total_input"]
                    ttft_ratio = last["ttft_mean_ms"] / first["ttft_mean_ms"]
                    scaling = ttft_ratio / input_ratio
                    print(f"  → Input {input_ratio:.1f}x larger, TTFT {ttft_ratio:.1f}x larger")
                    print(f"  → Scaling factor: {scaling:.2f} (1.0 = linear/prefill-like, 0 = constant/decode-like)")

    # Analyze TTFT by cache rate ranges
    print("\n2. TTFT BY CACHE RATE (Impact of cached context vs new tokens)")
    print("-"*60)

    low_r = [r for r in results if r["cache_rate"] < 0.3]
    mid_r = [r for r in results if 0.3 <= r["cache_rate"] < 0.7]
    high_r = [r for r in results if r["cache_rate"] >= 0.7]

    def summarize_group(group, name):
        if not group:
            return
        avg_ttft = statistics.mean(r["ttft_mean_ms"] for r in group)
        avg_total = statistics.mean(r["total_time_mean_ms"] for r in group)
        avg_ms_per_token = statistics.mean(r["ms_per_input_token"] for r in group)
        prefill_fraction = avg_ttft / avg_total * 100

        print(f"\n{name}:")
        print(f"  Average TTFT: {avg_ttft:.1f}ms ({prefill_fraction:.1f}% of total)")
        print(f"  Average ms/input_token: {avg_ms_per_token:.4f}")
        print(f"  Samples: {len(group)}")

    summarize_group(low_r, "LOW cache rate (r < 0.3) - Many new tokens")
    summarize_group(mid_r, "MID cache rate (0.3 <= r < 0.7) - Balanced")
    summarize_group(high_r, "HIGH cache rate (r >= 0.7) - Few new tokens")

    # Key insight: Does TTFT depend on m (append length)?
    print("\n3. CRITICAL TEST: Does TTFT scale with m (append length)?")
    print("-"*60)
    print("   If TTFT scales with m → append-prefill is PREFILL-LIKE (compute-bound)")
    print("   If TTFT is constant → append-prefill is DECODE-LIKE (memory-bound)")

    # Find experiments where L is fixed but m varies
    fixed_context_results = [r for r in results if "fixed context" in r.get("experiment", "").lower()]
    if fixed_context_results:
        sorted_by_m = sorted(fixed_context_results, key=lambda x: x["append_length"])
        print(f"\n   Fixed context (L={sorted_by_m[0]['context_length']}), varying append (m):")
        for r in sorted_by_m:
            print(f"     m={r['append_length']:>4}: TTFT={r['ttft_mean_ms']:.1f}ms")

        if len(sorted_by_m) >= 2:
            first, last = sorted_by_m[0], sorted_by_m[-1]
            m_ratio = last["append_length"] / first["append_length"]
            ttft_ratio = last["ttft_mean_ms"] / first["ttft_mean_ms"]
            print(f"\n   → When m increases {m_ratio:.0f}x, TTFT changes {ttft_ratio:.2f}x")
            if ttft_ratio > 1.5:
                print(f"   → CONCLUSION: Append-prefill is PREFILL-LIKE (TTFT scales with m)")
            elif ttft_ratio < 1.2:
                print(f"   → CONCLUSION: Append-prefill is DECODE-LIKE (TTFT mostly constant)")
            else:
                print(f"   → CONCLUSION: Append-prefill shows MIXED characteristics")


def print_summary_table(results: List[Dict]):
    """Print a summary table of all results"""
    print("\n" + "="*80)
    print("DETAILED RESULTS TABLE")
    print("="*80)

    print(f"\n{'L':>6} {'m':>6} {'Total':>6} {'r':>5} | "
          f"{'TTFT(ms)':>12} {'Decode(ms)':>12} {'Total(ms)':>12} | "
          f"{'ms/input':>10}")
    print("-"*80)

    for r in sorted(results, key=lambda x: (x["cache_rate"], x["total_input"])):
        print(f"{r['context_length']:>6} {r['append_length']:>6} {r['total_input']:>6} "
              f"{r['cache_rate']:>5.2f} | "
              f"{r['ttft_mean_ms']:>6.1f}±{r['ttft_std_ms']:>4.0f} "
              f"{r['decode_time_mean_ms']:>12.1f} "
              f"{r['total_time_mean_ms']:>12.1f} | "
              f"{r['ms_per_input_token']:>10.4f}")


def main():
    parser = argparse.ArgumentParser(description="Streaming TTFT Analysis for Append-Prefill")
    parser.add_argument("--server-url", type=str, default="http://127.0.0.1:30000")
    parser.add_argument("--runs", type=int, default=3, help="Number of runs per test case")
    parser.add_argument("--output", type=str, default="/workspace/ppd/results/ttft_analysis.json")
    parser.add_argument("--no-streaming", action="store_true", help="Use non-streaming API")
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

    analyzer = StreamingTTFTAnalyzer(args.server_url)
    use_streaming = not args.no_streaming

    all_results = []

    if args.quick:
        # Quick test
        experiments = [
            ("Quick: Fixed total, vary r", [
                (102, 922),   # r≈0.1
                (512, 512),   # r=0.5
                (922, 102),   # r≈0.9
            ]),
            ("Quick: Fixed context, vary m", [
                (512, 64),
                (512, 256),
                (512, 512),
            ]),
        ]
    else:
        # Full experiments
        experiments = [
            # Experiment 1: Fixed total input (1024), vary cache rate
            ("Exp1: Fixed total (1024), vary cache rate", [
                (102, 922),   # r≈0.1
                (204, 820),   # r≈0.2
                (307, 717),   # r≈0.3
                (409, 615),   # r≈0.4
                (512, 512),   # r=0.5
                (614, 410),   # r≈0.6
                (716, 308),   # r≈0.7
                (819, 205),   # r≈0.8
                (922, 102),   # r≈0.9
            ]),

            # Experiment 2: Fixed cache rate (0.5), vary scale
            ("Exp2: Fixed cache rate (0.5), vary scale", [
                (128, 128),   # 256 total
                (256, 256),   # 512 total
                (512, 512),   # 1024 total
                (1024, 1024), # 2048 total
            ]),

            # Experiment 3: Fixed context (512), vary append - KEY TEST
            ("Exp3: Fixed context (L=512), vary append", [
                (512, 32),
                (512, 64),
                (512, 128),
                (512, 256),
                (512, 512),
                (512, 1024),
            ]),

            # Experiment 4: Fixed append (128), vary context
            ("Exp4: Fixed append (m=128), vary context", [
                (64, 128),
                (128, 128),
                (256, 128),
                (512, 128),
                (1024, 128),
                (2048, 128),
            ]),
        ]

    for exp_name, test_cases in experiments:
        results = run_experiment(
            analyzer, exp_name, test_cases,
            num_runs=args.runs,
            use_streaming=use_streaming
        )
        all_results.extend(results)

    # Save results
    with open(args.output, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to: {args.output}")

    # Print summary table
    print_summary_table(all_results)

    # Analyze characteristics
    analyze_prefill_vs_decode_characteristics(all_results)

    # Final conclusions
    print("\n" + "="*80)
    print("CONCLUSIONS FOR P/D DISAGGREGATION ROUTING")
    print("="*80)
    print("""
Based on the append-prefill characteristics observed:

1. APPEND-PREFILL BEHAVIOR:
   - If TTFT scales with m: The append operation is COMPUTE-BOUND (prefill-like)
     → Benefit from routing to P-machine for optimized prefill kernels
   - If TTFT is constant: The append operation is MEMORY-BOUND (decode-like)
     → May be better to keep on D-machine to avoid transfer overhead

2. ROUTING DECISION FRAMEWORK:
   For cache rate r = L/(L+m):

   LOW r (< 0.3): Large m, significant compute work
   → Likely beneficial to route to P-machine

   HIGH r (> 0.7): Small m, minimal new compute
   → Likely better to stay on D-machine

   The crossover threshold depends on:
   - KV cache transfer cost (bandwidth, L size)
   - Compute advantage of P-machine
   - Network latency

3. EMPIRICAL THRESHOLDS (from this analysis):
   Use the TTFT measurements to determine optimal routing point where:
   T_transfer(L) + T_prefill_P(m) < T_append_D(m)
""")


if __name__ == "__main__":
    main()
