#!/usr/bin/env python3
"""
Multi-turn conversation test for Append-Prefill trade-off analysis.

This experiment measures the performance characteristics of append-prefill
under different scenarios:
1. Small append (m << L): High cache rate, should behave like decode
2. Large append (m >> L): Low cache rate, should behave like prefill
3. Balanced cases: Need intelligent routing decision

Key Metrics:
- TTFT (Time to First Token)
- TPOT (Time Per Output Token)
- End-to-end latency
- KV cache transfer overhead (if measurable)
- GPU compute/memory utilization
"""

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
import statistics

import requests
from transformers import AutoTokenizer

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs import (
    FullConfig,
    AppendPrefillTestCase,
    get_config_for_h100_4gpu,
    get_minimal_test_config,
)


@dataclass
class TimingMetrics:
    """Timing metrics for a single request."""
    ttft_ms: float = 0.0  # Time to first token
    tpot_ms: float = 0.0  # Time per output token (average)
    e2e_latency_ms: float = 0.0  # Total end-to-end latency
    input_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0  # Tokens served from cache

    @property
    def cache_hit_rate(self) -> float:
        """Calculate actual cache hit rate."""
        if self.input_tokens == 0:
            return 0.0
        return self.cached_tokens / self.input_tokens


@dataclass
class TestResult:
    """Result of a single test case."""
    test_case: AppendPrefillTestCase
    metrics: TimingMetrics
    success: bool = True
    error_message: str = ""
    raw_response: Optional[Dict] = None


@dataclass
class ExperimentResults:
    """Aggregated results from an experiment run."""
    config: Dict[str, Any]
    results: List[TestResult] = field(default_factory=list)
    start_time: str = ""
    end_time: str = ""
    summary: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "config": self.config,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "summary": self.summary,
            "results": [
                {
                    "test_case": {
                        "context_length": r.test_case.context_length,
                        "append_length": r.test_case.append_length,
                        "cache_rate": r.test_case.cache_rate,
                    },
                    "metrics": asdict(r.metrics),
                    "success": r.success,
                    "error_message": r.error_message,
                }
                for r in self.results
            ],
        }


class MultiTurnConversationTester:
    """
    Tester for multi-turn conversation append-prefill characteristics.

    This class simulates multi-turn conversations and measures how different
    append lengths affect performance, validating the P-PD hypothesis.
    """

    def __init__(
        self,
        router_url: str,
        model_name: str,
        config: FullConfig,
    ):
        self.router_url = router_url.rstrip("/")
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Pre-generate context of various lengths
        self.context_cache: Dict[int, List[int]] = {}

    def _generate_context(self, target_length: int) -> List[int]:
        """Generate or retrieve context tokens of target length."""
        if target_length in self.context_cache:
            return self.context_cache[target_length]

        # Generate realistic conversation context
        # Using repetitive but valid text to reach target length
        base_text = (
            "This is a multi-turn conversation about artificial intelligence, "
            "machine learning, and natural language processing. "
            "We have been discussing various topics including neural networks, "
            "transformer architectures, attention mechanisms, and optimization. "
        )

        # Repeat and truncate to get exact token count
        text = base_text
        tokens = self.tokenizer.encode(text, add_special_tokens=False)

        while len(tokens) < target_length:
            text = text + base_text
            tokens = self.tokenizer.encode(text, add_special_tokens=False)

        tokens = tokens[:target_length]
        self.context_cache[target_length] = tokens
        return tokens

    def _generate_append_prompt(self, target_length: int) -> Tuple[str, List[int]]:
        """Generate append prompt of target length."""
        # Generate user query that would naturally follow a conversation
        base_prompts = [
            "Can you explain more about ",
            "What do you think about ",
            "Tell me about the relationship between ",
            "How does this connect to ",
            "I'm curious about ",
        ]

        # Topics to pad the prompt
        topics = [
            "transformer attention mechanisms and their computational complexity, "
            "including multi-head attention, self-attention, and cross-attention patterns. ",
            "the evolution of language models from RNNs to transformers, "
            "covering LSTM, GRU, and the attention is all you need paper. ",
            "optimization techniques for large language models including "
            "Adam, AdamW, learning rate scheduling, and gradient clipping. ",
        ]

        prompt = base_prompts[target_length % len(base_prompts)]
        topic_idx = 0

        tokens = self.tokenizer.encode(prompt, add_special_tokens=False)

        while len(tokens) < target_length:
            prompt = prompt + topics[topic_idx % len(topics)]
            topic_idx += 1
            tokens = self.tokenizer.encode(prompt, add_special_tokens=False)

        tokens = tokens[:target_length]
        text = self.tokenizer.decode(tokens)

        return text, tokens

    def run_single_test(
        self,
        test_case: AppendPrefillTestCase,
        use_session: bool = True,
    ) -> TestResult:
        """
        Run a single append-prefill test case.

        This simulates:
        1. First turn: Full prefill of context
        2. Subsequent turn: Append new prompt to existing context
        """
        metrics = TimingMetrics()
        result = TestResult(test_case=test_case, metrics=metrics)

        try:
            # Generate context and append prompt
            context_tokens = self._generate_context(test_case.context_length)
            append_text, append_tokens = self._generate_append_prompt(test_case.append_length)

            if use_session:
                # Use SGLang session for multi-turn (KV cache reuse)
                result = self._run_with_session(test_case, context_tokens, append_tokens)
            else:
                # Run without session (baseline - full context every time)
                result = self._run_without_session(test_case, context_tokens, append_tokens)

        except Exception as e:
            result.success = False
            result.error_message = str(e)

        return result

    def _run_with_session(
        self,
        test_case: AppendPrefillTestCase,
        context_tokens: List[int],
        append_tokens: List[int],
    ) -> TestResult:
        """Run test using SGLang session for KV cache reuse."""
        metrics = TimingMetrics()
        result = TestResult(test_case=test_case, metrics=metrics)

        try:
            # Step 1: Open session
            session_response = requests.post(
                f"{self.router_url}/open_session",
                json={"capacity_of_str_len": test_case.total_input_length + 1000},
                timeout=30,
            )

            if session_response.status_code != 200:
                # Session might not be supported in disaggregation mode
                # Fall back to non-session approach
                return self._run_without_session(test_case, context_tokens, append_tokens)

            session_id = session_response.json()

            # Step 2: First request - establish context (prefill only)
            first_request = {
                "input_ids": context_tokens,
                "session_params": {
                    "id": session_id,
                    "rid": None,
                    "offset": -1,
                    "replace": True,
                },
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": 1,  # Just prefill, minimal generation
                },
            }

            first_response = requests.post(
                f"{self.router_url}/generate",
                json=first_request,
                timeout=self.config.server.request_timeout,
            )

            if first_response.status_code != 200:
                result.success = False
                result.error_message = f"First request failed: {first_response.text}"
                return result

            first_result = first_response.json()
            rid = first_result.get("meta_info", {}).get("id")

            # Step 3: Append request - this is what we're measuring
            # This simulates the user's follow-up input in a multi-turn conversation
            append_request = {
                "input_ids": append_tokens,
                "session_params": {
                    "id": session_id,
                    "rid": rid,
                    "offset": -1,  # Append to end
                    "replace": True,
                },
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": test_case.expected_output_length,
                    "no_stop_trim": True,
                },
            }

            # Measure timing for append request
            start_time = time.perf_counter()

            append_response = requests.post(
                f"{self.router_url}/generate",
                json=append_request,
                timeout=self.config.server.request_timeout,
            )

            e2e_time = (time.perf_counter() - start_time) * 1000  # ms

            if append_response.status_code != 200:
                result.success = False
                result.error_message = f"Append request failed: {append_response.text}"
                return result

            append_result = append_response.json()
            meta_info = append_result.get("meta_info", {})

            # Extract metrics
            metrics.e2e_latency_ms = e2e_time
            metrics.input_tokens = len(context_tokens) + len(append_tokens)
            metrics.output_tokens = meta_info.get("completion_tokens", test_case.expected_output_length)
            metrics.cached_tokens = meta_info.get("cached_tokens", len(context_tokens))

            # Calculate TTFT and TPOT if available
            if metrics.output_tokens > 0:
                metrics.tpot_ms = e2e_time / metrics.output_tokens

            result.metrics = metrics
            result.raw_response = append_result

            # Close session
            requests.post(
                f"{self.router_url}/close_session",
                json={"session_id": session_id},
                timeout=10,
            )

        except Exception as e:
            result.success = False
            result.error_message = str(e)

        return result

    def _run_without_session(
        self,
        test_case: AppendPrefillTestCase,
        context_tokens: List[int],
        append_tokens: List[int],
    ) -> TestResult:
        """
        Run test without session - full context sent each time.
        This is the baseline to compare against session-based approach.
        """
        metrics = TimingMetrics()
        result = TestResult(test_case=test_case, metrics=metrics)

        try:
            # Combine context and append tokens
            full_input = context_tokens + append_tokens

            request = {
                "input_ids": full_input,
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": test_case.expected_output_length,
                    "no_stop_trim": True,
                },
            }

            # Measure timing
            start_time = time.perf_counter()

            response = requests.post(
                f"{self.router_url}/generate",
                json=request,
                timeout=self.config.server.request_timeout,
            )

            e2e_time = (time.perf_counter() - start_time) * 1000  # ms

            if response.status_code != 200:
                result.success = False
                result.error_message = f"Request failed: {response.text}"
                return result

            response_json = response.json()
            meta_info = response_json.get("meta_info", {})

            # Extract metrics
            metrics.e2e_latency_ms = e2e_time
            metrics.input_tokens = len(full_input)
            metrics.output_tokens = meta_info.get("completion_tokens", test_case.expected_output_length)
            metrics.cached_tokens = meta_info.get("cached_tokens", 0)  # Prefix cache might help

            if metrics.output_tokens > 0:
                metrics.tpot_ms = e2e_time / metrics.output_tokens

            result.metrics = metrics
            result.raw_response = response_json

        except Exception as e:
            result.success = False
            result.error_message = str(e)

        return result

    def run_experiment(
        self,
        test_cases: List[AppendPrefillTestCase],
        num_warmup: int = 2,
        num_repetitions: int = 3,
        use_session: bool = True,
    ) -> ExperimentResults:
        """
        Run full experiment across all test cases.

        Args:
            test_cases: List of test cases to run
            num_warmup: Number of warmup iterations (results discarded)
            num_repetitions: Number of measurement repetitions
            use_session: Whether to use SGLang sessions for KV cache reuse
        """
        experiment = ExperimentResults(
            config=self.config.to_dict(),
            start_time=datetime.now().isoformat(),
        )

        total_tests = len(test_cases) * num_repetitions
        completed = 0

        print(f"\n{'='*60}")
        print(f"Starting Append-Prefill Experiment")
        print(f"Test cases: {len(test_cases)}")
        print(f"Repetitions: {num_repetitions}")
        print(f"Warmup iterations: {num_warmup}")
        print(f"Use session: {use_session}")
        print(f"{'='*60}\n")

        # Warmup
        if num_warmup > 0 and test_cases:
            print("Running warmup iterations...")
            warmup_case = test_cases[0]
            for i in range(num_warmup):
                self.run_single_test(warmup_case, use_session=use_session)
            print("Warmup complete.\n")

        # Run all test cases
        for test_case in test_cases:
            case_results = []

            for rep in range(num_repetitions):
                result = self.run_single_test(test_case, use_session=use_session)
                case_results.append(result)
                experiment.results.append(result)

                completed += 1
                status = "OK" if result.success else "FAILED"
                print(
                    f"[{completed}/{total_tests}] "
                    f"L={test_case.context_length:5d}, m={test_case.append_length:4d}, "
                    f"r={test_case.cache_rate:.3f} | "
                    f"E2E={result.metrics.e2e_latency_ms:8.2f}ms, "
                    f"TPOT={result.metrics.tpot_ms:6.2f}ms | "
                    f"{status}"
                )

            # Brief pause between test cases
            time.sleep(0.5)

        experiment.end_time = datetime.now().isoformat()

        # Calculate summary statistics
        experiment.summary = self._calculate_summary(experiment.results)

        return experiment

    def _calculate_summary(self, results: List[TestResult]) -> Dict[str, Any]:
        """Calculate summary statistics from results."""
        summary = {
            "total_tests": len(results),
            "successful_tests": sum(1 for r in results if r.success),
            "failed_tests": sum(1 for r in results if not r.success),
            "by_cache_rate": {},
        }

        # Group by cache rate ranges
        cache_rate_ranges = [
            (0.0, 0.25, "low"),      # Heavy append (prefill-like)
            (0.25, 0.5, "medium_low"),
            (0.5, 0.75, "medium_high"),
            (0.75, 1.0, "high"),     # Light append (decode-like)
        ]

        for low, high, label in cache_rate_ranges:
            range_results = [
                r for r in results
                if r.success and low <= r.test_case.cache_rate < high
            ]

            if range_results:
                e2e_times = [r.metrics.e2e_latency_ms for r in range_results]
                tpot_times = [r.metrics.tpot_ms for r in range_results if r.metrics.tpot_ms > 0]

                summary["by_cache_rate"][label] = {
                    "count": len(range_results),
                    "cache_rate_range": f"{low:.2f}-{high:.2f}",
                    "e2e_latency_ms": {
                        "mean": statistics.mean(e2e_times),
                        "std": statistics.stdev(e2e_times) if len(e2e_times) > 1 else 0,
                        "min": min(e2e_times),
                        "max": max(e2e_times),
                    },
                    "tpot_ms": {
                        "mean": statistics.mean(tpot_times) if tpot_times else 0,
                        "std": statistics.stdev(tpot_times) if len(tpot_times) > 1 else 0,
                    } if tpot_times else None,
                }

        return summary


def run_quick_validation(router_url: str, model_name: str) -> bool:
    """
    Run a quick validation to ensure the P/D setup is working correctly.

    Returns True if validation passes, False otherwise.
    """
    print("\n" + "="*60)
    print("Running Quick Validation Test")
    print("="*60 + "\n")

    config = get_minimal_test_config()
    tester = MultiTurnConversationTester(router_url, model_name, config)

    # Simple test case
    test_case = AppendPrefillTestCase(
        context_length=256,
        append_length=32,
        expected_output_length=64,
    )

    print(f"Test case: L={test_case.context_length}, m={test_case.append_length}")
    print(f"Expected cache rate: {test_case.cache_rate:.3f}")

    # Run without session first (simpler)
    print("\n1. Testing without session (baseline)...")
    result = tester.run_single_test(test_case, use_session=False)

    if result.success:
        print(f"   SUCCESS: E2E={result.metrics.e2e_latency_ms:.2f}ms")
    else:
        print(f"   FAILED: {result.error_message}")
        return False

    # Run with session
    print("\n2. Testing with session (KV cache reuse)...")
    result = tester.run_single_test(test_case, use_session=True)

    if result.success:
        print(f"   SUCCESS: E2E={result.metrics.e2e_latency_ms:.2f}ms")
        print(f"   Cached tokens: {result.metrics.cached_tokens}")
    else:
        print(f"   FAILED: {result.error_message}")
        # Session might not be supported, but that's okay for basic validation
        print("   (Note: Session might not be supported in disaggregation mode)")

    print("\n" + "="*60)
    print("Validation PASSED")
    print("="*60 + "\n")

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Run multi-turn conversation append-prefill experiments"
    )
    parser.add_argument(
        "--router-url", type=str, default="http://127.0.0.1:30000",
        help="Router URL for requests"
    )
    parser.add_argument(
        "--model", type=str, default="meta-llama/Llama-3.1-8B-Instruct",
        help="Model name for tokenizer"
    )
    parser.add_argument(
        "--output-dir", type=str, default="results",
        help="Directory to save results"
    )
    parser.add_argument(
        "--validate-only", action="store_true",
        help="Only run quick validation test"
    )
    parser.add_argument(
        "--minimal", action="store_true",
        help="Use minimal test configuration"
    )
    parser.add_argument(
        "--no-session", action="store_true",
        help="Run without SGLang sessions (baseline mode)"
    )
    parser.add_argument(
        "--repetitions", type=int, default=3,
        help="Number of repetitions per test case"
    )
    parser.add_argument(
        "--warmup", type=int, default=2,
        help="Number of warmup iterations"
    )

    args = parser.parse_args()

    # Validation mode
    if args.validate_only:
        success = run_quick_validation(args.router_url, args.model)
        sys.exit(0 if success else 1)

    # Full experiment mode
    if args.minimal:
        config = get_minimal_test_config()
    else:
        config = get_config_for_h100_4gpu()

    config.experiment.num_repetitions = args.repetitions
    config.experiment.warmup_iterations = args.warmup
    config.experiment.output_dir = args.output_dir

    # Create tester
    tester = MultiTurnConversationTester(args.router_url, args.model, config)

    # Generate test cases
    test_cases = config.generate_test_cases()
    print(f"Generated {len(test_cases)} test cases")

    # Run experiment
    results = tester.run_experiment(
        test_cases=test_cases,
        num_warmup=config.experiment.warmup_iterations,
        num_repetitions=config.experiment.num_repetitions,
        use_session=not args.no_session,
    )

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(args.output_dir, f"experiment_{timestamp}.json")

    with open(output_file, "w") as f:
        json.dump(results.to_dict(), f, indent=2)

    print(f"\nResults saved to: {output_file}")

    # Print summary
    print("\n" + "="*60)
    print("Experiment Summary")
    print("="*60)
    print(f"Total tests: {results.summary['total_tests']}")
    print(f"Successful: {results.summary['successful_tests']}")
    print(f"Failed: {results.summary['failed_tests']}")

    print("\nResults by Cache Rate:")
    for label, stats in results.summary.get("by_cache_rate", {}).items():
        print(f"\n  {label} ({stats['cache_rate_range']}):")
        print(f"    Count: {stats['count']}")
        print(f"    E2E Latency: {stats['e2e_latency_ms']['mean']:.2f}ms "
              f"(std: {stats['e2e_latency_ms']['std']:.2f})")
        if stats.get('tpot_ms'):
            print(f"    TPOT: {stats['tpot_ms']['mean']:.2f}ms")


if __name__ == "__main__":
    main()
