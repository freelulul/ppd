#!/usr/bin/env python3
"""
Quick validation script to verify the P/D disaggregation setup is working.

This script performs basic health checks and simple inference tests
to ensure the environment is correctly configured before running
full experiments.

Usage:
    python validate_setup.py --router-url http://127.0.0.1:30000
    python validate_setup.py --prefill-url http://127.0.0.1:30100 --decode-url http://127.0.0.1:30200
"""

import argparse
import sys
import time
import requests
from typing import Optional, Tuple


def check_server_health(url: str, name: str = "Server") -> bool:
    """Check if a server is healthy."""
    health_url = f"{url.rstrip('/')}/health"
    try:
        response = requests.get(health_url, timeout=10)
        if response.status_code == 200:
            print(f"  [OK] {name} at {url} is healthy")
            return True
        else:
            print(f"  [FAIL] {name} at {url} returned status {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print(f"  [FAIL] {name} at {url} - Connection refused")
        return False
    except requests.exceptions.Timeout:
        print(f"  [FAIL] {name} at {url} - Connection timeout")
        return False
    except Exception as e:
        print(f"  [FAIL] {name} at {url} - Error: {e}")
        return False


def get_server_info(url: str) -> Optional[dict]:
    """Get server information."""
    try:
        response = requests.get(f"{url.rstrip('/')}/get_server_info", timeout=10)
        if response.status_code == 200:
            return response.json()
    except Exception:
        pass
    return None


def get_model_info(url: str) -> Optional[dict]:
    """Get model information."""
    try:
        response = requests.get(f"{url.rstrip('/')}/get_model_info", timeout=10)
        if response.status_code == 200:
            return response.json()
    except Exception:
        pass
    return None


def run_simple_inference(url: str, prompt: str = "Hello, how are you?") -> Tuple[bool, str, float]:
    """
    Run a simple inference request.

    Returns: (success, output_text, latency_ms)
    """
    try:
        start_time = time.perf_counter()

        response = requests.post(
            f"{url.rstrip('/')}/generate",
            json={
                "text": prompt,
                "sampling_params": {
                    "temperature": 0.7,
                    "max_new_tokens": 32,
                },
            },
            timeout=60,
        )

        latency = (time.perf_counter() - start_time) * 1000

        if response.status_code == 200:
            result = response.json()
            output_text = result.get("text", "")
            return True, output_text, latency
        else:
            return False, f"Status {response.status_code}: {response.text}", latency

    except Exception as e:
        return False, str(e), 0


def run_multi_turn_simulation(url: str) -> Tuple[bool, str]:
    """
    Simulate a simple multi-turn conversation to test KV cache handling.

    Returns: (success, message)
    """
    turns = [
        "Let me tell you about artificial intelligence.",
        "What aspects are most interesting?",
        "Can you elaborate on that point?",
    ]

    print("\n  Simulating multi-turn conversation:")

    for i, prompt in enumerate(turns):
        success, output, latency = run_simple_inference(url, prompt)
        if not success:
            return False, f"Turn {i+1} failed: {output}"

        print(f"    Turn {i+1}: {latency:.1f}ms - OK")

    return True, "All turns completed successfully"


def validate_pd_disaggregation(
    router_url: str,
    prefill_url: Optional[str] = None,
    decode_url: Optional[str] = None,
) -> bool:
    """
    Validate the P/D disaggregation setup.

    Returns True if all checks pass.
    """
    print("\n" + "="*60)
    print("P/D Disaggregation Setup Validation")
    print("="*60)

    all_passed = True

    # Step 1: Check server health
    print("\n1. Checking Server Health:")

    if prefill_url:
        if not check_server_health(prefill_url, "Prefill Server"):
            all_passed = False

    if decode_url:
        if not check_server_health(decode_url, "Decode Server"):
            all_passed = False

    if not check_server_health(router_url, "Router"):
        all_passed = False
        print("\n  ERROR: Router is not accessible. Cannot continue.")
        return False

    # Step 2: Get server/model info
    print("\n2. Checking Server Configuration:")

    model_info = get_model_info(router_url)
    if model_info:
        model_path = model_info.get("model_path", "Unknown")
        print(f"  Model: {model_path}")
    else:
        print("  [WARN] Could not retrieve model info")

    server_info = get_server_info(router_url)
    if server_info:
        if "prefill" in server_info:
            print(f"  Prefill servers: {len(server_info['prefill'])}")
        if "decode" in server_info:
            print(f"  Decode servers: {len(server_info['decode'])}")
    else:
        print("  [WARN] Could not retrieve server info")

    # Step 3: Run simple inference
    print("\n3. Testing Simple Inference:")

    success, output, latency = run_simple_inference(router_url)
    if success:
        print(f"  [OK] Inference successful ({latency:.1f}ms)")
        # Truncate output for display
        display_output = output[:100] + "..." if len(output) > 100 else output
        print(f"  Output: {display_output}")
    else:
        print(f"  [FAIL] Inference failed: {output}")
        all_passed = False

    # Step 4: Multi-turn simulation
    print("\n4. Testing Multi-Turn Conversation:")

    success, message = run_multi_turn_simulation(router_url)
    if success:
        print(f"  [OK] {message}")
    else:
        print(f"  [FAIL] {message}")
        all_passed = False

    # Step 5: Test different input sizes
    print("\n5. Testing Different Input Sizes:")

    test_prompts = [
        ("Short (10 tokens)", "Hello!"),
        ("Medium (50 tokens)", "Please explain the concept of machine learning and how it differs from traditional programming approaches."),
        ("Long (100+ tokens)", "I would like you to provide a comprehensive explanation of how transformer architectures work in modern natural language processing systems, including the attention mechanism, positional encodings, and the overall encoder-decoder structure that has revolutionized the field."),
    ]

    for name, prompt in test_prompts:
        success, output, latency = run_simple_inference(router_url, prompt)
        status = "OK" if success else "FAIL"
        print(f"  [{status}] {name}: {latency:.1f}ms")
        if not success:
            all_passed = False

    # Summary
    print("\n" + "="*60)
    if all_passed:
        print("VALIDATION PASSED - Environment is ready for experiments")
    else:
        print("VALIDATION FAILED - Please check the errors above")
    print("="*60 + "\n")

    return all_passed


def main():
    parser = argparse.ArgumentParser(
        description="Validate P/D disaggregation setup"
    )
    parser.add_argument(
        "--router-url", type=str, default="http://127.0.0.1:30000",
        help="Router URL"
    )
    parser.add_argument(
        "--prefill-url", type=str, default=None,
        help="Prefill server URL (optional, for direct health check)"
    )
    parser.add_argument(
        "--decode-url", type=str, default=None,
        help="Decode server URL (optional, for direct health check)"
    )

    args = parser.parse_args()

    success = validate_pd_disaggregation(
        router_url=args.router_url,
        prefill_url=args.prefill_url,
        decode_url=args.decode_url,
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
