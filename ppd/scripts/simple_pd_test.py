#!/usr/bin/env python3
"""
Simple P/D disaggregation test script.

This script starts prefill and decode servers on separate GPUs and runs
a simple inference test to verify the setup is working correctly.

Usage:
    python simple_pd_test.py --model meta-llama/Llama-3.1-8B-Instruct
    python simple_pd_test.py --model /path/to/model --prefill-gpu 0 --decode-gpu 1
"""

import argparse
import os
import sys
import time
import signal
import subprocess
import requests
import threading
from typing import Optional

# Add sglang_router to path
sys.path.insert(0, '/workspace/ppd/sgl-router/bindings/python')


def stream_output(process, name):
    """Stream process output to console with prefix."""
    for line in iter(process.stdout.readline, b''):
        if line:
            print(f"[{name}] {line.decode().rstrip()}")


def wait_for_server(url: str, name: str, timeout: int = 300) -> bool:
    """Wait for a server to become healthy."""
    health_url = f"{url}/health"
    start_time = time.time()

    print(f"Waiting for {name} at {url}...")

    while time.time() - start_time < timeout:
        try:
            response = requests.get(health_url, timeout=5)
            if response.status_code == 200:
                print(f"  {name} is ready!")
                return True
        except requests.exceptions.RequestException:
            pass

        # Print progress dot every 10 seconds
        elapsed = int(time.time() - start_time)
        if elapsed % 10 == 0 and elapsed > 0:
            print(f"  Still waiting... ({elapsed}s)")

        time.sleep(2)

    print(f"  {name} failed to start within {timeout}s")
    return False


def run_simple_inference(url: str) -> bool:
    """Run a simple inference test."""
    print("\nRunning simple inference test...")

    prompt = "Hello! Can you tell me a short joke?"

    try:
        response = requests.post(
            f"{url}/generate",
            json={
                "text": prompt,
                "sampling_params": {
                    "temperature": 0.7,
                    "max_new_tokens": 64,
                },
            },
            timeout=60,
        )

        if response.status_code == 200:
            result = response.json()
            output = result.get("text", "")
            print(f"\nPrompt: {prompt}")
            print(f"Response: {output}")
            print("\nInference test PASSED!")
            return True
        else:
            print(f"Request failed with status {response.status_code}")
            print(f"Response: {response.text}")
            return False

    except Exception as e:
        print(f"Inference test failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Simple P/D disaggregation test")
    parser.add_argument(
        "--model", type=str, default="meta-llama/Llama-3.1-8B-Instruct",
        help="Model path or HuggingFace model ID"
    )
    parser.add_argument("--prefill-gpu", type=int, default=0, help="GPU ID for prefill server")
    parser.add_argument("--decode-gpu", type=int, default=1, help="GPU ID for decode server")
    parser.add_argument("--prefill-port", type=int, default=30100, help="Port for prefill server")
    parser.add_argument("--decode-port", type=int, default=30200, help="Port for decode server")
    parser.add_argument("--router-port", type=int, default=30000, help="Port for router")
    parser.add_argument("--bootstrap-port", type=int, default=9000, help="Bootstrap port for KV transfer")
    parser.add_argument(
        "--transfer-backend", type=str, default="mooncake",
        choices=["mooncake", "nixl", "fake"],
        help="KV cache transfer backend"
    )
    parser.add_argument(
        "--use-tcp", action="store_true", default=True,
        help="Use TCP transport for mooncake (for environments without RDMA hardware)"
    )
    parser.add_argument("--keep-running", action="store_true", help="Keep servers running after test")

    args = parser.parse_args()

    processes = []

    def cleanup():
        print("\nCleaning up processes...")
        for name, proc in processes:
            print(f"  Terminating {name}...")
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
        print("Cleanup complete")

    # Handle signals
    def signal_handler(signum, frame):
        cleanup()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        # ============================================
        # Start Prefill Server
        # ============================================
        print("\n" + "="*60)
        print("Starting Prefill Server")
        print("="*60)

        prefill_cmd = [
            sys.executable, "-m", "sglang.launch_server",
            "--model-path", args.model,
            "--host", "127.0.0.1",
            "--port", str(args.prefill_port),
            "--disaggregation-mode", "prefill",
            "--disaggregation-transfer-backend", args.transfer_backend,
            "--disaggregation-bootstrap-port", str(args.bootstrap_port),
            "--tp", "1",
            "--trust-remote-code",
            "--mem-fraction-static", "0.85",
        ]

        prefill_env = os.environ.copy()
        prefill_env["CUDA_VISIBLE_DEVICES"] = str(args.prefill_gpu)
        if args.use_tcp:
            prefill_env["SGLANG_MOONCAKE_USE_TCP"] = "true"

        print(f"Command: CUDA_VISIBLE_DEVICES={args.prefill_gpu} {' '.join(prefill_cmd)}")

        prefill_proc = subprocess.Popen(
            prefill_cmd,
            env=prefill_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        processes.append(("Prefill", prefill_proc))

        # Start output streaming thread
        prefill_thread = threading.Thread(
            target=stream_output, args=(prefill_proc, "Prefill"), daemon=True
        )
        prefill_thread.start()

        # ============================================
        # Start Decode Server
        # ============================================
        print("\n" + "="*60)
        print("Starting Decode Server")
        print("="*60)

        decode_cmd = [
            sys.executable, "-m", "sglang.launch_server",
            "--model-path", args.model,
            "--host", "127.0.0.1",
            "--port", str(args.decode_port),
            "--disaggregation-mode", "decode",
            "--disaggregation-transfer-backend", args.transfer_backend,
            "--tp", "1",
            # Note: Do NOT use --base-gpu-id with CUDA_VISIBLE_DEVICES
            # When CUDA_VISIBLE_DEVICES=1, the device appears as GPU 0 internally
            "--trust-remote-code",
            "--mem-fraction-static", "0.85",
        ]

        decode_env = os.environ.copy()
        decode_env["CUDA_VISIBLE_DEVICES"] = str(args.decode_gpu)
        if args.use_tcp:
            decode_env["SGLANG_MOONCAKE_USE_TCP"] = "true"

        print(f"Command: CUDA_VISIBLE_DEVICES={args.decode_gpu} {' '.join(decode_cmd)}")

        decode_proc = subprocess.Popen(
            decode_cmd,
            env=decode_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        processes.append(("Decode", decode_proc))

        # Start output streaming thread
        decode_thread = threading.Thread(
            target=stream_output, args=(decode_proc, "Decode"), daemon=True
        )
        decode_thread.start()

        # ============================================
        # Wait for servers to be ready
        # ============================================
        print("\n" + "="*60)
        print("Waiting for servers to initialize...")
        print("="*60)

        prefill_url = f"http://127.0.0.1:{args.prefill_port}"
        decode_url = f"http://127.0.0.1:{args.decode_port}"

        if not wait_for_server(prefill_url, "Prefill Server"):
            print("ERROR: Prefill server failed to start")
            cleanup()
            sys.exit(1)

        if not wait_for_server(decode_url, "Decode Server"):
            print("ERROR: Decode server failed to start")
            cleanup()
            sys.exit(1)

        # ============================================
        # Start Router (Mini Load Balancer)
        # ============================================
        print("\n" + "="*60)
        print("Starting Router (Mini Load Balancer)")
        print("="*60)

        router_cmd = [
            sys.executable, "-m", "sglang_router.launch_router",
            "--pd-disaggregation",
            "--mini-lb",
            "--host", "127.0.0.1",
            "--port", str(args.router_port),
            "--prefill", prefill_url, str(args.bootstrap_port),
            "--decode", decode_url,
        ]

        # Add sglang_router to PYTHONPATH
        router_env = os.environ.copy()
        router_env["PYTHONPATH"] = "/workspace/ppd/sgl-router/bindings/python:" + router_env.get("PYTHONPATH", "")

        print(f"Command: {' '.join(router_cmd)}")

        router_proc = subprocess.Popen(
            router_cmd,
            env=router_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        processes.append(("Router", router_proc))

        # Start output streaming thread
        router_thread = threading.Thread(
            target=stream_output, args=(router_proc, "Router"), daemon=True
        )
        router_thread.start()

        router_url = f"http://127.0.0.1:{args.router_port}"
        if not wait_for_server(router_url, "Router", timeout=60):
            print("ERROR: Router failed to start")
            cleanup()
            sys.exit(1)

        # ============================================
        # Run inference test
        # ============================================
        print("\n" + "="*60)
        print("All servers ready! Running inference test...")
        print("="*60)

        success = run_simple_inference(router_url)

        if not success:
            print("\nInference test FAILED!")
            cleanup()
            sys.exit(1)

        # ============================================
        # Summary
        # ============================================
        print("\n" + "="*60)
        print("P/D Disaggregation Test PASSED!")
        print("="*60)
        print(f"\nServer URLs:")
        print(f"  Prefill: {prefill_url}")
        print(f"  Decode:  {decode_url}")
        print(f"  Router:  {router_url}")

        if args.keep_running:
            print("\nServers are running. Press Ctrl+C to stop.")
            while True:
                time.sleep(1)
        else:
            print("\nTest complete. Cleaning up...")
            cleanup()

    except Exception as e:
        print(f"\nError: {e}")
        cleanup()
        sys.exit(1)


if __name__ == "__main__":
    main()
