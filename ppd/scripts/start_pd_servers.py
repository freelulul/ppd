#!/usr/bin/env python3
"""
Script to start Prefill/Decode disaggregated servers for P-PD experiments.

This script launches:
1. Prefill server(s) on specified GPU(s)
2. Decode server(s) on specified GPU(s)
3. Load balancer/Router to coordinate requests

Usage:
    python start_pd_servers.py --model meta-llama/Llama-3.1-8B-Instruct
    python start_pd_servers.py --model /path/to/model --prefill-gpus 0 --decode-gpus 1
"""

import argparse
import os
import sys
import time
import signal
import subprocess
import requests
from typing import List, Optional, Tuple
from dataclasses import dataclass

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs import FullConfig, get_config_for_h100_4gpu, get_minimal_test_config


@dataclass
class ServerProcess:
    """Container for a server process and its metadata."""
    process: subprocess.Popen
    name: str
    url: str
    gpu_id: int


class PDServerManager:
    """Manager for Prefill/Decode disaggregated server cluster."""

    def __init__(self, config: FullConfig):
        self.config = config
        self.processes: List[ServerProcess] = []
        self.router_process: Optional[subprocess.Popen] = None

    def _get_server_command(
        self,
        mode: str,  # "prefill" or "decode"
        gpu_id: int,
        host: str,
        port: int,
        bootstrap_port: Optional[int] = None,
    ) -> List[str]:
        """Build command to launch a server."""
        cmd = [
            sys.executable, "-m", "sglang.launch_server",
            "--model-path", self.config.model.model_path,
            "--host", host,
            "--port", str(port),
            "--disaggregation-mode", mode,
            "--tp", str(self.config.hardware.prefill_tp if mode == "prefill" else self.config.hardware.decode_tp),
            "--trust-remote-code",
            "--mem-fraction-static", "0.85",
        ]

        # Add transfer backend
        backend = self.config.server.transfer_backend.value
        cmd.extend(["--disaggregation-transfer-backend", backend])

        # Add bootstrap port for prefill servers
        if mode == "prefill" and bootstrap_port is not None:
            cmd.extend(["--disaggregation-bootstrap-port", str(bootstrap_port)])

        # Add IB device if specified
        if self.config.server.ib_device:
            cmd.extend(["--disaggregation-ib-device", self.config.server.ib_device])

        # For decode server, specify base GPU ID
        if mode == "decode":
            cmd.extend(["--base-gpu-id", str(gpu_id)])

        return cmd

    def start_prefill_server(self, gpu_id: int, port: int, bootstrap_port: int) -> ServerProcess:
        """Start a prefill server on specified GPU."""
        host = self.config.server.base_host
        url = f"http://{host}:{port}"

        cmd = self._get_server_command(
            mode="prefill",
            gpu_id=gpu_id,
            host=host,
            port=port,
            bootstrap_port=bootstrap_port,
        )

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

        print(f"[Prefill] Starting server on GPU {gpu_id}, port {port}")
        print(f"[Prefill] Command: {' '.join(cmd)}")

        process = subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )

        server = ServerProcess(
            process=process,
            name=f"prefill-gpu{gpu_id}",
            url=url,
            gpu_id=gpu_id,
        )
        self.processes.append(server)
        return server

    def start_decode_server(self, gpu_id: int, port: int) -> ServerProcess:
        """Start a decode server on specified GPU."""
        host = self.config.server.base_host
        url = f"http://{host}:{port}"

        cmd = self._get_server_command(
            mode="decode",
            gpu_id=gpu_id,
            host=host,
            port=port,
        )

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

        print(f"[Decode] Starting server on GPU {gpu_id}, port {port}")
        print(f"[Decode] Command: {' '.join(cmd)}")

        process = subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )

        server = ServerProcess(
            process=process,
            name=f"decode-gpu{gpu_id}",
            url=url,
            gpu_id=gpu_id,
        )
        self.processes.append(server)
        return server

    def start_router(
        self,
        prefill_urls: List[Tuple[str, int]],  # (url, bootstrap_port)
        decode_urls: List[str],
    ) -> subprocess.Popen:
        """Start the load balancer/router."""
        host = self.config.server.base_host
        port = self.config.server.router_port

        cmd = [
            sys.executable, "-m", "sglang_router.launch_router",
            "--pd-disaggregation",
            "--mini-lb",  # Use mini load balancer for testing
            "--host", host,
            "--port", str(port),
        ]

        # Add prefill servers with bootstrap ports
        for url, bootstrap_port in prefill_urls:
            cmd.extend(["--prefill", url, str(bootstrap_port)])

        # Add decode servers
        for url in decode_urls:
            cmd.extend(["--decode", url])

        print(f"[Router] Starting on port {port}")
        print(f"[Router] Command: {' '.join(cmd)}")

        self.router_process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        return self.router_process

    def wait_for_server(self, url: str, timeout: int = 300) -> bool:
        """Wait for a server to become healthy."""
        health_url = f"{url}/health"
        start_time = time.time()

        while time.time() - start_time < timeout:
            try:
                response = requests.get(health_url, timeout=5)
                if response.status_code == 200:
                    print(f"  Server {url} is ready")
                    return True
            except requests.exceptions.RequestException:
                pass
            time.sleep(2)

        print(f"  Server {url} failed to start within {timeout}s")
        return False

    def start_all(self) -> bool:
        """Start all servers and router."""
        prefill_servers = []
        decode_servers = []

        # Start prefill servers
        print("\n=== Starting Prefill Servers ===")
        for i, gpu_id in enumerate(self.config.hardware.prefill_gpu_ids):
            port = self.config.server.prefill_port + i
            bootstrap_port = self.config.server.bootstrap_port + i
            server = self.start_prefill_server(gpu_id, port, bootstrap_port)
            prefill_servers.append((server.url, bootstrap_port))

        # Start decode servers
        print("\n=== Starting Decode Servers ===")
        for i, gpu_id in enumerate(self.config.hardware.decode_gpu_ids):
            port = self.config.server.decode_port + i
            server = self.start_decode_server(gpu_id, port)
            decode_servers.append(server.url)

        # Wait for all servers to be ready
        print("\n=== Waiting for Servers to Initialize ===")
        all_ready = True
        for server in self.processes:
            if not self.wait_for_server(server.url, self.config.server.server_launch_timeout):
                all_ready = False
                break

        if not all_ready:
            print("ERROR: Some servers failed to start")
            self.shutdown_all()
            return False

        # Start router
        print("\n=== Starting Router ===")
        self.start_router(prefill_servers, decode_servers)

        router_url = f"http://{self.config.server.base_host}:{self.config.server.router_port}"
        if not self.wait_for_server(router_url, 60):
            print("ERROR: Router failed to start")
            self.shutdown_all()
            return False

        print("\n=== All Servers Ready ===")
        print(f"Router URL: {router_url}")
        print(f"Prefill servers: {[s[0] for s in prefill_servers]}")
        print(f"Decode servers: {decode_servers}")

        return True

    def shutdown_all(self):
        """Shutdown all servers gracefully."""
        print("\n=== Shutting Down Servers ===")

        # Shutdown router first
        if self.router_process:
            print("Terminating router...")
            self.router_process.terminate()
            try:
                self.router_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.router_process.kill()

        # Shutdown all other servers
        for server in self.processes:
            print(f"Terminating {server.name}...")
            server.process.terminate()
            try:
                server.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                server.process.kill()

        self.processes.clear()
        self.router_process = None
        print("All servers shut down")

    def get_router_url(self) -> str:
        """Get the router URL for client connections."""
        return f"http://{self.config.server.base_host}:{self.config.server.router_port}"


def main():
    parser = argparse.ArgumentParser(
        description="Start P/D disaggregated servers for experiments"
    )
    parser.add_argument(
        "--model", type=str, default="meta-llama/Llama-3.1-8B-Instruct",
        help="Model path or HuggingFace model ID"
    )
    parser.add_argument(
        "--prefill-gpus", type=int, nargs="+", default=[0],
        help="GPU IDs for prefill servers"
    )
    parser.add_argument(
        "--decode-gpus", type=int, nargs="+", default=[1],
        help="GPU IDs for decode servers"
    )
    parser.add_argument(
        "--prefill-port", type=int, default=30100,
        help="Base port for prefill servers"
    )
    parser.add_argument(
        "--decode-port", type=int, default=30200,
        help="Base port for decode servers"
    )
    parser.add_argument(
        "--router-port", type=int, default=30000,
        help="Router port"
    )
    parser.add_argument(
        "--transfer-backend", type=str, default="mooncake",
        choices=["mooncake", "nixl", "fake"],
        help="KV cache transfer backend"
    )
    parser.add_argument(
        "--ib-device", type=str, default=None,
        help="InfiniBand device (e.g., mlx5_0)"
    )
    parser.add_argument(
        "--minimal", action="store_true",
        help="Use minimal configuration for quick testing"
    )
    parser.add_argument(
        "--keep-running", action="store_true",
        help="Keep servers running after startup (don't exit)"
    )

    args = parser.parse_args()

    # Create configuration
    if args.minimal:
        config = get_minimal_test_config()
    else:
        config = get_config_for_h100_4gpu()

    # Override with command line arguments
    config.model.model_path = args.model
    config.hardware.prefill_gpu_ids = args.prefill_gpus
    config.hardware.decode_gpu_ids = args.decode_gpus
    config.server.prefill_port = args.prefill_port
    config.server.decode_port = args.decode_port
    config.server.router_port = args.router_port

    if args.transfer_backend:
        from configs import TransferBackend
        config.server.transfer_backend = TransferBackend(args.transfer_backend)

    config.server.ib_device = args.ib_device

    # Create and start server manager
    manager = PDServerManager(config)

    # Handle signals for graceful shutdown
    def signal_handler(signum, frame):
        print("\nReceived shutdown signal...")
        manager.shutdown_all()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Start all servers
    if not manager.start_all():
        sys.exit(1)

    # Keep running or exit
    if args.keep_running:
        print("\nServers are running. Press Ctrl+C to stop.")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass
        finally:
            manager.shutdown_all()
    else:
        print("\nServers started successfully. Use --keep-running to keep them active.")
        print(f"Router URL: {manager.get_router_url()}")


if __name__ == "__main__":
    main()
