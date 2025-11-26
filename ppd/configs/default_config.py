"""
Default configuration for P-PD (Prefill-Append Prefill-Decode) experiments.

This configuration defines the experimental setup for studying the trade-off
between treating append-prefill as prefill-like vs decode-like operations
in multi-turn conversations.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from enum import Enum


class TransferBackend(Enum):
    """KV cache transfer backend options."""
    MOONCAKE = "mooncake"
    NIXL = "nixl"
    FAKE = "fake"  # For testing without actual transfer


@dataclass
class HardwareConfig:
    """Hardware configuration for experiments."""
    # GPU configuration
    prefill_gpu_ids: List[int] = field(default_factory=lambda: [0])
    decode_gpu_ids: List[int] = field(default_factory=lambda: [1])

    # Tensor parallelism
    prefill_tp: int = 1
    decode_tp: int = 1

    # Memory and compute characteristics (will be profiled)
    # H100 specs as baseline
    gpu_memory_bandwidth_gbps: float = 3350.0  # H100 HBM3
    gpu_compute_tflops: float = 989.0  # H100 FP16 Tensor Core

    # Network configuration
    inter_gpu_bandwidth_gbps: float = 900.0  # NVLink 4.0
    network_bandwidth_gbps: float = 400.0  # For cross-node (if applicable)


@dataclass
class ModelConfig:
    """Model configuration."""
    model_path: str = "meta-llama/Llama-3.1-8B-Instruct"

    # Model architecture (will be auto-detected, these are defaults for Llama-3.1-8B)
    num_layers: int = 32
    num_heads: int = 32
    head_dim: int = 128
    hidden_size: int = 4096

    # KV cache configuration
    dtype: str = "float16"  # float16, bfloat16, float8
    kv_cache_dtype: str = "auto"  # auto will use model dtype

    # Context length
    max_context_length: int = 8192

    @property
    def kv_size_per_token_bytes(self) -> int:
        """Calculate KV cache size per token in bytes."""
        # 2 for K and V, dtype size
        dtype_size = 2 if self.dtype in ["float16", "bfloat16"] else 1
        return 2 * self.num_layers * self.num_heads * self.head_dim * dtype_size


@dataclass
class ServerConfig:
    """Server configuration for P/D disaggregation."""
    # Base networking
    base_host: str = "127.0.0.1"
    prefill_port: int = 30100
    decode_port: int = 30200
    router_port: int = 30000
    bootstrap_port: int = 9000

    # Transfer backend
    transfer_backend: TransferBackend = TransferBackend.MOONCAKE
    ib_device: Optional[str] = None  # e.g., "mlx5_0" for InfiniBand

    # Timeouts
    server_launch_timeout: int = 300  # seconds
    request_timeout: int = 120  # seconds

    # Batch configuration
    max_batch_size: int = 256
    max_prefill_tokens: int = 16384


@dataclass
class ExperimentConfig:
    """Configuration for append-prefill experiments."""

    # Append prompt length sweep (m in the formulation)
    append_lengths: List[int] = field(default_factory=lambda: [
        1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048
    ])

    # Context length sweep (L in the formulation)
    context_lengths: List[int] = field(default_factory=lambda: [
        512, 1024, 2048, 4096, 8192, 16384
    ])

    # Number of conversation turns to simulate
    num_turns: int = 5

    # Repetitions for statistical significance
    num_repetitions: int = 3

    # Warmup iterations
    warmup_iterations: int = 2

    # Metrics to collect
    collect_ttft: bool = True  # Time to first token
    collect_tpot: bool = True  # Time per output token
    collect_e2e_latency: bool = True  # End-to-end latency
    collect_throughput: bool = True
    collect_bandwidth_usage: bool = True
    collect_gpu_utilization: bool = True

    # Output configuration
    output_dir: str = "results"
    experiment_name: str = "append_prefill_tradeoff"


@dataclass
class AppendPrefillTestCase:
    """A single test case for append-prefill analysis."""
    # Context from previous turns
    context_length: int  # L: total tokens from history

    # New user input
    append_length: int  # m: new prompt tokens

    # Expected output (for testing)
    expected_output_length: int = 128

    @property
    def cache_rate(self) -> float:
        """Calculate cache hit rate r = L / (L + m)."""
        return self.context_length / (self.context_length + self.append_length)

    @property
    def total_input_length(self) -> int:
        """Total input length after append."""
        return self.context_length + self.append_length

    def is_prefill_like(self, threshold: float = 0.5) -> bool:
        """
        Heuristic: if cache_rate < threshold, treat as prefill-like.
        Low cache rate means more new tokens to process.
        """
        return self.cache_rate < threshold

    def estimated_kv_transfer_bytes(self, kv_size_per_token: int) -> int:
        """Estimate bytes needed to transfer KV cache."""
        return self.context_length * kv_size_per_token


@dataclass
class FullConfig:
    """Complete configuration combining all components."""
    hardware: HardwareConfig = field(default_factory=HardwareConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    server: ServerConfig = field(default_factory=ServerConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)

    def generate_test_cases(self) -> List[AppendPrefillTestCase]:
        """Generate all test cases from configuration."""
        test_cases = []
        for L in self.experiment.context_lengths:
            for m in self.experiment.append_lengths:
                if L + m <= self.model.max_context_length:
                    test_cases.append(AppendPrefillTestCase(
                        context_length=L,
                        append_length=m
                    ))
        return test_cases

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for serialization."""
        import dataclasses
        return {
            "hardware": dataclasses.asdict(self.hardware),
            "model": dataclasses.asdict(self.model),
            "server": dataclasses.asdict(self.server),
            "experiment": dataclasses.asdict(self.experiment),
        }


# Default configuration instance
DEFAULT_CONFIG = FullConfig()


def get_config_for_h100_4gpu() -> FullConfig:
    """Get configuration optimized for 4x H100 setup."""
    config = FullConfig()

    # Use 1 GPU for prefill, 1 for decode (conservative start)
    config.hardware.prefill_gpu_ids = [0]
    config.hardware.decode_gpu_ids = [1]
    config.hardware.prefill_tp = 1
    config.hardware.decode_tp = 1

    # H100 specs
    config.hardware.gpu_memory_bandwidth_gbps = 3350.0
    config.hardware.gpu_compute_tflops = 989.0
    config.hardware.inter_gpu_bandwidth_gbps = 900.0  # NVLink

    return config


def get_minimal_test_config() -> FullConfig:
    """Get minimal configuration for quick validation tests."""
    config = FullConfig()

    # Minimal sweep for quick testing
    config.experiment.append_lengths = [8, 64, 512]
    config.experiment.context_lengths = [512, 2048]
    config.experiment.num_turns = 2
    config.experiment.num_repetitions = 1
    config.experiment.warmup_iterations = 1

    return config
