"""
P-PD (Prefill-Append Prefill-Decode) Experiment Framework

This package provides tools for studying the trade-off between treating
append-prefill as prefill-like vs decode-like operations in multi-turn
LLM inference conversations.

Key Components:
- configs/: Configuration classes for experiments
- scripts/: Server startup and validation scripts
- experiments/: Experiment implementations
- results/: Output directory for experiment results
- utils/: Utility functions

Core Hypothesis:
In multi-turn conversations, the "append-prefill" phase (processing new user
input that appends to existing context) has varying characteristics:
- High cache rate (small append, large context): More decode-like
- Low cache rate (large append, small context): More prefill-like

This framework enables systematic measurement of these characteristics
to inform intelligent routing decisions in P/D disaggregated deployments.
"""

__version__ = "0.1.0"
