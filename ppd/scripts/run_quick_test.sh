#!/bin/bash
# Quick test script for P/D disaggregation setup
#
# Usage:
#   ./run_quick_test.sh [model_path]
#
# Example:
#   ./run_quick_test.sh meta-llama/Llama-3.1-8B-Instruct
#   ./run_quick_test.sh /path/to/local/model

set -e

# Configuration
MODEL_PATH="${1:-meta-llama/Llama-3.1-8B-Instruct}"
PREFILL_GPU=0
DECODE_GPU=1
PREFILL_PORT=30100
DECODE_PORT=30200
ROUTER_PORT=30000
BOOTSTRAP_PORT=9000

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PPD_DIR="$(dirname "$SCRIPT_DIR")"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}P/D Disaggregation Quick Test${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "Configuration:"
echo "  Model: $MODEL_PATH"
echo "  Prefill GPU: $PREFILL_GPU (port $PREFILL_PORT)"
echo "  Decode GPU: $DECODE_GPU (port $DECODE_PORT)"
echo "  Router port: $ROUTER_PORT"
echo ""

# Check if GPUs are available
echo -e "${YELLOW}Checking GPU availability...${NC}"
if command -v nvidia-smi &> /dev/null; then
    GPU_COUNT=$(nvidia-smi --query-gpu=count --format=csv,noheader | head -1)
    echo "  Found $GPU_COUNT GPU(s)"

    if [ "$GPU_COUNT" -lt 2 ]; then
        echo -e "${RED}ERROR: At least 2 GPUs required for P/D disaggregation${NC}"
        echo "  You can modify this script to use a single GPU for testing"
        exit 1
    fi
else
    echo -e "${RED}ERROR: nvidia-smi not found${NC}"
    exit 1
fi

# Function to cleanup processes
cleanup() {
    echo -e "\n${YELLOW}Cleaning up processes...${NC}"
    pkill -f "sglang.launch_server.*--disaggregation-mode" 2>/dev/null || true
    pkill -f "sglang_router.launch_router" 2>/dev/null || true
    sleep 2
}

# Set trap to cleanup on exit
trap cleanup EXIT

# Clean up any existing processes
cleanup

# Start Prefill Server
echo -e "\n${GREEN}Starting Prefill Server on GPU $PREFILL_GPU...${NC}"
CUDA_VISIBLE_DEVICES=$PREFILL_GPU python3 -m sglang.launch_server \
    --model-path "$MODEL_PATH" \
    --host 127.0.0.1 \
    --port $PREFILL_PORT \
    --disaggregation-mode prefill \
    --disaggregation-transfer-backend mooncake \
    --disaggregation-bootstrap-port $BOOTSTRAP_PORT \
    --tp 1 \
    --trust-remote-code \
    --mem-fraction-static 0.85 \
    > /tmp/prefill_server.log 2>&1 &
PREFILL_PID=$!
echo "  PID: $PREFILL_PID"
echo "  Log: /tmp/prefill_server.log"

# Start Decode Server
echo -e "\n${GREEN}Starting Decode Server on GPU $DECODE_GPU...${NC}"
CUDA_VISIBLE_DEVICES=$DECODE_GPU python3 -m sglang.launch_server \
    --model-path "$MODEL_PATH" \
    --host 127.0.0.1 \
    --port $DECODE_PORT \
    --disaggregation-mode decode \
    --disaggregation-transfer-backend mooncake \
    --tp 1 \
    --base-gpu-id $DECODE_GPU \
    --trust-remote-code \
    --mem-fraction-static 0.85 \
    > /tmp/decode_server.log 2>&1 &
DECODE_PID=$!
echo "  PID: $DECODE_PID"
echo "  Log: /tmp/decode_server.log"

# Wait for servers to initialize
echo -e "\n${YELLOW}Waiting for servers to initialize (this may take a few minutes)...${NC}"

wait_for_server() {
    local url=$1
    local name=$2
    local timeout=300  # 5 minutes
    local start_time=$(date +%s)

    while true; do
        if curl -s -f "$url/health" > /dev/null 2>&1; then
            echo -e "  ${GREEN}[OK]${NC} $name is ready"
            return 0
        fi

        local current_time=$(date +%s)
        local elapsed=$((current_time - start_time))

        if [ $elapsed -ge $timeout ]; then
            echo -e "  ${RED}[FAIL]${NC} $name failed to start within ${timeout}s"
            return 1
        fi

        # Check if process is still running
        if ! ps -p $3 > /dev/null 2>&1; then
            echo -e "  ${RED}[FAIL]${NC} $name process died"
            echo "  Check log at $4"
            return 1
        fi

        echo -n "."
        sleep 5
    done
}

echo -n "  Prefill server"
if ! wait_for_server "http://127.0.0.1:$PREFILL_PORT" "Prefill server" $PREFILL_PID "/tmp/prefill_server.log"; then
    echo -e "\n${RED}Prefill server failed to start. Last 50 lines of log:${NC}"
    tail -50 /tmp/prefill_server.log
    exit 1
fi

echo -n "  Decode server"
if ! wait_for_server "http://127.0.0.1:$DECODE_PORT" "Decode server" $DECODE_PID "/tmp/decode_server.log"; then
    echo -e "\n${RED}Decode server failed to start. Last 50 lines of log:${NC}"
    tail -50 /tmp/decode_server.log
    exit 1
fi

# Start Router
echo -e "\n${GREEN}Starting Router...${NC}"
python3 -m sglang_router.launch_router \
    --pd-disaggregation \
    --mini-lb \
    --host 127.0.0.1 \
    --port $ROUTER_PORT \
    --prefill "http://127.0.0.1:$PREFILL_PORT" $BOOTSTRAP_PORT \
    --decode "http://127.0.0.1:$DECODE_PORT" \
    > /tmp/router.log 2>&1 &
ROUTER_PID=$!
echo "  PID: $ROUTER_PID"
echo "  Log: /tmp/router.log"

# Wait for router
sleep 5
echo -n "  Router"
if ! wait_for_server "http://127.0.0.1:$ROUTER_PORT" "Router" $ROUTER_PID "/tmp/router.log"; then
    echo -e "\n${RED}Router failed to start. Last 20 lines of log:${NC}"
    tail -20 /tmp/router.log
    exit 1
fi

echo -e "\n${GREEN}========================================${NC}"
echo -e "${GREEN}All servers are running!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "Server URLs:"
echo "  Prefill: http://127.0.0.1:$PREFILL_PORT"
echo "  Decode:  http://127.0.0.1:$DECODE_PORT"
echo "  Router:  http://127.0.0.1:$ROUTER_PORT"
echo ""

# Run validation
echo -e "${YELLOW}Running validation tests...${NC}"
python3 "$SCRIPT_DIR/validate_setup.py" \
    --router-url "http://127.0.0.1:$ROUTER_PORT" \
    --prefill-url "http://127.0.0.1:$PREFILL_PORT" \
    --decode-url "http://127.0.0.1:$DECODE_PORT"

VALIDATION_RESULT=$?

if [ $VALIDATION_RESULT -eq 0 ]; then
    echo -e "\n${GREEN}========================================${NC}"
    echo -e "${GREEN}Quick test completed successfully!${NC}"
    echo -e "${GREEN}========================================${NC}"
    echo ""
    echo "You can now run experiments with:"
    echo "  python3 $PPD_DIR/experiments/multi_turn_append_test.py \\"
    echo "    --router-url http://127.0.0.1:$ROUTER_PORT \\"
    echo "    --model $MODEL_PATH \\"
    echo "    --minimal"
    echo ""
    echo "Servers will continue running. Press Ctrl+C to stop."

    # Keep script running
    wait
else
    echo -e "\n${RED}Validation failed!${NC}"
    exit 1
fi
