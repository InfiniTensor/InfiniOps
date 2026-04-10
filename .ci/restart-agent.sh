#!/bin/bash
# Usage: bash .ci/restart-agent.sh [port] [webhook-secret]
#
# Restart the CI agent with proxy configured.
# Edit the HTTPS_PROXY line below for your environment, then:
#   bash .ci/restart-agent.sh
#   bash .ci/restart-agent.sh 8080 my-webhook-secret

set -euo pipefail

PORT="${1:-8080}"
WEBHOOK_SECRET="${2:-}"

# --- Proxy config (edit this) ---
export HTTPS_PROXY="http://your-proxy:port"
export HTTP_PROXY="$HTTPS_PROXY"
export NO_PROXY="localhost,127.0.0.1"
export https_proxy="$HTTPS_PROXY"
export http_proxy="$HTTP_PROXY"
export no_proxy="$NO_PROXY"

# --- Kill existing agent ---
if pgrep -f "agent.py serve" > /dev/null 2>&1; then
    echo "Stopping existing agent..."
    pkill -f "agent.py serve" || true
    sleep 2
fi

# --- Start agent ---
CI_DIR="$(cd "$(dirname "$0")" && pwd)"

if [ ! -f "$CI_DIR/agent.py" ]; then
    echo "error: $CI_DIR/agent.py not found"
    exit 1
fi

ARGS="serve --port $PORT"
if [ -n "$WEBHOOK_SECRET" ]; then
    ARGS="$ARGS --webhook-secret $WEBHOOK_SECRET"
fi

echo "Starting CI agent on port $PORT..."
nohup python "$CI_DIR/agent.py" $ARGS > /tmp/ci-agent.log 2>&1 &

HOST_IP=$(hostname -I 2>/dev/null | awk '{print $1}' || hostname)

echo "PID:    $!"
echo "Listen: http://${HOST_IP}:${PORT}"
echo "Log:    /tmp/ci-agent.log"
echo "Proxy:  $HTTPS_PROXY"
