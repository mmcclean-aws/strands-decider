#!/usr/bin/env bash
# One `strands-decider serve --device neuron` per logical NeuronCore, on loopback ports.
#
#   deploy/trainium2/scripts/serve_replicas.sh CHECKPOINT [REPLICAS] [FIRST_CORE] [FIRST_PORT]
#
# Replica i runs on logical core FIRST_CORE + i and listens on 127.0.0.1:(FIRST_PORT + i). The
# first replica starts alone and compiles every graph into the Neuron SDK's cache; the others
# start once it is healthy and load the compiled graphs from the cache. Logs go to
# $LOG_DIR (default ./replica-logs). Stop them with: kill $(cat $LOG_DIR/pids)
set -euo pipefail

CKPT=${1:?usage: serve_replicas.sh CHECKPOINT [REPLICAS] [FIRST_CORE] [FIRST_PORT]}
N=${2:-64}
CORE0=${3:-0}
PORT0=${4:-8100}
LOG_DIR=${LOG_DIR:-./replica-logs}
# One cache for every replica, so that only the first one compiles.
export NEURON_LIBTORCH_CACHE_ROOT=${NEURON_LIBTORCH_CACHE_ROOT:-$HOME/.cache/neuron_libtorch}
mkdir -p "$LOG_DIR"
: > "$LOG_DIR/pids"

start() {
  local i=$1 core=$((CORE0 + $1)) port=$((PORT0 + $1))
  NEURON_RT_VISIBLE_CORES=$core \
    strands-decider serve "$CKPT" --device neuron --host 127.0.0.1 --port "$port" \
    > "$LOG_DIR/replica-$i.log" 2>&1 &
  echo $! >> "$LOG_DIR/pids"
}

wait_healthy() {
  local port=$((PORT0 + $1))
  until curl -sf "http://127.0.0.1:$port/health" > /dev/null; do
    if ! kill -0 "$(sed -n "$(($1 + 1))p" "$LOG_DIR/pids")" 2> /dev/null; then
      echo "replica $1 exited; see $LOG_DIR/replica-$1.log" >&2
      exit 1
    fi
    sleep 5
  done
}

start 0
wait_healthy 0
echo "replica 0 healthy on 127.0.0.1:$PORT0; starting $((N - 1)) more"
for ((i = 1; i < N; i++)); do start "$i"; done
for ((i = 1; i < N; i++)); do wait_healthy "$i"; done
echo "$N replicas healthy on 127.0.0.1:$PORT0-$((PORT0 + N - 1))"
