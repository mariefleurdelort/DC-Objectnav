#!/usr/bin/env bash
# Launches all 4 model servers, each in its own conda env and on its own port
# (see orchestrator/ports.yaml), as background processes logging to ../logs/.
#
# Usage:
#   ./orchestrator/launch_servers.sh start
#   ./orchestrator/launch_servers.sh stop
#   ./orchestrator/launch_servers.sh status
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

mkdir -p logs
PIDFILE_DIR="logs/pids"
mkdir -p "$PIDFILE_DIR"

# name:conda_env:dir:port  (mirrors orchestrator/ports.yaml — kept as plain bash
# data here so this script has no yaml-parsing dependency of its own)
SERVERS=(
  "yoloworld:vps-yoloworld:model_servers/yoloworld:8001"
  "groundingdino:vps-groundingdino:model_servers/groundingdino:8002"
  "owlv2:vps-owlv2:model_servers/owlv2:8003"
  "sam3:vps-sam3:model_servers/sam3:8004"
)

CONDA_BASE=$(conda info --base)

start_one() {
  local name=$1 env=$2 dir=$3 port=$4
  local pidfile="$PIDFILE_DIR/${name}.pid"
  if [ -f "$pidfile" ] && kill -0 "$(cat "$pidfile")" 2>/dev/null; then
    echo "$name already running (pid $(cat "$pidfile"))"
    return
  fi
  echo "starting $name on port $port (env $env)"
  (
    source "$CONDA_BASE/etc/profile.d/conda.sh"
    conda activate "$env"
    cd "$dir"
    exec uvicorn server:app --host 0.0.0.0 --port "$port" \
      >> "../../logs/${name}.log" 2>&1
  ) &
  echo $! > "$pidfile"
}

stop_one() {
  local name=$1
  local pidfile="$PIDFILE_DIR/${name}.pid"
  if [ -f "$pidfile" ] && kill -0 "$(cat "$pidfile")" 2>/dev/null; then
    echo "stopping $name (pid $(cat "$pidfile"))"
    kill "$(cat "$pidfile")"
  fi
  rm -f "$pidfile"
}

status_one() {
  local name=$1 port=$2
  local pidfile="$PIDFILE_DIR/${name}.pid"
  if [ -f "$pidfile" ] && kill -0 "$(cat "$pidfile")" 2>/dev/null; then
    echo "$name: running (pid $(cat "$pidfile"), port $port)"
  else
    echo "$name: stopped"
  fi
}

cmd="${1:-status}"
for entry in "${SERVERS[@]}"; do
  IFS=":" read -r name env dir port <<< "$entry"
  case "$cmd" in
    start)  start_one "$name" "$env" "$dir" "$port" ;;
    stop)   stop_one "$name" ;;
    status) status_one "$name" "$port" ;;
    *) echo "Usage: $0 [start|stop|status]" >&2; exit 1 ;;
  esac
done
