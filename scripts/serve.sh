#!/usr/bin/env bash
# Helper: (re)start llmon serve detached, tracking PID in a file.
# Usage: ./scripts/serve.sh [port]
PORT="${1:-11435}"
PIDFILE=/tmp/llmon-serve.pid
if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
  echo "already running (pid $(cat "$PIDFILE"))"
  exit 0
fi
export PATH="$HOME/.local/bin:$PATH"
nohup env -u LD_LIBRARY_PATH llmon serve --port "$PORT" >/tmp/llmon-serve.log 2>&1 &
echo $! > "$PIDFILE"
sleep 2
curl -s "localhost:$PORT/health" && echo && echo "serving (pid $(cat "$PIDFILE"))"
