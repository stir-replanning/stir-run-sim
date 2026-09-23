#!/usr/bin/env bash
HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/_env.sh"
for p in "${OWN_PROCS[@]}"; do pkill -f "$p" 2>/dev/null; done
sleep 2
for p in "${OWN_PROCS[@]}"; do pkill -9 -f "$p" 2>/dev/null; done
if pgrep -f "serve_policy.py --port $SIM_POLICY_PORT( |$)" >/dev/null 2>&1; then
  echo "[kill] the sim policy server on :$SIM_POLICY_PORT is still running (bash stack.sh policy stop)."
fi
echo "[kill] stir-run-sim processes stopped."
