#!/usr/bin/env bash
usage_text() {
  cat <<'USAGE'
STIR sim policy rollout -- the robosuite env chain + a sim policy server + the runner
seed. No ROS, no robot, no headset.

Each component runs in the FOREGROUND (logs straight to your terminal, Ctrl-C stops
just that one). Open a terminal per component and go in THIS ORDER:

  bash stack.sh check            # 1. env chain + gripper asset + data + checkpoints present?
  bash stack.sh policy pi05-sim  # 2. serve the sim checkpoint on ws://localhost:8001  [wait for the port]
  bash stack.sh seed --limit-per-task 16 --label pi05-sim   # 3. closed-loop rollout against it

Policy server (model switching; checkpoints in ~/models/kinova_sim -- see steps/policy.sh):
  bash stack.sh policy list      # models + checkpoint paths
  bash stack.sh policy pi0-sim   # serve the pi0 one instead (restarts the :8001 server)
  bash stack.sh policy pi05-sim --task 1   # CONDITION = the prompt for the yellow cube (0=red 1=yellow)
  bash stack.sh policy prompts   # the 2 exact training prompts
  bash stack.sh policy check     # checkpoints + runtime present?
  bash stack.sh policy stop      # stop it -- ONLY the :8001 server (stack.sh kill leaves it running)

Runner seed (runner/eval_kinova_sim.py: KinovaTwoObjSimple, restores each demo's state):
  bash stack.sh seed --random-policy --limit-per-task 2   # harness smoke test, no server
  bash stack.sh seed --limit-per-task 16 --label pi05-sim --csv /tmp/pi05-sim.csv
  bash stack.sh seed --limit-per-task 16 --swap-prompt    # language-grounding check
  bash stack.sh seed --help                               # every seed flag

Env checks:
  bash stack.sh check --make     # ...and BUILD + reset the envs (their __main__ self-tests)

One-shots:
  bash stack.sh status           # what's running right now
  bash stack.sh kill             # stop THIS repo's runner / env processes (never a policy server)
USAGE
}
HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/_env.sh"
usage() {
  usage_text
  exit "${1:-1}"
}
status() {
  local pats=(
    "runner=${OWN_PROCS[0]}"
    "env_selftest=${OWN_PROCS[1]}"
    "sim_policy=serve_policy.py --port $SIM_POLICY_PORT( |$)")
  for p in "${pats[@]}"; do
    local name="${p%%=*}" pat="${p#*=}" pid
    pid=$(pgrep -af "$pat" | grep -v "stack.sh" | awk '{print $1}' | head -1)
    if [ -n "$pid" ]; then
      printf "  %-15s UP (pid %s)\n" "$name" "$pid"
    else printf "  %-15s down\n" "$name"; fi
  done
  if ss -ltn 2>/dev/null | grep -q ":$SIM_POLICY_PORT "; then
    echo "[status] ws://localhost:$SIM_POLICY_PORT is listening (bash stack.sh policy list)"
  else
    echo "[status] nothing listening on :$SIM_POLICY_PORT (bash stack.sh policy pi05-sim)"
  fi
  local all other
  all=$(pgrep -af "serve_policy.py")
  other=$(echo "$all" | grep -vE "serve_policy.py --port $SIM_POLICY_PORT( |$)" | grep -o "serve_policy[^ ]* --port [0-9]*" | sort -u)
  [ -n "$other" ] && echo "[status] other policy servers running (not ours; they share the GPU): $(echo "$other" | awk '{print ":"$3}' | tr '\n' ' ')"
  return 0
}
cmd="${1:-}"
[ $# -gt 0 ] && shift
case "$cmd" in
  policy) exec bash "$HERE/steps/policy.sh" "$@" ;;
  check) exec bash "$HERE/steps/check.sh" "$@" ;;
  seed) exec bash "$HERE/steps/seed.sh" "$@" ;;
  status) status ;;
  kill) exec bash "$HERE/kill.sh" "$@" ;;
  -h | --help | help | "") usage 0 ;;
  *)
    echo "unknown component: $cmd"
    usage 1
    ;;
esac
