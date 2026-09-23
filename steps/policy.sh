#!/usr/bin/env bash
usage_text() {
  cat <<'USAGE'
STEP: SIM policy server -- SWITCH the served sim model. One command per model: running
one stops the sim policy server on :$SIM_POLICY_PORT (only that one) and starts the
chosen model in the FOREGROUND (Ctrl-C stops it). Checkpoints live OUTSIDE this repo in
$SIM_MODELS_DIR (~/models/kinova_sim/<config>/29999). Serves on
ws://localhost:$SIM_POLICY_PORT (8001; openpi websocket protocol -> openpi_client in the
runner seed). Policy servers on other ports are never touched.

  bash stack.sh policy list              # model table: name, kind, checkpoint, config
  bash stack.sh policy pi05-sim          # pi0.5 full fine-tune on kinova/sim (openpi serve_policy.py)
  bash stack.sh policy pi0-sim           # pi0   full fine-tune on kinova/sim
  bash stack.sh policy <m> --print       # print the exact command, run nothing
  bash stack.sh policy <m> --task 0      # CONDITION: 0=red 1=yellow
  bash stack.sh policy <m> --prompt "..."  # or the raw prompt string
  bash stack.sh policy prompts           # the 2 sim training prompts
  bash stack.sh policy check             # every checkpoint + runtime present? (runs nothing)
  bash stack.sh policy stop              # stop the sim policy server (:$SIM_POLICY_PORT only)

openpi models need an openpi checkout at $OPENPI_DIR that DEFINES the training config
(pi0_kinova_sim / pi05_kinova_sim come from openpi/openpi_thor.patch, not upstream
openpi) plus `uv` on PATH. The server process gets
XLA_PYTHON_CLIENT_MEM_FRACTION=$SIM_XLA_MEM_FRACTION so it can share the GPU with
other processes. Override it with SIM_XLA_MEM_FRACTION: an exported
XLA_PYTHON_CLIENT_MEM_FRACTION is IGNORED here (a warning says so), so a shell set
up for another server cannot resize this one:
  SIM_XLA_MEM_FRACTION=0.3 bash stack.sh policy pi05-sim
This repo's kill.sh never stops it (a reload is slow);
`policy stop` stops only the :$SIM_POLICY_PORT server.

CONDITION (which cube to pick) = the LANGUAGE PROMPT. --task 0|1 expands to the exact
  string the model was trained on. --default_prompt makes it the server's default; the
  openpi client can also send `prompt` per observation, which OVERRIDES the default --
  the runner seed does exactly that (each demo's own `language` attribute), so the
  server default is optional for `stack.sh seed`.
USAGE
}
HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/../_env.sh"
SM="$SIM_MODELS_DIR"
DEFAULT_MODEL=pi05-sim
MODELS=(pi05-sim pi0-sim)
usage() {
  usage_text
  exit "${1:-1}"
}
TASK_IDS=(0 1)
task_prompt() {
  case "$(echo "$1" | tr '[:upper:]' '[:lower:]')" in
    0 | red | r)
      TASK_ID=0
      TASK_COLOR=red
      PROMPT="pick up the red cube and place it in the basket"
      ;;
    1 | yellow | y)
      TASK_ID=1
      TASK_COLOR=yellow
      PROMPT="pick up the yellow cube and place it in the basket"
      ;;
    *)
      echo "[policy] unknown task: $1 (0|1, red|yellow, r|y)" >&2
      return 1
      ;;
  esac
}
list_prompts() {
  echo "CONDITION for pi05-sim / pi0-sim = the language prompt (task_index / colour / instruction):"
  local t
  for t in "${TASK_IDS[@]}"; do
    task_prompt "$t"
    printf "  --task %s  (%-6s)  \"%s\"\n" "$t" "$TASK_COLOR" "$PROMPT"
  done
  echo
  echo "  These set the server's DEFAULT prompt (--default_prompt). The runner seed sends"
  echo "  each demo's own \`language\` per observation, which OVERRIDES it:"
  echo "    bash stack.sh seed --limit-per-task 16 --label pi05-sim"
}
model_info() {
  case "$1" in
    pi05-sim)
      KIND=openpi
      DIR="$SM/pi05_kinova_sim/29999"
      CFG=pi05_kinova_sim
      DESC="pi0.5 full fine-tune on kinova/sim (simple.hdf5: 644 demos, KinovaTwoObjSimple red/yellow cubes + guideline square), 30k steps"
      ;;
    pi0-sim)
      KIND=openpi
      DIR="$SM/pi0_kinova_sim/29999"
      CFG=pi0_kinova_sim
      DESC="pi0 full fine-tune, same data, 30k steps"
      ;;
    *) return 1 ;;
  esac
}
list_models() {
  printf "%-10s %-9s    %s\n" MODEL KIND CHECKPOINT
  local m mark
  for m in "${MODELS[@]}"; do
    model_info "$m"
    mark="  "
    [ -e "$DIR" ] || mark="!!"
    printf "%-10s %-9s %s %s\n" "$m" "$KIND" "$mark" "$DIR"
    printf "%-23s config=%s   %s\n" "" "$CFG" "$DESC"
  done
  echo
  echo "!! = checkpoint missing.   server: ws://localhost:$SIM_POLICY_PORT   openpi: $OPENPI_DIR"
  echo "default model: $DEFAULT_MODEL   (bash stack.sh policy <model> [--print] [--prompt \"...\"])"
}
chk() {
  local label="$1"
  shift
  if test "$@"; then echo "OK   $label"; else
    echo "MISS $label"
    BAD=1
  fi
}
check_openpi_runtime() {
  chk "uv on PATH" -n "$(command -v uv)"
  chk "openpi checkout $OPENPI_DIR/scripts/serve_policy.py" -f "$OPENPI_DIR/scripts/serve_policy.py"
  local cfgpy="$OPENPI_DIR/src/openpi/training/config.py"
  if [ -f "$cfgpy" ]; then
    chk "config '$CFG' defined in $cfgpy" -n "$(grep -F "\"$CFG\"" "$cfgpy" 2>/dev/null | head -1)"
  else
    echo "MISS config '$CFG' (no $cfgpy to look in)"
    BAD=1
  fi
}
check_openpi_ckpt() {
  chk "$1 params/      $DIR/params" -d "$DIR/params"
  local ns
  ns=$(find "$DIR/assets" -name norm_stats.json 2>/dev/null | head -1)
  chk "$1 norm_stats   ${ns:-$DIR/assets/**/norm_stats.json}" -n "$ns"
}
check_all() {
  BAD=0
  echo "=== checkpoints ($SIM_MODELS_DIR) ==="
  local m
  for m in "${MODELS[@]}"; do
    model_info "$m"
    check_openpi_ckpt "$m"
  done
  echo "=== openpi runtime (${MODELS[*]}) ==="
  local first=1
  for m in "${MODELS[@]}"; do
    model_info "$m"
    if [ "$first" = 1 ]; then
      check_openpi_runtime
      first=0
    else chk "config '$CFG' defined" -n "$(grep -F "\"$CFG\"" "$OPENPI_DIR/src/openpi/training/config.py" 2>/dev/null | head -1)"; fi
  done
  echo
  [ "$BAD" = 0 ] && echo "[policy] everything present." || echo "[policy] something is MISSING (see above); serving will refuse until fixed."
  return "$BAD"
}
stop_server() {
  local pat="serve_policy.py --port $SIM_POLICY_PORT( |$)"
  if pgrep -f "$pat" >/dev/null 2>&1; then
    echo "[policy] stopping the sim policy server on :$SIM_POLICY_PORT"
    pkill -f "$pat" 2>/dev/null
    sleep 1
    pkill -9 -f "$pat" 2>/dev/null
  fi
  local i
  for i in 1 2 3 4 5 6 7 8 9 10; do
    ss -ltn 2>/dev/null | grep -q ":$SIM_POLICY_PORT " || return 0
    sleep 1
  done
  echo "[policy] WARNING: something is still listening on :$SIM_POLICY_PORT" >&2
  ss -ltnp 2>/dev/null | grep ":$SIM_POLICY_PORT " >&2
  return 1
}
build_openpi_cmd() {
  CMD=(uv run scripts/serve_policy.py --port "$SIM_POLICY_PORT")
  [ -n "$PROMPT" ] && CMD+=(--default_prompt "$PROMPT")
  CMD+=(policy:checkpoint "--policy.config=$CFG" "--policy.dir=$DIR")
}
print_cmd() {
  [ "$KIND" = openpi ] && printf 'cd %q && XLA_PYTHON_CLIENT_MEM_FRACTION=%q ' "$OPENPI_DIR" "$SIM_XLA_MEM_FRACTION"
  printf '%q ' "${CMD[@]}"
  echo
}
PRINT=0
PROMPT=""
cmd="${1:-$DEFAULT_MODEL}"
[ $# -gt 0 ] && shift
while [ $# -gt 0 ]; do
  case "$1" in
    --print) PRINT=1 ;;
    --prompt)
      PROMPT="$2"
      shift
      ;;
    --task)
      RAW_TASK="$2"
      shift
      ;;
    -h | --help) usage 0 ;;
    *)
      echo "unknown option: $1"
      usage 1
      ;;
  esac
  shift
done
case "$cmd" in
  list)
    list_models
    exit 0
    ;;
  prompts)
    list_prompts
    exit 0
    ;;
  check)
    check_all
    exit $?
    ;;
  stop)
    stop_server
    exit $?
    ;;
  -h | --help | help) usage 0 ;;
esac
model_info "$cmd" || {
  echo "[policy] unknown model: $cmd"
  echo
  list_models
  exit 1
}
if [ -n "${RAW_TASK:-}" ]; then
  task_prompt "$RAW_TASK" || exit 1
fi
build_openpi_cmd
if [ -n "${XLA_PYTHON_CLIENT_MEM_FRACTION:-}" ] && [ "$XLA_PYTHON_CLIENT_MEM_FRACTION" != "$SIM_XLA_MEM_FRACTION" ]; then
  echo "[policy] WARNING: ignoring exported XLA_PYTHON_CLIENT_MEM_FRACTION=$XLA_PYTHON_CLIENT_MEM_FRACTION (set SIM_XLA_MEM_FRACTION instead);" \
    "this server gets SIM_XLA_MEM_FRACTION=$SIM_XLA_MEM_FRACTION" >&2
fi
if [ "$PRINT" = 1 ]; then
  print_cmd
  exit 0
fi
BAD=0
echo "[policy] $cmd: $DESC"
check_openpi_ckpt "$cmd"
check_openpi_runtime
if [ "$BAD" != 0 ]; then
  echo "[policy] refusing to start $cmd -- fix the MISS lines above. The command it would run:"
  print_cmd
  exit 1
fi
stop_server
echo "[policy] serving $cmd on ws://localhost:$SIM_POLICY_PORT  config=$CFG"
echo "[policy] condition: ${TASK_ID:+task=$TASK_ID:$TASK_COLOR  }default prompt=${PROMPT:-<none: the client must send one per observation>}"
print_cmd
cd "$OPENPI_DIR" && XLA_PYTHON_CLIENT_MEM_FRACTION="$SIM_XLA_MEM_FRACTION" exec "${CMD[@]}"
