#!/usr/bin/env bash
HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/../_env.sh"
RANDOM_POLICY=0
HELP=0
for a in "$@"; do
  case "$a" in
    --random-policy) RANDOM_POLICY=1 ;;
    -h | --help) HELP=1 ;;
  esac
done
if [ "$HELP" = 0 ]; then
  [ -f "$SIM_EVAL_HDF5" ] || echo "[seed] WARNING: no sim data at $SIM_EVAL_HDF5 (SIM_EVAL_HDF5 / SIM_DATA_DIR; SETUP.md)" >&2
  (cd "$SIM_DIR" && "$VENV_PY" -B -c "import kinova_env; kinova_env.menagerie_gripper_xml()" </dev/null >/dev/null 2>&1) || echo "[seed] WARNING: Menagerie gripper XML not found -- every env build will fail (generate it: sim/assets/build_gripper.py)" >&2
  if [ "$RANDOM_POLICY" = 0 ]; then
    "$VENV_PY" -c "import openpi_client" </dev/null >/dev/null 2>&1 || echo "[seed] WARNING: openpi_client is not importable in $VENV_PY -- only --random-policy works (SETUP.md)" >&2
  fi
fi
echo "[seed] env=$SIM_DIR  data=$SIM_EVAL_HDF5  server=ws://$SIM_POLICY_HOST:$SIM_POLICY_PORT  MUJOCO_GL=$MUJOCO_GL"
exec "$VENV_PY" "$REPO/runner/eval_kinova_sim.py" \
  --env-dir "$SIM_DIR" --hdf5 "$SIM_EVAL_HDF5" \
  --host "$SIM_POLICY_HOST" --port "$SIM_POLICY_PORT" "$@"
