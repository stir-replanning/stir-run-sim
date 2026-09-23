#!/usr/bin/env bash
usage_text() {
  cat <<'USAGE'
STEP: check -- is everything the sim rollout needs present? Prints OK/MISS lines (like
`policy check`) and starts nothing.

  bash stack.sh check            # venv, env-chain imports, identical-copy hashes, gripper
                                 # asset + its meshes, LIBERO, sim data, `policy check`
  bash stack.sh check --make     # ...then BUILD + reset the envs: the __main__ self-tests of
                                 # sim/kinova_simple_env.py and sim/kinova_circle_env.py
                                 # (MUJOCO_GL=egl). Needs the gripper asset.

An import-only pass is not enough: the modules import fine without the Menagerie gripper
XML, and only building an env fails. Hence the separate asset lines.
USAGE
}
HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/../_env.sh"
MAKE=0
case "${1:-}" in
  --make) MAKE=1 ;;
  "") ;;
  -h | --help | help)
    usage_text
    exit 0
    ;;
  *)
    echo "unknown option: $1"
    exit 1
    ;;
esac
chk() {
  local label="$1"
  shift
  if test "$@"; then echo "OK   $label"; else
    echo "MISS $label"
    BAD=1
  fi
}
pyrun() { (cd "$SIM_DIR" && "$VENV_PY" -B -c "$1" </dev/null 2>/dev/null); }
BAD=0
echo "=== venv + env chain ($SIM_DIR) ==="
chk "libero_venv python $VENV_PY" -x "$VENV_PY"
chk "chain files match sim/env_chain.sha256" \
  -n "$(cd "$SIM_DIR" && sha256sum -c --quiet env_chain.sha256 >/dev/null 2>&1 && echo ok)"
if [ -z "$COLLECT_SIM_DIR" ]; then
  echo "--   COLLECT_SIM_DIR not set: no other env-chain copy compared (set it to a directory containing sim/)"
elif [ -d "$COLLECT_SIM_DIR/sim" ]; then
  CS_SIM="$COLLECT_SIM_DIR/sim"
  chk "$CS_SIM: same env_chain.sha256 + assets/README.md + assets/build_gripper.py, and its chain files match it" \
    -n "$(cmp -s "$SIM_DIR/env_chain.sha256" "$CS_SIM/env_chain.sha256" && cmp -s "$SIM_DIR/assets/README.md" "$CS_SIM/assets/README.md" && cmp -s "$SIM_DIR/assets/build_gripper.py" "$CS_SIM/assets/build_gripper.py" && (cd "$CS_SIM" && sha256sum -c --quiet env_chain.sha256 >/dev/null 2>&1) && echo ok)"
else
  echo "--   $COLLECT_SIM_DIR/sim not found (COLLECT_SIM_DIR): env-chain copy not compared"
fi
LIBERO_CFG="${LIBERO_CONFIG_PATH:-$HOME/.libero}/config.yaml"
chk "LIBERO config $LIBERO_CFG (else LIBERO's first import asks on stdin)" -f "$LIBERO_CFG"
chk "import kinova_env, kinova_grid_env, kinova_circle_env, kinova_simple_env" \
  -n "$(pyrun 'import kinova_env, kinova_grid_env, kinova_circle_env, kinova_simple_env; print("ok")')"
echo "=== assets ==="
GRIPPER=$(pyrun 'import kinova_env; print(kinova_env.menagerie_gripper_xml())')
chk "Menagerie 2F-85 gripper XML ${GRIPPER:-(not found: run sim/assets/build_gripper.py)}" -n "$GRIPPER"
if [ -n "$GRIPPER" ]; then
  MESH_MISS=$(pyrun "
import os, xml.etree.ElementTree as ET
xml = '$GRIPPER'; d = os.path.dirname(xml)
files = [e.get('file') for e in ET.parse(xml).iter() if e.get('file')]
print(' '.join(f for f in files if not os.path.isfile(os.path.join(d, f))) or 'none')")
  chk "gripper meshes next to it (missing: ${MESH_MISS:-?})" "${MESH_MISS:-?}" = none
fi
LIBERO_PKG=$(pyrun 'import importlib.util; print(importlib.util.find_spec("libero.libero").submodule_search_locations[0])')
chk "LIBERO package ${LIBERO_PKG:-(libero not importable)}" -n "$LIBERO_PKG"
for x in stable_hope_objects/milk/milk.xml stable_hope_objects/tomato_sauce/tomato_sauce.xml \
  stable_scanned_objects/basket/basket.xml; do
  chk "LIBERO asset $x" -f "$LIBERO_PKG/assets/$x"
done
echo "=== sim data ==="
chk "eval data $SIM_EVAL_HDF5" -f "$SIM_EVAL_HDF5"
if [ -f "$SIM_EVAL_HDF5" ]; then
  NDEMO=$("$VENV_PY" -c "
import h5py
with h5py.File('$SIM_EVAL_HDF5', 'r') as f:
    d = f['data']
    assert 'env_args' in d.attrs and 'states' in d['demo_0']
    print(len(d.keys()))" </dev/null 2>/dev/null)
  chk "$SIM_EVAL_HDF5 has data.attrs['env_args'] + data/demo_0/states (${NDEMO:-0} demos)" -n "$NDEMO"
fi
echo "=== seed client ==="
chk "openpi_client importable in $VENV_PY (not needed for --random-policy)" \
  -n "$("$VENV_PY" -c 'import openpi_client; print("ok")' </dev/null 2>/dev/null)"
echo "=== policy server ==="
POLICY_OUT=$(bash "$HERE/policy.sh" check) || BAD=1
echo "$POLICY_OUT" | sed '/^\[policy\]/d; /^$/d'
echo
if [ "$BAD" = 0 ]; then
  echo "[check] everything present."
else echo "[check] something is MISSING (see above). SETUP.md says where each piece comes from."; fi
if [ "$MAKE" = 1 ]; then
  if [ -z "$GRIPPER" ]; then
    echo "[check] --make skipped: no gripper XML, so no env can be built."
    exit 1
  fi
  for m in kinova_simple_env.py kinova_circle_env.py; do
    echo "=== build + reset: $m (MUJOCO_GL=$MUJOCO_GL) ==="
    PYTHONUNBUFFERED=1 "$VENV_PY" -B "$SIM_DIR/$m" </dev/null || BAD=1
  done
fi
exit "$BAD"
