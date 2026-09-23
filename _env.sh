#!/usr/bin/env bash
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SIM_DIR="$REPO/sim"
VENV_PY=${VENV_PY:-"$(dirname "$REPO")/libero_venv/bin/python"}
COLLECT_SIM_DIR=${COLLECT_SIM_DIR:-}
OWN_PROCS=(
  "python[0-9.]*( -[^ ]+)* $REPO/runner/"
  "python[0-9.]*( -[^ ]+)* $SIM_DIR/")
MODELS_DIR=${MODELS_DIR:-$HOME/models}
SIM_MODELS_DIR=${SIM_MODELS_DIR:-$MODELS_DIR/kinova_sim}
OPENPI_DIR=${OPENPI_DIR:-$HOME/openpi}
SIM_POLICY_PORT=${SIM_POLICY_PORT:-8001}
SIM_POLICY_HOST=${SIM_POLICY_HOST:-127.0.0.1}
SIM_XLA_MEM_FRACTION=${SIM_XLA_MEM_FRACTION:-0.4}
SIM_DATA_DIR=${SIM_DATA_DIR:-$HOME/kinova_dataset}
SIM_EVAL_HDF5=${SIM_EVAL_HDF5:-$SIM_DATA_DIR/data/simple.hdf5}
export MUJOCO_GL=${MUJOCO_GL:-egl}
if [ -n "${KINOVA_SIM_ASSETS:-}" ]; then export KINOVA_SIM_ASSETS; fi
