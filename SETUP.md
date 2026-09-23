# Setup

This repo runs next to a venv, the LIBERO source tree and one openpi checkout:

```
~/workspace/
├── stir-run-sim/       this repo (env chain, launchers, runner seed)
├── libero_venv/        Python 3.12 venv for robosuite + the seed (VENV_PY default: a sibling of the repo)
└── LIBERO/             LIBERO source, editable-installed into libero_venv
~/openpi/               openpi + openpi_thor.patch, with its own uv venv: the policy server
~/models/kinova_sim/    sim checkpoints (SIM_MODELS_DIR)
~/kinova_dataset/data/  sim data, simple.hdf5 (SIM_DATA_DIR)
```

Tested platform: Jetson Thor (aarch64, JetPack 7.2.1, CUDA 13.2, Ubuntu 24.04,
Python 3.12). No ROS is needed.

`bash stack.sh check` tells you which of the steps below are still missing.

---

## 1. This repo

```bash
git clone <remote>/stir-run-sim.git ~/workspace/stir-run-sim
```

## 2. Python environment (env chain + seed)

The seed and the env self-tests run under a venv, by default `~/workspace/libero_venv`
(Python 3.12; override with `VENV_PY`). The venv can be shared with other projects.
LIBERO provides the object assets, and `kinova_grid_env` imports it at module load. It is
pinned at `8f1084e`, and the editable install needs an empty `libero/__init__.py`:

```bash
python3 -m venv ~/workspace/libero_venv
~/workspace/libero_venv/bin/pip install -r ~/workspace/stir-run-sim/requirements.txt
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git ~/workspace/LIBERO
git -C ~/workspace/LIBERO checkout 8f1084e
~/workspace/libero_venv/bin/pip install -e ~/workspace/LIBERO
touch ~/workspace/LIBERO/libero/__init__.py
```

LIBERO's `setup.py` declares no dependencies, so `pip install -e` pulls in nothing. The
modules `libero.libero.envs` imports at load time (bddl, future, gym, cloudpickle,
easydict, matplotlib, pyyaml) are listed in this repo's `requirements.txt`, which the
first `pip install` above already covered. Do **not** install LIBERO's own
`requirements.txt`: it pins an old training stack (numpy 1.22, torch, robomimic) that this
venv does not use.

LIBERO's first import asks for its paths on stdin unless `~/.libero/config.yaml` exists
(or `$LIBERO_CONFIG_PATH/config.yaml`). Under `nohup` or a closed stdin it hangs or fails
instead, so create the file up front:

```bash
mkdir -p ~/.libero && cat > ~/.libero/config.yaml <<EOF
assets: $HOME/workspace/LIBERO/libero/libero/./assets
bddl_files: $HOME/workspace/LIBERO/libero/libero/./bddl_files
benchmark_root: $HOME/workspace/LIBERO/libero/libero
datasets: $HOME/workspace/LIBERO/libero/libero/../datasets
init_states: $HOME/workspace/LIBERO/libero/libero/./init_files
EOF
```

Keep robosuite at 1.4.x: the chain calls `robosuite.load_controller_config`, which 1.5
removed.

## 3. Menagerie gripper asset (generate it)

```bash
~/workspace/libero_venv/bin/python ~/workspace/stir-run-sim/sim/assets/build_gripper.py
```

Every env in the chain builds with the Menagerie Robotiq 2F-85 gripper, and robosuite does
not ship it. The script downloads MuJoCo Menagerie's `robotiq_2f85` at a pinned commit
(network needed), checks the conversion, and writes the XML, the meshes and the
BSD-2-Clause LICENSE to `sim/assets/grippers/` (gitignored). See
[`sim/assets/README.md`](sim/assets/README.md). Alternatively, run it with `--out <dir>` and
export `KINOVA_SIM_ASSETS=<dir>`. Do **not** copy it into a shared venv's site-packages.

Check: `bash stack.sh check` then shows `OK Menagerie 2F-85 gripper XML` and
`OK gripper meshes`, and `bash stack.sh check --make` builds and resets the envs.

## 4. Sim checkpoints

The two servable sim checkpoints are `pi05_kinova_sim` and `pi0_kinova_sim` at step 29999.
An openpi training run stores them at `checkpoints/<config>/<config>/29999`. With the train
state each step dir is about 41 GB, but serving needs only `params/` and `assets/`:

```bash
for cfg in pi05_kinova_sim pi0_kinova_sim; do
  mkdir -p ~/models/kinova_sim/$cfg/29999
  rsync -a <source>/checkpoints/$cfg/$cfg/29999/params \
           <source>/checkpoints/$cfg/$cfg/29999/assets \
           ~/models/kinova_sim/$cfg/29999/
done
bash stack.sh policy check
```

`policy check` confirms that `params/` and `assets/kinova/sim/norm_stats.json` are present.
A different location works too: set `SIM_MODELS_DIR` (or `MODELS_DIR`).

## 5. Sim data

The seed restores each demo's simulator state from `simple.hdf5` (644 demos, with
`states (T, 57)`):

```bash
mkdir -p ~/kinova_dataset/data
rsync -a <source>/kinova_dataset/data/simple.hdf5 ~/kinova_dataset/data/
```

To use another location, set `SIM_DATA_DIR`, or point `SIM_EVAL_HDF5` at the file itself.

## 6. Optional: openpi-client for the seed

The seed talks to the policy server through `openpi_client`, which `requirements.txt` does
not install. (`--random-policy` does not need it.) To install it:

```bash
~/workspace/libero_venv/bin/pip install -e ~/openpi/packages/openpi-client
```

That also pulls `dm-tree`, `msgpack`, `pillow` and `websockets` into libero_venv, which
other projects may share. The alternative is to vendor a drop-in client
(`runner/README.md`, item 6).

## 7. Policy server (openpi)

The server runs from `~/openpi` in openpi's own uv venv, never in libero_venv. If
`~/openpi` is already built with this patch, reuse it and do not re-apply the patch. To
build it from scratch, see [`openpi/README.md`](openpi/README.md): base `15a9616`, then
`git apply openpi/openpi_thor.patch` and `uv sync`.
[`docs/policy_server.md`](docs/policy_server.md) explains why each change is needed, and
covers the memory cap for running beside another policy server. `uv` must be on PATH.

## 8. Run it

```bash
cd ~/workspace/stir-run-sim
bash stack.sh check
bash stack.sh policy pi05-sim
bash stack.sh seed --limit-per-task 16 --label pi05-sim --csv /tmp/pi05-sim.csv
```

Run `policy` in one terminal (server on :8001) and `seed` in another. `bash stack.sh`
prints the full command menu.

`MUJOCO_GL=egl` is exported by `_env.sh`. MuJoCo EGL rendering works on Jetson Thor with
the NVIDIA Tegra libEGL. On x86 images that ship only the NVIDIA EGL vendor driver, a
libglvnd loader on `LD_LIBRARY_PATH` was needed (`docs/sim_eval.md`, "Reference setup").
If EGL context creation fails, check that first. Next, check memory: the XLA
preallocation of any running policy server shares the same unified memory.
