# sim/: the MuJoCo env chain (robosuite 1.4)

The envs the sim policy rolls out in. The orchestration lives at the repo root
(`stack.sh`, `_env.sh`, `steps/`). There is no ROS and no server here.

**Env chain, pinned by hash.** Keep any other copy byte-identical:
- `kinova_circle_env.py` -> `kinova_grid_env.py` -> `kinova_env.py`: the chain.
  `KinovaTwoObjCircle` is the env the circle data is recorded in.
- `circle_schedule.py`, `env_config.py`, `env_configs/`: the episode schedule and the
  scene configs of the circle data. The seed does not import them. They are here so a
  circle-env runner can rebuild a collection scene.
- `assets/README.md`, `assets/build_gripper.py`: the generator of the Menagerie Robotiq
  2F-85 gripper XML + meshes (`assets/grippers/`, gitignored), and where
  `kinova_env.menagerie_gripper_xml()` looks for them. They are not in `env_chain.sha256`;
  keep both identical in every copy (`cmp`).
- `env_chain.sha256`: the pinned hashes of the `.py`/`.json` files above.
  `cd sim && sha256sum -c env_chain.sha256`, or `bash stack.sh check`. When
  `$COLLECT_SIM_DIR` is set to a directory containing another copy of `sim/` (unset by
  default), check also requires that copy to have the same `env_chain.sha256`,
  `assets/README.md` and `assets/build_gripper.py`, and its chain files to match those
  hashes.

**This repo only:**
- `kinova_simple_env.py`: `KinovaTwoObjSimple`, two 4 cm cubes and a flat guideline
  square, with no basket. It is the env of `simple.hdf5`, which is the training data of
  `pi0_kinova_sim` / `pi05_kinova_sim`, and the env the runner seed rolls out in. It is not
  used for collection. It imports the chain with flat imports, so it must stay in this
  directory. It is not covered by `env_chain.sha256`.
- `README.md`: this file.

**Assets the chain loads at env build (not at import):**
- The Menagerie gripper, via `$KINOVA_SIM_ASSETS/grippers/`, then `assets/grippers/`,
  then robosuite's own asset tree (`assets/README.md`).
- The LIBERO objects (`milk`, `tomato_sauce`, the `basket`), from the LIBERO package
  (`~/workspace/LIBERO/libero/libero/assets/`). `kinova_grid_env` imports
  `libero.libero.envs.objects` at module load, so LIBERO must be installed and
  `~/.libero/config.yaml` must exist even for the cube-only envs.

Each module has a `__main__` self-test that builds, resets and checks an env:
`MUJOCO_GL=egl ~/workspace/libero_venv/bin/python ~/workspace/stir-run-sim/sim/kinova_simple_env.py`
(`bash stack.sh check --make` runs the simple and circle ones). Give the absolute path:
`stack.sh kill` and `status` match only launches under the absolute `$REPO/sim/`.

See the top-level [README.md](../README.md).
