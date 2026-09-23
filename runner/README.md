# runner/: the sim rollout client

`eval_kinova_sim.py` is the **seed** of this repo's runner: the script that produced the
sim results in [`../docs/sim_eval.md`](../docs/sim_eval.md). Its defaults: `--port` is
`$SIM_POLICY_PORT` (8001), `--hdf5` is `$SIM_DATA_DIR/data/simple.hdf5`, and `--env-dir`
is `$SIM_DATA_DIR/simple_env`. Run it through `bash stack.sh seed ...`, which always passes
`--env-dir` = this repo's `../sim`.

What it does, per episode:

1. Restore a recorded demo's `states[0]` into `KinovaTwoObjSimple`.
2. Send `observation/image` (agentview), `observation/wrist_image` (eye-in-hand), and
   `observation/state` (8: `ee_pos`, `ee_quat` xyzw, `gripper_qpos[0]`). `prompt` is the
   demo's own `language` attribute.
3. Execute the first `--exec-horizon` (5) of the 10 returned actions.
4. Score with the env's own `_check_success()` and classify the failures.

A CSV row is flushed per episode (`--csv`). Put it outside the repo, since `*.csv` is
ignored.

## Check before comparing with the reference numbers

The reference numbers were produced with `--env-dir` = `$SIM_DATA_DIR/simple_env/`.
`stack.sh seed` rolls out in this repo's `sim/` instead: `kinova_simple_env.py` plus the
env chain it imports (`kinova_circle_env`, `kinova_grid_env`, `kinova_env`). The two
have not been compared. On a machine that has `simple_env/`:

```bash
cd ~/workspace/stir-run-sim && source _env.sh
for f in "$SIM_DATA_DIR"/simple_env/*.py; do
  cmp -s "$f" "sim/${f##*/}" && echo "same     ${f##*/}" || echo "DIFFERS  ${f##*/}"
done
```

All `same`: `stack.sh seed` results compare with the reference table. Any `DIFFERS` (or a
file missing from `sim/`): the envs differ. Reproduce with `bash stack.sh seed --env-dir
"$SIM_DATA_DIR/simple_env" ...` (your flag comes last and wins), and note the difference
next to any numbers from `sim/`.

## Extending the runner to KinovaTwoObjCircle

A runner for **`KinovaTwoObjCircle`** (`sim/kinova_circle_env.py`), the env the circle
data is recorded in, needs the changes below.

1. **Env import.** The seed does `sys.path.insert(0, args.env_dir); import kinova_simple_env`.
   Make it a repo-relative import of `sim/`, and use `kinova_circle_env.make_circle_env`
   for circle data.
2. **Env constructor.** `make_simple_env(128, obj_center, obj_sep, obj_axis, guideline_xy,
   guideline_size, cube_half=0.02)` becomes `make_circle_env(view_size, obj_names,
   obj_center, obj_sep, obj_axis, basket_xy)`. The circle env has no `guideline_xy`,
   `guideline_size` or `_success_rest_z`, so rewrite the `wrong_cube` and
   `final_dist_to_zone` logic against `basket_pos()`.
3. **Default paths.** These already come from `$SIM_DATA_DIR`, and `stack.sh seed` passes
   `../sim`. Point `--hdf5` at the circle data (`$SIM_DATA_DIR/data/circle.hdf5` by
   default), and default `--env-dir` to `../sim` once the import is repo-relative
   (item 1).
4. **`env_args`.** Circle data stores `env_name: KinovaTwoObjCircle` with `env_kwargs`
   `objects` (not `obj_names`) and without `obj_axis`, `basket_xy` or `guideline_*`, so
   `kwargs["guideline_xy"]` raises `KeyError`. Read the scene from `env_args`, or from the
   `env_configs/*.json` the data was collected with (`sim/env_config.py`).
5. **State length.** `set_state_from_flattened` needs a model that matches the recording:
   57 for two cubes, but LIBERO objects and a basket body add qpos/qvel. The demo's
   `model_file` attr holds the XML it was recorded with.
6. **Client library.** `from openpi_client import websocket_client_policy` is not
   installed by `requirements.txt`. Either `pip install -e $OPENPI_DIR/packages/openpi-client`
   (SETUP.md §6), or vendor a drop-in `WebsocketClientPolicy(host, port)` that needs only
   `websockets` and `msgpack`.
7. **Host default.** The `--port` default is already 8001. `--host` still defaults to
   `0.0.0.0`: make it `127.0.0.1` / `$SIM_POLICY_HOST`.
8. **State vector.** `build_state` (`ee_pos`, `eef_quat` xyzw, `gripper_qpos[0]`) must
   match what the LeRobot converter wrote (`examples/kinova/convert_kinova_data_to_lerobot.py`
   in `openpi/openpi_thor.patch`).
   It matters only for pi0, because pi0.5 never sees the state.
9. **Prompts and checkpoints.** Circle data says "pick up the <obj> and place it in the
   basket". `pi0/pi05_kinova_sim` are **out of distribution** in the circle env: they
   were trained without a basket body, on 4 cm cubes (cube.json uses 5 cm) with
   `obj_center (0, 0.10)` / `sep 0.05` (cube.json: `(0, 0.16)` / `0.07`), and without
   LIBERO objects. A circle-env runner needs checkpoints trained on circle-env data.
10. **Private env attributes.** The seed uses `env._spec`, `env._check_success()`,
    `env._get_observations(force_update=True)` and `env._success_rest_z`. Keep that set
    small, or give the env public accessors (then regenerate `sim/env_chain.sha256`).
11. **Rendering.** The caller must set `MUJOCO_GL=egl` (`_env.sh` does). The x86
    libglvnd `LD_LIBRARY_PATH` workaround in `docs/sim_eval.md` does not apply on Thor.
12. **Images.** Observations are raw robosuite frames, with no flip. That is also what
    the circle data stores, so training and rollout match.
13. **Recording rollouts.** A circle-env runner may want an HDF5 writer in the circle-data layout,
    and `env_config.py` + `env_configs/` to rebuild a collection scene. `env_config.py`,
    `env_configs/` and `circle_schedule.py` are already in `sim/`.
14. **Kill scope.** Keep launching python on the absolute `$REPO/runner/...` path, so that
    `kill.sh` (`OWN_PROCS`) and `stack.sh status` still find the runner.
