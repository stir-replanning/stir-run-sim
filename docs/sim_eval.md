# Sim evaluation: the seed's I/O contract and reference results

What `runner/eval_kinova_sim.py` sends and expects, and the results it produced.

## What the sim data looks like

`simple.hdf5` is robomimic-style HDF5: `data/demo_N/{actions, obs/*, dones, rewards,
states}`, with a `language` attribute per demo.

|                       | sim (`simple.hdf5`)       |
| --------------------- | ------------------------- |
| env                   | `KinovaTwoObjSimple`      |
| demos / frames        | 644 / 95,402              |
| episode length        | 76 – 362                  |
| images                | 128×128                   |
| `obs/gripper_states`  | `(T, 8)`                  |
| distinct instructions | 2 (red, yellow cube)      |
| action dims 0–5 std   | 0.005 – 0.30              |
| simulator states      | `states (T, 57)`          |
| object layout         | 2 cubes at fixed cells    |
| arm start             | 55-point grid × 3 heights |

**Actions are already deltas.** The controller is `OSC_POSE`, so dims 0–5 are pose
increments and dim 6 is a binary gripper command taking exactly `{-1, +1}`. openpi's
`extra_delta_transform` subtracts the current state from the action chunk, which is the
right thing to do for absolute targets and the wrong thing here — it would subtract the
state a second time. All four configs set `extra_delta_transform=False`. Note that
upstream's `pi0_libero` sets it to `True`, so this is a real divergence, and it is
driven by the data.

**`ee_quat` is xyzw, not wxyz.** In sim the dominant component is index 0 at ≈ −0.9998
across every frame. Read as xyzw that is a 180° flip about x — a downward-pointing
gripper, which is what top-down grasping needs. Read as wxyz, the sim value would mean a
near-identity orientation, i.e. the gripper pointing up. This only matters for keeping
the converter and the eval client consistent, since the model sees the state as four
opaque normalized numbers, but it is worth knowing which is which.

**The sim gripper's 8 columns are two copies of 4.** `corr(d0, d4) == 1.0` exactly. Of
the 4 unique columns, the first covariance principal component explains 99.6% of the
variance (74.8% on the correlation matrix), so the block is effectively one degree of
freedom. Column 0 spans `[-0.0003, 0.459]`, close to real's single gripper column
`[0.007, 0.485]`, so taking column 0 makes the sim and real state vectors mean roughly the
same thing.

Hence the state vector the seed's `build_state` must reproduce, 8-dim:

    state = concat(ee_pos(3), ee_quat(4), gripper_states[:, 0](1))

**π₀.₅ never sees the state.** `pi0.py` adds a state token only when `pi05` is false, and
`discrete_state_input=False` keeps the state out of the tokenized prompt too. So `pi05`
is conditioned on images and language alone. State quality therefore only affects `pi0`.
`ResizeImages(224, 224)` upsamples both cameras, so the 128×128 frames are fine as sent.

## Action horizon

`action_horizon=10` for all runs, not `pi0`'s default of 50.

The data loader builds `delta_timestamps` over `range(action_horizon)` and openpi never
reads the `actions_is_pad` flag LeRobot returns alongside the chunk. Any chunk that runs
past the end of an episode is silently trained on clamped targets. With sim episodes
averaging 148 frames, a horizon of 50 would corrupt roughly a third of sim samples. A
horizon of 10 also matches `pi05_libero` and keeps `pi0` and `pi05` directly comparable.
The seed executes the first 5 of each 10-action chunk (`--exec-horizon 5`).

## Evaluating

**Sim is closed-loop.** `simple.hdf5` carries `states (T, 57)`, so `eval_kinova_sim.py`
restores each demo's exact simulator state and rolls the policy out against the env's own
success test (target cube resting, released and settled inside the guideline square).
robosuite and openpi's JAX stack cannot share a venv, so the policy runs behind
`scripts/serve_policy.py` and the eval client talks to it over websockets — the same
server/client arrangement openpi uses for LIBERO. In this repo, the server (openpi venv,
:8001) and the client (libero_venv, `MUJOCO_GL=egl`) are:

    bash stack.sh policy pi0-sim
    bash stack.sh seed --limit-per-task 16 --label pi0

A caveat on the sim number: the training frames were rendered on the machine that
collected them and the eval frames are rendered elsewhere. Geometry matches exactly
(`ee_pos`, `ee_quat`, `gripper_qpos` reproduce bit-for-bit after `set_state_from_flattened`)
but pixels differ slightly at object edges — mean |Δ| ≈ 2.7/255 on `agentview`, 0.16/255 on
the wrist camera. Small, but it is a real train/eval shift. The Thor venv also differs
(mujoco 3.3.0 and numpy 2, vs mujoco 3.2.3 and numpy 1.26.4 for the reference runs), so the
table below is a reference, not a guaranteed reproduction. The env code may differ as well:
`stack.sh seed` uses this repo's `sim/`, not the `simple_env/` the table was made with, and
the two have not been compared (`../runner/README.md` has the check).

## Reference results (x86 GPU server)

Sim, closed-loop, 400-step cap, 5 actions executed per inference. Median steps counts only
successful episodes.

| policy            | step | n  | success   | median steps | final distance to zone centre |
| ----------------- | ---- | -- | --------- | ------------ | ----------------------------- |
| `pi0_kinova_sim`  | 1000 | 32 | 27/32     | 200          | 0.049 m                       |
| `pi0_kinova_sim`  | 3000 | 48 | **48/48** | 140          | 0.017 m                       |
| `pi0_kinova_sim`  | 5000 | 48 | 47/48     | 172          | 0.019 m                       |
| `pi05_kinova_sim` | 1000 | 32 | **32/32** | 136          | 0.014 m                       |
| `pi05_kinova_sim` | 3000 | 48 | **48/48** | 114          | 0.011 m                       |
| `pi05_kinova_sim` | 5000 | 48 | 46/48     | 129          | 0.019 m                       |
| random actions    | —    | 4  | 0/4       | —            | —                             |

None of these rows is the step-29999 checkpoint this repo serves. Steps 1000, 3000 and
4000 were not kept (see "Reference setup" below), so of the rows above only the step-5000
ones refer to a checkpoint that still exists.

`pi0`'s five failures at step 1000 were three `grasp_fail`, one `carry_fail` and one
`wrong_cube` — that last one is a genuine language error, the policy moved the cube the
prompt did not name. By step 3000 `pi0` has caught up on success rate, so `pi05`'s early
lead is faster convergence rather than a ceiling difference.

What survives is efficiency. At every checkpoint `pi05` reaches the goal in 20–30% fewer
steps than `pi0` (114 vs 140 at step 3000; 129 vs 172 at step 5000), and it never once
picked the wrong cube.

Success is not monotonic in training step — `pi05` goes 32/32 → 48/48 → 46/48 and `pi0` goes
27/32 → 48/48 → 47/48. At n=48 a 46-vs-48 difference is noise (Fisher exact p ≈ 0.49). Read
a single checkpoint's 100% as "somewhere near the ceiling", not as a guarantee.

### Language is the only cue for which cube to pick

Across all 644 sim demos the two cubes sit at the *same fixed cells* — `cellA = (-0.05, 0.1)`
and `cellB = (0.05, 0.1)`, never anywhere else. What varies is the arm: `init_xyz` (the
initial end-effector pose, not a cube position) takes 55 distinct xy values on a grid at
three heights. From there the end effector is a median 0.153 m from the target cube (max
0.32 m), and only 1.9% of demos start within 3 cm of it. `is_grasp` is 1 in 4 of 644 demos,
and the gripper first closes at a median of step 66 — the arm spends the first third of an
episode travelling.

The decisive number: the end effector starts closer to the target cube than to the distractor
in **49.7%** of demos. The start pose is a coin flip. It carries no information about which
cube is the target, so `wrong_cube = 0` across every normal rollout above is already evidence
that the policy is reading the instruction. Nothing else in the observation distinguishes the
two cubes except their colour and the prompt naming it.

`eval_kinova_sim.py --swap-prompt` confirms it directly. It feeds the *other* cube's
instruction while still scoring against the demo's own target, so the prompt names cube B
while the demo — and the scoring — is about cube A:

| policy                  | n  | moved the cube the prompt named | moved the demo's own target |
| ----------------------- | -- | ------------------------------- | --------------------------- |
| `pi0_kinova_sim` @3000  | 32 | 30 (94%)                        | **0**                       |
| `pi05_kinova_sim` @4000 | 32 | 31 (97%)                        | **0**                       |

Not once does either policy fall back on the demo's target. Change the word, and it goes to
the other cube. Language grounding is real and strong.

Two reading notes. In `--swap-prompt` runs the `outcome` column is computed against the
original target, so labels like `wrong_cube` and `no_reach` invert their usual meaning, and
median steps hit the 400-step cap because the scored cube is never moved — read `success` and
`wrong_cube` as raw flags, not as the outcome label. And `wrong_cube` tests *inside the
guideline square* and *released*, but not *settled*, so it is marginally weaker than
`_check_success`; after 400 steps the cube is at rest, so this does not change the counts.

### Caveats

The rollouts restore each demo's recorded simulator state, and every one of those layouts
was in the training set. This measures fit, not generalization. There is no held-out sim
split. `examples/kinova/convert_kinova_data_to_lerobot.py --split sim --subset train
--repo-id kinova/sim_train` (from `openpi/openpi_thor.patch`, run in `$OPENPI_DIR`) would
make one; `--subset val` gives the held-out tenth.

A first pass over only 6 episodes scored `pi0` at 6/6; the 32-episode run put it at 27/32.
Small rollout counts mislead.

A spacing-based generalization test is not available off the shelf: `simple_sp09.hdf5` and
`simple_sp13.hdf5` carry no `states` array, and their `env_args` still records
`obj_sep = 0.05`, so the spacing they are named for is not recoverable from the file.
`simple_d10.hdf5` does have `states`, but its init cells are a subset of the d100 grid the
model already trained on.

## Reference setup

- **Checkpoints are garbage collected.** `save_interval=1000` writes one every 1000 steps
  but `keep_period=5000` only makes multiples of 5000 permanent; everything else is deleted
  as soon as a newer checkpoint lands. Step 1000 disappears the moment step 2000 is written.
  If you want to compare two runs at the same step, evaluate before the next save, or raise
  `keep_period`. Each checkpoint is 41 GB (params + Adam state + EMA + assets).
- The reference eval venv had robosuite 1.4.0, mujoco 3.2.3 and numpy 1.26.4, matching the
  machine the data came from. It needs the custom `robotiq_2f85_menagerie` gripper assets
  (here: `sim/assets/README.md`), and on an x86 image that ships only the NVIDIA EGL vendor
  driver and no loader, a separately installed libglvnd. The client command was:

      LD_LIBRARY_PATH=$GLVND_DIR/lib:/usr/lib/x86_64-linux-gnu \
      __EGL_VENDOR_LIBRARY_DIRS=/usr/share/glvnd/egl_vendor.d MUJOCO_GL=egl \
        $KINOVA_EVAL_VENV/bin/python examples/kinova/eval_kinova_sim.py --port 8000

  `examples/kinova/inject_sweeps/run_sweep*.sh` in the patch read the same two variables
  (`KINOVA_EVAL_VENV`, default `~/envs/kinova_eval`; `GLVND_DIR`, default `~/envs/glvnd`).
