# openpi patch for Jetson Thor

`openpi_thor.patch` turns an upstream openpi checkout into the tree the sim policy server
runs from (`~/openpi`, `OPENPI_DIR`) on Jetson Thor (aarch64, CUDA 13).

| | |
|---|---|
| upstream | https://github.com/Physical-Intelligence/openpi |
| base commit | `15a9616a00943ada6c20a0f158e3adb39df2ccac` (2026-06-16, "update output objects to support batching (#975)") |
| `openpi_thor.patch` | changes to 10 tracked files, plus 17 new files under `examples/kinova/` |

## Apply

```bash
git clone https://github.com/Physical-Intelligence/openpi.git ~/openpi
git -C ~/openpi checkout 15a9616a00943ada6c20a0f158e3adb39df2ccac
git -C ~/openpi apply --check /path/to/openpi_thor.patch
git -C ~/openpi apply /path/to/openpi_thor.patch
cd ~/openpi && uv sync
```

The patch creates `examples/kinova/`, so nothing else needs to be copied in.
[`../docs/policy_server.md`](../docs/policy_server.md) explains why each Thor change is
needed and gives measured timings.

## What the patch changes

- Packaging for Thor (outside `src/`): `.python-version` 3.11 to 3.12,
  `requires-python >=3.12`, `jax[cuda12]==0.5.3` to `jax[cuda13]==0.11.1`, numpy `>=2.0` in
  both `openpi` and `packages/openpi-client`, `flax>=0.12.0`, `orbax-checkpoint>=0.12.4`,
  `gym-aloha` and the `rlds` group removed, the `ml-dtypes` and `tensorstore` overrides
  dropped, and the regenerated `uv.lock`.
  [`../docs/policy_server.md`](../docs/policy_server.md) gives the reason for each change.
- `src/openpi/models/model.py`: `restore_params` reads both the old mapping and the
  orbax >= 0.12 `StepMetadata` (`.item_metadata`) shape.
- `src/openpi/models/pi0.py`, `pi0_config.py`, `transforms.py`: the d_img auxiliary loss
  (`Pi0Config.dimg_*` fields, the aux branch in `Pi0.compute_loss`, the
  `AttachDimgTarget` transform). It is off by default (`dimg_aux_weight == 0.0`), so the
  stock loss is unchanged. `pi05_kinova_sim_dimgaux` turns it on through the
  `DIMG_AUX_WEIGHT`, `DIMG_AUX_MODE` and `DIMG_MARGIN` env vars, with targets read from
  `assets/dimg_targets/kinova/sim.npz`.
- `src/openpi/policies/libero_policy.py`: `LiberoInputs` passes `dimg_target` /
  `dimg_target_mask` through when present (a no-op on the stock path).
- `src/openpi/training/config.py`: the `TrainConfig`s below.
- `examples/kinova/`: the Kinova converter, sim and goal-switch evals, probes, injection
  sweeps and `serve_policy_inject.py`. Serving never imports them.

## TrainConfigs defined in `config.py` (32)

All are added by this patch; none exist upstream. `name` = the value passed to
`serve_policy.py` / `train.py`.

| group | configs |
|---|---|
| LIBERO-on-Kinova | `pi05_libero_grid`, `pi05_libero_grid500`, `pi05_libero_home500` |
| Kinova sim, base | `pi0_kinova_sim`, `pi05_kinova_sim`, `pi05_kinova_sim_dimgaux` |
| Kinova sim, data-size sweeps | `pi05_kinova_sim_d10`, `pi0_kinova_sim_d10`, `pi05_kinova_sim_d30`, `pi0_kinova_sim_d30`, `pi05_kinova_sim_d50`, `pi0_kinova_sim_d50`, `pi05_kinova_sim_d80`, `pi0_kinova_sim_d80` |
| Kinova sim, spacing sweeps | `pi05_kinova_sp09`, `pi0_kinova_sp09`, `pi05_kinova_sp13`, `pi0_kinova_sp13` |
| Kinova sim, home-start | `pi05_kinova_sim_home64`, `pi05_kinova_sim_home192`, `pi05_kinova_sim_home64_v2`, `pi05_kinova_sim_home192_v2`, `pi05_kinova_sim_home320_v2`, `pi05_kinova_sim_home516_v2` |
| Kinova real, 3-block set | `pi0_kinova_real`, `pi05_kinova_real`, `pi05_kinova_real_rt` |
| Kinova real, Thor rig | `pi05_kinova_thor`, `pi05_kinova_thor_home`, `pi05_kinova_real2_home`, `pi05_kinova_real2_home_v2`, `pi05_kinova_real2_grid` |

Each Thor-rig config's `repo_id` (`kinova/thor`, `kinova/thor_home`, `kinova/real2_home`,
`kinova/real2_home_v2`, `kinova/real2_grid`) selects the norm stats its checkpoint ships
under `assets/`.
