# The sim policy server on Jetson Thor

The sim policy server runs locally on Jetson Thor (JetPack 7.2.1, CUDA 13.2, compute
capability **11.0 / sm_110**). Upstream openpi cannot be installed there unmodified. This
page covers what that takes and how the server runs next to another policy server.

`../openpi/openpi_thor.patch` is the full diff against upstream openpi
`15a9616a00943ada6c20a0f158e3adb39df2ccac`. It contains the Kinova training configs,
including `pi0_kinova_sim` and `pi05_kinova_sim`, and the Thor changes below.
[`../openpi/README.md`](../openpi/README.md) lists every config and every file it touches.

## Building the server

If `~/openpi` is already built with this patch, reuse it: do not re-apply the patch or
`git checkout` in it. On a fresh machine:

```bash
git clone https://github.com/Physical-Intelligence/openpi.git ~/openpi
cd ~/openpi && git checkout 15a9616a00943ada6c20a0f158e3adb39df2ccac
git apply ~/workspace/stir-run-sim/openpi/openpi_thor.patch
uv sync
```

Serving never imports `examples/kinova/`. The patch creates it anyway, and nothing else
needs to be copied in. Then, from this repo, run `bash stack.sh policy pi05-sim` (serves
:8001). The first serve downloads the PaliGemma tokenizer into `$OPENPI_DATA_HOME`
(`~/.cache/openpi`).

## What had to change, and why

Upstream openpi is pinned to CUDA 12 / x86. Each of these is a hard blocker: the install
or the first GPU op fails without it.

| change | reason |
|---|---|
| `jax[cuda12]==0.5.3` → `jax[cuda13]==0.11.1` | CUDA 12 builds cannot target sm_110. Even jax **0.7.2** (the first with a CUDA 13 plugin) fails at runtime: `PTX .version 8.8 does not support .target sm_110a`. The PTX ISA that can name this GPU only arrives in the newer jaxlib. |
| `numpy<2.0.0` → `>=2.0`, in **both** `openpi` and `packages/openpi-client` | jax ≥ 0.7 requires numpy 2. |
| `requires-python >=3.11` → `>=3.12`, `.python-version` 3.11 → 3.12 | jax 0.11 dropped 3.11. |
| `ml-dtypes==0.4.1` override dropped | jax ≥ 0.7 requires ml_dtypes ≥ 0.5. |
| `tensorstore==0.1.74` override dropped | it was chosen to match the old orbax. |
| `orbax-checkpoint==0.11.13` → `>=0.12.4` | 0.11.13 imports `jax.experimental.layout.DeviceLocalLayout`, removed in jax 0.11. |
| `flax==0.10.2` → `>=0.12.0` | 0.10.2 calls `jax.core.get_opaque_trace_state`, removed in jax 0.11. |
| `rlds` dependency group removed | it pinned `tensorflow-cpu==2.15.0`, whose numpy<2 requirement cannot coexist with the jax Thor needs. It is an RLDS→LeRobot training-data conversion path that serving never touches, and it wanted python 3.11 for the only wheels that exist. |
| `gym-aloha` removed | it drags in mujoco 2.3.7, which has no cp312 aarch64 wheel and fails to build from source. Nothing under `src/` or `scripts/` imports `gym_aloha` or `mujoco`. |
| `model.py: restore_params` | orbax ≥ 0.12 returns a `StepMetadata` object where it used to return a mapping; the tree now hangs off `.item_metadata`. The fix reads either shape, so it works on both. |

## Measured on Jetson Thor

Measured with `pi05_kinova_thor` (a real-robot pi0.5 checkpoint of the same size). The sim
checkpoints have not been timed yet.

| | |
|---|---|
| JAX bf16 2048³ matmul | ~133 TFLOP/s |
| checkpoint load | ~10 s (6.2 GiB at ~1 GiB/s) |
| first inference (XLA compile) | ~28 s |
| steady-state inference | ~210 ms |

The seed calls `infer()` synchronously, one chunk at a time, so the first call blocks for
the ~28 s compile and every later call takes ~210 ms. At 5 executed actions per call, that
dominates a rollout's wall time.

`pi0_kinova_sim` (pi0 with a state token) has not been served on sm_110. Only pi0.5
checkpoints have been verified on Thor.

## Running next to another policy server

Another openpi server can serve from the same `~/openpi` on a different port; this repo
uses :8001. `policy stop` / restart matches only its own port
(`serve_policy.py --port <PORT>( |$)`), and this repo's kill script never stops a policy
server.

Memory is the constraint. By default JAX preallocates 75% of device memory, so on Thor's
unified memory the second server to start runs out unless each server caps its process.
Here the cap is `SIM_XLA_MEM_FRACTION` (default 0.4, passed as
`XLA_PYTHON_CLIENT_MEM_FRACTION` to the server only). This repo **ignores** an exported
`XLA_PYTHON_CLIENT_MEM_FRACTION` and prints a warning, so a shell set up for another
server cannot resize the :8001 server. Set `SIM_XLA_MEM_FRACTION` instead, for example
`SIM_XLA_MEM_FRACTION=0.3 bash stack.sh policy pi05-sim`. Each pi0/pi0.5 checkpoint is
about 6.2 GiB of params plus XLA workspace. MuJoCo EGL rendering for the seed, and any
other MuJoCo process, need the same memory, so lower the fractions if EGL context
creation fails.

Two servers started at the same moment also contend on the uv lock in `~/openpi`. That
makes the start slower, but it does no harm.
