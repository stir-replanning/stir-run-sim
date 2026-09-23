# STIR Simulation Rollouts

Serve trained OpenPI policies, run closed-loop Kinova Gen3 rollouts in MuJoCo and
robosuite, and evaluate them against the simulation dataset.

## Trained weights

The pretrained simulation policies used in the paper are available in the
[stir-libero-policies collection](https://huggingface.co/stir-replanning/stir-libero-policies).

## Quick start

```bash
cd ~/workspace/stir-run-sim
~/workspace/libero_venv/bin/python sim/assets/build_gripper.py
bash stack.sh check
bash stack.sh policy pi05-sim
bash stack.sh seed --limit-per-task 16 --label pi05-sim --csv /tmp/pi05-sim.csv
```

## Common commands

| Command | Purpose |
| --- | --- |
| `bash stack.sh policy list` | List configured models and checkpoint paths |
| `bash stack.sh policy prompts` | Show the training prompts |
| `bash stack.sh status` | Show rollout and environment status |
| `bash stack.sh kill` | Stop rollout and environment processes |
| `bash stack.sh help` | List all stack commands |

The policy server runs on `:8001`; the seed is the rollout client.

## Requirements

- Python 3.12 virtual environment with `robosuite==1.4.0`
- LIBERO checkout and MuJoCo/EGL GPU environment
- OpenPI checkout at `$OPENPI_DIR`
- Sim checkpoints in `$MODELS_DIR/kinova_sim`
- Generated Menagerie gripper asset

## Documentation

- [Setup](SETUP.md)
- [Policy server](docs/policy_server.md)
- [Simulation evaluation](docs/sim_eval.md)

## License

Apache-2.0. See [LICENSE](LICENSE).
