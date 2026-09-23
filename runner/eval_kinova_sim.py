import argparse
import collections
import contextlib
import csv
import json
import os
import pathlib
import sys

import h5py
import numpy as np

SIM_DATA_DIR = os.path.expanduser(os.environ.get("SIM_DATA_DIR", "~/kinova_dataset"))
DEFAULT_HDF5 = os.path.join(SIM_DATA_DIR, "data", "simple.hdf5")
DEFAULT_ENV_DIR = os.path.join(SIM_DATA_DIR, "simple_env")

REACH_DIST = 0.06
LIFT_HEIGHT = 0.03


def read_language(group) -> str:
    lang = group.attrs["language"]
    return lang.decode() if isinstance(lang, bytes) else str(lang)


def build_state(obs: dict) -> np.ndarray:
    return np.concatenate(
        [
            np.asarray(obs["robot0_eef_pos"], np.float32),
            np.asarray(obs["robot0_eef_quat"], np.float32),
            np.asarray(obs["robot0_gripper_qpos"], np.float32).reshape(-1)[:1],
        ]
    ).astype(np.float32)


def classify(*, success: bool, wrong_cube: bool, reached: bool, grasped: bool) -> str:
    if success:
        return "success"
    if wrong_cube:
        return "wrong_cube"
    if not reached:
        return "no_reach"
    if not grasped:
        return "grasp_fail"
    return "carry_fail"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=int(os.environ.get("SIM_POLICY_PORT", 8001)))
    ap.add_argument("--hdf5", default=DEFAULT_HDF5)
    ap.add_argument("--env-dir", default=DEFAULT_ENV_DIR)
    ap.add_argument("--limit-per-task", type=int, default=16, help="demos per language/target, 0 = all")
    ap.add_argument("--exec-horizon", type=int, default=5, help="actions executed per inference (chunk is 10)")
    ap.add_argument("--max-steps", type=int, default=400)
    ap.add_argument("--csv", default=None)
    ap.add_argument("--label", default="policy")
    ap.add_argument(
        "--random-policy",
        action="store_true",
        help="skip the server and emit random actions: exercises this harness, and gives a chance baseline",
    )
    ap.add_argument(
        "--swap-prompt",
        action="store_true",
        help="feed the other cube's instruction; score against the demo's own target",
    )
    args = ap.parse_args()

    sys.path.insert(0, args.env_dir)
    import kinova_simple_env as kse

    if args.random_policy:
        rng = np.random.default_rng(0)

        class RandomPolicy:
            def infer(self, _obs: dict) -> dict:
                pose = rng.normal(0.0, 0.2, size=(10, 6))
                grip = rng.choice([-1.0, 1.0], size=(10, 1))
                return {"actions": np.concatenate([pose, grip], axis=1)}

        policy = RandomPolicy()
        print("using random policy (no server)", flush=True)
    else:
        from openpi_client import websocket_client_policy

        policy = websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
        print(f"connected to policy server {args.host}:{args.port}", flush=True)

    handle = h5py.File(args.hdf5, "r")
    data = handle["data"]
    kwargs = json.loads(data.attrs["env_args"])["env_kwargs"]
    env = kse.make_simple_env(
        128,
        obj_center=tuple(kwargs["obj_center"]),
        obj_sep=float(kwargs["obj_sep"]),
        obj_axis=kwargs.get("obj_axis", "x"),
        guideline_xy=tuple(kwargs["guideline_xy"]),
        guideline_size=float(kwargs["guideline_size"]),
        cube_half=0.02,
    )
    guide_x, guide_y = env.guideline_xy
    half = env.guideline_size / 2.0
    rest_z = float(env.table_offset[2]) + 0.002 + env._success_rest_z

    by_target = collections.defaultdict(list)
    language_of = {}
    for name in sorted(data.keys(), key=lambda k: int(k.split("_")[1])):
        idx = int(data[name].attrs["target_idx"])
        by_target[idx].append(name)
        language_of.setdefault(idx, read_language(data[name]))

    selected = []
    for target_idx in sorted(by_target):
        names = by_target[target_idx]
        selected += names if args.limit_per_task <= 0 else names[: args.limit_per_task]
    available = {k: len(v) for k, v in by_target.items()}
    print(f"evaluating {len(selected)} episodes ({available} available)", flush=True)

    rows = []
    tally = collections.Counter()
    per_task = collections.defaultdict(collections.Counter)

    fields = [
        "demo",
        "prompt",
        "target_idx",
        "outcome",
        "success",
        "reached",
        "grasped",
        "wrong_cube",
        "steps",
        "final_dist_to_zone",
    ]
    csv_file = None
    writer = None
    if args.csv:
        path = pathlib.Path(args.csv)
        path.parent.mkdir(parents=True, exist_ok=True)
        csv_file = path.open("w", newline="")
        writer = csv.DictWriter(csv_file, fieldnames=fields)
        writer.writeheader()
        csv_file.flush()

    for episode, name in enumerate(selected):
        demo = data[name]
        target_idx = int(demo.attrs["target_idx"])
        prompt = language_of[1 - target_idx] if args.swap_prompt else read_language(demo)

        env.reset()
        env.sim.set_state_from_flattened(np.asarray(demo["states"][0], np.float64))
        env.sim.forward()
        for robot in env.robots:
            with contextlib.suppress(Exception):
                robot.controller.reset_goal()
        env._spec["target_idx"] = target_idx
        env._spec["obj_left_idx"] = int(demo.attrs["obj_left_idx"])

        obs = env._get_observations(force_update=True)
        start_target = np.array(env.sim.data.body_xpos[env.obj_body_ids[target_idx]])

        reached = grasped = success = False
        steps = 0
        while steps < args.max_steps and not success:
            result = policy.infer(
                {
                    "observation/image": np.asarray(obs["agentview_image"], np.uint8),
                    "observation/wrist_image": np.asarray(obs["robot0_eye_in_hand_image"], np.uint8),
                    "observation/state": build_state(obs),
                    "prompt": prompt,
                }
            )
            chunk = np.asarray(result["actions"], np.float64)
            for action in chunk[: args.exec_horizon]:
                obs, _, _, _ = env.step(action)
                steps += 1
                ee = np.asarray(obs["robot0_eef_pos"])
                target = np.array(env.sim.data.body_xpos[env.obj_body_ids[target_idx]])
                reached |= float(np.linalg.norm(ee - target)) < REACH_DIST
                grasped |= float(target[2] - start_target[2]) > LIFT_HEIGHT
                if env._check_success():
                    success = True
                    break
                if steps >= args.max_steps:
                    break

        target = np.array(env.sim.data.body_xpos[env.obj_body_ids[target_idx]])
        other = np.array(env.sim.data.body_xpos[env.obj_body_ids[1 - target_idx]])
        wrong_cube = abs(other[0] - guide_x) <= half and abs(other[1] - guide_y) <= half and other[2] < rest_z
        outcome = classify(success=success, wrong_cube=wrong_cube, reached=reached, grasped=grasped)

        tally[outcome] += 1
        per_task[prompt][outcome] += 1
        row = {
            "demo": name,
            "prompt": prompt,
            "target_idx": target_idx,
            "outcome": outcome,
            "success": int(success),
            "reached": int(reached),
            "grasped": int(grasped),
            "wrong_cube": int(wrong_cube),
            "steps": steps,
            "final_dist_to_zone": round(float(np.hypot(target[0] - guide_x, target[1] - guide_y)), 4),
        }
        rows.append(row)
        if writer is not None:
            writer.writerow(row)
            csv_file.flush()
        done = episode + 1
        print(
            f"[{done}/{len(selected)}] {name} {outcome:10s} steps={steps:3d} "
            f"running_success={tally['success']}/{done} ({100.0 * tally['success'] / done:.1f}%)",
            flush=True,
        )

    total = len(rows)
    print(
        f"\n=== {args.label}: success {tally['success']}/{total} = {100.0 * tally['success'] / max(total, 1):.1f}% ==="
    )
    for outcome, count in tally.most_common():
        print(f"  {outcome:12s} {count:4d}  ({100.0 * count / max(total, 1):.1f}%)")
    print("\nper task:")
    for prompt, counter in per_task.items():
        n = sum(counter.values())
        print(f"  {prompt[:48]:50s} success {counter['success']}/{n} ({100.0 * counter['success'] / max(n, 1):.1f}%)")

    if csv_file is not None:
        csv_file.close()
        print(f"\nwrote {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
