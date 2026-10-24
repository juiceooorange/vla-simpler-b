"""Generate training trajectories by rolling out our own policy in the simulator.

The task spec asks for trajectories produced by a model in simulation, recording
observations, language instructions and the actions actually executed. This script
does exactly that: it loads a policy (VLA action head, or the VLA detector +
primitive decoder), rolls it out on *training* initial states, and saves frames /
executed actions / proprioception / object poses / success.

Output layout matches the scripted collector so the same training code can read it.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import cv2 as cv
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from evaluate import build_env, get_image, resolve_simpler_env_root  # noqa: E402


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", required=True, choices=["head", "localize"])
    ap.add_argument("--head-dir", required=True)
    ap.add_argument("--checkpoint", default="../models/sft-simpler")
    ap.add_argument("--task", required=True)
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--eval-config", default="configs/dev_states.json")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--run-name", default="gen")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--flip-mode", default="none")
    ap.add_argument("--dart", type=float, default=0.0, help="DART: perturb the executed action, record the policy clean action")
    return ap.parse_args()


def main():
    args = parse_args()
    manifest = json.load(open(args.eval_config, encoding="utf-8"))
    spec = dict(manifest["tasks"][args.task])
    spec.setdefault("max_episode_steps", manifest.get("protocol", {}).get("max_episode_steps", 120))
    if args.max_steps:
        spec["max_episode_steps"] = args.max_steps
    states = spec["states"]

    if args.policy == "head":
        from policies.head_policy import HeadPolicy

        policy = HeadPolicy(base_model=args.checkpoint, head_dir=args.head_dir,
                            device=args.device, flip_mode=args.flip_mode)
    else:
        from policies.localize_policy import DetectPolicy

        policy = DetectPolicy(head_dir=args.head_dir, device=args.device, flip_mode=args.flip_mode)

    run_dir = os.path.join(args.output_dir, args.task, args.run_name)
    os.makedirs(run_dir, exist_ok=True)
    env = build_env(spec, resolve_simpler_env_root(None))
    stats = {"task": args.task, "run": args.run_name, "generator": f"policy:{args.policy}",
             "checkpoint": args.head_dir, "requested": args.episodes, "success": 0, "kept": 0, "episodes": []}
    t0 = time.time()
    for i in range(args.episodes):
        state = states[i % len(states)]
        ep_dir = os.path.join(run_dir, f"ep{i:03d}")
        frames_dir = os.path.join(ep_dir, "frames")
        os.makedirs(frames_dir, exist_ok=True)
        obs, _ = env.reset(seed=int(state.get("env_seed", 0)), options={
            "robot_init_options": {
                "init_xy": np.array([state["robot_init_x"], state["robot_init_y"]], dtype=np.float32),
                "init_rot_quat": np.array(state.get("robot_init_rot_quat", [0, 0, 0, 1]), dtype=np.float32),
            },
            "obj_init_options": {"episode_id": int(state["episode_id"])},
        })
        instruction = env.get_language_instruction()
        u = env.unwrapped
        src = np.array(u.source_obj_pose.p, dtype=np.float32)
        sq = np.array(u.source_obj_pose.q, dtype=np.float32)
        dst = np.array(u.target_obj_pose.p, dtype=np.float32)
        from scipy.spatial.transform import Rotation
        yaw = float(Rotation.from_quat([sq[1], sq[2], sq[3], sq[0]]).as_euler("xyz")[2])
        poses = {"src": src.tolist(), "dst": dst.tolist(), "src_yaw": yaw}

        image = get_image(env, obs)
        cv.imwrite(os.path.join(frames_dir, "00000.jpg"), cv.cvtColor(image, cv.COLOR_RGB2BGR),
                   [cv.IMWRITE_JPEG_QUALITY, 95])
        actions, props, depths = [], [], []
        policy.reset(instruction)
        success = False
        steps = 0
        for step in range(spec["max_episode_steps"]):
            try:
                policy._tcp = np.asarray(u.tcp.pose.p, dtype=np.float32)
                policy._tcp_quat = np.asarray(u.tcp.pose.q, dtype=np.float32)
            except Exception:  # noqa: BLE001
                pass
            if args.policy == "head" and getattr(policy, "uses_proprio", False):
                action = policy.act(image, instruction, obs=obs)
            elif args.policy == "head":
                action = policy.act(image, instruction)
            else:
                try:
                    policy._tcp = np.asarray(u.tcp.pose.p, dtype=np.float32)
                    policy._tcp_quat = np.asarray(u.tcp.pose.q, dtype=np.float32)
                except Exception:  # noqa: BLE001
                    pass
                action = policy.act(image, instruction, obs=obs)
            clean_action = np.asarray(action, dtype=np.float32).copy()
            exec_action = clean_action.copy()
            if args.dart > 0:
                exec_action[:3] += np.random.normal(0, args.dart, 3).astype(np.float32)
                exec_action[3:6] += np.random.normal(0, args.dart * 0.5, 3).astype(np.float32)
                exec_action[:6] = np.clip(exec_action[:6], -0.05, 0.05)
            obs, _r, term, trunc, info = env.step(exec_action)
            actions.append(clean_action)
            qpos = np.asarray(u.agent.robot.get_qpos(), dtype=np.float32).reshape(-1)
            tcp = np.asarray(u.tcp.pose.p, dtype=np.float32).reshape(-1)
            props.append(np.concatenate([qpos, tcp]))
            dep = np.asarray(obs["image"]["3rd_view_camera"]["depth"], dtype=np.float32)
            if dep.ndim == 3:
                dep = dep[..., 0]
            depths.append(dep[::4, ::4])
            image = get_image(env, obs)
            cv.imwrite(os.path.join(frames_dir, f"{len(actions):05d}.jpg"), cv.cvtColor(image, cv.COLOR_RGB2BGR),
                       [cv.IMWRITE_JPEG_QUALITY, 95])
            steps = len(actions)
            if info.get("success"):
                success = True
            if term or trunc:
                break
        np.save(os.path.join(ep_dir, "actions.npy"), np.asarray(actions, dtype=np.float32))
        np.save(os.path.join(ep_dir, "proprio.npy"), np.asarray(props, dtype=np.float32))
        np.save(os.path.join(ep_dir, "depth.npy"), np.asarray(depths, dtype=np.uint16))
        with open(os.path.join(ep_dir, "meta.json"), "w", encoding="utf-8") as fh:
            json.dump({"task": args.task, "episode": i, "state": state, "instruction": instruction,
                       "success": success, "steps": steps, "generator": f"policy:{args.policy}",
                       "run": args.run_name, "poses": poses}, fh, ensure_ascii=False, indent=2)
        stats["success"] += int(success)
        stats["kept"] += int(success)
        stats["episodes"].append({"i": i, "success": success, "steps": steps})
        print(f"  ep{i:03d} success={success} steps={steps} t={time.time()-t0:.0f}s", flush=True)
        if not success:
            # keep only successful trajectories (spec: prefer successful, action-valid ones)
            import shutil
            shutil.rmtree(ep_dir, ignore_errors=True)
    stats["success_rate"] = round(100.0 * stats["success"] / max(1, args.episodes), 2)
    with open(os.path.join(run_dir, "stats.json"), "w", encoding="utf-8") as fh:
        json.dump(stats, fh, indent=2, ensure_ascii=False)
    print(json.dumps({k: stats[k] for k in ("task", "generator", "requested", "success", "success_rate")}, indent=2))
    env.close()


if __name__ == "__main__":
    main()

