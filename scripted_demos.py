"""Collect successful demonstrations with a privileged scripted expert.

Why: the public OpenVLA family checkpoints produce ~0% success on this SimplerEnv
WidowX setup even after fixing the action frame, so they cannot generate the
successful trajectories needed for behaviour cloning. This script uses the
simulator's ground-truth object poses to run a simple pick-and-place controller
and records the resulting (image, instruction, action) episodes in exactly the
same format as ``collect_trajectories.py``.

The action frame correction (x/y inverted) was verified independently with
``scripted_expert.py``.

Usage:
    python scripted_demos.py --task widowx_spoon_on_towel --episodes 40 \
        --eval-config configs/dev_states.json --output-dir ../data/rollouts --run-name expert
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import cv2 as cv
import numpy as np

from evaluate import build_env, get_image, resolve_simpler_env_root

FLIP = np.array([-1.0, -1.0, 1.0], dtype=np.float32)
TRAIN_EPISODE_IDS = list(range(10, 24))  # 0..9 are the formal test placements  # action frame correction


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=[
        "widowx_spoon_on_towel", "widowx_put_eggplant_in_basket",
        "widowx_carrot_on_plate", "widowx_stack_cube",
    ])
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--eval-config", required=True)
    ap.add_argument("--output-dir", default="../data/rollouts")
    ap.add_argument("--run-name", default="expert")
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--jpeg-quality", type=int, default=90)
    ap.add_argument("--grasp-z", type=float, default=0.012, help="grasp height above object origin")
    ap.add_argument("--lift-z", type=float, default=0.12)
    ap.add_argument("--place-z", type=float, default=0.04, help="release height above target origin")
    ap.add_argument("--dart", type=float, default=0.0, help="DART: perturb the executed action by this std, but record the clean expert action")
    ap.add_argument("--strategy", default="grasp", choices=["grasp", "push", "reactive"])
    ap.add_argument("--push-height", type=float, default=0.03)
    ap.add_argument("--random-obj", action="store_true", help="randomise the object xy per episode")
    ap.add_argument("--obj-jitter", type=float, default=0.025, help="isotropic object jitter (m)")
    ap.add_argument("--obj-jitter-x", type=float, default=None)
    ap.add_argument("--obj-jitter-y", type=float, default=None)
    ap.add_argument("--align-yaw", action="store_true", help="rotate gripper yaw to match the object")
    ap.add_argument("--yaw-offset", type=float, default=0.0, help="extra yaw offset (rad)")
    ap.add_argument("--grasp-offset", type=float, default=0.0, help="shift grasp point along the object local x (m)")
    ap.add_argument("--jitter", type=float, default=0.008, help="robot xy jitter (m)")
    ap.add_argument("--action-noise", type=float, default=0.0015, help="gaussian noise added to deltas (m)")
    return ap.parse_args()


class Expert:
    def __init__(self, env, spec, grasp_offset=0.0, strategy="grasp", push_height=0.03, align_yaw=False, yaw_offset=0.0, action_scale=1.0):
        self.env = env
        self.u = env.unwrapped
        self.spec = spec
        self.base_p = np.array(self.u.agent.robot.pose.p)
        self.max_steps = spec["max_episode_steps"]
        self.grasp_offset = grasp_offset
        self.strategy = strategy
        self.align_yaw = align_yaw
        self.yaw_offset = yaw_offset
        self.push_height = push_height
        self.steps = 0
        self.done = False
        self.info = {}

    def tcp(self):
        return np.array(self.u.tcp.pose.p) - self.base_p

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        obs, _r, done, trunc, info = self.env.step(action)
        self.steps += 1
        self.info = info
        self.done = bool(done or trunc)
        return obs, info

    def move_to(self, target, gripper, max_delta=0.03, tol=0.008, budget=80):
        for _ in range(budget):
            if self.done or self.steps >= self.max_steps:
                return
            d = np.clip(target - self.tcp(), -max_delta, max_delta)
            a = np.zeros(7, dtype=np.float32)
            a[:3] = d * FLIP
            a[6] = gripper
            self.step(a)
            if np.linalg.norm(target - self.tcp()) < tol:
                return

    def hold(self, gripper, n):
        for _ in range(n):
            if self.done or self.steps >= self.max_steps:
                return
            a = np.zeros(7, dtype=np.float32)
            a[6] = gripper
            self.step(a)

    def run_push(self):
        u = self.u
        s = np.array(u.source_obj_pose.p) - self.base_p
        t = np.array(u.target_obj_pose.p) - self.base_p
        d = t[:2] - s[:2]
        n = float(np.linalg.norm(d))
        d = d / n if n > 1e-6 else np.array([1.0, 0.0])
        behind = np.array([s[0] - d[0] * 0.07, s[1] - d[1] * 0.07, s[2] + self.push_height])
        self.move_to(behind, 1.0, budget=45)
        self.hold(-1.0, 6)
        target_z = s[2] + self.push_height
        for _ in range(int((n + 0.18) / 0.01)):
            if self.done or self.steps >= self.max_steps or self.info.get("success"):
                break
            a = np.zeros(7, dtype=np.float32)
            a[0] = d[0] * 0.01 * FLIP[0]
            a[1] = d[1] * 0.01 * FLIP[1]
            a[2] = float(np.clip((target_z - self.tcp()[2]) * 0.6, -0.01, 0.01)) * FLIP[2]
            a[6] = -1.0
            self.step(a)
        return self.info

    def align(self, target_yaw):
        from scipy.spatial.transform import Rotation
        for _ in range(25):
            if self.done or self.steps >= self.max_steps:
                return
            q = np.array(self.u.tcp.pose.q)
            yaw = Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_euler("xyz")[2]
            err = (target_yaw - yaw + np.pi) % (2 * np.pi) - np.pi
            if abs(err) < 0.05:
                return
            a = np.zeros(7, dtype=np.float32)
            a[5] = float(np.clip(err * 0.7, -0.12, 0.12))
            a[6] = 1.0
            self.step(a)

    def run_reactive(self):
        """Potential-field controller: every action is a pure function of state.

        No phase machine and no step counters. The vertical target rises with the
        horizontal distance to the goal, which replaces the approach/descend
        switch with a continuous function, and the gripper command is a state
        predicate. This keeps the demonstrations Markovian so single-frame BC can
        imitate them.
        """
        u = self.u
        src = np.array(u.source_obj_pose.p) - self.base_p
        dst = np.array(u.target_obj_pose.p) - self.base_p
        if self.align_yaw:
            from scipy.spatial.transform import Rotation
            q = np.array(u.source_obj_pose.q)
            self.align(Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_euler("xyz")[2])
        for _ in range(self.max_steps):
            if self.done or self.steps >= self.max_steps:
                return self.info
            tcp = self.tcp()
            qpos = np.asarray(u.agent.robot.get_qpos(), dtype=np.float64).reshape(-1)
            finger = float(qpos[-1]) if qpos.size else 1.0
            obj = np.array(u.source_obj_pose.p) - self.base_p
            in_hand = (np.linalg.norm(obj - tcp) < 0.05) and finger < 0.02
            goal = dst if in_hand else src
            xy_err = float(np.linalg.norm(goal[:2] - tcp[:2]))
            lift = 0.09 * min(1.0, xy_err / 0.06)
            target = np.array([goal[0], goal[1], goal[2] + lift + (0.04 if in_hand else 0.012)])
            if in_hand:
                grip = 1.0 if (xy_err < 0.010 and abs(tcp[2] - target[2]) < 0.010) else -1.0
            else:
                grip = -1.0 if (xy_err < 0.010 and tcp[2] < src[2] + 0.020) else 1.0
            d = np.clip(target - tcp, -0.03, 0.03)
            a = np.zeros(7, dtype=np.float32)
            a[:3] = d * FLIP
            a[6] = grip
            self.step(a)
        return self.info

    def run(self):
        if self.strategy == "push":
            return self.run_push()
        u = self.u
        src = np.array(u.source_obj_pose.p) - self.base_p
        dst = np.array(u.target_obj_pose.p) - self.base_p
        if self.grasp_offset:
            from scipy.spatial.transform import Rotation
            q = np.array(u.source_obj_pose.q)
            rot = Rotation.from_quat([q[1], q[2], q[3], q[0]])
            src = src + rot.apply([self.grasp_offset, 0.0, 0.0])
        if self.align_yaw:
            from scipy.spatial.transform import Rotation
            q = np.array(u.source_obj_pose.q)
            spoon_yaw = Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_euler("xyz")[2]
            self.align(spoon_yaw + self.yaw_offset)
        # approach above object
        self.move_to(src + np.array([0, 0, 0.10]), 1.0, budget=45)
        # descend
        self.move_to(src + np.array([0, 0, 0.012]), 1.0, max_delta=0.015, tol=0.006, budget=35)
        # grasp
        self.hold(-1.0, 12)
        # lift
        self.move_to(src + np.array([0, 0, 0.12]), -1.0, budget=30)
        # transfer above target
        self.move_to(dst + np.array([0, 0, 0.10]), -1.0, budget=60)
        # lower
        self.move_to(dst + np.array([0, 0, 0.04]), -1.0, max_delta=0.015, tol=0.008, budget=35)
        # release
        self.hold(1.0, 12)
        return self.info


def main():
    args = parse_args()
    with open(args.eval_config, encoding="utf-8") as fh:
        manifest = json.load(fh)
    spec = dict(manifest["tasks"][args.task])
    spec.setdefault("max_episode_steps", manifest.get("protocol", {}).get("max_episode_steps", 120))
    if args.max_steps:
        spec["max_episode_steps"] = args.max_steps

    states = spec["states"]
    plan = [states[i % len(states)] for i in range(args.episodes)]

    run_dir = os.path.join(args.output_dir, args.task, args.run_name)
    os.makedirs(run_dir, exist_ok=True)
    env = build_env(spec, resolve_simpler_env_root(None))

    OBJ_CENTER = None
    _obs0, _ = env.reset(seed=0, options={"obj_init_options": {"episode_id": int(plan[0]["episode_id"])}})
    OBJ_CENTER = tuple(float(v) for v in env.unwrapped.source_obj_pose.p[:2])
    print("[demos] object centre", OBJ_CENTER, flush=True)

    stats = {"task": args.task, "run": args.run_name, "requested": len(plan),
             "success": 0, "kept": 0, "episodes": []}
    t0 = time.time()
    for i, state in enumerate(plan):
        ep_dir = os.path.join(run_dir, f"ep{i:03d}")
        frames_dir = os.path.join(ep_dir, "frames")
        os.makedirs(frames_dir, exist_ok=True)
        obj_opt = {"episode_id": int(state["episode_id"])}
        if args.random_obj:
            # NOTE: these environments ignore `obj_init_options["init_xy"]` and rebuild the
            # placement from `episode_id` (24 precomputed configs). We therefore sample an
            # episode_id from the training-only range 10..23; 0..9 are the formal test
            # placements and must never be used for data collection.
            obj_opt = {"episode_id": int(np.random.choice(TRAIN_EPISODE_IDS))}
        jx = float(np.random.uniform(-args.jitter, args.jitter))
        jy = float(np.random.uniform(-args.jitter, args.jitter))
        obs, _ = env.reset(seed=int(state.get("env_seed", 0)) + i, options={
            "robot_init_options": {
                "init_xy": np.array([state["robot_init_x"] + jx, state["robot_init_y"] + jy], dtype=np.float32),
                "init_rot_quat": np.array(state.get("robot_init_rot_quat", [0, 0, 0, 1]), dtype=np.float32),
            },
            "obj_init_options": obj_opt,
        })
        instruction = env.get_language_instruction()
        from scipy.spatial.transform import Rotation as _Rot
        _sp = np.array(env.unwrapped.source_obj_pose.p, dtype=np.float32)
        _sq = np.array(env.unwrapped.source_obj_pose.q, dtype=np.float32)
        _dp = np.array(env.unwrapped.target_obj_pose.p, dtype=np.float32)
        _yaw = float(_Rot.from_quat([_sq[1], _sq[2], _sq[3], _sq[0]]).as_euler("xyz")[2])
        _poses = {"src": _sp.tolist(), "dst": _dp.tolist(), "src_yaw": _yaw}
        image = get_image(env, obs)
        cv.imwrite(os.path.join(frames_dir, "00000.jpg"), cv.cvtColor(image, cv.COLOR_RGB2BGR),
                   [cv.IMWRITE_JPEG_QUALITY, args.jpeg_quality])

        expert = Expert(env, spec, grasp_offset=args.grasp_offset, strategy=args.strategy, push_height=args.push_height, align_yaw=args.align_yaw, yaw_offset=args.yaw_offset)
        actions = []

        # wrap env.step to record frames/actions
        orig_step = expert.step
        frames_root = frames_dir

        props = []
        depths = []

        def step_and_record(action, _orig=orig_step, _root=frames_root, _actions=actions, _env=env):
            clean = np.asarray(action, dtype=np.float32).copy()
            action = clean.copy()
            if args.dart > 0:
                action[:3] += np.random.normal(0, args.dart, 3).astype(np.float32)
                action[3:6] += np.random.normal(0, args.dart * 0.5, 3).astype(np.float32)
                action[:6] = np.clip(action[:6], -0.05, 0.05)
            action[:3] += np.random.normal(0, args.action_noise, 3).astype(np.float32)
            obs, info = _orig(action)
            dep = np.asarray(obs["image"]["3rd_view_camera"]["depth"], dtype=np.float32)
            if dep.ndim == 3:
                dep = dep[..., 0]
            dep = dep.reshape(112, 4, 112, 4).mean(axis=(1, 3)) if dep.shape == (448, 448) else dep[::4, ::4]
            depths.append(np.clip(dep * 1000.0, 0, 65535).astype(np.uint16))
            qpos = np.asarray(_env.unwrapped.agent.robot.get_qpos(), dtype=np.float32).reshape(-1)
            tcp = np.asarray(_env.unwrapped.tcp.pose.p, dtype=np.float32).reshape(-1)
            props.append(np.concatenate([qpos, tcp]))
            _actions.append(clean if args.dart > 0 else np.asarray(action, dtype=np.float32))
            img = get_image(_env, obs)
            cv.imwrite(os.path.join(_root, f"{len(_actions):05d}.jpg"), cv.cvtColor(img, cv.COLOR_RGB2BGR),
                       [cv.IMWRITE_JPEG_QUALITY, args.jpeg_quality])
            return obs, info

        expert.step = step_and_record
        info = expert.run_reactive() if args.strategy == "reactive" else expert.run()
        success = bool(info.get("success", False))
        np.save(os.path.join(ep_dir, "actions.npy"), np.asarray(actions, dtype=np.float32))
        if props:
            np.save(os.path.join(ep_dir, "proprio.npy"), np.asarray(props, dtype=np.float32))
        if depths:
            np.save(os.path.join(ep_dir, "depth.npy"), np.asarray(depths, dtype=np.uint16))
        with open(os.path.join(ep_dir, "meta.json"), "w", encoding="utf-8") as fh:
            json.dump({"task": args.task, "episode": i, "state": state, "instruction": instruction,
                       "success": success, "steps": len(actions), "generator": "scripted_expert",
                       "run": args.run_name, "poses": _poses}, fh, ensure_ascii=False, indent=2)
        stats["success"] += int(success)
        stats["kept"] += int(success)
        stats["episodes"].append({"i": i, "success": success, "steps": len(actions)})
        print(f"  ep{i:03d} success={success} steps={len(actions)} t={time.time()-t0:.0f}s", flush=True)

    stats["success_rate"] = round(100.0 * stats["success"] / max(1, len(plan)), 2)
    with open(os.path.join(run_dir, "stats.json"), "w", encoding="utf-8") as fh:
        json.dump(stats, fh, indent=2, ensure_ascii=False)
    print(json.dumps({k: stats[k] for k in ("task", "requested", "success", "success_rate")}, indent=2))
    env.close()


if __name__ == "__main__":
    main()

