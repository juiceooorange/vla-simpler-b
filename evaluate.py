"""Closed-loop SimplerEnv / WidowX evaluator.

Course protocol (see 26ml_project.md sections 6 and 8):

* tasks: widowx_spoon_on_towel (T1), widowx_put_eggplant_in_basket (T2),
  widowx_carrot_on_plate (T3), widowx_stack_cube (T4)
* 5 Hz control, 500 Hz sim, 120 control steps (24 s) per episode
* initial states come from a pre-registered manifest; Base / Final / ablations
  must use the same manifest
* success is read from the environment's native ``info["success"]``; the
  episode-termination flag is never treated as success on its own

Usage:
    python evaluate.py --checkpoint /path/to/ckpt --policy openvla \
        --task widowx_spoon_on_towel --episodes 10 \
        --eval-config configs/eval_states.json --output-dir results/base_t1
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import traceback

import numpy as np

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS_DIR)

CONTROL_MODE = "arm_pd_ee_target_delta_pose_align2_gripper_pd_joint_pos"
CONTROL_FREQ = 5
SIM_FREQ = 500


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, help="local checkpoint dir (HF format)")
    ap.add_argument("--policy", default="openvla", choices=["openvla", "head", "vision", "depth", "localize", "oracle"])
    ap.add_argument("--task", required=True, choices=[
        "widowx_spoon_on_towel", "widowx_put_eggplant_in_basket",
        "widowx_carrot_on_plate", "widowx_stack_cube",
    ])
    ap.add_argument("--episodes", type=int, default=10, help="number of episodes to run (10 for B, 30 for A)")
    ap.add_argument("--eval-config", default=os.path.join(THIS_DIR, "configs", "eval_states.json"))
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--run-name", default=None, help="label written to the CSV, e.g. base / final")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--model-seed", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=None, help="override the manifest max steps")
    ap.add_argument("--save-video", action="store_true", default=True)
    ap.add_argument("--no-video", dest="save_video", action="store_false")
    ap.add_argument("--dry-run", action="store_true", help="validate env + manifest without loading the policy")
    ap.add_argument("--debug-actions", action="store_true", help="print every executed action")
    ap.add_argument("--action-scale", type=float, default=1.0, help="multiplier on predicted deltas")
    ap.add_argument("--head-dir", default=None, help="action-head directory (comma separated for an ensemble)")
    ap.add_argument("--flip-mode", default="xy_rot", choices=["none", "xy", "xy_rot"],
                    help="action-frame correction (SimplerEnv WidowX is rotated 180 deg about z)")
    ap.add_argument("--zero-rotation", action="store_true", help="ignore predicted rotation deltas")
    ap.add_argument("--temporal-ensemble", type=float, default=None,
                    help="ACT-style temporal ensembling coefficient m (None disables)")
    ap.add_argument("--simpler-env-root", default=None, help="path to the SimplerEnv repo (for overlay assets)")
    return ap.parse_args()


def resolve_simpler_env_root(explicit: str | None) -> str:
    if explicit:
        return explicit
    try:
        import simpler_env

        return os.path.dirname(os.path.dirname(os.path.abspath(simpler_env.__file__)))
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("cannot locate simpler_env package; pass --simpler-env-root") from exc


def build_env(spec: dict, repo_root: str):
    from simpler_env.utils.env.env_builder import build_maniskill2_env

    overlay = spec.get("rgb_overlay")
    overlay_path = os.path.join(repo_root, overlay) if overlay else None
    kwargs = dict(
        obs_mode="rgbd",
        robot=spec["robot"],
        sim_freq=SIM_FREQ,
        control_mode=os.environ.get("WIDOWX_CONTROL_MODE", CONTROL_MODE),
        control_freq=CONTROL_FREQ,
        max_episode_steps=spec.get("max_episode_steps", 120),
        scene_name=spec["scene_name"],
        camera_cfgs={"add_segmentation": True},
    )
    if overlay_path:
        kwargs["rgb_overlay_path"] = overlay_path
    return build_maniskill2_env(spec["env_name"], **kwargs)


def get_image(env, obs) -> np.ndarray:
    from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict

    return get_image_from_maniskill2_obs_dict(env, obs)


def save_video(path: str, frames: list[np.ndarray], fps: int = CONTROL_FREQ) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    try:
        import mediapy

        mediapy.write_video(path, frames, fps=fps)
    except Exception:  # noqa: BLE001
        import imageio

        imageio.mimsave(path, frames, fps=fps, macro_block_size=1)


def run_episode(env, policy, spec, state, instruction_override=None, debug_actions=False):
    reset_options = {
        "robot_init_options": {
            "init_xy": np.array([state["robot_init_x"], state["robot_init_y"]], dtype=np.float32),
            "init_rot_quat": np.array(state.get("robot_init_rot_quat", [0, 0, 0, 1]), dtype=np.float32),
        },
        "obj_init_options": (
            {"init_xy": np.array(state["obj_init_xy"], dtype=np.float32)}
            if state.get("obj_init_xy") is not None
            else {"episode_id": int(state["episode_id"])}
        ),
    }
    obs, _ = env.reset(seed=int(state.get("env_seed", 0)), options=reset_options)
    instruction = instruction_override or env.get_language_instruction()
    if policy is not None:
        policy.reset(instruction)

    image = get_image(env, obs)
    frames = [image]
    success = False
    grasped = False
    obj_z0 = None
    steps = 0
    err = ""
    terminated = truncated = False
    t0 = time.time()
    max_steps = spec["max_episode_steps"]
    try:
        for steps in range(1, max_steps + 1):
            if policy is None:
                action = np.zeros(7, dtype=np.float32)
            elif getattr(policy, "uses_proprio", False):
                if getattr(policy, "oracle", False):
                    from scipy.spatial.transform import Rotation as _R
                    _u = env.unwrapped
                    _sq = np.array(_u.source_obj_pose.q, dtype=np.float32)
                    try:
                        policy._base_p = np.asarray(env.unwrapped.agent.robot.pose.p, dtype=np.float32)
                    except Exception:  # noqa: BLE001
                        policy._base_p = np.zeros(3, dtype=np.float32)
                    policy._gt_poses = {
                        "src": np.array(_u.source_obj_pose.p, dtype=np.float64),
                        "dst": np.array(_u.target_obj_pose.p, dtype=np.float64),
                        "yaw": float(_R.from_quat([_sq[1], _sq[2], _sq[3], _sq[0]]).as_euler("xyz")[2]),
                    }
                try:
                    policy._tcp = np.asarray(env.unwrapped.tcp.pose.p, dtype=np.float32)
                    policy._tcp_quat = np.asarray(env.unwrapped.tcp.pose.q, dtype=np.float32)
                except Exception:  # noqa: BLE001
                    policy._tcp = np.zeros(3, dtype=np.float32)
                action = policy.act(image, instruction, obs=obs)
            else:
                action = policy.act(image, instruction)
            obs, _reward, terminated, truncated, info = env.step(action)
            try:
                _z = float(np.asarray(env.unwrapped.source_obj_pose.p)[2])
                if obj_z0 is None:
                    obj_z0 = _z
                elif _z - obj_z0 > 0.02:
                    grasped = True
            except Exception:  # noqa: BLE001
                pass
            if debug_actions:
                print(f"    step {steps}: action={np.round(action, 4).tolist()} success={info.get('success')}", flush=True)
            image = get_image(env, obs)
            frames.append(image)
            success = bool(info.get("success", False))
            if terminated or truncated:
                break
        else:
            success = bool(info.get("success", False))
    except Exception:  # noqa: BLE001
        err = traceback.format_exc(limit=3).replace("\n", " | ")
        success = False
    elapsed = time.time() - t0
    return {
        "success": success,
        "grasped": grasped,
        "steps": steps,
        "frames": frames,
        "instruction": instruction,
        "elapsed_s": round(elapsed, 2),
        "error": err,
        "terminated": bool(terminated),
        "truncated": bool(truncated),
    }


def main():
    args = parse_args()
    with open(args.eval_config, encoding="utf-8") as fh:
        manifest = json.load(fh)
    if args.task not in manifest["tasks"]:
        raise SystemExit(f"task {args.task} missing from {args.eval_config}")
    spec = dict(manifest["tasks"][args.task])
    spec.setdefault("max_episode_steps", manifest.get("protocol", {}).get("max_episode_steps", 120))
    if args.max_steps is not None:
        spec["max_episode_steps"] = args.max_steps

    states = spec["states"][: args.episodes]
    if args.num_shards > 1:
        states = [s for s in states if s["index"] % args.num_shards == args.shard_index]
    if len(states) < args.episodes and args.num_shards == 1:
        raise SystemExit(f"manifest has only {len(states)} states for {args.task}, need {args.episodes}")

    run_name = args.run_name or os.path.basename(os.path.normpath(args.checkpoint))
    out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)
    videos_dir = os.path.join(out_dir, "videos")

    repo_root = resolve_simpler_env_root(args.simpler_env_root)
    print(f"[evaluate] task={args.task} episodes={len(states)} run={run_name} device={args.device}")
    print(f"[evaluate] simpler_env root: {repo_root}")

    env = build_env(spec, repo_root)
    print("[evaluate] env built:", spec["env_name"], "control_mode:", env.unwrapped.control_mode)

    policy = None
    if not args.dry_run:
        t0 = time.time()
        if args.policy == "oracle":
            from policies.oracle_policy import OracleMLPPolicy

            policy = OracleMLPPolicy(head_dir=args.head_dir, device=args.device, flip_mode=args.flip_mode)
        elif args.policy == "localize":
            from policies.localize_policy import DetectPolicy, HeatmapPolicy, LocalizePolicy, TwoStagePolicy

            kind = os.environ.get("LOCALIZE_KIND", "reg")
            cls = {"2s": TwoStagePolicy, "hm": HeatmapPolicy, "det": DetectPolicy}.get(kind, LocalizePolicy)
            policy = cls(head_dir=args.head_dir, device=args.device, flip_mode=args.flip_mode)
            policy.oracle = os.environ.get("LOCALIZE_ORACLE") == "1"
        elif args.policy == "depth":
            from policies.depth_policy import DepthPolicy

            policy = DepthPolicy(base_model=args.checkpoint, head_dir=args.head_dir,
                                 device=args.device, flip_mode=args.flip_mode)
        elif args.policy == "vision":
            from policies.vision_policy import VisionPolicy

            policy = VisionPolicy(base_model=args.checkpoint, head_dir=args.head_dir,
                                  device=args.device, flip_mode=args.flip_mode)
        elif args.policy == "head":
            from policies.head_policy import HeadPolicy

            policy = HeadPolicy(base_model=args.checkpoint, head_dir=args.head_dir,
                                device=args.device, flip_mode=args.flip_mode,
                                temporal_ensemble=args.temporal_ensemble)
        else:
            from policies.openvla_policy import OpenVLAPolicy

            policy = OpenVLAPolicy(saved_model_path=args.checkpoint, device=args.device,
                                   action_scale=args.action_scale, flip_mode=args.flip_mode,
                                   zero_rotation=args.zero_rotation)
        print(f"[evaluate] policy loaded in {time.time()-t0:.1f}s")

    rows = []
    for state in states:
        res = run_episode(env, policy, spec, state, debug_actions=args.debug_actions)
        row = {
            "run": run_name,
            "checkpoint": args.checkpoint,
            "task": args.task,
            "episode_index": state["index"],
            "episode_id": state.get("episode_id", -1),
            "obj_init_xy": state.get("obj_init_xy", ""),
            "robot_init_x": state["robot_init_x"],
            "robot_init_y": state["robot_init_y"],
            "env_seed": state.get("env_seed", 0),
            "model_seed": args.model_seed,
            "success": int(res["success"]),
        "grasped": int(res.get("grasped", False)),
            "steps": res["steps"],
            "terminated": int(res["terminated"]),
            "truncated": int(res["truncated"]),
            "elapsed_s": res["elapsed_s"],
            "instruction": res["instruction"],
            "error": res["error"],
        }
        rows.append(row)
        if args.save_video and res["frames"]:
            vid = os.path.join(videos_dir, f"{args.task}_ep{state['index']:02d}_{'ok' if res['success'] else 'fail'}.mp4")
            try:
                save_video(vid, res["frames"])
                row["video"] = vid
            except Exception as exc:  # noqa: BLE001
                print("[evaluate] video failed:", exc)
        print(
            f"  ep{state['index']:02d} obj{state.get('episode_id', -1):02d} "
            f"success={res['success']} steps={res['steps']} {res['elapsed_s']}s"
            + (f" ERROR" if res["error"] else ""),
            flush=True,
        )

    csv_path = os.path.join(out_dir, f"episodes_{args.task}.csv")
    fieldnames = list(rows[0].keys()) if rows else ["run"]
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    n_success = sum(r["success"] for r in rows)
    n_grasp = sum(r.get("grasped", 0) for r in rows)
    summary = {
        "run": run_name,
        "grasp_rate": round(100.0 * n_grasp / max(1, len(rows)), 2),
        "checkpoint": args.checkpoint,
        "task": args.task,
        "episodes": len(rows),
        "success": n_success,
        "success_rate": round(100.0 * n_success / max(1, len(rows)), 2),
        "mean_steps": round(float(np.mean([r["steps"] for r in rows])), 2) if rows else None,
        "mean_elapsed_s": round(float(np.mean([r["elapsed_s"] for r in rows])), 2) if rows else None,
        "policy": args.policy,
        "eval_config": os.path.abspath(args.eval_config),
        "device": args.device,
    }
    with open(os.path.join(out_dir, f"summary_{args.task}.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    env.close()


if __name__ == "__main__":
    main()

