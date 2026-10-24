"""Depth-conditioned action head (T1 focus).

The RGB-only heads never reached the spoon; depth gives explicit 3D geometry for
thin objects. Inputs:
    LLM last-token feature (4096) + proprio (11) + depth map (1x120x160, metres)
    -> depth CNN (3 conv layers, global pool) -> 128
    -> MLP -> chunk x 7

Usage:
    python train_head_depth.py --base-model ../models/sft-simpler --data ../data/rollouts \
        --task widowx_spoon_on_towel --run depth --out ../runs/head_depth --chunk 5 --epochs 40
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time

import numpy as np
import torch
from PIL import Image
from torch import nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--task", default="widowx_spoon_on_towel")
    ap.add_argument("--run", default="depth")
    ap.add_argument("--out", required=True)
    ap.add_argument("--chunk", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--max-episodes", type=int, default=0)
    return ap.parse_args()


def list_episodes(data_root, task, run, chunk, seed, val_frac, max_episodes=0):
    root = os.path.join(data_root, task, run)
    eps = []
    for ep in sorted(os.listdir(root)):
        d = os.path.join(root, ep)
        mp = os.path.join(d, "meta.json")
        if not os.path.isfile(mp):
            continue
        meta = json.load(open(mp, encoding="utf-8"))
        if not meta.get("success"):
            continue
        dep_path = os.path.join(d, "depth.npy")
        if not os.path.isfile(dep_path):
            continue
        actions = np.load(os.path.join(d, "actions.npy")).astype(np.float32)
        prop = np.load(os.path.join(d, "proprio.npy")).astype(np.float32)
        frames = sorted(os.listdir(os.path.join(d, "frames")))
        n = min(len(frames), len(actions), len(prop))
        if n < 3:
            continue
        padded = np.concatenate([actions, np.repeat(actions[-1:], chunk, axis=0)], axis=0)
        eps.append({"frames": [os.path.join(d, "frames", f) for f in frames[:n]],
                    "depth": dep_path, "proprio": prop[:n],
                    "targets": [padded[t:t + chunk].reshape(-1) for t in range(n)],
                    "instruction": meta["instruction"]})
    rng = random.Random(seed)
    rng.shuffle(eps)
    if max_episodes:
        eps = eps[:max_episodes]
    n_val = max(1, int(len(eps) * val_frac))
    return eps[n_val:], eps[:n_val]


class DepthHead(nn.Module):
    def __init__(self, feat_dim=4107, chunk=5):
        super().__init__()
        self.chunk = chunk
        self.depth_net = nn.Sequential(
            nn.Conv2d(1, 16, 5, stride=2, padding=2), nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(3), nn.Flatten(),
        )
        self.mlp = nn.Sequential(
            nn.LayerNorm(feat_dim + 64 * 9),
            nn.Linear(feat_dim + 64 * 9, 512), nn.GELU(),
            nn.Linear(512, 512), nn.GELU(),
            nn.Linear(512, 7 * chunk),
        )

    def forward(self, feat, depth):
        return self.mlp(torch.cat([feat, self.depth_net(depth)], dim=1))


@torch.inference_mode()
def build_dataset(model, processor, episodes, device, chunk, cache_path, batch=8):
    if os.path.isfile(cache_path):
        print("[depth] reusing", cache_path, flush=True)
        return torch.load(cache_path, map_location="cpu")
    feats, depths, ys = [], [], []
    t0 = time.time()
    for ep in episodes:
        prompt = f"In: What action should the robot take to {ep['instruction'].strip().lower()}?\nOut:"
        dep = np.load(ep["depth"])  # (T, H, W) uint16 mm
        for i in range(0, len(ep["frames"]), batch):
            sel = range(i, min(i + batch, len(ep["frames"])))
            imgs = [Image.open(ep["frames"][t]).convert("RGB") for t in sel]
            inputs = processor([prompt] * len(imgs), imgs)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)
            out = model(**inputs, output_hidden_states=True)
            f = out.hidden_states[-1][:, -1].float().cpu()
            prop = torch.tensor(ep["proprio"][i:i + len(imgs)], dtype=torch.float32)
            feats.append(torch.cat([f, prop], dim=1).half())
            d = torch.tensor(dep[i:i + len(imgs)].astype(np.float32) / 1000.0).unsqueeze(1)
            depths.append(d.half())
            for t in sel:
                ys.append(torch.tensor(ep["targets"][t], dtype=torch.float32))
        if len(feats) % 50 == 0:
            print(f"  [depth] {len(feats)} eps ({time.time()-t0:.0f}s)", flush=True)
    data = {"feat": torch.cat(feats), "depth": torch.cat(depths), "y": torch.stack(ys)}
    torch.save(data, cache_path)
    print("[depth] cached", {k: tuple(v.shape) for k, v in data.items()}, flush=True)
    return data


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    os.makedirs(args.out, exist_ok=True)

    from transformers import AutoModelForVision2Seq, AutoProcessor

    tr_eps, va_eps = list_episodes(args.data, args.task, args.run, args.chunk, args.seed,
                                   args.val_frac, args.max_episodes)
    print(f"[depth] episodes train={len(tr_eps)} val={len(va_eps)}", flush=True)

    processor = AutoProcessor.from_pretrained(args.base_model, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        args.base_model, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
        trust_remote_code=True, attn_implementation="sdpa",
    ).to(args.device).eval()
    for p in model.parameters():
        p.requires_grad = False

    tr = build_dataset(model, processor, tr_eps, args.device, args.chunk, os.path.join(args.out, "train.pt"))
    va = build_dataset(model, processor, va_eps, args.device, args.chunk, os.path.join(args.out, "val.pt"))
    del model

    head = DepthHead(tr["feat"].shape[1], args.chunk).to(args.device)
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    log = open(os.path.join(args.out, "train_log.jsonl"), "w", encoding="utf-8")
    best = 1e9
    n = len(tr["y"])
    for epoch in range(args.epochs):
        head.train()
        perm = torch.randperm(n)
        tot = 0.0
        for i in range(0, n, args.batch_size):
            idx = perm[i:i + args.batch_size]
            f = tr["feat"][idx].float().to(args.device)
            d = tr["depth"][idx].float().to(args.device)
            y = tr["y"][idx].to(args.device)
            loss = torch.nn.functional.mse_loss(head(f, d), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(idx)
        sched.step()
        head.eval()
        with torch.no_grad():
            pred = head(va["feat"].float().to(args.device), va["depth"].float().to(args.device))
            vmse = torch.nn.functional.mse_loss(pred, va["y"].to(args.device)).item()
            mae = (pred - va["y"].to(args.device)).abs().mean().item() * 1000
        msg = {"epoch": epoch, "train_mse": round(tot / n, 6), "val_mse": round(vmse, 6), "val_mae_mm": round(mae, 3)}
        print("[depth]", msg, flush=True)
        log.write(json.dumps(msg) + "\n")
        log.flush()
        if vmse < best:
            best = vmse
            torch.save({"state_dict": head.state_dict(), "chunk": args.chunk, "feat_dim": int(tr["feat"].shape[1])},
                       os.path.join(args.out, "head.pt"))
    log.close()
    with open(os.path.join(args.out, "head_config.json"), "w", encoding="utf-8") as fh:
        json.dump({"arch": "depth", "chunk": args.chunk, "task": args.task, "best_val_mse": best}, fh, indent=2)
    print("[depth] saved", args.out, best)


if __name__ == "__main__":
    main()

