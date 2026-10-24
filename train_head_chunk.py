"""Action-chunking head on the frozen OpenVLA backbone.

Why: a single-step head (train_head.py) accumulates compounding errors in the
closed loop. This version predicts the next K=5 actions from one observation,
executes them open-loop, then re-plans. Targets are the recorded environment-space
actions (no frame flip), so evaluation must use --flip-mode none.

Usage:
    python train_head_chunk.py --base-model ../models/sft-simpler --data ../data/rollouts \
        --runs expert,expert2 --out ../runs/head_chunk --chunk 5 --epochs 60
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
    ap.add_argument("--runs", default="expert,expert2")
    ap.add_argument("--tasks", default="widowx_spoon_on_towel,widowx_put_eggplant_in_basket")
    ap.add_argument("--out", required=True)
    ap.add_argument("--chunk", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--feature-batch", type=int, default=8)
    ap.add_argument("--simple-prompt", action="store_true", help="replace the env instruction with a short object-centric prompt")
    ap.add_argument("--localizer", default="", help="heatmap localizer dir; appends 8-D predicted poses to the feature")
    ap.add_argument("--proprio", action="store_true", help="append 11-D proprio (qpos+tcp) to the feature")
    ap.add_argument("--crop", default="", help="x0,y0,w,h crop applied before the processor")
    ap.add_argument("--max-episodes", type=int, default=0)
    return ap.parse_args()


class ChunkHead(nn.Module):
    def __init__(self, dim=4096, hidden=512, out=35):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, out),
        )

    def forward(self, x):
        return self.net(x)


def list_episodes(data_root, runs, tasks, seed, val_frac, max_episodes=0, simple_prompt=False):
    eps = []
    for task in tasks:
        for run in runs:
            run_dir = os.path.join(data_root, task, run)
            if not os.path.isdir(run_dir):
                continue
            for ep in sorted(os.listdir(run_dir)):
                d = os.path.join(run_dir, ep)
                mp = os.path.join(d, "meta.json")
                if not os.path.isfile(mp):
                    continue
                meta = json.load(open(mp, encoding="utf-8"))
                if not meta.get("success"):
                    continue
                actions = np.load(os.path.join(d, "actions.npy")).astype(np.float32)
                frames = sorted(os.listdir(os.path.join(d, "frames")))
                n = min(len(frames), len(actions))
                if n < 3:
                    continue
                prop_path = os.path.join(d, "proprio.npy")
                prop = np.load(prop_path).astype(np.float32) if os.path.isfile(prop_path) else None
                if simple_prompt:
                    obj = "spoon" if "spoon" in task else ("eggplant" if "eggplant" in task else "object")
                    instruction = f"pick up the {obj}"
                else:
                    instruction = meta["instruction"]
                eps.append({"task": task, "ep": ep,
                            "frames": [os.path.join(d, "frames", f) for f in frames[:n]],
                            "actions": actions[:n], "instruction": instruction,
                            "proprio": prop})
    rng = random.Random(seed)
    rng.shuffle(eps)
    if max_episodes:
        eps = eps[:max_episodes]
    n_val = max(1, int(len(eps) * val_frac))
    return eps[n_val:], eps[:n_val]


@torch.inference_mode()
def extract(model, processor, episodes, device, batch_size, chunk, crop=None, proprio=False, loc=None):
    X, Y, meta = [], [], []
    t0 = time.time()
    total = sum(len(e["frames"]) for e in episodes)
    done = 0
    for ep in episodes:
        prompt = f"In: What action should the robot take to {ep['instruction'].strip().lower()}?\nOut:"
        frames, acts = ep["frames"], ep["actions"]
        padded = np.concatenate([acts, np.repeat(acts[-1:], chunk, axis=0)], axis=0)
        for i in range(0, len(frames), batch_size):
            chunk_frames = frames[i:i + batch_size]
            imgs = [Image.open(p).convert("RGB") for p in chunk_frames]
            if crop:
                imgs = [im.crop(crop) for im in imgs]
            inputs = processor([prompt] * len(imgs), imgs)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)
            out = model(**inputs, output_hidden_states=True)
            feat = out.hidden_states[-1][:, -1].float().cpu()
            if loc is not None:
                from train_localizer_hm import decode_batch
                lx = torch.stack([loc["tf"](im) for im in imgs]).to(device)
                with torch.inference_mode():
                    lh = loc["model"](lx)
                lf = torch.tensor(decode_batch(lh, loc["Hm"]), dtype=torch.float16)
                feat = torch.cat([feat, lf], dim=1)
            if proprio:
                rows = []
                for j in range(len(chunk_frames)):
                    t = i + j
                    # NOTE: empirically the original pairing (proprio[t]) trains far
                    # better in closed loop than the shifted one, so it is kept.
                    pr = ep.get("proprio")
                    st = t
                    rows.append(pr[st] if pr is not None and st < len(pr) else np.zeros(11, dtype=np.float32))
                feat = torch.cat([feat, torch.tensor(np.stack(rows), dtype=torch.float32)], dim=1)
            X.append(feat)
            for j, f in enumerate(chunk_frames):
                t = i + j
                Y.append(torch.tensor(padded[t:t + chunk].reshape(-1), dtype=torch.float32))
                meta.append((ep["task"], ep["ep"], t))
            done += len(chunk_frames)
            if done % (batch_size * 20) == 0:
                print(f"  features {done}/{total} ({time.time()-t0:.0f}s)", flush=True)
    return torch.cat(X), torch.stack(Y), meta


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    os.makedirs(args.out, exist_ok=True)

    from transformers import AutoModelForVision2Seq, AutoProcessor

    tasks = [t for t in args.tasks.split(",") if t]
    runs = [r for r in args.runs.split(",") if r]
    train_eps, val_eps = list_episodes(args.data, runs, tasks, args.seed, args.val_frac, args.max_episodes, args.simple_prompt)
    print(f"[chunk] episodes train={len(train_eps)} val={len(val_eps)}")

    processor = AutoProcessor.from_pretrained(args.base_model, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        args.base_model, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
        trust_remote_code=True, attn_implementation="sdpa",
    ).to(args.device).eval()
    for p in model.parameters():
        p.requires_grad = False

    print("[chunk] extracting train features", flush=True)
    crop = tuple(int(v) for v in args.crop.split(",")) if args.crop else None
    if crop:
        crop = (crop[0], crop[1], crop[0] + crop[2], crop[1] + crop[3])
    print("[chunk] crop", crop, flush=True)
    loc = None
    if args.localizer:
        import torch as _t
        from train_localizer_hm import HeatLocalizer
        _ck = _t.load(os.path.join(args.localizer, "localizer_hm.pt"), map_location="cpu", weights_only=False)
        _m = HeatLocalizer(4).to(args.device).float().eval()
        _m.load_state_dict(_ck["state_dict"])
        def _tf(im):
            _i = im.resize((_ck["image_size"], _ck["image_size"]), Image.BILINEAR)
            return torch.from_numpy(np.asarray(_i, dtype=np.float32) / 255.0).permute(2, 0, 1)
        loc = {"model": _m, "Hm": _ck["Hm"], "tf": _tf}
    Xtr, Ytr, _ = extract(model, processor, train_eps, args.device, args.feature_batch, args.chunk, crop, args.proprio, loc)
    Xva, Yva, _ = extract(model, processor, val_eps, args.device, args.feature_batch, args.chunk, crop, args.proprio, loc)
    print("[chunk] shapes", tuple(Xtr.shape), tuple(Ytr.shape), tuple(Xva.shape))
    torch.save({"X": Xtr, "Y": Ytr}, os.path.join(args.out, "train_features.pt"))
    torch.save({"X": Xva, "Y": Yva}, os.path.join(args.out, "val_features.pt"))

    head = ChunkHead(Xtr.shape[1], 512, 7 * args.chunk).to(args.device)
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    log = open(os.path.join(args.out, "train_log.jsonl"), "w", encoding="utf-8")
    best = 1e9
    for epoch in range(args.epochs):
        perm = torch.randperm(len(Xtr))
        tot = 0.0
        head.train()
        for i in range(0, len(Xtr), args.batch_size):
            idx = perm[i:i + args.batch_size]
            xb, yb = Xtr[idx].to(args.device), Ytr[idx].to(args.device)
            loss = torch.nn.functional.mse_loss(head(xb), yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(idx)
        sched.step()
        head.eval()
        with torch.no_grad():
            pred = head(Xva.to(args.device))
            vmse = torch.nn.functional.mse_loss(pred, Yva.to(args.device)).item()
            mae = (pred - Yva.to(args.device)).abs().mean().item() * 1000
        msg = {"epoch": epoch, "train_mse": round(tot / len(Xtr), 6), "val_mse": round(vmse, 6), "val_mae_mm": round(mae, 3)}
        print("[chunk]", msg, flush=True)
        log.write(json.dumps(msg) + "\n")
        log.flush()
        if vmse < best:
            best = vmse
            torch.save({"state_dict": head.state_dict(), "dim": Xtr.shape[1], "chunk": args.chunk},
                       os.path.join(args.out, "head.pt"))
    log.close()
    with open(os.path.join(args.out, "head_config.json"), "w", encoding="utf-8") as fh:
        json.dump({"dim": int(Xtr.shape[1]), "hidden": 512, "chunk": args.chunk,
                   "base_model": args.base_model, "outputs": "env-space 7-D action chunk",
                   "flip_mode": "none", "crop": list(crop) if crop else None, "proprio": bool(args.proprio), "localizer": args.localizer or None, "prompt_override": ("pick up the " + ("spoon" if "spoon" in args.tasks else "eggplant")) if args.simple_prompt else None, "best_val_mse": best}, fh, indent=2)
    print("[chunk] saved", args.out, "best val mse", best)


if __name__ == "__main__":
    main()

