"""Vision-encoder + action head, trained end-to-end on images.

Motivation: every head trained on the *frozen* LLM last-token feature plateaued
(T1 ~0, T2 40%). The research notes the backbone representation itself may be the
bottleneck. Here we drop the LLM from the control path and train the VLA's own
vision backbone (frozen, or with LoRA) together with a convolutional action head
directly on images.

Architecture:
    224x224 RGB -> OpenVLA vision backbone (DINOv2+SigLIP, frozen or LoRA)
    -> 16x16x2176 token map -> 3x3 conv trunk -> global pool + task embedding
    -> MLP -> action chunk (K x 7)

The instruction selects a learned task embedding (two tasks in this project).

Usage:
    python train_head_vision.py --base-model ../models/sft-simpler --data ../data/rollouts \
        --runs clean --out ../runs/head_vision --chunk 5 --epochs 20 --lora
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
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--runs", default="clean")
    ap.add_argument("--tasks", default="widowx_spoon_on_towel,widowx_put_eggplant_in_basket")
    ap.add_argument("--out", required=True)
    ap.add_argument("--chunk", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--head-lr", type=float, default=1e-3)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--lora", action="store_true", help="apply LoRA to the vision backbone")
    ap.add_argument("--crop-task0", default="", help="x0,y0,w,h crop applied only to task 0 (spoon)")
    ap.add_argument("--crop-task1", default="", help="x0,y0,w,h crop applied only to task 1 (eggplant)")
    ap.add_argument("--lora-rank", type=int, default=16)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max-episodes", type=int, default=0)
    return ap.parse_args()


def build_index(data_root, runs, tasks, seed, val_frac, chunk, max_episodes=0):
    eps = []
    for ti, task in enumerate(tasks):
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
                padded = np.concatenate([actions, np.repeat(actions[-1:], chunk, axis=0)], axis=0)
                eps.append({"task_idx": ti, "frames": [os.path.join(d, "frames", f) for f in frames[:n]],
                            "targets": [padded[t:t + chunk].reshape(-1) for t in range(n)]})
    rng = random.Random(seed)
    rng.shuffle(eps)
    if max_episodes:
        eps = eps[:max_episodes]
    n_val = max(1, int(len(eps) * val_frac))
    samples = []
    for ep in eps:
        samples.extend(list(zip(ep["frames"], [ep["task_idx"]] * len(ep["frames"]), ep["targets"])))
    n_cut = sum(len(e["frames"]) for e in eps[:n_val])
    return samples[n_cut:], samples[:n_cut]


class ChunkDataset(Dataset):
    """Uses the VLA's own image transform; supports a per-task crop box."""

    def __init__(self, samples, transform, crops=None):
        self.samples = samples
        self.transform = transform
        self.crops = crops or {}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, task_idx, target = self.samples[i]
        img = Image.open(path).convert("RGB")
        box = self.crops.get(int(task_idx))
        if box:
            img = img.crop(box)
        x = self.transform(img)
        return x, int(task_idx), torch.tensor(target, dtype=torch.float32)


class VisionHead(nn.Module):
    def __init__(self, vis_dim=2176, d=512, chunk=5, n_tasks=2, n_tokens=256):
        super().__init__()
        self.chunk = chunk
        self.trunk = nn.Sequential(
            nn.Conv2d(vis_dim, d, 1),
            nn.GroupNorm(8, d),
            nn.GELU(),
            nn.Conv2d(d, d, 3, padding=1),
            nn.GroupNorm(8, d),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(2),
            nn.Flatten(),
        )
        self.task_emb = nn.Embedding(n_tasks, d)
        self.mlp = nn.Sequential(
            nn.LayerNorm(4 * d + d),
            nn.Linear(4 * d + d, d),
            nn.GELU(),
            nn.Linear(d, 7 * chunk),
        )

    def forward(self, tokens, task_idx):
        # tokens: (B, 2176, 16, 16)
        h = self.trunk(tokens)
        t = self.task_emb(task_idx)
        return self.mlp(torch.cat([h, t], dim=1))


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    tasks = [t for t in args.tasks.split(",") if t]
    runs = [r for r in args.runs.split(",") if r]

    from transformers import AutoModelForVision2Seq, AutoProcessor

    base = AutoModelForVision2Seq.from_pretrained(
        args.base_model, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
        trust_remote_code=True, attn_implementation="sdpa",
    ).to(args.device).eval()
    vision = base.vision_backbone
    transform = AutoProcessor.from_pretrained(args.base_model, trust_remote_code=True).image_processor.apply_transform
    for p in base.parameters():
        p.requires_grad = False
    del base

    if args.lora:
        from peft import LoraConfig, get_peft_model

        cfg = LoraConfig(r=args.lora_rank, lora_alpha=args.lora_rank, lora_dropout=0.0,
                         target_modules=["qkv", "proj", "fc1", "fc2"], bias="none")
        vision = get_peft_model(vision, cfg)
        vision.print_trainable_parameters()

    def parse_crop(v):
        if not v:
            return None
        a = [int(x) for x in v.split(",")]
        return (a[0], a[1], a[0] + a[2], a[1] + a[3])

    crops = {}
    if parse_crop(args.crop_task0):
        crops[0] = parse_crop(args.crop_task0)
    if parse_crop(args.crop_task1):
        crops[1] = parse_crop(args.crop_task1)
    print("[vision] crops", crops, flush=True)
    train_s, val_s = build_index(args.data, runs, tasks, args.seed, args.val_frac, args.chunk, args.max_episodes)
    print(f"[vision] train={len(train_s)} val={len(val_s)} lora={args.lora}", flush=True)
    dl_tr = DataLoader(ChunkDataset(train_s, transform, crops), batch_size=args.batch_size, shuffle=True,
                       num_workers=args.workers, drop_last=True, persistent_workers=True)
    dl_va = DataLoader(ChunkDataset(val_s, transform, crops), batch_size=args.batch_size, shuffle=False,
                       num_workers=max(2, args.workers // 2))

    head = VisionHead(2176, 512, args.chunk, len(tasks)).to(args.device)
    params = [{"params": head.parameters(), "lr": args.head_lr}]
    if args.lora:
        params.append({"params": [p for p in vision.parameters() if p.requires_grad], "lr": args.lr})
    opt = torch.optim.AdamW(params, weight_decay=1e-4)
    total_steps = args.epochs * max(1, len(dl_tr))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[args.head_lr, args.lr] if args.lora else args.head_lr,
                                                total_steps=total_steps)

    log = open(os.path.join(args.out, "train_log.jsonl"), "w", encoding="utf-8")
    best = 1e9
    t0 = time.time()
    for epoch in range(args.epochs):
        head.train()
        vision.train() if args.lora else vision.eval()
        tot, nb = 0.0, 0
        for x, t, y in dl_tr:
            x = x.to(args.device, dtype=torch.bfloat16)
            t = t.to(args.device)
            y = y.to(args.device)
            with torch.set_grad_enabled(True):
                feats = vision(x).float()  # (B, 256, 2176)
                b, n, c = feats.shape
                side = int(round(n ** 0.5))
                fmap = feats.reshape(b, side, side, c).permute(0, 3, 1, 2)
                pred = head(fmap, t)
                loss = torch.nn.functional.mse_loss(pred, y)
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
                opt.step()
                sched.step()
            tot += loss.item()
            nb += 1
            if nb % 50 == 0:
                print(f"  ep{epoch} {nb}/{len(dl_tr)} loss {tot/nb:.5f} ({time.time()-t0:.0f}s)", flush=True)
        head.eval()
        vl, vn = 0.0, 0
        with torch.no_grad():
            for x, t, y in dl_va:
                x = x.to(args.device, dtype=torch.bfloat16)
                feats = vision(x).float()
                b, n, c = feats.shape
                side = int(round(n ** 0.5))
                fmap = feats.reshape(b, side, side, c).permute(0, 3, 1, 2)
                pred = head(fmap, t.to(args.device))
                vl += torch.nn.functional.mse_loss(pred, y.to(args.device)).item()
                vn += 1
        msg = {"epoch": epoch, "train_mse": round(tot / max(1, nb), 6), "val_mse": round(vl / max(1, vn), 6)}
        print("[vision]", msg, flush=True)
        log.write(json.dumps(msg) + "\n")
        log.flush()
        if vl / max(1, vn) < best:
            best = vl / max(1, vn)
            torch.save({"head": head.state_dict(), "chunk": args.chunk, "lora": args.lora}, os.path.join(args.out, "head.pt"))
            if args.lora:
                vision.save_pretrained(os.path.join(args.out, "vision_lora"))
    log.close()
    with open(os.path.join(args.out, "head_config.json"), "w", encoding="utf-8") as fh:
        json.dump({"arch": "vision", "chunk": args.chunk, "base_model": args.base_model,
                   "lora": args.lora, "flip_mode": "none", "tasks": tasks, "crops": {str(k): list(v) for k, v in crops.items()}, "best_val_mse": best}, fh, indent=2)
    print("[vision] saved", args.out, "best", best)


if __name__ == "__main__":
    main()

