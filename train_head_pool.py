"""MLP head on pooled spatial tokens + global feature.

The ACT transformer under-fitted on this dataset (train L1 stayed ~19 mm), so this
variant keeps the same spatial information but replaces the transformer with a
simple MLP: mean/max pooling over the 8x8 token grid, concatenated with the
global feature.

Reuses the feature caches produced by train_head_act.py (no re-extraction).

Usage:
    python train_head_pool.py --cache-dir ../runs/head_act2 --out ../runs/head_pool
"""

from __future__ import annotations

import argparse
import json
import os

import torch
from torch import nn


class PoolHead(nn.Module):
    def __init__(self, in_dim, hidden=1024, out=56):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, out),
        )

    def forward(self, x):
        return self.net(x)


def featurize(cache):
    g = cache["global"].float()
    tok = cache["tokens"].float()
    return torch.cat([g, tok.mean(1), tok.amax(1)], dim=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    tr = torch.load(os.path.join(args.cache_dir, "train_cache.pt"), map_location="cpu")
    va = torch.load(os.path.join(args.cache_dir, "val_cache.pt"), map_location="cpu")
    Xtr, Ytr = featurize(tr), tr["y"]
    Xva, Yva = featurize(va), va["y"]
    chunk = int(Ytr.shape[1] // 7)
    print(f"[pool] train {tuple(Xtr.shape)} val {tuple(Xva.shape)} chunk {chunk}", flush=True)

    head = PoolHead(Xtr.shape[1], 1024, Ytr.shape[1]).to(args.device)
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    log = open(os.path.join(args.out, "train_log.jsonl"), "w", encoding="utf-8")
    best = 1e9
    n = len(Xtr)
    for epoch in range(args.epochs):
        head.train()
        perm = torch.randperm(n)
        tot = 0.0
        for i in range(0, n, args.batch_size):
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
        msg = {"epoch": epoch, "train_mse": round(tot / n, 6), "val_mse": round(vmse, 6), "val_mae_mm": round(mae, 3)}
        print("[pool]", msg, flush=True)
        log.write(json.dumps(msg) + "\n")
        log.flush()
        if vmse < best:
            best = vmse
            torch.save({"state_dict": head.state_dict(), "chunk": chunk, "in_dim": int(Xtr.shape[1])},
                       os.path.join(args.out, "head.pt"))
    log.close()
    with open(os.path.join(args.out, "head_config.json"), "w", encoding="utf-8") as fh:
        json.dump({"arch": "pool", "chunk": chunk, "flip_mode": "none", "best_val_mse": best}, fh, indent=2)
    print("[pool] saved", args.out, "best", best)


if __name__ == "__main__":
    main()

