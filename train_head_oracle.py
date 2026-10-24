"""Oracle controller probe: can the action interface be learned at all?

Input is the privileged state (object poses, yaw, joint positions, tcp) and the
target is the expert action chunk. No vision. If this MLP reaches a high success
rate in closed loop, the action interface is sound and the end-to-end VLA failure
is purely a perception problem. If it does not, there is a bug in the action/
observation interface that all our learned heads inherited.
"""

from __future__ import annotations

import argparse
import json
import os
import random

import numpy as np
import torch
from torch import nn


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--task", default="widowx_spoon_on_towel")
    ap.add_argument("--run", default="pose")
    ap.add_argument("--out", required=True)
    ap.add_argument("--chunk", type=int, default=5)
    ap.add_argument("--history", type=int, default=5, help="number of previous actions appended to the input")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")
    return ap.parse_args()


def build(args):
    root = os.path.join(args.data, args.task, args.run)
    X, Y = [], []
    for ep in sorted(os.listdir(root)):
        d = os.path.join(root, ep)
        mp = os.path.join(d, "meta.json")
        if not os.path.isfile(mp):
            continue
        meta = json.load(open(mp, encoding="utf-8"))
        if not meta.get("success") or "poses" not in meta:
            continue
        prop_p = os.path.join(d, "proprio.npy")
        if not os.path.isfile(prop_p):
            continue
        acts = np.load(os.path.join(d, "actions.npy")).astype(np.float32)
        prop = np.load(prop_p).astype(np.float32)
        n = min(len(acts), len(prop))
        p = meta["poses"]
        vec = np.concatenate([p["src"], p["dst"], [np.sin(p["src_yaw"]), np.cos(p["src_yaw"])]]).astype(np.float32)
        padded = np.concatenate([acts, np.repeat(acts[-1:], args.chunk, axis=0)], axis=0)
        hist = np.zeros((n, args.history * 7), dtype=np.float32)
        for t in range(n):
            for k in range(1, args.history + 1):
                if t - k >= 0:
                    hist[t, (k - 1) * 7:k * 7] = acts[t - k]
        # ALIGN: proprio[t] was recorded AFTER stepping with actions[t], i.e. it is
        # the state at t+1. Pair frame/action t with proprio[t-1].
        for t in range(1, n):
            X.append(np.concatenate([vec, prop[t - 1], hist[t]]))
            Y.append(padded[t:t + args.chunk].reshape(-1))
    X = torch.tensor(np.stack(X), dtype=torch.float32)
    Y = torch.tensor(np.stack(Y), dtype=torch.float32)
    idx = torch.randperm(len(X), generator=torch.Generator().manual_seed(args.seed))
    n_val = max(1, int(len(X) * args.val_frac))
    return X[idx[n_val:]], Y[idx[n_val:]], X[idx[:n_val]], Y[idx[:n_val]]


class OracleHead(nn.Module):
    """Two-headed: continuous 6-D pose chunk + discrete gripper logits per step."""

    def __init__(self, in_dim, chunk, hidden=256):
        super().__init__()
        self.chunk = chunk
        self.trunk = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
        )
        self.pose = nn.Linear(hidden, 6 * chunk)
        self.grip_logits = nn.Linear(hidden, chunk)

    def forward(self, x):
        h = self.trunk(x)
        return self.pose(h), self.grip_logits(h)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    Xtr, Ytr, Xva, Yva = build(args)
    print(f"[oracle] train={tuple(Xtr.shape)} val={tuple(Xva.shape)}", flush=True)
    chunk = args.chunk
    model = OracleHead(Xtr.shape[1], chunk).to(args.device)
    def split(Y):
        Y = Y.reshape(-1, chunk, 7)
        pose = torch.cat([Y[:, :, :6].reshape(len(Y), -1)], dim=1)
        grip = (Y[:, :, 6] > 0).float()
        return pose, grip
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    for epoch in range(args.epochs):
        model.train()
        perm = torch.randperm(len(Xtr))
        tot, nb = 0.0, 0
        for i in range(0, len(Xtr), args.batch_size):
            idx = perm[i:i + args.batch_size]
            xb = Xtr[idx].to(args.device)
            p_t, g_t = split(Ytr[idx].to(args.device))
            p_p, g_l = model(xb)
            loss = torch.nn.functional.mse_loss(p_p, p_t) + 0.5 * torch.nn.functional.binary_cross_entropy_with_logits(g_l, g_t)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item(); nb += 1
        sched.step()
        if epoch % 10 == 0 or epoch == args.epochs - 1:
            model.eval()
            with torch.no_grad():
                p_p, g_l = model(Xva.to(args.device))
                p_t, g_t = split(Yva.to(args.device))
                mae = (p_p - p_t).abs().mean().item() * 1000
                acc = ((g_l > 0).float() == g_t).float().mean().item() * 100
            print(f"[oracle] epoch {epoch} train {tot/nb:.5f} val_pos_mae_mm {mae:.2f} grip_acc {acc:.1f}", flush=True)
            torch.save({"state_dict": model.state_dict(), "chunk": args.chunk, "in_dim": int(Xtr.shape[1])},
                       os.path.join(args.out, "head.pt"))
    print("[oracle] saved", args.out)


if __name__ == "__main__":
    main()

