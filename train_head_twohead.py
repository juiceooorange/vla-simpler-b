"""Two-headed VLA head: continuous pose regression + discrete gripper classifier.

Reuses the feature cache produced by train_head_chunk.py (no re-extraction).
"""
import argparse, json, os
import torch
from torch import nn


class TwoHead(nn.Module):
    def __init__(self, in_dim, chunk, hidden=512):
        super().__init__()
        self.chunk = chunk
        self.trunk = nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, hidden), nn.GELU(),
                                   nn.Linear(hidden, hidden), nn.GELU())
        self.pose = nn.Linear(hidden, 6 * chunk)
        self.grip = nn.Linear(hidden, chunk)

    def forward(self, x):
        h = self.trunk(x)
        return self.pose(h), self.grip(h)


def split(Y, chunk):
    Y = Y.reshape(-1, chunk, 7)
    return Y[:, :, :6].reshape(len(Y), -1), (Y[:, :, 6] > 0).float()


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
    tr = torch.load(os.path.join(args.cache_dir, "train_features.pt"), map_location="cpu", weights_only=False)
    va = torch.load(os.path.join(args.cache_dir, "val_features.pt"), map_location="cpu", weights_only=False)
    chunk = int(tr["Y"].shape[1] // 7)
    Xtr, Ytr = tr["X"].float(), tr["Y"]
    Xva, Yva = va["X"].float(), va["Y"]
    model = TwoHead(Xtr.shape[1], chunk).to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    best = 1e9
    for epoch in range(args.epochs):
        model.train()
        perm = torch.randperm(len(Xtr))
        tot = 0.0
        for i in range(0, len(Xtr), args.batch_size):
            idx = perm[i:i + args.batch_size]
            p_t, g_t = split(Ytr[idx].to(args.device), chunk)
            p_p, g_l = model(Xtr[idx].to(args.device))
            loss = torch.nn.functional.mse_loss(p_p, p_t) + 0.5 * torch.nn.functional.binary_cross_entropy_with_logits(g_l, g_t)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(idx)
        sched.step()
        model.eval()
        with torch.no_grad():
            p_p, g_l = model(Xva.to(args.device))
            p_t, g_t = split(Yva.to(args.device), chunk)
            mae = (p_p - p_t).abs().mean().item() * 1000
            acc = ((g_l > 0).float() == g_t).float().mean().item() * 100
            vmse = torch.nn.functional.mse_loss(p_p, p_t).item()
        if epoch % 10 == 0 or epoch == args.epochs - 1:
            print(f"[2h] epoch {epoch} pos_mae_mm {mae:.2f} grip_acc {acc:.1f}", flush=True)
        if vmse < best:
            best = vmse
            torch.save({"state_dict": model.state_dict(), "chunk": chunk, "in_dim": int(Xtr.shape[1])},
                       os.path.join(args.out, "head.pt"))
    with open(os.path.join(args.out, "head_config.json"), "w") as fh:
        json.dump({"arch": "twohead", "chunk": chunk, "proprio": True, "flip_mode": "none"}, fh)
    print("[2h] saved", args.out)


if __name__ == "__main__":
    main()
