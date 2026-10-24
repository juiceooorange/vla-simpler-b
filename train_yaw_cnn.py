"""Small CNN that regresses the object's grasp yaw from a centred, zoomed crop.

The VLA-side yaw head collapsed to a constant (its pooled features are dominated
by position), while a 160 px crop centred on the detected object shows the spoon's
shape clearly. Predicting (sin 2胃, cos 2胃) removes the 180掳 ambiguity of a
parallel-jaw gripper.
"""

from __future__ import annotations

import argparse
import json
import os
import random

import numpy as np
import torch
from PIL import Image
from torch import nn

H, W = 480, 640


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--task", default="widowx_spoon_on_towel")
    ap.add_argument("--run", default="pose")
    ap.add_argument("--camera", default="configs/camera_t1.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--crop", type=int, default=160)
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--jitter-px", type=float, default=35.0)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--frames-per-ep", type=int, default=4)
    ap.add_argument("--val-frac", type=float, default=0.12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")
    return ap.parse_args()


def project(world_xyz, K, Ecv):
    p4 = np.array([world_xyz[0], world_xyz[1], world_xyz[2], 1.0], dtype=np.float64)
    uv = K @ (Ecv @ p4)[:3]
    return uv[:2] / uv[2]


class YawNet(nn.Module):
    def __init__(self, size=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 5, stride=2, padding=2), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 128, 3, stride=2, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 2),
        )

    def forward(self, x):
        return self.net(x)


def load_items(args, K, Ecv):
    items = []
    roots = [os.path.join(args.data, args.task, r) for r in args.run.split(",") if r]
    entries = []
    for root in roots:
        if os.path.isdir(root):
            entries.extend((root, e) for e in sorted(os.listdir(root)))
    for root, ep in entries:
        d = os.path.join(root, ep)
        mp = os.path.join(d, "meta.json")
        if not os.path.isfile(mp):
            continue
        meta = json.load(open(mp, encoding="utf-8"))
        if not meta.get("success") or "poses" not in meta:
            continue
        frames = sorted(os.listdir(os.path.join(d, "frames")))
        if not frames:
            continue
        p = meta["poses"]
        px = project(p["src"], K, Ecv)
        for i in np.arange(min(args.frames_per_ep, len(frames))):
            items.append((os.path.join(d, "frames", frames[i]), px, p["src_yaw"]))
    rng = random.Random(args.seed)
    rng.shuffle(items)
    n_val = max(1, int(len(items) * args.val_frac))
    return items[n_val:], items[:n_val]


class YawDS(torch.utils.data.Dataset):
    def __init__(self, items, args):
        self.items = items
        self.a = args

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, px, yaw = self.items[i]
        img = Image.open(path).convert("RGB")
        c = self.a.crop
        left = int(np.clip(px[0] - c / 2 + random.uniform(-self.a.jitter_px, self.a.jitter_px), 0, W - c))
        top = int(np.clip(px[1] - c / 2 + random.uniform(-self.a.jitter_px, self.a.jitter_px), 0, H - c))
        crop = img.crop((left, top, left + c, top + c)).resize((self.a.size, self.a.size), Image.BILINEAR)
        x = torch.from_numpy(np.asarray(crop, dtype=np.float32) / 255.0).permute(2, 0, 1)
        y = torch.tensor([np.sin(2 * yaw), np.cos(2 * yaw)], dtype=torch.float32)
        return x, y


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    cam = json.load(open(args.camera, encoding="utf-8"))
    K = np.array(cam["K"])
    F = np.diag([1.0, -1.0, -1.0, 1.0])
    Ecv = F @ np.linalg.inv(np.array(cam["cam2world_gl"]))
    tr, va = load_items(args, K, Ecv)
    print(f"[yaw] train={len(tr)} val={len(va)}", flush=True)
    dl_tr = torch.utils.data.DataLoader(YawDS(tr, args), batch_size=args.batch_size, shuffle=True,
                                        num_workers=8, drop_last=True)
    dl_va = torch.utils.data.DataLoader(YawDS(va, args), batch_size=args.batch_size, shuffle=False, num_workers=4)
    model = YawNet(args.size).to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    best = 1e9
    for epoch in range(args.epochs):
        model.train()
        tot, nb = 0.0, 0
        for x, y in dl_tr:
            x, y = x.to(args.device), y.to(args.device)
            loss = torch.nn.functional.mse_loss(model(x), y)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss); nb += 1
        sched.step()
        model.eval()
        errs = []
        with torch.no_grad():
            for x, y in dl_va:
                y = y.to(args.device)
                pred = model(x.to(args.device))
                a = torch.atan2(pred[:, 0], pred[:, 1]) / 2
                b = torch.atan2(y[:, 0], y[:, 1]) / 2
                d = (a - b + np.pi / 2) % np.pi - np.pi / 2
                errs.extend(np.degrees(np.abs(d.cpu().numpy())).tolist())
        err = float(np.mean(errs))
        if epoch % 5 == 0 or epoch == args.epochs - 1:
            print(f"[yaw] epoch {epoch} loss {tot/max(1,nb):.5f} val_yaw_deg {err:.1f}", flush=True)
        if err < best:
            best = err
            torch.save({"state_dict": model.state_dict(), "crop": args.crop, "size": args.size},
                       os.path.join(args.out, "yawnet.pt"))
    print("[yaw] saved", args.out, "best deg", best)


if __name__ == "__main__":
    main()

