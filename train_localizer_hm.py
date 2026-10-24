"""Heatmap localizer: finds the spoon in the image, then maps pixels to world
coordinates with the table-plane homography.

The coordinate-regression version memorised the 14 training placements (109 mm
error on unseen test placements). A detection-style model generalises because it
only has to locate the object in image space; the pixel->world mapping is pure
geometry (the object lies on the table plane).

Outputs three heatmaps: object centre, and the two ends of the object's local
x-axis (used to recover the grasp yaw).
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
    ap.add_argument("--image-size", type=int, default=384)
    ap.add_argument("--heat", type=int, default=96)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=32)
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


class HeatLocalizer(nn.Module):
    def __init__(self, out_ch=3):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Conv2d(3, 32, 5, stride=2, padding=2), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=1, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 128, 3, stride=1, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, out_ch, 1),
        )

    def forward(self, x):
        return self.head(self.trunk(x))


def gaussian_heat(px, py, size, sigma=2.0):
    ys, xs = np.mgrid[0:size, 0:size]
    g = np.exp(-((xs - px) ** 2 + (ys - py) ** 2) / (2 * sigma ** 2))
    return g.astype(np.float32)


def load_samples(args, K, Ecv):
    items = []
    roots = [os.path.join(args.data, args.task, r) for r in args.run.split(",") if r]
    eps = []
    for root in roots:
        if os.path.isdir(root):
            eps.extend((root, e) for e in sorted(os.listdir(root)))
    for root, ep in eps:
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
        src = np.array(p["src"], dtype=np.float64)
        yaw = p["src_yaw"]
        axis = np.array([np.cos(yaw), np.sin(yaw), 0.0])
        e1 = src + 0.11 * axis
        e2 = src - 0.11 * axis
        dst = np.array(p["dst"], dtype=np.float64)
        pts = [project(src, K, Ecv), project(e1, K, Ecv), project(e2, K, Ecv), project(dst, K, Ecv)]
        sel = np.arange(min(args.frames_per_ep, len(frames)))
        for i in sel:
            items.append((os.path.join(d, "frames", frames[i]), pts, src, dst, yaw))
    rng = random.Random(args.seed)
    rng.shuffle(items)
    n_val = max(1, int(len(items) * args.val_frac))
    return items[n_val:], items[:n_val]


class HeatDataset(torch.utils.data.Dataset):
    def __init__(self, items, image_size, heat):
        self.items = items
        self.image_size = image_size
        self.heat = heat

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, pts, _src, _dst, _yaw = self.items[i]
        img = Image.open(path).convert("RGB").resize((self.image_size, self.image_size), Image.BILINEAR)
        x = torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0).permute(2, 0, 1)
        hm = np.stack([gaussian_heat(px * self.heat / W, py * self.heat / H, self.heat) for (px, py) in pts])
        return x, torch.from_numpy(hm)


def fit_homography(items):
    """pixel (640x480) -> world xy, least squares over the training projections."""
    # use centre point of each sample
    A, b = [], []
    for _path, pts, src, _dst, _yaw in items:
        px, py = pts[0]
        for (X, Y) in [src[:2]]:
            A.append([px, py, 1, 0, 0, 0, -X * px, -X * py])
            A.append([0, 0, 0, px, py, 1, -Y * px, -Y * py])
            b.extend([X, Y])
    A = np.asarray(A); b = np.asarray(b)
    h, *_ = np.linalg.lstsq(A, b, rcond=None)
    Hm = np.array([[h[0], h[1], h[2]], [h[3], h[4], h[5]], [h[6], h[7], 1.0]])
    return Hm


def apply_h(Hm, px, py):
    v = Hm @ np.array([px, py, 1.0])
    return v[:2] / v[2]


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    cam = json.load(open(args.camera, encoding="utf-8"))
    K = np.array(cam["K"]); Ecv = np.array(cam["cam2world_gl"])
    # cv convention extrinsic: derive from cam2world_gl by flipping y/z of the camera frame
    F = np.diag([1.0, -1.0, -1.0, 1.0])
    Ecv = F @ np.linalg.inv(np.array(cam["cam2world_gl"]))

    tr, va = load_samples(args, K, Ecv)
    print(f"[hm] train={len(tr)} val={len(va)}", flush=True)
    dl_tr = torch.utils.data.DataLoader(HeatDataset(tr, args.image_size, args.heat), batch_size=args.batch_size,
                                        shuffle=True, num_workers=6, drop_last=True)
    dl_va = torch.utils.data.DataLoader(HeatDataset(va, args.image_size, args.heat), batch_size=args.batch_size,
                                        shuffle=False, num_workers=3)
    model = HeatLocalizer(4).to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    Hm = fit_homography(tr)
    best = 1e9
    for epoch in range(args.epochs):
        model.train()
        tot, nb = 0.0, 0
        for x, hm in dl_tr:
            x, hm = x.to(args.device), hm.to(args.device)
            loss = torch.nn.functional.mse_loss(model(x), hm)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item(); nb += 1
        sched.step()
        model.eval()
        errs = []
        with torch.no_grad():
            idx = 0
            for x, hm in dl_va:
                out = model(x.to(args.device))
                b, c, hh, ww = out.shape
                flat = out.reshape(b, c, -1).argmax(-1)
                px = (flat % ww).float() * W / ww
                py = (flat // ww).float() * H / hh
                for i in range(b):
                    gt = va[idx][2]
                    wx, wy = apply_h(Hm, float(px[i][0]), float(py[i][0]))
                    errs.append(float(np.hypot(wx - gt[0], wy - gt[1])) * 1000)
                    idx += 1
        err_mm = float(np.mean(errs)) if errs else -1.0
        msg = {"epoch": epoch, "train": round(tot / max(1, nb), 6), "val_xy_mm": round(err_mm, 2)}
        print("[hm]", msg, flush=True)
        if err_mm < best:
            best = err_mm
            _sz = float(np.mean([it[2][2] for it in tr]))
            _dz = float(np.mean([it[3][2] for it in tr]))
            torch.save({"state_dict": model.state_dict(), "Hm": Hm, "image_size": args.image_size,
                        "heat": args.heat, "src_z": _sz, "dst_z": _dz}, os.path.join(args.out, "localizer_hm.pt"))
    print("[hm] saved", args.out, "best val mm", best)


if __name__ == "__main__":
    main()



def decode_batch(out, Hm, W=W, H=H):
    """out: (B, 4, h, w) heatmaps -> (B, 8): src xyz, dst xyz, sin(yaw), cos(yaw)."""
    import numpy as _np
    b, c, hh, ww = out.shape
    feats = []
    for i in range(b):
        peaks = []
        for ch in range(c):
            flat = int(out[i, ch].reshape(-1).argmax())
            py = (flat // ww) * H / ww
            px = (flat % ww) * W / hh
            peaks.append(apply_h(Hm, px, py))
        src = _np.array([peaks[0][0], peaks[0][1], 0.881])
        dst = _np.array([peaks[3][0], peaks[3][1], 0.874])
        v = _np.asarray(peaks[1]) - _np.asarray(peaks[2])
        yaw = float(_np.arctan2(v[1], v[0]))
        feats.append(_np.concatenate([src, dst, [np.sin(yaw), np.cos(yaw)]]))
    return _np.stack(feats).astype(_np.float32)
