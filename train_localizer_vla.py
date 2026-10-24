"""VLA-grounded perception: OpenVLA vision backbone (+LoRA) -> object heatmaps.

This keeps the project on the VLA track: instead of a stand-alone CNN, the
perception module is the OpenVLA vision encoder lightly fine-tuned (LoRA) to
localise the objects, and a small set of motion primitives executes the grasp and
place. The end-to-end action head remains as the baseline for comparison.

Outputs 4 heatmaps (object centre, two ends of its local x-axis, target), which
are decoded to world coordinates with the table-plane homography.
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
    ap.add_argument("--base-model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--task", default="widowx_spoon_on_towel")
    ap.add_argument("--run", default="pose")
    ap.add_argument("--camera", default="configs/camera_t1.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--head-lr", type=float, default=1e-3)
    ap.add_argument("--lora", action="store_true")
    ap.add_argument("--lora-rank", type=int, default=8)
    ap.add_argument("--frames-per-ep", type=int, default=4)
    ap.add_argument("--val-frac", type=float, default=0.12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")
    return ap.parse_args()


def project(world_xyz, K, Ecv):
    p4 = np.array([world_xyz[0], world_xyz[1], world_xyz[2], 1.0], dtype=np.float64)
    uv = K @ (Ecv @ p4)[:3]
    return uv[:2] / uv[2]


def gaussian_heat(px, py, size, sigma=2.0):
    ys, xs = np.mgrid[0:size, 0:size]
    return np.exp(-((xs - px) ** 2 + (ys - py) ** 2) / (2 * sigma ** 2)).astype(np.float32)


class HeatDecoder(nn.Module):
    """16x16 VLA token grid -> upsampled heatmaps."""

    def __init__(self, in_dim=2176, ch=256, heat=56):
        super().__init__()
        self.heat = heat
        self.proj = nn.Sequential(nn.Conv2d(in_dim, ch, 1), nn.GroupNorm(8, ch), nn.ReLU())
        self.up = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1), nn.GroupNorm(8, ch), nn.ReLU(),
            nn.Upsample(scale_factor=4, mode="bilinear", align_corners=False),
            nn.Conv2d(ch, 128, 3, padding=1), nn.GroupNorm(8, 128), nn.ReLU(),
            nn.Conv2d(128, 4, 1),
        )

    def forward(self, tokens):
        # tokens: (B, 256, 2176)
        b, n, c = tokens.shape
        side = int(round(n ** 0.5))
        x = tokens.reshape(b, side, side, c).permute(0, 3, 1, 2)
        return self.up(self.proj(x))


def load_items(args, K, Ecv):
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
        dst = np.array(p["dst"], dtype=np.float64)
        yaw = p["src_yaw"]
        axis = np.array([np.cos(yaw), np.sin(yaw), 0.0])
        pts = [project(src, K, Ecv), project(src + 0.11 * axis, K, Ecv),
               project(src - 0.11 * axis, K, Ecv), project(dst, K, Ecv)]
        for i in np.arange(min(args.frames_per_ep, len(frames))):
            items.append((os.path.join(d, "frames", frames[i]), pts, src, dst))
    rng = random.Random(args.seed)
    rng.shuffle(items)
    n_val = max(1, int(len(items) * args.val_frac))
    return items[n_val:], items[:n_val]


class HeatDS(torch.utils.data.Dataset):
    def __init__(self, items, transform, heat):
        self.items = items
        self.transform = transform
        self.heat = heat

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, pts, _src, _dst = self.items[i]
        img = Image.open(path).convert("RGB")
        x = self.transform(img).float() / 255.0
        hm = np.stack([gaussian_heat(px * self.heat / W, py * self.heat / H, self.heat) for (px, py) in pts])
        return x, torch.from_numpy(hm)


def fit_homography(items):
    A, b = [], []
    for _path, pts, src, _dst in items:
        px, py = pts[0]
        X, Y = src[0], src[1]
        A.append([px, py, 1, 0, 0, 0, -X * px, -X * py])
        A.append([0, 0, 0, px, py, 1, -Y * px, -Y * py])
        b.extend([X, Y])
    h, *_ = np.linalg.lstsq(np.asarray(A), np.asarray(b), rcond=None)
    return np.array([[h[0], h[1], h[2]], [h[3], h[4], h[5]], [h[6], h[7], 1.0]])


def soft_argmax(heat, W=W, H=H):
    """Sub-pixel decode: expectation of the pixel coordinates under the heatmap."""
    h = heat.clamp(min=0)
    s = h.sum()
    if float(s) <= 1e-8:
        return None
    hh, ww = h.shape
    ys = torch.arange(hh, dtype=torch.float32, device=h.device).view(-1, 1)
    xs = torch.arange(ww, dtype=torch.float32, device=h.device).view(1, -1)
    py = float((h * ys).sum() / s) * H / hh
    px = float((h * xs).sum() / s) * W / ww
    return px, py


def apply_h(Hm, px, py):
    v = Hm @ np.array([px, py, 1.0])
    return v[:2] / v[2]


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    os.makedirs(args.out, exist_ok=True)

    from transformers import AutoModelForVision2Seq, AutoProcessor

    cam = json.load(open(args.camera, encoding="utf-8"))
    K = np.array(cam["K"])
    F = np.diag([1.0, -1.0, -1.0, 1.0])
    Ecv = F @ np.linalg.inv(np.array(cam["cam2world_gl"]))

    base = AutoModelForVision2Seq.from_pretrained(
        args.base_model, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
        trust_remote_code=True, attn_implementation="sdpa",
    ).to(args.device).eval()
    vision = base.vision_backbone
    for p in base.parameters():
        p.requires_grad = False
    del base
    if args.lora:
        from peft import LoraConfig, get_peft_model
        vision = get_peft_model(vision, LoraConfig(r=args.lora_rank, lora_alpha=args.lora_rank,
                                                   lora_dropout=0.0,
                                                   target_modules=["qkv", "proj", "fc1", "fc2"], bias="none"))
        vision.print_trainable_parameters()

    processor = AutoProcessor.from_pretrained(args.base_model, trust_remote_code=True)
    tf = processor.image_processor.apply_transform
    tr, va = load_items(args, K, Ecv)
    print(f"[vloc] train={len(tr)} val={len(va)}", flush=True)
    dl_tr = torch.utils.data.DataLoader(HeatDS(tr, tf, 56), batch_size=args.batch_size, shuffle=True,
                                        num_workers=6, drop_last=True)
    dl_va = torch.utils.data.DataLoader(HeatDS(va, tf, 56), batch_size=args.batch_size, shuffle=False,
                                        num_workers=3)
    dec = HeatDecoder(2176, 256, 56).to(args.device)
    params = [{"params": dec.parameters(), "lr": args.head_lr}]
    if args.lora:
        params.append({"params": [p for p in vision.parameters() if p.requires_grad], "lr": args.lr})
    opt = torch.optim.AdamW(params, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    Hm = fit_homography(tr)
    best = 1e9
    for epoch in range(args.epochs):
        dec.train()
        tot, nb = 0.0, 0
        for x, hm in dl_tr:
            x = x.to(args.device, dtype=torch.bfloat16)
            hm = hm.to(args.device)
            with torch.set_grad_enabled(True):
                tok = vision(x).float()
                out = dec(tok)
                # downsample target to the decoder's resolution
                tgt = torch.nn.functional.interpolate(hm, size=out.shape[-2:], mode="bilinear", align_corners=False)
                loss = torch.nn.functional.mse_loss(out, tgt)
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(dec.parameters(), 1.0)
                opt.step()
            tot += loss.item(); nb += 1
        sched.step()
        dec.eval()
        errs = []
        idx = 0
        with torch.no_grad():
            for x, hm in dl_va:
                tok = vision(x.to(args.device, dtype=torch.bfloat16)).float()
                out = dec(tok)
                b, c, hh, ww = out.shape
                for i in range(b):
                    px, py = soft_argmax(out[i, 0])
                    wx, wy = apply_h(Hm, px, py)
                    gt = va[idx][2]
                    errs.append(float(np.hypot(wx - gt[0], wy - gt[1])) * 1000)
                    idx += 1
        err = float(np.mean(errs)) if errs else -1
        print(f"[vloc] epoch {epoch} train {tot/max(1,nb):.5f} val_xy_mm {err:.2f}", flush=True)
        if err < best:
            best = err
            torch.save({"decoder": dec.state_dict(), "Hm": Hm, "lora": args.lora,
                        "src_z": float(np.mean([it[2][2] for it in tr])),
                        "dst_z": float(np.mean([it[3][2] for it in tr]))},
                       os.path.join(args.out, "vla_localizer.pt"))
            if args.lora:
                vision.save_pretrained(os.path.join(args.out, "vision_lora"))
    print("[vloc] saved", args.out, "best", best)


if __name__ == "__main__":
    main()

