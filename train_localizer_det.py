"""Sliding-window object detector on the VLA vision encoder.

Motivation: the two-stage policy's accuracy was capped by the stage-1 coarse
localiser (~5 cm on unseen placements) because it chose the crop window. Here the
window choice is removed entirely: the model is trained on *randomly cropped*
windows (positive and negative) and predicts, for each window,
  - 4 heatmaps (object centre, two ends of its axis, target) in crop coordinates
  - a presence logit (is the object inside this crop?)
At inference we slide the window over a grid, keep confident windows, and fuse
their world-space predictions. No coarse prior is needed.

Usage:
    python train_localizer_det.py --base-model ../models/sft-simpler --data ../data/rollouts \
        --task widowx_spoon_on_towel --run pose,pose2 --camera configs/camera_t1.json \
        --out ../runs/det_t1 --crop 192 --epochs 25 --lora
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
    ap.add_argument("--crop", type=int, default=192)
    ap.add_argument("--heat", type=int, default=56)
    ap.add_argument("--neg-ratio", type=float, default=0.5)
    ap.add_argument("--jitter-px", type=float, default=70.0)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--frames-per-ep", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=12)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--head-lr", type=float, default=1e-3)
    ap.add_argument("--lora", action="store_true")
    ap.add_argument("--lora-rank", type=int, default=8)
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


class Detector(nn.Module):
    def __init__(self, in_dim=2176, ch=256, heat=56, out_ch=4):
        super().__init__()
        self.heat = heat
        self.proj = nn.Sequential(nn.Conv2d(in_dim, ch, 1), nn.GroupNorm(8, ch), nn.ReLU())
        self.up = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1), nn.GroupNorm(8, ch), nn.ReLU(),
            nn.Upsample(scale_factor=4, mode="bilinear", align_corners=False),
            nn.Conv2d(ch, 128, 3, padding=1), nn.GroupNorm(8, 128), nn.ReLU(),
        )
        self.heat_head = nn.Conv2d(128, out_ch, 1)
        self.pres_head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                                       nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))
        self.yaw_head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                                      nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 2))

    def forward(self, tokens):
        b, n, c = tokens.shape
        side = int(round(n ** 0.5))
        x = tokens.reshape(b, side, side, c).permute(0, 3, 1, 2)
        h = self.up(self.proj(x))
        return self.heat_head(h), self.pres_head(h).squeeze(-1), self.yaw_head(h)


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
        src = np.array(p["src"], dtype=np.float64)
        dst = np.array(p["dst"], dtype=np.float64)
        yaw = p["src_yaw"]
        axis = np.array([np.cos(yaw), np.sin(yaw), 0.0])
        pts = [project(src, K, Ecv), project(src + 0.11 * axis, K, Ecv),
               project(src - 0.11 * axis, K, Ecv), project(dst, K, Ecv)]
        for i in np.arange(min(args.frames_per_ep, len(frames))):
            items.append((os.path.join(d, "frames", frames[i]), pts, src, dst, yaw))
    rng = random.Random(args.seed)
    rng.shuffle(items)
    n_val = max(1, int(len(items) * args.val_frac))
    return items[n_val:], items[:n_val]


class DetDS(torch.utils.data.Dataset):
    def __init__(self, items, transform, args):
        self.items = items
        self.transform = transform
        self.a = args

    def __len__(self):
        return len(self.items)

    def _crop_at(self, img, left, top):
        c = self.a.crop
        crop = img.crop((left, top, left + c, top + c)).resize((224, 224), Image.BILINEAR)
        return self.transform(crop).float() / 255.0

    def __getitem__(self, i):
        path, pts, src, dst, yaw = self.items[i]
        img = Image.open(path).convert("RGB")
        c = self.a.crop
        sx, sy = pts[0]
        # positive crop: object roughly inside, with jitter
        if random.random() > self.a.neg_ratio:
            # rejection: the object must actually be inside the crop, otherwise the
            # heatmap target is empty and the model collapses to predicting zeros
            left = top = None
            for _try in range(12):
                _l = int(np.clip(sx - c / 2 + random.uniform(-self.a.jitter_px, self.a.jitter_px), 0, W - c))
                _t = int(np.clip(sy - c / 2 + random.uniform(-self.a.jitter_px, self.a.jitter_px), 0, H - c))
                if _l + 8 <= sx <= _l + c - 8 and _t + 8 <= sy <= _t + c - 8:
                    left, top = _l, _t
                    break
            if left is None:
                left = int(np.clip(sx - c / 2, 0, W - c)); top = int(np.clip(sy - c / 2, 0, H - c))
            present = 1.0
            hm = np.stack([gaussian_heat(px * self.a.heat / c, py * self.a.heat / c, self.a.heat)
                           for (px, py) in [(p[0] - left, p[1] - top) for p in pts]])
        else:
            left, top = 0, 0
            for _ in range(6):
                left = random.randint(0, W - c)
                top = random.randint(0, H - c)
                if not (left - 20 <= sx <= left + c + 20 and top - 20 <= sy <= top + c + 20):
                    break
            present = 0.0
            hm = np.zeros((4, self.a.heat, self.a.heat), dtype=np.float32)
        yaw2 = torch.tensor([np.sin(2 * yaw), np.cos(2 * yaw)], dtype=torch.float32)
        return self._crop_at(img, left, top), torch.from_numpy(hm), torch.tensor(present, dtype=torch.float32), yaw2


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
        vision = get_peft_model(vision, LoraConfig(r=args.lora_rank, lora_alpha=args.lora_rank, lora_dropout=0.0,
                                                   target_modules=["qkv", "proj", "fc1", "fc2"], bias="none"))
        vision.print_trainable_parameters()

    processor = AutoProcessor.from_pretrained(args.base_model, trust_remote_code=True)
    tf = processor.image_processor.apply_transform
    tr, va = load_items(args, K, Ecv)
    print(f"[det] train={len(tr)} val={len(va)}", flush=True)
    dl_tr = torch.utils.data.DataLoader(DetDS(tr, tf, args), batch_size=args.batch_size, shuffle=True,
                                        num_workers=6, drop_last=True)
    dl_va = torch.utils.data.DataLoader(DetDS(va, tf, args), batch_size=args.batch_size, shuffle=False,
                                        num_workers=3)
    dec = Detector(2176, 256, args.heat, 4).to(args.device)
    params = [{"params": dec.parameters(), "lr": args.head_lr}]
    if args.lora:
        params.append({"params": [p for p in vision.parameters() if p.requires_grad], "lr": args.lr})
    opt = torch.optim.AdamW(params, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    best = 1e9
    for epoch in range(args.epochs):
        dec.train()
        tot, nb = 0.0, 0
        for x, hm, pres, yaw2 in dl_tr:
            x = x.to(args.device, dtype=torch.bfloat16)
            hm = hm.to(args.device)
            pres = pres.to(args.device)
            yaw2 = yaw2.to(args.device)
            with torch.set_grad_enabled(True):
                out = dec(vision(x).float())
                hh, pl, yl = out
                tgt = torch.nn.functional.interpolate(hm, size=hh.shape[-2:], mode="bilinear", align_corners=False)
                loss = (torch.nn.functional.mse_loss(hh, tgt)
                        + 0.5 * torch.nn.functional.binary_cross_entropy_with_logits(pl, pres)
                        + 0.5 * torch.nn.functional.mse_loss(yl, yaw2.to(args.device)))
                opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss); nb += 1
        sched.step()
        dec.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x, hm, pres, yaw2 in dl_va:
                _hh, pl, _yl = dec(vision(x.to(args.device, dtype=torch.bfloat16)).float())
                correct += int(((pl > 0).float() == pres.to(args.device)).sum())
                total += len(pres)
        acc = 100.0 * correct / max(1, total)
        print(f"[det] epoch {epoch} loss {tot/max(1,nb):.5f} presence_acc {acc:.1f}", flush=True)
        if tot / max(1, nb) < best:
            best = tot / max(1, nb)
            torch.save({"decoder": dec.state_dict(), "lora": args.lora, "crop": args.crop,
                        "heat": args.heat}, os.path.join(args.out, "detector.pt"))
            if args.lora:
                vision.save_pretrained(os.path.join(args.out, "vision_lora"))
    print("[det] saved", args.out, "best", best)


if __name__ == "__main__":
    main()

