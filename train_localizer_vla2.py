"""Two-stage (foveated) VLA localizer.

Stage 1 runs the VLA localizer on the full image (coarse, ~5 cm on unseen poses).
Its prediction centres a small crop; stage 2 runs the same VLA backbone on the
zoomed crop, where each 16x16 token covers ~5 mm instead of ~2 cm, so the fine
stage can resolve the object precisely. The crop centre is known, so stage 2
regresses the absolute world position.

Crop jitter during training matches the stage-1 error distribution at test time.
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
    ap.add_argument("--stage1", required=True, help="dir of the stage-1 VLA localizer")
    ap.add_argument("--data", required=True)
    ap.add_argument("--task", default="widowx_spoon_on_towel")
    ap.add_argument("--run", default="pose")
    ap.add_argument("--camera", default="configs/camera_t1.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--crop", type=int, default=160)
    ap.add_argument("--jitter-m", type=float, default=0.06)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--head-lr", type=float, default=1e-3)
    ap.add_argument("--lora", action="store_true")
    ap.add_argument("--lora-rank", type=int, default=8)
    ap.add_argument("--frames-per-ep", type=int, default=3)
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


def soft_argmax_t(heat):
    """heat: (B, h, w) -> (nx, ny) each (B,) in [0,1]."""
    h = heat.clamp(min=0)
    b, hh, ww = h.shape
    s = h.sum(dim=(1, 2)) + 1e-8
    ys = torch.arange(hh, dtype=torch.float32, device=h.device).view(1, -1, 1)
    xs = torch.arange(ww, dtype=torch.float32, device=h.device).view(1, 1, -1)
    nx = (h * xs).sum(dim=(1, 2)) / s / ww
    ny = (h * ys).sum(dim=(1, 2)) / s / hh
    return nx, ny


def soft_argmax(heat):
    """Normalised (nx, ny) in [0,1] over the input crop."""
    h = heat.clamp(min=0)
    s = h.sum()
    if float(s) <= 1e-8:
        return None
    hh, ww = h.shape
    ys = torch.arange(hh, dtype=torch.float32, device=h.device).view(-1, 1)
    xs = torch.arange(ww, dtype=torch.float32, device=h.device).view(1, -1)
    return float((h * xs).sum() / s) / ww, float((h * ys).sum() / s) / hh


def apply_h(Hm, px, py):
    v = Hm @ np.array([px, py, 1.0])
    return v[:2] / v[2]


class HeatDecoder(nn.Module):
    def __init__(self, in_dim=2176, ch=256, heat=56, out_ch=2):
        super().__init__()
        self.heat = heat
        self.proj = nn.Sequential(nn.Conv2d(in_dim, ch, 1), nn.GroupNorm(8, ch), nn.ReLU())
        self.up = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1), nn.GroupNorm(8, ch), nn.ReLU(),
            nn.Upsample(scale_factor=4, mode="bilinear", align_corners=False),
            nn.Conv2d(ch, 128, 3, padding=1), nn.GroupNorm(8, 128), nn.ReLU(),
            nn.Conv2d(128, out_ch, 1),
        )

    def forward(self, tokens):
        b, n, c = tokens.shape
        side = int(round(n ** 0.5))
        x = tokens.reshape(b, side, side, c).permute(0, 3, 1, 2)
        return self.up(self.proj(x))


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    os.makedirs(args.out, exist_ok=True)

    from transformers import AutoModelForVision2Seq, AutoProcessor
    from train_localizer_vla import soft_argmax as sa1

    cam = json.load(open(args.camera, encoding="utf-8"))
    K = np.array(cam["K"])
    F = np.diag([1.0, -1.0, -1.0, 1.0])
    Ecv = F @ np.linalg.inv(np.array(cam["cam2world_gl"]))

    ck1 = torch.load(os.path.join(args.stage1, "vla_localizer.pt"), map_location="cpu", weights_only=False)
    dec1 = HeatDecoder(2176, 256, 56, 4).to(args.device).float().eval()
    dec1.load_state_dict(ck1["decoder"])
    Hm1 = ck1["Hm"]

    base = AutoModelForVision2Seq.from_pretrained(
        args.base_model, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
        trust_remote_code=True, attn_implementation="sdpa",
    ).to(args.device).eval()
    vision = base.vision_backbone
    for p in base.parameters():
        p.requires_grad = False
    del base
    if ck1.get("lora"):
        from peft import PeftModel
        vision = PeftModel.from_pretrained(vision, os.path.join(args.stage1, "vision_lora")).merge_and_unload()
    vision = vision.to(args.device).eval()
    processor = AutoProcessor.from_pretrained(args.base_model, trust_remote_code=True)
    tf = processor.image_processor.apply_transform

    if args.lora:
        from peft import LoraConfig, get_peft_model
        vision = get_peft_model(vision, LoraConfig(r=args.lora_rank, lora_alpha=args.lora_rank, lora_dropout=0.0,
                                                   target_modules=["qkv", "proj", "fc1", "fc2"], bias="none"))
        vision.print_trainable_parameters()

    # collect (frame, coarse centre, gt src, gt dst, yaw)
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
        for i in np.arange(min(args.frames_per_ep, len(frames))):
            items.append((os.path.join(d, "frames", frames[i]), np.array(p["src"]), np.array(p["dst"]), p["src_yaw"]))
    rng = random.Random(args.seed)
    rng.shuffle(items)
    n_val = max(1, int(len(items) * args.val_frac))
    va_items, tr_items = items[:n_val], items[n_val:]

    # stage-1 predictions (cached)
    cache = os.path.join(args.out, "coarse.pt")
    coarse = {}
    if os.path.isfile(cache):
        coarse = torch.load(cache, map_location="cpu", weights_only=False)
    for path, src, dst, yaw in items:
        if path in coarse:
            continue
        img = Image.open(path).convert("RGB")
        x = tf(img).float().div(255.0)[None].cuda().bfloat16()
        with torch.inference_mode():
            out = dec1(vision(x).float())[0]
        px, py = sa1(out[0])
        coarse[path] = apply_h(Hm1, px, py)
    torch.save(coarse, cache)

    class DS(torch.utils.data.Dataset):
        def __init__(self, entries, jitter):
            self.entries = entries
            self.jitter = jitter

        def __len__(self):
            return len(self.entries)

        def __getitem__(self, i):
            path, src, dst, yaw = self.entries[i]
            img = Image.open(path).convert("RGB")
            cx, cy = coarse[path]
            jx = self.jitter * random.uniform(-1, 1)
            jy = self.jitter * random.uniform(-1, 1)
            # shift in metres -> pixels using the homography Jacobian approximation
            J = np.array([[1.0, 0.0], [0.0, 1.0]])
            px = project([cx + jx, cy + jy, src[2]], K, Ecv)
            half = args.crop // 2
            left = int(np.clip(px[0] - half, 0, W - args.crop))
            top = int(np.clip(px[1] - half, 0, H - args.crop))
            crop = img.crop((left, top, left + args.crop, top + args.crop)).resize((224, 224), Image.BILINEAR)
            x = tf(crop).float() / 255.0
            tgt = torch.tensor(np.concatenate([src[:2], dst[:2], [np.sin(yaw), np.cos(yaw)]]), dtype=torch.float32)
            axis = np.array([np.cos(yaw), np.sin(yaw), 0.0])
            p_src = project([src[0], src[1], src[2]], K, Ecv)
            p_e1 = project(list(np.asarray(src) + 0.11 * axis), K, Ecv)
            p_e2 = project(list(np.asarray(src) - 0.11 * axis), K, Ecv)
            p_dst = project([dst[0], dst[1], dst[2]], K, Ecv)
            def norm(pt):
                return (float(np.clip((pt[0] - left) / args.crop, 0, 1)),
                        float(np.clip((pt[1] - top) / args.crop, 0, 1)))
            n1, n2, n3, n4 = norm(p_src), norm(p_e1), norm(p_e2), norm(p_dst)
            off = torch.tensor([left, top, n1[0], n1[1], n2[0], n2[1], n3[0], n3[1], n4[0], n4[1]], dtype=torch.float32)
            return x, tgt, off

    dl_tr = torch.utils.data.DataLoader(DS(tr_items, args.jitter_m), batch_size=args.batch_size, shuffle=True,
                                        num_workers=6, drop_last=True)
    dl_va = torch.utils.data.DataLoader(DS(va_items, 0.0), batch_size=args.batch_size, shuffle=False, num_workers=3)

    dec = HeatDecoder(2176, 256, 56, 4).to(args.device)
    params = [{"params": dec.parameters(), "lr": args.head_lr}]
    if args.lora:
        params.append({"params": [p for p in vision.parameters() if p.requires_grad], "lr": args.lr})
    opt = torch.optim.AdamW(params, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    best = 1e9
    for epoch in range(args.epochs):
        dec.train()
        tot, nb = 0.0, 0
        for x, tgt, _off in dl_tr:
            x = x.to(args.device, dtype=torch.bfloat16)
            _off = _off.to(args.device)
            with torch.set_grad_enabled(True):
                out = dec(vision(x).float())
                # target heatmaps: we cannot build them here cheaply, so use a soft-argmax
                # regression loss against the true world position converted to crop pixels
                loss = torch.zeros((), device=args.device)
                for ch in range(4):
                    nx, ny = soft_argmax_t(out[:, ch])
                    loss = loss + torch.nn.functional.mse_loss(nx, _off[:, 2 + 2 * ch]) \
                                + torch.nn.functional.mse_loss(ny, _off[:, 3 + 2 * ch])
                opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss); nb += 1
        sched.step()
        dec.eval()
        errs = []
        with torch.no_grad():
            k = 0
            for x, tgt, _off in dl_va:
                _off = _off.to(args.device)
                out = dec(vision(x.to(args.device, dtype=torch.bfloat16)).float())
                for i in range(out.shape[0]):
                    res = soft_argmax(out[i, 0])
                    if res is None:
                        errs.append(1000.0); k += 1; continue
                    nx, ny = res
                    off = _off[i]
                    px = float(off[0]) + nx * args.crop
                    py = float(off[1]) + ny * args.crop
                    wx, wy = apply_h(Hm1, px, py)
                    errs.append(float(np.hypot(wx - float(tgt[i, 0]), wy - float(tgt[i, 1]))) * 1000)
                    k += 1
        err = float(np.mean(errs)) if errs else -1
        print(f"[s2] epoch {epoch} train {tot/max(1,nb):.6f} val_xy_mm {err:.2f}", flush=True)
        if err < best:
            best = err
            torch.save({"decoder": dec.state_dict(), "Hm": Hm1, "lora": args.lora, "crop": args.crop,
                        "src_z": ck1["src_z"], "dst_z": ck1["dst_z"], "out_ch": 4}, os.path.join(args.out, "stage2.pt"))
            if args.lora:
                vision.save_pretrained(os.path.join(args.out, "vision_lora"))
    print("[s2] saved", args.out, "best", best)


if __name__ == "__main__":
    main()

