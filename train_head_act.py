"""ACT-style head on frozen OpenVLA vision tokens.

Motivation (see subagent research): the global last-token feature + 5-step MLP
loses spatial layout and has a short action horizon. ACT keeps a spatial token
grid, uses a transformer decoder with learned queries to emit a longer action
chunk, and is trained with an L1 loss; temporal ensembling at inference smooths
the chunk boundaries.

Pipeline:
    vision tokens (256 x 2176) --avgpool 2x2--> 64 x 2176
    + global feature (4096) as one extra token
    -> linear projection to d=512 + learned position embedding
    -> 2-layer encoder -> 16 learned queries through a 4-layer decoder -> 16 x 7 actions

Usage:
    python train_head_act.py --base-model ../models/sft-simpler --data ../data/rollouts \
        --runs clean --out ../runs/head_act --chunk 16 --epochs 60
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
    ap.add_argument("--runs", default="clean")
    ap.add_argument("--tasks", default="widowx_spoon_on_towel,widowx_put_eggplant_in_basket")
    ap.add_argument("--out", required=True)
    ap.add_argument("--chunk", type=int, default=16)
    ap.add_argument("--stride", type=int, default=2, help="cache every Nth frame to save disk")
    ap.add_argument("--pool", type=int, default=2, help="spatial pooling factor (2 -> 8x8 tokens)")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--feature-batch", type=int, default=8)
    ap.add_argument("--max-episodes", type=int, default=0)
    return ap.parse_args()


class ACTHead(nn.Module):
    def __init__(self, vis_dim=2176, global_dim=4096, d=512, nheads=8, enc_layers=2,
                 dec_layers=4, chunk=16, n_tokens=64):
        super().__init__()
        self.chunk = chunk
        self.token_proj = nn.Linear(vis_dim, d)
        self.pos = nn.Parameter(torch.zeros(1, n_tokens, d))
        nn.init.trunc_normal_(self.pos, std=0.02)
        self.global_proj = nn.Linear(global_dim, d)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d, nheads, 4 * d, dropout=0.1, batch_first=True), enc_layers)
        self.queries = nn.Parameter(torch.zeros(1, chunk, d))
        nn.init.trunc_normal_(self.queries, std=0.02)
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d, nheads, 4 * d, dropout=0.1, batch_first=True), dec_layers)
        self.out = nn.Linear(d, 7)

    def forward(self, global_feat, tokens):
        # tokens: (B, N, 2176), global_feat: (B, 4096)
        x = self.token_proj(tokens) + self.pos[:, : tokens.shape[1]]
        g = self.global_proj(global_feat).unsqueeze(1)
        x = torch.cat([g, x], dim=1)
        mem = self.encoder(x)
        q = self.queries.expand(tokens.shape[0], -1, -1)
        h = self.decoder(q, mem)
        return self.out(h)  # (B, chunk, 7)


def list_episodes(data_root, runs, tasks, seed, val_frac, max_episodes=0):
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
                eps.append({"task": task, "ep": ep,
                            "frames": [os.path.join(d, "frames", f) for f in frames[:n]],
                            "actions": actions[:n], "instruction": meta["instruction"]})
    rng = random.Random(seed)
    rng.shuffle(eps)
    if max_episodes:
        eps = eps[:max_episodes]
    n_val = max(1, int(len(eps) * val_frac))
    return eps[n_val:], eps[:n_val]


@torch.inference_mode()
def cache_features(model, processor, episodes, device, batch_size, chunk, stride, pool, path):
    if os.path.isfile(path):
        print("[act] reusing cache", path, flush=True)
        return torch.load(path, map_location="cpu")
    globs, toks, ys = [], [], []
    t0 = time.time()
    total = sum(len(range(0, len(e["frames"]), stride)) for e in episodes)
    done = 0
    for ep in episodes:
        prompt = f"In: What action should the robot take to {ep['instruction'].strip().lower()}?\nOut:"
        frames, acts = ep["frames"], ep["actions"]
        padded = np.concatenate([acts, np.repeat(acts[-1:], chunk, axis=0)], axis=0)
        idxs = list(range(0, len(frames), stride))
        for i in range(0, len(idxs), batch_size):
            sel = idxs[i:i + batch_size]
            imgs = [Image.open(frames[t]).convert("RGB") for t in sel]
            inputs = processor([prompt] * len(imgs), imgs)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)
            out = model(**inputs, output_hidden_states=True)
            globs.append(out.hidden_states[-1][:, -1].float().cpu().half())
            vt = model.vision_backbone(inputs["pixel_values"]).float()
            b, n, c = vt.shape
            side = int(round(n ** 0.5))
            vt = vt.reshape(b, side, side, c).permute(0, 3, 1, 2)
            vt = torch.nn.functional.avg_pool2d(vt, pool).flatten(2).permute(0, 2, 1)
            toks.append(vt.cpu().half())
            for t in sel:
                ys.append(torch.tensor(padded[t:t + chunk].reshape(-1), dtype=torch.float32))
            done += len(sel)
            if done % (batch_size * 20) == 0:
                print(f"  cache {done}/{total} ({time.time()-t0:.0f}s)", flush=True)
    data = {"global": torch.cat(globs), "tokens": torch.cat(toks), "y": torch.stack(ys)}
    torch.save(data, path)
    print(f"[act] cached {path} global={tuple(data['global'].shape)} tokens={tuple(data['tokens'].shape)}", flush=True)
    return data


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    os.makedirs(args.out, exist_ok=True)

    from transformers import AutoModelForVision2Seq, AutoProcessor

    tasks = [t for t in args.tasks.split(",") if t]
    runs = [r for r in args.runs.split(",") if r]
    train_eps, val_eps = list_episodes(args.data, runs, tasks, args.seed, args.val_frac, args.max_episodes)
    print(f"[act] episodes train={len(train_eps)} val={len(val_eps)}", flush=True)

    processor = AutoProcessor.from_pretrained(args.base_model, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        args.base_model, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
        trust_remote_code=True, attn_implementation="sdpa",
    ).to(args.device).eval()
    for p in model.parameters():
        p.requires_grad = False

    tr = cache_features(model, processor, train_eps, args.device, args.feature_batch,
                        args.chunk, args.stride, args.pool, os.path.join(args.out, "train_cache.pt"))
    va = cache_features(model, processor, val_eps, args.device, args.feature_batch,
                        args.chunk, args.stride, args.pool, os.path.join(args.out, "val_cache.pt"))

    n_tokens = tr["tokens"].shape[1]
    head = ACTHead(tr["tokens"].shape[2], tr["global"].shape[1], 512, 8, 2, 4, args.chunk, n_tokens).to(args.device)
    print(f"[act] params {sum(p.numel() for p in head.parameters())/1e6:.2f}M tokens={n_tokens}", flush=True)
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    log = open(os.path.join(args.out, "train_log.jsonl"), "w", encoding="utf-8")
    best = 1e9
    n = len(tr["y"])
    for epoch in range(args.epochs):
        head.train()
        perm = torch.randperm(n)
        tot = 0.0
        for i in range(0, n, args.batch_size):
            idx = perm[i:i + args.batch_size]
            g = tr["global"][idx].float().to(args.device)
            tk = tr["tokens"][idx].float().to(args.device)
            y = tr["y"][idx].to(args.device).reshape(-1, args.chunk, 7)
            loss = torch.nn.functional.l1_loss(head(g, tk), y)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            opt.step()
            tot += loss.item() * len(idx)
        sched.step()
        head.eval()
        with torch.no_grad():
            preds = []
            for i in range(0, len(va["y"]), 64):
                g = va["global"][i:i + 64].float().to(args.device)
                tk = va["tokens"][i:i + 64].float().to(args.device)
                preds.append(head(g, tk).cpu())
            pred = torch.cat(preds).reshape(-1, args.chunk, 7)
            yv = va["y"].reshape(-1, args.chunk, 7)
            vl1 = torch.nn.functional.l1_loss(pred, yv).item()
            mae = (pred - yv).abs().mean().item() * 1000
        msg = {"epoch": epoch, "train_l1": round(tot / n, 6), "val_l1": round(vl1, 6), "val_mae_mm": round(mae, 3)}
        print("[act]", msg, flush=True)
        log.write(json.dumps(msg) + "\n")
        log.flush()
        if vl1 < best:
            best = vl1
            torch.save({"state_dict": head.state_dict(), "chunk": args.chunk,
                        "vis_dim": int(tr["tokens"].shape[2]), "global_dim": int(tr["global"].shape[1]),
                        "n_tokens": int(n_tokens)}, os.path.join(args.out, "head.pt"))
    log.close()
    with open(os.path.join(args.out, "head_config.json"), "w", encoding="utf-8") as fh:
        json.dump({"arch": "act", "chunk": args.chunk, "base_model": args.base_model,
                   "flip_mode": "none", "pool": args.pool, "best_val_l1": best}, fh, indent=2)
    print("[act] saved", args.out, "best val l1", best)


if __name__ == "__main__":
    main()

