"""LoRA fine-tuning for OpenVLA on self-collected WidowX rollouts.

This is a compact re-implementation of OpenVLA's official LoRA recipe
(`vla-scripts/finetune.py`) with the RLDS/TFDS data pipeline replaced by a plain
PyTorch dataset over the rollouts produced by ``collect_trajectories.py``.

Training objective: next-token cross-entropy on the 7 discretised action tokens,
exactly as in OpenVLA (``ActionTokenizer`` over the 256 least-used vocabulary
entries). The prompt is OpenVLA's official template; the final prompt token is
replaced by the action token sequence, so the loss is computed on actions only.

Usage:
    python train_lora.py --base-model ../models/openvla-7b --data ../data/rollouts \
        --tasks widowx_spoon_on_towel,widowx_put_eggplant_in_basket \
        --out ../runs/lora_r16 --lora-rank 16 --steps 1500 --batch-size 6
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", required=True)
    ap.add_argument("--data", required=True, help="root with <task>/<run>/epNNN folders")
    ap.add_argument("--runs", default="base_dev", help="comma-separated rollout run names to use")
    ap.add_argument("--tasks", default="", help="comma-separated task names (default: all found)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--lora-rank", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=16)
    ap.add_argument("--lora-dropout", type=float, default=0.0)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--batch-size", type=int, default=6)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--warmup-steps", type=int, default=50)
    ap.add_argument("--max-grad-norm", type=float, default=1.0)
    ap.add_argument("--image-size", type=int, default=224)
    ap.add_argument("--max-samples", type=int, default=0, help="cap on training samples (0 = all)")
    ap.add_argument("--val-frac", type=float, default=0.1, help="fraction of *episodes* held out")
    ap.add_argument("--only-success", action="store_true", default=True)
    ap.add_argument("--include-failures", dest="only_success", action="store_false")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--flip-targets", action="store_true", default=True, help="recorded actions already contain the frame flip; undo it to obtain model-space targets")
    ap.add_argument("--save-every", type=int, default=500)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--gradient-checkpointing", action="store_true", default=True)
    return ap.parse_args()


class RolloutDataset(Dataset):
    """(image, instruction, action) samples from collected episodes."""

    def __init__(self, samples, processor, image_size, augment=False):
        self.samples = samples
        self.processor = processor
        self.image_size = image_size
        self.augment = augment

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, instruction, action = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.augment and random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        pixel_values = self.processor.image_processor.apply_transform(img)
        return {
            "pixel_values": pixel_values,
            "instruction": instruction,
            "action": torch.tensor(action, dtype=torch.float32),
        }


@dataclass
class Collator:
    tokenizer: object
    action_tokenizer: object
    model_max_length: int = 512

    def __call__(self, batch):
        input_ids, labels, pixel_values = [], [], []
        for item in batch:
            prompt = f"In: What action should the robot take to {item['instruction'].strip().lower()}?\nOut:"
            ids = self.tokenizer(prompt, add_special_tokens=True).input_ids
            action_ids = list(self.action_tokenizer(item["action"].numpy()))
            ids = ids[:-1] + action_ids
            lab = [-100] * (len(ids) - len(action_ids)) + action_ids
            input_ids.append(torch.tensor(ids, dtype=torch.long))
            labels.append(torch.tensor(lab, dtype=torch.long))
            pixel_values.append(item["pixel_values"])

        max_len = min(max(len(x) for x in input_ids), self.model_max_length)
        padded_ids = torch.full((len(batch), max_len), self.tokenizer.pad_token_id, dtype=torch.long)
        padded_labels = torch.full((len(batch), max_len), -100, dtype=torch.long)
        for i, (ids, lab) in enumerate(zip(input_ids, labels)):
            n = min(len(ids), max_len)
            padded_ids[i, :n] = ids[:n]
            padded_labels[i, :n] = lab[:n]
        return {
            "input_ids": padded_ids,
            "attention_mask": padded_ids.ne(self.tokenizer.pad_token_id),
            "labels": padded_labels,
            "pixel_values": torch.stack(pixel_values),
        }


def load_norm_stats(base_model: str, model=None, unnorm_key: str = "bridge_orig"):
    """q01/q99 used by OpenVLA to unnormalise bridge actions.

    Stored actions are in metres / radians (the env consumes unnormalised actions),
    while the action tokenizer expects [-1, 1]. We therefore re-normalise with the
    same statistics before discretising.
    """
    stats = getattr(model, "norm_stats", None) if model is not None else None
    if not stats:
        path = os.path.join(base_model, "dataset_statistics.json")
        if not os.path.isfile(path):
            return None
        with open(path, encoding="utf-8") as fh:
            stats = json.load(fh)
    ns = stats[unnorm_key]["action"]
    return np.array(ns["q01"], dtype=np.float32), np.array(ns["q99"], dtype=np.float32)


def normalize_actions(actions: np.ndarray, q01: np.ndarray, q99: np.ndarray) -> np.ndarray:
    span = np.where(np.abs(q99 - q01) < 1e-6, 1.0, q99 - q01)
    return np.clip(2.0 * (actions - q01) / span - 1.0, -1.0, 1.0).astype(np.float32)


def discover(data_root, runs, tasks, only_success, seed, val_frac, q01=None, q99=None, flip_targets=True):
    train, val, episodes = [], [], []
    for task in sorted(tasks):
        for run in runs:
            run_dir = os.path.join(data_root, task, run)
            if not os.path.isdir(run_dir):
                continue
            for ep in sorted(os.listdir(run_dir)):
                ep_dir = os.path.join(run_dir, ep)
                meta_path = os.path.join(ep_dir, "meta.json")
                if not os.path.isfile(meta_path):
                    continue
                with open(meta_path, encoding="utf-8") as fh:
                    meta = json.load(fh)
                if only_success and not meta.get("success"):
                    continue
                actions = np.load(os.path.join(ep_dir, "actions.npy")).astype(np.float32)
                if flip_targets:
                    actions = actions * np.array([-1, -1, 1, -1, -1, 1, 1], dtype=np.float32)
                if q01 is not None:
                    actions = normalize_actions(actions, q01, q99)
                frames = sorted(os.listdir(os.path.join(ep_dir, "frames")))
                n = min(len(frames), len(actions))
                if n < 4:
                    continue
                episodes.append({
                    "task": task, "run": run, "ep": ep,
                    "samples": [
                        (os.path.join(ep_dir, "frames", frames[t]), meta["instruction"], actions[t].astype(np.float32))
                        for t in range(n)
                    ],
                })

    rng = random.Random(seed)
    rng.shuffle(episodes)
    n_val = max(1, int(len(episodes) * val_frac)) if episodes else 0
    for i, ep in enumerate(episodes):
        (val if i < n_val else train).extend(ep["samples"])
    return train, val, len(episodes) - n_val


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "train_args.json"), "w", encoding="utf-8") as fh:
        json.dump(vars(args), fh, indent=2)

    from transformers import AutoModelForVision2Seq, AutoProcessor
    from peft import LoraConfig, get_peft_model

    print("[train] loading processor/model:", args.base_model, flush=True)
    processor = AutoProcessor.from_pretrained(args.base_model, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        args.base_model, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
        trust_remote_code=True, attn_implementation="sdpa",
    )
    model.to(args.device)
    raw_model = model

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    model.config.use_cache = False

    lora_cfg = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        init_lora_weights="gaussian",
        bias="none",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"[train] trainable {trainable/1e6:.2f}M / total {total/1e9:.2f}B", flush=True)

    # action tokenizer over the model's own tokenizer (256 least-used tokens)
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from openvla_action_tokenizer import ActionTokenizer

    action_tokenizer = ActionTokenizer(processor.tokenizer)

    tasks = [t for t in args.tasks.split(",") if t] or sorted(os.listdir(args.data))
    runs = [r for r in args.runs.split(",") if r]
    norm = load_norm_stats(args.base_model, model=raw_model)
    if norm is None:
        raise SystemExit(f"dataset_statistics.json with bridge_orig not found in {args.base_model}")
    q01, q99 = norm
    print("[train] action q01:", np.round(q01, 4).tolist(), "q99:", np.round(q99, 4).tolist())
    train_samples, val_samples, n_train_eps = discover(
        args.data, runs, tasks, args.only_success, args.seed, args.val_frac, q01=q01, q99=q99,
        flip_targets=args.flip_targets
    )
    print(f"[train] episodes={n_train_eps + (1 if val_samples else 0)} train_samples={len(train_samples)} val_samples={len(val_samples)}")
    if not train_samples:
        raise SystemExit("no training samples found")
    if args.max_samples:
        train_samples = train_samples[: args.max_samples]

    train_ds = RolloutDataset(train_samples, processor, args.image_size, augment=False)
    val_ds = RolloutDataset(val_samples, processor, args.image_size, augment=False)
    collator = Collator(processor.tokenizer, action_tokenizer)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collator, num_workers=4, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            collate_fn=collator, num_workers=2) if len(val_ds) else None

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda s: min(1.0, (s + 1) / max(1, args.warmup_steps)) * max(0.0, 1 - s / max(1, args.steps)),
    )

    log_path = os.path.join(args.out, "train_log.jsonl")
    log_fh = open(log_path, "a", encoding="utf-8")
    model.train()
    step = 0
    t0 = time.time()
    running = 0.0
    while step < args.steps:
        for batch in train_loader:
            batch = {k: v.to(args.device) for k, v in batch.items()}
            batch["pixel_values"] = batch["pixel_values"].to(torch.bfloat16)
            out = model(**batch)
            loss = out.loss / args.grad_accum
            loss.backward()
            running += out.loss.item()
            if (step + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(params, args.max_grad_norm)
                optimizer.step()
                sched.step()
                optimizer.zero_grad(set_to_none=True)
            if step % args.log_every == 0:
                rec = {"step": step, "loss": round(running / max(1, (step % args.log_every) + 1), 5),
                       "lr": sched.get_last_lr()[0], "elapsed_s": round(time.time() - t0, 1)}
                print("[train]", rec, flush=True)
                log_fh.write(json.dumps(rec) + "\n")
                log_fh.flush()
                running = 0.0
            step += 1
            if step % args.save_every == 0 or step == args.steps:
                ckpt = os.path.join(args.out, f"checkpoint-{step}")
                model.save_pretrained(ckpt)
                print("[train] saved", ckpt, flush=True)
                if val_loader is not None:
                    model.eval()
                    vl, nb = 0.0, 0
                    with torch.no_grad():
                        for vb in val_loader:
                            vb = {k: v.to(args.device) for k, v in vb.items()}
                            vb["pixel_values"] = vb["pixel_values"].to(torch.bfloat16)
                            vl += model(**vb).loss.item()
                            nb += 1
                    rec = {"step": step, "val_loss": round(vl / max(1, nb), 5)}
                    print("[train]", rec, flush=True)
                    log_fh.write(json.dumps(rec) + "\n")
                    log_fh.flush()
                    model.train()
            if step >= args.steps:
                break
    log_fh.close()
    print("[train] done in", round(time.time() - t0, 1), "s")


if __name__ == "__main__":
    main()
