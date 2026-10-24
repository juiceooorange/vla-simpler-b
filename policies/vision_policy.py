"""Vision-backbone policy: OpenVLA vision encoder (+ optional LoRA) -> conv head."""

from __future__ import annotations

import json
import os
from collections import deque
from typing import Optional

import numpy as np
import torch
from PIL import Image


class VisionPolicy:
    def __init__(self, base_model: str, head_dir: str, device: str = "cuda:0",
                 flip_mode: str = "none", **_) -> None:
        from transformers import AutoModelForVision2Seq, AutoProcessor

        cfg = json.load(open(os.path.join(head_dir, "head_config.json"), encoding="utf-8"))
        ckpt = torch.load(os.path.join(head_dir, "head.pt"), map_location="cpu")
        self.chunk = int(ckpt["chunk"])
        self.tasks = cfg.get("tasks", [])
        self.crops = {int(k): tuple(v) for k, v in (cfg.get("crops") or {}).items()}
        self.flip_mode = flip_mode
        self.device = device
        self.queue: deque[np.ndarray] = deque()
        self.task_idx = 0

        base = AutoModelForVision2Seq.from_pretrained(
            base_model, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
            trust_remote_code=True, attn_implementation="sdpa",
        ).to(device).eval()
        vision = base.vision_backbone
        del base
        if ckpt.get("lora"):
            from peft import PeftModel

            vision = PeftModel.from_pretrained(vision, os.path.join(head_dir, "vision_lora"))
            vision = vision.merge_and_unload()
        self.vision = vision.to(device).eval()
        self.transform = AutoProcessor.from_pretrained(base_model, trust_remote_code=True).image_processor.apply_transform

        from train_head_vision import VisionHead

        self.head = VisionHead(2176, 512, self.chunk, max(2, len(self.tasks) or 2)).to(device).float().eval()
        self.head.load_state_dict(ckpt["head"])
        print(f"[vision] loaded chunk={self.chunk} tasks={self.tasks} lora={bool(ckpt.get('lora'))}")

    def reset(self, instruction: str) -> None:
        self.queue.clear()
        inst = instruction.lower()
        self.task_idx = 1 if ("eggplant" in inst or "basket" in inst) else 0

    def _post(self, action: np.ndarray) -> np.ndarray:
        if self.flip_mode in ("xy", "xy_rot"):
            action[[0, 1]] *= -1.0
        if self.flip_mode == "xy_rot":
            action[[3, 4]] *= -1.0
        return action

    @torch.inference_mode()
    def act(self, image: np.ndarray, instruction: Optional[str] = None) -> np.ndarray:
        if self.queue:
            return self._post(np.array(self.queue.popleft(), dtype=np.float32))
        img = Image.fromarray(image).convert("RGB")
        box = self.crops.get(self.task_idx)
        if box:
            img = img.crop(box)
        x = self.transform(img)[None]
        x = x.to(self.device, dtype=torch.bfloat16)
        feats = self.vision(x).float()
        b, n, c = feats.shape
        side = int(round(n ** 0.5))
        fmap = feats.reshape(b, side, side, c).permute(0, 3, 1, 2)
        pred = self.head(fmap, torch.tensor([self.task_idx], device=self.device))[0].cpu().numpy()
        pred = pred.reshape(self.chunk, 7)
        for row in pred[1:]:
            self.queue.append(row.copy())
        return self._post(pred[0].copy())

