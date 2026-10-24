"""Policy using the depth-conditioned head (LLM feature + proprio + depth map)."""

from __future__ import annotations

import json
import os
from collections import deque
from typing import Optional

import numpy as np
import torch
from PIL import Image


class DepthPolicy:
    uses_proprio = True  # tells the evaluator to pass the full observation

    def __init__(self, base_model: str, head_dir: str, device: str = "cuda:0",
                 flip_mode: str = "none", **_) -> None:
        from transformers import AutoModelForVision2Seq, AutoProcessor

        from train_head_depth import DepthHead

        cfg = json.load(open(os.path.join(head_dir, "head_config.json"), encoding="utf-8"))
        ckpt = torch.load(os.path.join(head_dir, "head.pt"), map_location="cpu")
        self.chunk = int(ckpt["chunk"])
        self.head = DepthHead(int(ckpt["feat_dim"]), self.chunk).to(device).float().eval()
        self.head.load_state_dict(ckpt["state_dict"])
        self.processor = AutoProcessor.from_pretrained(base_model, trust_remote_code=True)
        self.vla = AutoModelForVision2Seq.from_pretrained(
            base_model, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
            trust_remote_code=True, attn_implementation="sdpa",
        ).to(device).eval()
        self.device = device
        self.flip_mode = flip_mode
        self.instruction: Optional[str] = None
        self.queue: deque[np.ndarray] = deque()
        self._tcp = np.zeros(3, dtype=np.float32)
        print(f"[depth] loaded chunk={self.chunk} feat_dim={ckpt['feat_dim']}")

    def reset(self, instruction: str) -> None:
        self.instruction = instruction
        self.queue.clear()

    def _post(self, action: np.ndarray) -> np.ndarray:
        if self.flip_mode in ("xy", "xy_rot"):
            action[[0, 1]] *= -1.0
        if self.flip_mode == "xy_rot":
            action[[3, 4]] *= -1.0
        return action

    @torch.inference_mode()
    def act(self, image: np.ndarray, instruction: Optional[str] = None, obs=None) -> np.ndarray:
        if instruction is not None and instruction != self.instruction:
            self.reset(instruction)
        if self.queue:
            return self._post(np.array(self.queue.popleft(), dtype=np.float32))

        prompt = f"In: What action should the robot take to {self.instruction.strip().lower()}?\nOut:"
        inputs = self.processor(prompt, Image.fromarray(image))
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)
        out = self.vla(**inputs, output_hidden_states=True)
        feat = out.hidden_states[-1][0, -1].float()

        qpos = np.asarray(obs["agent"]["qpos"], dtype=np.float32).reshape(-1)[:8]
        prop = np.concatenate([qpos, np.asarray(self._tcp, dtype=np.float32).reshape(-1)[:3]])
        feat = torch.cat([feat, torch.tensor(prop, device=self.device)], dim=0).unsqueeze(0)

        raw = np.asarray(obs["image"]["3rd_view_camera"]["depth"], dtype=np.float32)
        if raw.ndim == 3:
            raw = raw[..., 0]
        step = max(1, raw.shape[0] // 120)
        dep = raw[::step, ::step][:120, :160]
        dep = torch.tensor(dep, device=self.device, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

        pred = self.head(feat, dep)[0].cpu().numpy().reshape(self.chunk, 7)
        for row in pred[1:]:
            self.queue.append(row.copy())
        return self._post(pred[0].copy())

