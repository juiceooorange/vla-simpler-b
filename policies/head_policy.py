"""Frozen OpenVLA backbone + trained action head(s).

Supports:
* ``mlp``  - last-token LLM feature -> MLP -> action chunk (train_head_chunk.py)
* ``conv`` - vision-token map + LLM feature -> conv trunk -> chunk (train_head_conv.py)
* ensemble - ``--head-dir a,b`` averages the predicted chunks of several heads
* temporal ensembling (ACT-style): re-plan every step and average overlapping
  chunk predictions with exponential weights
"""

from __future__ import annotations

import json
import math
import os
from collections import deque
from typing import Optional

import numpy as np
import torch
from PIL import Image


def _build_mlp(sd):
    idxs = sorted({int(k.split(".")[0]) for k in sd})
    layers = []
    for idx in range(max(idxs) + 1):
        if idx not in idxs:
            layers.append(torch.nn.GELU())
            continue
        w = sd[f"{idx}.weight"]
        if w.dim() == 2:
            layers.append(torch.nn.Linear(w.shape[1], w.shape[0]))
        else:
            layers.append(torch.nn.LayerNorm(w.shape[0]))
    return torch.nn.Sequential(*layers)


class HeadPolicy:
    def __init__(self, base_model: str, head_dir: str, device: str = "cuda:0",
                 image_size: tuple[int, int] = (224, 224), flip_mode: str = "none",
                 temporal_ensemble: Optional[float] = None) -> None:
        from transformers import AutoModelForVision2Seq, AutoProcessor

        head_dirs = [d for d in head_dir.split(",") if d]
        self.heads, self.archs = [], []
        self.chunk = None
        for d in head_dirs:
            cfg = json.load(open(os.path.join(d, "head_config.json"), encoding="utf-8"))
            ckpt = torch.load(os.path.join(d, "head.pt"), map_location="cpu")
            arch = cfg.get("arch", "mlp")
            sd = {k.replace("net.", ""): v for k, v in ckpt["state_dict"].items()}
            if arch == "conv":
                from train_head_conv import ConvHead

                head = ConvHead(ckpt["vis_dim"], ckpt["llm_dim"], 512, 7 * int(ckpt["chunk"]))
                head.load_state_dict(sd)
            elif arch == "twohead":
                from train_head_twohead import TwoHead

                head = TwoHead(ckpt["in_dim"], int(ckpt["chunk"]))
                head.load_state_dict(sd)
            elif arch == "act":
                from train_head_act import ACTHead

                head = ACTHead(ckpt["vis_dim"], ckpt["global_dim"], 512, 8, 2, 4,
                               int(ckpt["chunk"]), int(ckpt["n_tokens"]))
                head.load_state_dict(sd)
                self.pool = int(cfg.get("pool", 2))
            else:
                head = _build_mlp(sd)
                head.load_state_dict(sd)
            head.to(device).float().eval()
            self.heads.append(head)
            self.archs.append(arch)
            self.chunk = int(ckpt.get("chunk", cfg.get("chunk", 1)))
            self.uses_proprio = bool(cfg.get("proprio", False))
            self.loc_dir = cfg.get("localizer")
            self.prompt_override = cfg.get("prompt_override")
            c = cfg.get("crop")
            if c:
                self.crop = tuple(int(v) for v in c)
        assert self.chunk is not None, "no head loaded"
        print(f"[head] {len(self.heads)} head(s): {self.archs}, chunk={self.chunk}, "
              f"temporal_ensemble={temporal_ensemble}")

        self.processor = AutoProcessor.from_pretrained(base_model, trust_remote_code=True)
        self.vla = AutoModelForVision2Seq.from_pretrained(
            base_model, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
            trust_remote_code=True, attn_implementation="sdpa",
        ).to(device).eval()
        self.device = device
        self.image_size = image_size
        self.flip_mode = flip_mode
        self.m = temporal_ensemble
        self.instruction: Optional[str] = None
        self.queue: deque[np.ndarray] = deque()
        self.crop = getattr(self, "crop", None)
        self.loc = None
        if getattr(self, "loc_dir", None):
            from train_localizer_hm import HeatLocalizer
            _ck = torch.load(os.path.join(self.loc_dir, "localizer_hm.pt"), map_location="cpu", weights_only=False)
            _m = HeatLocalizer(4).to(device).float().eval()
            _m.load_state_dict(_ck["state_dict"])
            self.loc = {"model": _m, "Hm": _ck["Hm"], "size": int(_ck["image_size"])}
            print(f"[head] localizer {self.loc_dir} size={self.loc['size']}")
        self.prompt_override = getattr(self, "prompt_override", None)
        self.pending: dict[int, list[tuple[int, np.ndarray]]] = {}
        self.t = 0

    def reset(self, instruction: str) -> None:
        self.instruction = instruction
        self.queue.clear()
        self.pending.clear()
        self.t = 0

    def _post(self, action: np.ndarray) -> np.ndarray:
        if self.flip_mode in ("xy", "xy_rot"):
            action[[0, 1]] *= -1.0
        if self.flip_mode == "xy_rot":
            action[[3, 4]] *= -1.0
        return action

    @torch.inference_mode()
    def _predict_chunk(self, image: np.ndarray, obs=None) -> np.ndarray:
        instr = self.prompt_override or self.instruction
        prompt = f"In: What action should the robot take to {instr.strip().lower()}?\nOut:"
        img = Image.fromarray(image)
        if self.crop:
            img = img.crop(self.crop)
        inputs = self.processor(prompt, img)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)
        out = self.vla(**inputs, output_hidden_states=True)
        feat = out.hidden_states[-1][0, -1].float().unsqueeze(0)

        vt = vt_act = None
        if "conv" in self.archs or "act" in self.archs:
            vt = self.vla.vision_backbone(inputs["pixel_values"]).float()
            b, n, c = vt.shape
            side = int(round(n ** 0.5))
            vt = vt.reshape(b, side, side, c).permute(0, 3, 1, 2)
            vt_act = torch.nn.functional.avg_pool2d(vt, self.pool).flatten(2).permute(0, 2, 1)

        if self.loc is not None:
            from train_localizer_hm import decode_batch
            _img = Image.fromarray(image).convert("RGB").resize((self.loc["size"], self.loc["size"]), Image.BILINEAR)
            _x = torch.from_numpy(np.asarray(_img, dtype=np.float32) / 255.0).permute(2, 0, 1)[None].to(self.device)
            with torch.inference_mode():
                _h = self.loc["model"](_x)
            _lf = torch.tensor(decode_batch(_h, self.loc["Hm"]), dtype=torch.float32, device=feat.device).half()
            feat = torch.cat([feat, _lf], dim=1)

        if getattr(self, "uses_proprio", False):
            qpos = np.asarray(obs["agent"]["qpos"], dtype=np.float32).reshape(-1)[:8]
            tcp = np.asarray(getattr(self, "_tcp", np.zeros(3)), dtype=np.float32).reshape(-1)[:3]
            pt = torch.tensor(np.concatenate([qpos, tcp])[None], dtype=torch.float32, device=feat.device).half()
            feat = torch.cat([feat, pt], dim=1)

        preds = []
        for head, arch in zip(self.heads, self.archs):
            if arch == "twohead":
                pose, logits = head(feat)
                g = torch.where(logits > 0, 1.0, -1.0)
                out = torch.cat([pose.reshape(1, -1, 6), g.unsqueeze(-1)], dim=-1).reshape(1, -1)
            elif arch == "conv":
                out = head(feat, vt)
            elif arch == "twohead":
                from train_head_twohead import TwoHead

                head = TwoHead(ckpt["in_dim"], int(ckpt["chunk"]))
                head.load_state_dict(sd)
            elif arch == "act":
                out = head(feat, vt_act).reshape(-1, 7 * self.chunk)
            else:
                out = head(feat)
            preds.append(out[0].cpu().numpy().astype(np.float32))
        return np.mean(preds, axis=0).reshape(self.chunk, 7)

    def _combine(self, step: int) -> np.ndarray:
        entries = self.pending.pop(step)
        if len(entries) == 1:
            return entries[0][1]
        ws = np.array([math.exp(-self.m * (step - t0)) for t0, _ in entries], dtype=np.float32)
        acts = np.stack([a for _, a in entries], axis=0)
        return (acts * ws[:, None]).sum(0) / ws.sum()

    @torch.inference_mode()
    def act(self, image: np.ndarray, instruction: Optional[str] = None, obs=None) -> np.ndarray:
        if instruction is not None and instruction != self.instruction:
            self.reset(instruction)

        if self.m is None:  # chunked open-loop execution (original behaviour)
            if self.queue:
                return self._post(np.array(self.queue.popleft(), dtype=np.float32))
            pred = self._predict_chunk(image, obs)
            for row in pred[1:]:
                self.queue.append(row.copy())
            return self._post(pred[0].copy())

        # ACT-style temporal ensembling: re-plan every step, average overlapping chunks
        s = self.t
        if s not in self.pending:
            pred = self._predict_chunk(image, obs)
            for i in range(pred.shape[0]):
                self.pending.setdefault(s + i, []).append((s, pred[i].copy()))
        action = self._combine(s)
        self.t += 1
        return self._post(action)

