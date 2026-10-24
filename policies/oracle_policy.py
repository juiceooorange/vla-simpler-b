"""Oracle MLP controller (diagnostic): privileged object poses -> action chunk."""

from __future__ import annotations

import json
import os
from collections import deque
from typing import Optional

import numpy as np
import torch


class OracleMLPPolicy:
    uses_proprio = True
    oracle = True

    def __init__(self, head_dir: str, device: str = "cuda:0", flip_mode: str = "none", **_) -> None:
        import sys
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from train_head_oracle import OracleHead

        ck = torch.load(os.path.join(head_dir, "head.pt"), map_location="cpu", weights_only=False)
        self.two_head = any(k.startswith("pose.") for k in ck["state_dict"])
        self.model = OracleHead(int(ck["in_dim"]), int(ck["chunk"])).to(device).float().eval()
        self.model.load_state_dict(ck["state_dict"])
        self.chunk = int(ck["chunk"])
        self.in_dim = int(ck["in_dim"])
        self.device = device
        self.flip_mode = flip_mode
        self.queue: deque[np.ndarray] = deque()
        self.hist: deque[np.ndarray] = deque(maxlen=5)
        self._tcp = np.zeros(3, dtype=np.float32)
        self._gt_poses = None
        self._base_p = np.zeros(3, dtype=np.float32)
        print(f"[oracle] loaded chunk={self.chunk}")

    def reset(self, instruction: str) -> None:
        self.queue.clear()
        self.hist.clear()

    def _post(self, a: np.ndarray) -> np.ndarray:
        if self.flip_mode in ("xy", "xy_rot"):
            a[[0, 1]] *= -1.0
        if self.flip_mode == "xy_rot":
            a[[3, 4]] *= -1.0
        return a

    @torch.inference_mode()
    def act(self, image: Optional[np.ndarray], instruction: Optional[str] = None, obs=None) -> np.ndarray:
        if self.queue:
            return self._post(np.array(self.queue.popleft(), dtype=np.float32))
        p = self._gt_poses
        b = np.asarray(self._base_p, dtype=np.float64).reshape(-1)[:3]
        base = np.concatenate([np.asarray(p["src"]) - b, np.asarray(p["dst"]) - b,
                               [np.sin(p["yaw"]), np.cos(p["yaw"])],
                               np.asarray(obs["agent"]["qpos"], dtype=np.float32).reshape(-1)[:8],
                               np.asarray(self._tcp, dtype=np.float32).reshape(-1)[:3] - b]).astype(np.float32)
        h = np.zeros(35, dtype=np.float32)
        for k, a in enumerate(reversed(list(self.hist))):
            h[k * 7:(k + 1) * 7] = a
        vec = np.concatenate([base, h]) if self.in_dim > len(base) else base
        x = torch.tensor(vec[None], device=self.device)
        out = self.model(x)
        if self.two_head:
            pose, logits = out
            pose = pose[0].cpu().numpy().reshape(self.chunk, 6)
            grip = np.where(logits[0].cpu().numpy() > 0, 1.0, -1.0)
            pred = np.concatenate([pose, grip[:, None]], axis=1).astype(np.float32)
        else:
            pred = out[0].cpu().numpy().reshape(self.chunk, 7)
        for row in pred[1:]:
            self.queue.append(row.copy())
        out = pred[0].copy()
        self.hist.append(out.copy())
        return self._post(out)
