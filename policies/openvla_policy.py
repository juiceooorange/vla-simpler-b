"""OpenVLA policy wrapper for SimplerEnv WidowX evaluation.

Adapted from simpler-env/SimplerEnv-OpenVLA (`simpler_env/policies/openvla/openvla_model.py`,
MIT-licensed). Two deliberate changes:

1. The VLM prompt uses OpenVLA's official template
   ``In: What action should the robot take to {instruction}?\nOut:`` instead of
   the raw instruction string.
2. Attention uses PyTorch SDPA instead of ``flash_attention_2`` so the model runs
   on Blackwell (sm_120) GPUs without building flash-attn.
"""

from __future__ import annotations

import os
from typing import Optional

import cv2 as cv
import numpy as np
import torch
from PIL import Image
from transforms3d.euler import euler2axangle


def format_prompt(instruction: str) -> str:
    return f"In: What action should the robot take to {instruction.strip().lower()}?\nOut:"


class OpenVLAPolicy:
    def __init__(
        self,
        saved_model_path: str,
        unnorm_key: str = "bridge_orig",
        device: str = "cuda:0",
        image_size: tuple[int, int] = (224, 224),
        action_scale: float = 1.0,
        attn_implementation: str = "sdpa",
        lora_path: str | None = None,
        flip_mode: str = "xy_rot",
        zero_rotation: bool = False,
    ) -> None:
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        from transformers import AutoModelForVision2Seq, AutoProcessor

        self.device = device
        self.unnorm_key = unnorm_key
        self.image_size = image_size
        self.action_scale = action_scale
        self.flip_mode = flip_mode
        self.zero_rotation = zero_rotation
        self.instruction: Optional[str] = None

        self.processor = AutoProcessor.from_pretrained(saved_model_path, trust_remote_code=True)
        kwargs = dict(
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            attn_implementation=attn_implementation,
        )
        try:
            self.vla = AutoModelForVision2Seq.from_pretrained(saved_model_path, **kwargs)
        except (ValueError, ImportError) as exc:  # older checkpoints may not accept sdpa
            print(f"[openvla] {attn_implementation} unavailable ({exc}); falling back to eager")
            kwargs["attn_implementation"] = "eager"
            self.vla = AutoModelForVision2Seq.from_pretrained(saved_model_path, **kwargs)
        self.vla.to(device).eval()

    def reset(self, instruction: str) -> None:
        self.instruction = instruction

    @torch.inference_mode()
    def act(self, image: np.ndarray, instruction: Optional[str] = None) -> np.ndarray:
        """image: HxWx3 uint8 RGB -> 7-D WidowX action (pos delta, rot ax-angle, gripper)."""
        if instruction is not None:
            self.instruction = instruction
        if self.instruction is None:
            raise RuntimeError("policy.reset(instruction) must be called first")

        # pass the raw observation; the processor performs the resize itself, which is the
        # same preprocessing used when the fine-tuning data was built (no double resizing)
        inputs = self.processor(format_prompt(self.instruction), Image.fromarray(image))
        inputs = inputs.to(self.device, dtype=torch.bfloat16)
        try:
            # ECoT-derived checkpoints forward **kwargs to generate() and need an
            # explicit max_new_tokens (their generation_config caps max_length at 20).
            out = self.vla.predict_action(
                **inputs, unnorm_key=self.unnorm_key, do_sample=False, max_new_tokens=8
            )
        except TypeError:
            out = self.vla.predict_action(**inputs, unnorm_key=self.unnorm_key, do_sample=False)
        if isinstance(out, tuple):  # ECoT returns (actions, generated_ids)
            out = out[0]
        raw = np.asarray(out, dtype=np.float32).reshape(-1)

        world_vector = raw[:3] * self.action_scale
        roll, pitch, yaw = raw[3:6]
        axis, angle = euler2axangle(roll, pitch, yaw)
        rot_axangle = axis * angle * self.action_scale
        gripper = 2.0 * (raw[6] > 0.5) - 1.0
        action = np.concatenate([world_vector, rot_axangle, np.array([gripper])]).astype(np.float32)
        # The SimplerEnv WidowX controller's action frame is rotated 180 deg about the
        # vertical axis with respect to the Bridge/OpenVLA base frame: commanding +x/+y
        # moves the end effector along -x/-y (verified with scripted_expert.py).
        if self.flip_mode in ("xy", "xy_rot"):
            action[[0, 1]] *= -1.0
        if self.flip_mode == "xy_rot":
            action[[3, 4]] *= -1.0
        if self.zero_rotation:
            action[3:6] = 0.0
        return action
