"""Localizer + scripted controller policy.

Phase 1 of the episode: run the learned localizer once on the current image to get
the object poses (predicted in world coordinates). Then execute the same
pick-and-place phase machine as the scripted expert, which is known to succeed
~98% when the poses are exact.

This splits the problem: perception is a single-frame regression (no compounding
error), control is scripted (no learning needed).
"""

from __future__ import annotations

import json
import os
from typing import Optional

import numpy as np
import torch
from PIL import Image
from scipy.spatial.transform import Rotation

FLIP = np.array([-1.0, -1.0, 1.0], dtype=np.float32)


class LocalizePolicy:
    uses_proprio = True  # evaluator passes obs + tcp

    def __init__(self, head_dir: str, device: str = "cuda:0", flip_mode: str = "none", **_) -> None:
        from train_localizer import Localizer

        ckpt = torch.load(os.path.join(head_dir, "localizer.pt"), map_location="cpu")
        self.model = Localizer(8).to(device).float().eval()
        self.model.load_state_dict(ckpt["state_dict"])
        self.mean = ckpt["mean"].to(device)
        self.std = ckpt["std"].to(device)
        cfgp = os.path.join(head_dir, "localizer_config.json")
        self.image_size = 224
        if os.path.isfile(cfgp):
            self.image_size = int(json.load(open(cfgp, encoding="utf-8")).get("image_size", 224))
        self.device = device
        self.flip_mode = flip_mode
        self._tcp = None
        self.pred = None
        self.phase = 0
        self.budget = 0
        self.counter = 0
        print(f"[localize] loaded localizer from {head_dir}")

    @property
    def p(self):
        if not hasattr(self, "_p"):
            self._p = {
                "approach_h": float(os.environ.get("SERVO_APPROACH_H", 0.10)),
                "descend_dz": float(os.environ.get("SERVO_DESCEND_DZ", 0.012)),
                "lift_h": float(os.environ.get("SERVO_LIFT_H", 0.12)),
                "place_dz": float(os.environ.get("SERVO_PLACE_DZ", 0.04)),
                "xy_tol": float(os.environ.get("SERVO_XY_TOL", 0.008)),
                "descend_tol": float(os.environ.get("SERVO_DESCEND_TOL", 0.006)),
                "close_dz": float(os.environ.get("SERVO_CLOSE_DZ", 0.0)),
                "grasp_along": float(os.environ.get("SERVO_GRASP_ALONG", 0.0)),
            }
        return self._p

    def reset(self, instruction: str) -> None:
        self.pred = None
        self.phase = 0
        self.counter = 0
        self.budget = 0

    # ---------------------------------------------------------------- perception
    @torch.inference_mode()
    def _predict(self, image: np.ndarray) -> dict:
        img = Image.fromarray(image).convert("RGB").resize((self.image_size, self.image_size), Image.BILINEAR)
        x = torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0).permute(2, 0, 1)[None].to(self.device)
        out = self.model(x)[0] * self.std + self.mean
        v = out.cpu().numpy().astype(np.float64)
        return {"src": v[0:3], "dst": v[3:6], "yaw": float(np.arctan2(v[6], v[7])), "raw_yaw": v[6:8]}

    # ---------------------------------------------------------------- control
    def _tcp_pos(self) -> np.ndarray:
        return np.asarray(self._tcp, dtype=np.float64)

    def _move_action(self, target, gripper, max_delta):
        d = np.clip(target - self._tcp_pos(), -max_delta, max_delta)
        a = np.zeros(7, dtype=np.float32)
        a[:3] = (d * FLIP).astype(np.float32)
        a[6] = gripper
        return a

    def _align_action(self) -> Optional[np.ndarray]:
        q = getattr(self, "_tcp_quat", None)
        if q is None:
            return None
        yaw = Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_euler("xyz")[2]
        err = (self.pred["yaw"] - yaw + np.pi) % (2 * np.pi) - np.pi
        if abs(err) < 0.05:
            return None
        a = np.zeros(7, dtype=np.float32)
        a[5] = float(np.clip(err * 0.7, -0.12, 0.12))
        a[6] = 1.0
        return a

    def _post(self, action: np.ndarray) -> np.ndarray:
        if self.flip_mode in ("xy", "xy_rot"):
            action[[0, 1]] *= -1.0
        if self.flip_mode == "xy_rot":
            action[[3, 4]] *= -1.0
        return action

    def act(self, image: np.ndarray, instruction: Optional[str] = None, obs=None) -> np.ndarray:
        if self.pred is None:
            if getattr(self, "oracle", False) and getattr(self, "_gt_poses", None) is not None:
                self.pred = dict(self._gt_poses)
                print("[localize] ORACLE poses", flush=True)
            else:
                self.pred = self._predict(image)
            self.phase = 0
            self.counter = 0
            print(f"[localize] pred src={np.round(self.pred['src'], 3)} dst={np.round(self.pred['dst'], 3)} "
                  f"yaw={self.pred['yaw']:.2f}", flush=True)

        src, dst = self.pred["src"], self.pred["dst"]
        p = self.p
        off = np.array([np.cos(self.pred["yaw"]), np.sin(self.pred["yaw"]), 0.0]) * p["grasp_along"]
        src = src + off
        phases = [
            ("align", None),
            ("move", (src + np.array([0, 0, p["approach_h"]]), 1.0, 0.03)),
            ("move", (src + np.array([0, 0, p["descend_dz"] + p["close_dz"]]), 1.0, 0.015)),
            ("hold", (-1.0, 12)),
            ("move", (src + np.array([0, 0, p["lift_h"]]), -1.0, 0.03)),
            ("move", (dst + np.array([0, 0, p["approach_h"]]), -1.0, 0.03)),
            ("move", (dst + np.array([0, 0, p["place_dz"]]), -1.0, 0.015)),
            ("hold", (1.0, 12)),
        ]
        while self.phase < len(phases):
            kind, arg = phases[self.phase]
            if kind == "align":
                a = self._align_action()
                self.counter += 1
                if a is None or self.counter >= 25:
                    self.phase += 1
                    self.counter = 0
                    if a is None:
                        continue
                return self._post(a)
            if kind == "hold":
                gripper, n = arg
                a = np.zeros(7, dtype=np.float32)
                a[6] = gripper
                self.counter += 1
                if self.counter >= n:
                    self.phase += 1
                    self.counter = 0
                return self._post(a)
            target, gripper, max_delta = arg
            budget = {0: 45, 2: 35, 4: 30, 5: 60, 6: 35}.get(self.phase, 60)
            d = target - self._tcp_pos()
            tol = self.p["descend_tol"] if self.phase in (2, 6) else self.p["xy_tol"]
            if np.linalg.norm(d) < tol or self.counter >= budget:
                self.phase += 1
                self.counter = 0
                continue
            self.counter += 1
            return self._post(self._move_action(target, gripper, max_delta))
        return np.zeros(7, dtype=np.float32)



class HeatmapPolicy(LocalizePolicy):
    """Same controller, but perception comes from the heatmap localizer."""

    def __init__(self, head_dir: str, device: str = "cuda:0", flip_mode: str = "none", **kw) -> None:
        import torch as _t
        from train_localizer_hm import HeatLocalizer, apply_h, W, H

        self._apply_h = staticmethod(apply_h)
        self._W, self._H = W, H
        ck = _t.load(os.path.join(head_dir, "localizer_hm.pt"), map_location="cpu", weights_only=False)
        self.model = HeatLocalizer(4).to(device).float().eval()
        self.model.load_state_dict(ck["state_dict"])
        self.Hm = ck["Hm"]
        self.src_z = float(ck.get("src_z", 0.881))
        self.dst_z = float(ck.get("dst_z", 0.874))
        self.image_size = int(ck["image_size"])
        self.device = device
        self.flip_mode = flip_mode
        self._tcp = None
        self._tcp_quat = None
        self.pred = None
        self.phase = 0
        self.counter = 0
        print(f"[heatmap] loaded {head_dir} size={self.image_size}")

    @torch.inference_mode()
    def _predict(self, image: np.ndarray) -> dict:
        img = Image.fromarray(image).convert("RGB").resize((self.image_size, self.image_size), Image.BILINEAR)
        x = torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0).permute(2, 0, 1)[None].to(self.device)
        out = self.model(x)[0]
        peaks = []
        for c in range(4):
            flat = int(out[c].reshape(-1).argmax())
            py = (flat // out.shape[-1]) * self._H / out.shape[-1]
            px = (flat % out.shape[-1]) * self._W / out.shape[-1]
            peaks.append(self._apply_h(self.Hm, px, py))
        centre = np.array([peaks[0][0], peaks[0][1], self.src_z])
        v = peaks[1] - peaks[2]
        yaw = float(np.arctan2(v[1], v[0]))
        dst = np.array([peaks[3][0], peaks[3][1], self.dst_z])
        return {"src": centre, "dst": dst, "yaw": yaw}


class TwoStagePolicy(HeatmapPolicy):
    """Stage-1 VLA localizer -> 160 px crop -> stage-2 VLA localizer -> servo."""

    def __init__(self, head_dir: str, device: str = "cuda:0", flip_mode: str = "none", **kw) -> None:
        import torch as _t
        from train_localizer_hm import apply_h
        from train_localizer_vla import HeatDecoder as Dec1, soft_argmax as sa1
        from train_localizer_vla2 import HeatDecoder as Dec2, soft_argmax as sa2

        self._apply_h = staticmethod(apply_h)
        ck2 = _t.load(os.path.join(head_dir, "stage2.pt"), map_location="cpu", weights_only=False)
        self.Hm = ck2["Hm"]
        self.crop = int(ck2["crop"])
        self.src_z = float(ck2.get("src_z", 0.881))
        self.dst_z = float(ck2.get("dst_z", 0.874))
        self.device = device
        self.flip_mode = flip_mode
        self._tcp = None
        self._tcp_quat = None
        self.pred = None
        self.phase = 0
        self.counter = 0

        # stage 1 lives next to stage 2 (passed via env)
        s1dir = os.environ.get("STAGE1_DIR", "")
        ck1 = _t.load(os.path.join(s1dir, "vla_localizer.pt"), map_location="cpu", weights_only=False)
        from transformers import AutoModelForVision2Seq, AutoProcessor
        base = AutoModelForVision2Seq.from_pretrained(ck1.get("base_model", "/data/renxuancheng/mlpj/models/sft-simpler"),
                                                      torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
                                                      trust_remote_code=True, attn_implementation="sdpa").to(device).eval()
        vision = base.vision_backbone
        del base
        if ck1.get("lora"):
            from peft import PeftModel
            vision = PeftModel.from_pretrained(vision, os.path.join(s1dir, "vision_lora")).merge_and_unload()
        if ck2.get("lora"):
            from peft import PeftModel
            vision = PeftModel.from_pretrained(vision, os.path.join(head_dir, "vision_lora")).merge_and_unload()
        self.vision = vision.to(device).eval()
        self.dec1 = Dec1(2176, 256, 56).to(device).float().eval(); self.dec1.load_state_dict(ck1["decoder"])
        self.dec2 = Dec2(2176, 256, 56, int(ck2["decoder"]["up.7.weight"].shape[0])).to(device).float().eval(); self.dec2.load_state_dict(ck2["decoder"])
        self.tf = AutoProcessor.from_pretrained("/data/renxuancheng/mlpj/models/sft-simpler",
                                                trust_remote_code=True).image_processor.apply_transform
        # full-image model used only for the (large, far) target object
        self.dst_full = None
        _dp = os.environ.get("DST_DIR", "")
        if _dp and os.path.isfile(os.path.join(_dp, "vla_localizer.pt")):
            from train_localizer_vla import HeatDecoder as _D1
            _c = _t.load(os.path.join(_dp, "vla_localizer.pt"), map_location="cpu", weights_only=False)
            self.dst_full = _D1(2176, 256, 56).to(device).float().eval()
            self.dst_full.load_state_dict(_c["decoder"])
            from transformers import AutoModelForVision2Seq as _AV
            _b = _AV.from_pretrained("/data/renxuancheng/mlpj/models/sft-simpler", torch_dtype=torch.bfloat16,
                                     low_cpu_mem_usage=True, trust_remote_code=True,
                                     attn_implementation="sdpa").to(device).eval()
            _v = _b.vision_backbone
            del _b
            if _c.get("lora"):
                from peft import PeftModel as _PM
                _v = _PM.from_pretrained(_v, os.path.join(_dp, "vision_lora")).merge_and_unload()
            self.vision_dst = _v.to(device).eval()
            print("[detect] full-image dst head loaded")
        self.yaw_model = None
        _yp = os.environ.get("YAWNET_DIR", "")
        if _yp and os.path.isfile(os.path.join(_yp, "yawnet.pt")):
            from train_yaw_cnn import YawNet
            _yc = _t.load(os.path.join(_yp, "yawnet.pt"), map_location="cpu", weights_only=False)
            self.yaw_model = YawNet(int(_yc["size"])).to(device).float().eval()
            self.yaw_model.load_state_dict(_yc["state_dict"])
            self.yaw_crop = int(_yc["crop"])
            self.yaw_size = int(_yc["size"])
            print(f"[detect] yawnet loaded crop={self.yaw_crop}")
        print(f"[2stage] crop={self.crop}")

    @torch.inference_mode()
    def _predict(self, image: np.ndarray) -> dict:
        from train_localizer_vla import soft_argmax as sa1
        from train_localizer_vla2 import soft_argmax as sa2
        img = Image.fromarray(image).convert("RGB")
        x = self.tf(img).float().div(255.0)[None].to(self.device, dtype=torch.bfloat16)
        out1 = self.dec1(self.vision(x).float())[0]
        p1 = sa1(out1[0])
        coarse = self._apply_h(self.Hm, p1[0], p1[1])
        # crop around the coarse estimate
        px, py = p1
        half = self.crop // 2
        left = int(np.clip(px - half, 0, 640 - self.crop))
        top = int(np.clip(py - half, 0, 480 - self.crop))
        crop = img.crop((left, top, left + self.crop, top + self.crop)).resize((224, 224), Image.BILINEAR)
        xc = self.tf(crop).float().div(255.0)[None].to(self.device, dtype=torch.bfloat16)
        out2 = self.dec2(self.vision(xc).float())[0]
        if out2.shape[0] >= 4:
            pts = []
            for ch in range(4):
                nx, ny = sa2(out2[ch])
                pts.append(self._apply_h(self.Hm, left + nx * self.crop, top + ny * self.crop))
            src_xy, p_e1, p_e2, dst_xy = pts
        else:
            nx, ny = sa2(out2[0])
            src_xy = self._apply_h(self.Hm, left + nx * self.crop, top + ny * self.crop)
            d1 = sa1(out1[3]); dst_xy = self._apply_h(self.Hm, d1[0], d1[1])
            e1 = sa1(out1[1]); e2 = sa1(out1[2])
            p_e1 = self._apply_h(self.Hm, e1[0], e1[1]); p_e2 = self._apply_h(self.Hm, e2[0], e2[1])
        v = np.asarray(p_e1) - np.asarray(p_e2)
        yaw = float(np.arctan2(v[1], v[0]))
        return {"src": np.array([src_xy[0], src_xy[1], self.src_z]),
                "dst": np.array([dst_xy[0], dst_xy[1], self.dst_z]), "yaw": yaw}


class DetectPolicy(LocalizePolicy):
    """Sliding-window VLA detector: no coarse prior, fuse confident windows."""

    def __init__(self, head_dir: str, device: str = "cuda:0", flip_mode: str = "none", **kw) -> None:
        import torch as _t
        from train_localizer_hm import apply_h
        from train_localizer_det import Detector

        self._apply_h = staticmethod(apply_h)
        ck = _t.load(os.path.join(head_dir, "detector.pt"), map_location="cpu", weights_only=False)
        self.crop = int(ck["crop"])
        self.heat = int(ck["heat"])
        self.stride = max(48, self.crop // 2)
        self.device = device
        self.flip_mode = flip_mode
        self._tcp = None
        self._tcp_quat = None
        self.pred = None
        self.phase = 0
        self.counter = 0
        # flat table heights measured from the recordings
        self.src_z = float(os.environ.get("DET_SRC_Z", 0.881))
        self.dst_z = float(os.environ.get("DET_DST_Z", 0.874))
        self.topk = int(os.environ.get("DET_TOPK", 5))
        from transformers import AutoModelForVision2Seq, AutoProcessor
        base = AutoModelForVision2Seq.from_pretrained("/data/renxuancheng/mlpj/models/sft-simpler",
                                                      torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
                                                      trust_remote_code=True, attn_implementation="sdpa").to(device).eval()
        vision = base.vision_backbone
        del base
        if ck.get("lora"):
            from peft import PeftModel
            vision = PeftModel.from_pretrained(vision, os.path.join(head_dir, "vision_lora")).merge_and_unload()
        self.vision = vision.to(device).eval()
        self.dec = Detector(2176, 256, self.heat, 4).to(device).float().eval()
        self.dec.load_state_dict(ck["decoder"])
        self.tf = AutoProcessor.from_pretrained("/data/renxuancheng/mlpj/models/sft-simpler",
                                                trust_remote_code=True).image_processor.apply_transform
        # full-image model used only for the (large, far) target object
        self.dst_full = None
        _dp = os.environ.get("DST_DIR", "")
        if _dp and os.path.isfile(os.path.join(_dp, "vla_localizer.pt")):
            from train_localizer_vla import HeatDecoder as _D1
            _c = _t.load(os.path.join(_dp, "vla_localizer.pt"), map_location="cpu", weights_only=False)
            self.dst_full = _D1(2176, 256, 56).to(device).float().eval()
            self.dst_full.load_state_dict(_c["decoder"])
            from transformers import AutoModelForVision2Seq as _AV
            _b = _AV.from_pretrained("/data/renxuancheng/mlpj/models/sft-simpler", torch_dtype=torch.bfloat16,
                                     low_cpu_mem_usage=True, trust_remote_code=True,
                                     attn_implementation="sdpa").to(device).eval()
            _v = _b.vision_backbone
            del _b
            if _c.get("lora"):
                from peft import PeftModel as _PM
                _v = _PM.from_pretrained(_v, os.path.join(_dp, "vision_lora")).merge_and_unload()
            self.vision_dst = _v.to(device).eval()
            print("[detect] full-image dst head loaded")
        self.yaw_model = None
        _yp = os.environ.get("YAWNET_DIR", "")
        if _yp and os.path.isfile(os.path.join(_yp, "yawnet.pt")):
            from train_yaw_cnn import YawNet
            _yc = _t.load(os.path.join(_yp, "yawnet.pt"), map_location="cpu", weights_only=False)
            self.yaw_model = YawNet(int(_yc["size"])).to(device).float().eval()
            self.yaw_model.load_state_dict(_yc["state_dict"])
            self.yaw_crop = int(_yc["crop"])
            self.yaw_size = int(_yc["size"])
            print(f"[detect] yawnet loaded crop={self.yaw_crop}")
        # world hsomography: pixels -> table plane
        self.Hm = ck["Hm"] if "Hm" in ck else _t.load(
            "/data/renxuancheng/mlpj/runs/vloc2_t1/vla_localizer.pt",
            map_location="cpu", weights_only=False)["Hm"]
        print(f"[detect] crop={self.crop} stride={self.stride} topk={self.topk}")

    def _world_to_pixel(self, xy):
        Hinv = np.linalg.inv(np.asarray(self.Hm))
        v = Hinv @ np.array([float(xy[0]), float(xy[1]), 1.0])
        return float(v[0] / v[2]), float(v[1] / v[2])

    @staticmethod
    def _soft_argmax(h):
        h = h.clamp(min=0)
        s = h.sum() + 1e-8
        hh, ww = h.shape
        xs = torch.arange(ww, dtype=torch.float32, device=h.device).view(1, -1)
        ys = torch.arange(hh, dtype=torch.float32, device=h.device).view(-1, 1)
        return float((h * xs).sum() / s) / ww, float((h * ys).sum() / s) / hh

    @torch.inference_mode()
    def _predict(self, image: np.ndarray) -> dict:
        img = Image.fromarray(image).convert("RGB")
        wins, crops = [], []
        for top in range(0, 480 - self.crop + 1, self.stride):
            for left in range(0, 640 - self.crop + 1, self.stride):
                wins.append((left, top))
                c = img.crop((left, top, left + self.crop, top + self.crop)).resize((224, 224), Image.BILINEAR)
                crops.append(self.tf(c).float() / 255.0)
        x = torch.stack(crops).to(self.device, dtype=torch.bfloat16)
        heat, _pres, yaw_out = self.dec(self.vision(x).float())
        cands = []
        dst_conf = []
        for i, (left, top) in enumerate(wins):
            conf = float(heat[i, 0].max())
            dst_conf.append(float(heat[i, 3].max()))
            pts = []
            for ch in range(4):
                nx, ny = self._soft_argmax(heat[i, ch])
                pts.append(self._apply_h(self.Hm, left + nx * self.crop, top + ny * self.crop))
            cands.append((conf, pts))
        _idx = sorted(range(len(cands)), key=lambda i: -cands[i][0])[: self.topk]
        _w = np.array([max(1e-6, cands[i][0]) for i in _idx])
        _s2 = np.average([np.sin(2 * np.arctan2(yaw_out[i, 0].item(), yaw_out[i, 1].item())) for i in _idx], weights=_w)
        _c2 = np.average([np.cos(2 * np.arctan2(yaw_out[i, 0].item(), yaw_out[i, 1].item())) for i in _idx], weights=_w)
        yaw_head = 0.5 * float(np.arctan2(_s2, _c2))
        # pick the window whose prediction sits closest to its own centre: that is
        # the window where the object is best centred, and the detector is most
        # accurate there.
        best = None
        for idx, (conf, pts) in enumerate(cands):
            left, top = wins[idx]
            cx = self._apply_h(self.Hm, left + self.crop / 2, top + self.crop / 2)
            d = float(np.linalg.norm(np.asarray(pts[0]) - np.asarray(cx)))
            score = d - 0.02 * conf
            if best is None or score < best[0]:
                best = (score, idx, pts, d)
        _score, bidx, bpts, _d = best
        if os.environ.get("DET_MODE", "conf") == "vote":
            # cluster the per-window world predictions; the windows that actually
            # contain the object agree with each other
            P = np.array([c[1][0] for c in cands])
            C = np.array([c[0] for c in cands])
            keep = C >= np.percentile(C, 50)
            P = P[keep]; C = C[keep]
            best = None
            for i in range(len(P)):
                d = np.linalg.norm(P - P[i], axis=1)
                inl = d < 0.03
                wsum = float(C[inl].sum())
                if best is None or wsum > best[0]:
                    best = (wsum, inl)
            _ws, inl = best
            src = np.average(P[inl], axis=0, weights=C[inl])
            # cluster the towel predictions as well
            P3 = np.array([c[1][3] for c in cands])
            C3 = np.asarray(dst_conf)
            best3 = None
            for i in range(len(P3)):
                d3 = np.linalg.norm(P3 - P3[i], axis=1)
                inl3 = d3 < 0.04
                w3 = float(C3[inl3].sum())
                if best3 is None or w3 > best3[0]:
                    best3 = (w3, inl3)
            _w3, inl3 = best3
            dst = np.average(P3[inl3], axis=0, weights=C3[inl3])
        if self.dst_full is not None:
            from train_localizer_vla import soft_argmax as _sa1
            _xf = self.tf(img).float().div(255.0)[None].to(self.device, dtype=torch.bfloat16)
            _o = self.dst_full(self.vision_dst(_xf).float())[0]
            _p3 = _sa1(_o[3])
            if _p3 is not None:
                dst = np.asarray(self._apply_h(self.Hm, _p3[0], _p3[1]))
                dst_full = True
            dst_vote = True
            j = int(np.argmax(C))
            e1 = np.asarray(cands[j][1][1]); e2 = np.asarray(cands[j][1][2])
        elif os.environ.get("DET_MODE", "conf") == "center":
            src, e1, e2, dst = [np.asarray(q) for q in bpts]
        else:
            top = cands[: max(1, self.topk)]
            w = np.array([max(1e-6, c[0]) for c in top])
            src = np.average(np.array([c[1][0] for c in top]), axis=0, weights=w)
            jd = int(np.argmax(np.asarray(dst_conf)))
            dst = np.asarray(cands[jd][1][3])
            e1 = np.average(np.array([c[1][1] for c in top]), axis=0, weights=w)
            e2 = np.average(np.array([c[1][2] for c in top]), axis=0, weights=w)
        if self.yaw_model is not None:
            _px = self._world_to_pixel(src)
            _half = self.yaw_crop // 2
            _l = int(np.clip(_px[0] - _half, 0, 640 - self.yaw_crop))
            _t = int(np.clip(_px[1] - _half, 0, 480 - self.yaw_crop))
            _c = img.crop((_l, _t, _l + self.yaw_crop, _t + self.yaw_crop)).resize((self.yaw_size, self.yaw_size), Image.BILINEAR)
            _x = torch.from_numpy(np.asarray(_c, dtype=np.float32) / 255.0).permute(2, 0, 1)[None].to(self.device)
            with torch.inference_mode():
                _o = self.yaw_model(_x)[0]
            yaw_head = 0.5 * float(np.arctan2(_o[0].item(), _o[1].item()))
        if os.environ.get("DET_CASCADE", "0") != "1":
            v = e1 - e2
            yaw = float(np.arctan2(v[1], v[0])) if os.environ.get("DET_YAW", "head") != "head" else yaw_head
            return {"src": np.array([src[0], src[1], self.src_z]),
                    "dst": np.array([dst[0], dst[1], self.dst_z]), "yaw": yaw}
        # second pass: crop centred on the winning prediction and decode again
        c1 = self._apply_h(self.Hm, 0, 0)  # placeholder to keep shapes obvious
        centre_world = np.asarray(bpts[0])
        px = self._world_to_pixel(centre_world)
        half = self.crop // 2
        left2 = int(np.clip(px[0] - half, 0, 640 - self.crop))
        top2 = int(np.clip(px[1] - half, 0, 480 - self.crop))
        c = img.crop((left2, top2, left2 + self.crop, top2 + self.crop)).resize((224, 224), Image.BILINEAR)
        x2 = self.tf(c).float().div(255.0)[None].to(self.device, dtype=torch.bfloat16)
        heat2, _ = self.dec(self.vision(x2).float())
        pts2 = []
        for ch in range(4):
            nx, ny = self._soft_argmax(heat2[0, ch])
            pts2.append(self._apply_h(self.Hm, left2 + nx * self.crop, top2 + ny * self.crop))
        src, e1, e2, dst = [np.asarray(q) for q in pts2]
        v = e1 - e2
        yaw = float(np.arctan2(v[1], v[0])) if os.environ.get("DET_YAW", "head") != "head" else yaw_head
        return {"src": np.array([src[0], src[1], self.src_z]),
                "dst": np.array([dst[0], dst[1], self.dst_z]), "yaw": yaw}
