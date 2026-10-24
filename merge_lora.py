"""Merge a LoRA adapter into the base OpenVLA weights so evaluation needs no PEFT.

Usage:
    python merge_lora.py --base-model ../models/openvla-7b \
        --adapter ../runs/lora_r16/checkpoint-1500 --out ../models/openvla-7b-final
"""

import argparse
import os

import torch
from peft import PeftModel
from transformers import AutoModelForVision2Seq, AutoProcessor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    print("[merge] loading base", args.base_model)
    model = AutoModelForVision2Seq.from_pretrained(
        args.base_model, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
        trust_remote_code=True, attn_implementation="sdpa",
    )
    print("[merge] applying adapter", args.adapter)
    model = PeftModel.from_pretrained(model, args.adapter)
    model = model.merge_and_unload()
    model.save_pretrained(args.out, safe_serialization=True, max_shard_size="5GB")
    processor = AutoProcessor.from_pretrained(args.base_model, trust_remote_code=True)
    processor.save_pretrained(args.out)

    # save_pretrained() does not copy the remote-code files, without which the
    # merged directory cannot be loaded with trust_remote_code=True.
    import glob
    import shutil

    patterns = ["*.py", "tokenizer*", "added_tokens.json", "special_tokens_map.json",
                "preprocessor_config.json", "processor_config.json", "generation_config.json"]
    for pattern in patterns:
        for path in glob.glob(os.path.join(args.base_model, pattern)):
            shutil.copy2(path, os.path.join(args.out, os.path.basename(path)))
    print("[merge] wrote", args.out)


if __name__ == "__main__":
    main()
