#!/usr/bin/env bash
# 复现 Final 模型的完整流程（数据采集 → 训练 → 评测）
# 依赖：见 README §4；基础模型放在 models/sft-simpler
set -e

BASE=models/sft-simpler
DATA=data/rollouts

# ---------------------------------------------------------------- 0. 数据采集
# 训练位姿：configs/dev_states.json（episode_id 10–19）；采集脚本内部会覆盖 10–23 的位姿网格
python scripted_demos.py --task widowx_spoon_on_towel --episodes 300 --run-name pose \
  --eval-config configs/dev_states.json --output-dir "$DATA"
python scripted_demos.py --task widowx_spoon_on_towel --episodes 500 --run-name pose2 \
  --eval-config configs/dev_states.json --output-dir "$DATA"
python scripted_demos.py --task widowx_put_eggplant_in_basket --episodes 200 --run-name prop \
  --eval-config configs/dev_states.json --output-dir "$DATA" --random-obj

# ---------------------------------------------------------------- 1. T1 定位器
python train_localizer_det.py --base-model "$BASE" --data "$DATA" \
  --task widowx_spoon_on_towel --run pose,pose2 --camera configs/camera_t1.json \
  --out runs/det_t1c --crop 192 --epochs 30 --batch-size 12 --lora --lr 3e-4 \
  --neg-ratio 0.0 --jitter-px 120

# ---------------------------------------------------------------- 2. T1 朝向 CNN
python train_yaw_cnn.py --data "$DATA" --task widowx_spoon_on_towel --run pose \
  --camera configs/camera_t1.json --out runs/yawnet_t1 --epochs 40

# ---------------------------------------------------------------- 3. T2 动作头
python train_head_chunk.py --base-model "$BASE" --data "$DATA" \
  --runs prop --tasks widowx_put_eggplant_in_basket --out runs/head_prop \
  --chunk 5 --epochs 40 --batch-size 64 --lr 1e-3 --proprio

# ---------------------------------------------------------------- 4. 评测（B 类 10 次口径）
export LOCALIZE_KIND=det DET_MODE=vote YAWNET_DIR=runs/yawnet_t1 DST_DIR=runs/vloc2_t1
python evaluate.py --policy localize --head-dir runs/det_t1c --checkpoint "$BASE" \
  --task widowx_spoon_on_towel --episodes 10 --eval-config configs/eval_states.json \
  --output-dir results/ --no-video

python evaluate.py --policy head --head-dir runs/head_prop --checkpoint "$BASE" \
  --task widowx_put_eggplant_in_basket --episodes 10 --eval-config configs/eval_states.json \
  --output-dir results/ --no-video

# Base 对照
python evaluate.py --policy openvla --checkpoint "$BASE" --task widowx_spoon_on_towel \
  --episodes 10 --eval-config configs/eval_states.json --output-dir results/ --no-video
python evaluate.py --policy openvla --checkpoint "$BASE" --task widowx_put_eggplant_in_basket \
  --episodes 10 --eval-config configs/eval_states.json --output-dir results/ --no-video

# 扩展口径（30 / 100 / 官方 24 trial）
#   --episodes 30  --eval-config configs/eval_states.json
#   --episodes 100 --eval-config configs/eval_states_ext.json
#   --episodes 24  --eval-config configs/eval_states_official.json

echo "done. 结果在 results/，与 results.csv / episodes.csv 对照即可"
