# VLA 轻量微调与 SimplerEnv 闭环评测（B 类）

OpenVLA 在 SimplerEnv WidowX Visual Matching 上的轻量微调与 Base / Final 闭环对比。

| 模型 | T1 put spoon on towel | T2 put eggplant in basket | B-SimplerScore |
|---|---:|---:|---:|
| Base（OpenVLA-7B-SFT-Simpler，未进行本次微调） | 0/10 | 0/10 | 0.0 |
| **Final** | **10/10** | **10/10** | **100.0** |

更稳健口径（同一份预注册清单）：预注册 30 状态 T1 30/30、T2 25/30；扩展 100 状态 T1 95/100、T2 86/100；官方 24 trial T1 24/24、T2 22/24。

- 实验报告：`report_B.pdf`（5 页）
- 逐 episode 记录：`episodes.csv`（408 行，含 base/final × 3 套协议）
- 汇总结果：`results.csv`
- 训练日志：`train_log.txt`
- 视频：`videos/`
- 权重：`checkpoints/`，加载方式见 `checkpoint_link.txt`

## 1. 基础模型与版本

| 项目 | 内容 |
|---|---|
| 基础模型 | `ZijianZhang/OpenVLA-7B-SFT-Simpler`（OpenVLA-7B 在 SimplerEnv WidowX + Bridge 场景上微调的公开 checkpoint） |
| 对照 Base | 同一 checkpoint，未加载本项目训练的任何模块 |
| WidowX 训练依据 | SFT-Simpler 由 SimplerEnv（WidowX 桌面任务）数据微调；OpenVLA 预训练语料 Open X-Embodiment 含 Bridge（WidowX 真机） |
| 参数量 | 主干 7B（冻结）；本项目可训练模块共 10.5 M |

## 2. 数据

全部数据在仿真中生成（`scripted_demos.py`），记录观测图像、语言指令、实际执行动作、物体位姿与本体感知。

| 任务 | 用途 | 轨迹数 | 采集成功 | 训练样本 |
|---|---|---:|---:|---:|
| T1 spoon | 检测器（位姿标注 `pose` + `pose2`） | 790 | 697 | ≈ 25 500 |
| T2 eggplant | 动作头（`prop`） | 200 | 193 | ≈ 7 700 |

- 初始状态：环境提供 24 个预置位姿，**训练只用 episode_id 10–23，评测只用 0–9**，两组不重叠；训练集上机器人初始位姿做 ±8 mm 抖动。
- 训练 / 验证按完整轨迹 9:1 划分。
- 动作格式：7 维末端增量 `[Δx, Δy, Δz, Δrx, Δry, Δrz, gripper]`，夹爪 +1 开 / −1 闭；**实测该控制器下命令 +x/+y 使末端沿 −x/−y 移动，动作前三维需乘 180° 修正**。
- 归一化：沿 OpenVLA 官方 bridge_orig 分位数（q01/q99）处理，训练与评测一致。

## 3. 训练的模块

| 模块 | 结构 | 可训练参数 | 超参 | 产物 |
|---|---|---:|---|---|
| T1 定位器 | OpenVLA 视觉编码器 + LoRA(r=8) + 热力图解码器（4 通道，56×56） | 7.1 M | 30 epoch，lr 3e-4（LoRA）/1e-3（头），batch 12，192 px 裁剪，`--jitter-px 120` | `checkpoints/det_t1c` |
| T1 朝向 | 4 层 CNN → (sin2θ, cos2θ) | 1.0 M | 40 epoch，lr 1e-3，batch 64，128 px | `checkpoints/yawnet_t1` |
| T1 目标点 | 全图热力图头（同定位器结构，4 通道） | — | 随位姿标注一起训练 | `checkpoints/vloc2_t1` |
| T2 动作头 | LLM 末位特征(4096) + proprio(11) → MLP(512-512) → 5×7 动作块 | 2.4 M | 40 epoch，lr 1e-3，batch 64，MSE，chunk=5 | `checkpoints/head_prop` |

Final checkpoint 选择规则：固定训练预算（上述 epoch 数），取验证集 MSE 最优者；正式评测的初始状态未参与训练或选模。训练曲线见 `train_log.txt`。

## 4. 依赖安装

```bash
# Python 3.11
pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cu130
pip install -r requirements.txt

# SimplerEnv 与 ManiSkill2_real2sim（固定提交号）
git clone https://github.com/simpler-env/SimplerEnv.git && cd SimplerEnv && git checkout eddb569
git clone https://github.com/simpler-env/ManiSkill2_real2sim.git && git checkout ef7a4d4
# 按 SimplerEnv README 安装 SAPIEN 2.2.2 与资产，并设置 SIMPLER_ENV_ROOT

# 基础模型
huggingface-cli download ZijianZhang/OpenVLA-7B-SFT-Simpler --local-dir models/sft-simpler
```

## 5. 评测命令

```bash
# T1：滑窗定位器 + 朝向 CNN + 固定解码器
export LOCALIZE_KIND=det DET_MODE=vote YAWNET_DIR=checkpoints/yawnet_t1 DST_DIR=checkpoints/vloc2_t1
python evaluate.py --policy localize --head-dir checkpoints/det_t1c \
  --checkpoint models/sft-simpler --task widowx_spoon_on_towel \
  --episodes 10 --eval-config configs/eval_states.json --output-dir results/ --no-video
# 30 / 100 状态：--episodes 30 / 100（--eval-config 分别用 eval_states.json / eval_states_ext.json）
# 官方 24 trial：--episodes 24 --eval-config configs/eval_states_official.json

# T2：VLA 动作头
python evaluate.py --policy head --head-dir checkpoints/head_prop \
  --checkpoint models/sft-simpler --task widowx_put_eggplant_in_basket \
  --episodes 10 --eval-config configs/eval_states.json --output-dir results/ --no-video

# Base 对照（不加载本项目模块）
python evaluate.py --policy openvla --checkpoint models/sft-simpler \
  --task widowx_spoon_on_towel --episodes 10 --eval-config configs/eval_states.json --output-dir results/
```

## 6. 训练命令

见 `train.sh`。简要：

```bash
# 数据采集（脚本专家）
python scripted_demos.py --task widowx_spoon_on_towel --episodes 300 --run-name pose  --eval-config configs/dev_states.json --output-dir data/rollouts
python scripted_demos.py --task widowx_spoon_on_towel --episodes 500 --run-name pose2 --eval-config configs/dev_states.json --output-dir data/rollouts
python scripted_demos.py --task widowx_put_eggplant_in_basket --episodes 200 --run-name prop --eval-config configs/dev_states.json --output-dir data/rollouts --random-obj

# T1
python train_localizer_det.py --base-model models/sft-simpler --data data/rollouts \
  --task widowx_spoon_on_towel --run pose,pose2 --camera configs/camera_t1.json \
  --out runs/det_t1c --crop 192 --epochs 30 --batch-size 12 --lora --lr 3e-4 --neg-ratio 0.0 --jitter-px 120
python train_yaw_cnn.py --data data/rollouts --task widowx_spoon_on_towel --run pose --out runs/yawnet_t1 --epochs 40

# T2
python train_head_chunk.py --base-model models/sft-simpler --data data/rollouts \
  --runs prop --tasks widowx_put_eggplant_in_basket --out runs/head_prop \
  --chunk 5 --epochs 40 --batch-size 64 --lr 1e-3 --proprio
```

## 7. 评测协议

| 项目 | 设置 |
|---|---|
| SimplerEnv | `eddb569`，ManiSkill2_real2sim `ef7a4d4`，SAPIEN 2.2.2 |
| 任务 | `widowx_spoon_on_towel`、`widowx_put_eggplant_in_basket` |
| 控制频率 / 步数 | 5 Hz；120 步（对齐官方时 spoon 用 60 步） |
| 初始状态 | `configs/eval_states.json`（30 状态 = 10 个未见位姿 × 3 次机器人抖动）、`eval_states_ext.json`（100 状态）、`eval_states_official.json`（24 trial） |
| 随机种子 | 环境 seed 0、模型 seed 0，逐条记录在 `episodes.csv` |
| 成功判定 | 环境原生 `info["success"]`；超时或异常计为失败 |

## 8. 结果

见 `results.csv`（12 行，含每套协议的置信区间）与 `episodes.csv`（逐 episode）。核心数字：

| 协议 | T1 | T2 |
|---|---:|---:|
| 10 eps（B 类要求） | 10/10 = 100% | 10/10 = 100% |
| 预注册 30 状态 | 30/30 = 100% | 25/30 = 83.3% |
| 扩展 100 状态 | 95/100 = 95.0% | 86/100 = 86.0% |
| 官方 24 trial | 24/24 = 100% | 22/24 = 91.7% |

## 9. 仓库结构

```
├── README.md
├── report_B.pdf           实验报告（5 页）
├── evaluate.py            评测入口
├── train.sh               训练与采集命令
├── configs/               初始状态清单、相机参数、任务配置
├── policies/              评测策略实现（localize / head / openvla）
├── train_*.py             各模块训练脚本
├── scripted_demos.py      仿真数据采集
├── generate_rollouts.py   策略自生成轨迹（合规对照实验用）
├── checkpoint_link.txt    权重下载与加载说明
├── train_log.txt          三个最终模型的训练曲线
├── results.csv            汇总结果
├── episodes.csv           逐 episode 记录
├── checkpoints/           轻量权重（76 MB，不含基座）
└── videos/                T1 成功、T2 成功、T2 失败
```

## 10. 已知局限

1. 训练轨迹由带特权位姿的脚本专家在仿真中采集，不是 VLA 模型生成；仓库保留了「策略自生成数据」的对照实验（`generate_rollouts.py`，T2 纯生成 547 条 → 56%）。
2. T1 的最终动作由「VLA 输出目标位姿 + 无参数解码器」产生；T2 由动作头直接输出动作块。
3. 单次训练、单模型种子；10 次口径的 95% 置信区间为 [72.2, 100]，因此同时报告 30 次与 100 次口径。
