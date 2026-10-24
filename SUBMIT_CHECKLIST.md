# 提交前检查清单（B 类，对照文档 §9 / §10）

## 必须替换的占位符

- [x] `README.md` 与 `checkpoint_link.txt` 中的仓库链接 → 已填 `https://github.com/juiceooorange/vla-simpler-b`
- [x] `report_B.pdf` 第 8 节「复现信息」里的仓库与 checkpoint 链接 → 已填并重新导出

## 已随包提供

- [x] `report_B.pdf` 5 页（≤10 页），注明 B 类与组员分工
- [x] `README.md`：基础模型及版本、数据量、训练模块、checkpoint 加载、依赖、训练与评测命令、主要结果
- [x] `checkpoint_link.txt`：权重清单 + 基座名称与版本 + 加载方式
- [x] `results.csv`：12 行汇总（base/final × 4 套协议），含 Wilson 95% CI
- [x] `episodes.csv`：408 行逐 episode（setting/run/task/episode_id/env_seed/model_seed/success/steps/error）
- [x] `train_log.txt`：三个最终模型的训练曲线
- [x] `configs/`：三套初始状态清单 + 相机参数 + 任务配置
- [x] `train.sh`：采集 → 训练 → 评测的完整命令
- [x] `checkpoints/`：76 MB 轻量权重（det_t1c / yawnet_t1 / vloc2_t1 / head_prop）
- [x] `videos/`：T1 成功、T2 成功、T2 失败（3 段，覆盖两个任务且含失败样例）

## 对照 §10 B 类检查项

- [x] checkpoint 有 WidowX 数据训练依据（SFT-Simpler）
- [x] 数据来源、生成/筛选方法、轨迹划分、动作归一化写明（README §2）
- [x] 训练过可学习模块，保存配置与日志（`train_log.txt`）
- [x] Final 可在独立进程加载（`evaluate.py`），checkpoint 选择规则写明
- [x] 正式评测初始状态预先确定，未用于训练/调参/选模
- [x] Base / Final 在 T1、T2 各完成 10 次评测
- [x] 报告训练前后成功率、平均分与变化
- [x] 分析至少两个代表性 episode（报告 §7.1）
- [x] 逐 episode 结果与视频随包
- [x] 报告 ≤10 页，含复现链接位置与组员分工

## 打包

```bash
# 仓库（推送用）
git init && git add . && git commit -m "B class submission: VLA lightweight finetuning on SimplerEnv"
# 注意：checkpoints/ 76 MB 若超过仓库限制，改为在 checkpoint_link.txt 中给外链

# 提交包（压缩）
zip -r B_submission.zip README.md report_B.pdf checkpoint_link.txt results.csv episodes.csv train_log.txt \
    configs policies evaluate.py train_*.py scripted_demos.py generate_rollouts.py train.sh requirements.txt videos
```
