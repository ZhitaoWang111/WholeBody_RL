# Whole-Body IK RL (简化版)

本仓库是一个精简的移动底盘 + 机械臂 IK 强化学习任务，基于 MuJoCo 与 PPO 训练。
目标：控制底盘与 6 关节机械臂，让 `ee_site` 到达目标点。

## 任务示意

![任务示意](figure/task.png)

图中展示了移动底盘 + 机械臂的到达任务、目标采样扇形范围及末端到目标的关系。

## 训练效果

![训练效果](video/result.gif)

任务设定（已简化）：
- 目标点位于底盘前方 ±60° 扇形
- 距离 1–2m
- 目标 z 相对 `ee_site` 在 ±10cm 内
- 观测仅 `state(10)` + `target(3)`
- 动作为 9 维：底盘(3) + 6 关节
- 夹爪固定开，不参与控制

课程机制（target_z）：
- 训练早期：`target_z` 在 `ee_z ± 10cm` 内均匀采样
- 训练后期：逐渐偏向上下两侧（更高/更低）

## 目录结构

```
WholeBody_RL/
├─ car_controller/                 # 底盘控制器
├─ figure/                         # 任务示意图
├─ model_assets/                   # MuJoCo 模型资源
│  └─ fw_mini_single_piper/
│     └─ fw_mini_single_piper_v2.xml
├─ model_encoder/                  # 编码器
│  └─ state_target_encoder.py
├─ model_policy/                   # 策略/价值网络
│  └─ mlp_agent.py
├─ scripts/                        # 工具脚本
│  └─ mp4_to_gif.py
├─ video/                          # 训练效果视频
│  ├─ result.mp4
│  └─ result.gif
├─ mobile_robot_env.py             # 环境定义（MuJoCo）
├─ ppo_main.py                     # PPO 训练逻辑
├─ train_ppo.py                    # 训练入口
├─ test_ik_task.py                 # 测试评估脚本（可视化）
├─ requirements.txt
└─ README.md
```

## ✅ 环境安装

### 创建虚拟环境并安装依赖

```bash
conda create -n rl_ik python=3.10.9
conda activate rl_ik
pip install -r requirements.txt
```

## 启动方式

### 训练
> 训练渲染仅建议 `--ppo.num-envs 1`，否则会禁用可视化。

```bash
python train_ppo.py
```

训练参数可以在 ppo_main.py ， 中的 class PPOArgs: 内修改
常用关键参数（含义）：
- `exp_name`：实验名（会自动拼接时间戳）
- `seed`：随机种子
- `cuda`：是否启用 GPU
- `num_envs`：并行环境数
- `num_steps`：每次 rollout 步数
- `total_timesteps`：总训练步数
- `learning_rate`：学习率
- `gamma`：折扣因子
- `gae_lambda`：GAE 系数
- `update_epochs`：每次更新的 epoch 数
- `num_minibatch`：小批量数
- `clip_coef`：PPO 裁剪系数
- `ent_coef`：熵奖励系数
- `vf_coef`：价值损失系数
- `reward_scale`：奖励缩放
- `save_model`：是否保存模型
- `track` / `wandb_project_name`：wandb 开关与项目名


### 测试评估（带可视化窗口）

```bash
python test_ik_task.py --checkpoint runs/piper_ik_try3/100.pt
```

常用选项：

```bash
python test_ik_task.py \
  --checkpoint runs/piper_ik_try3/100.pt \
  --episodes 10 \
  --max-steps 40 \
  --target-min 2.0 \
  --target-max 3.0 \
  --sim-steps 20
```



## 模型保存

训练默认每 50 次迭代保存一次：

```
runs/<exp_name>__<timestamp>/<iteration>.pt
```

如果关闭保存（`save_model=False`），启动时会打印“本应保存的路径”。

## 说明

- 观测：`state(10) = base_xy + base_heading_xy + 6 关节`，`target(3) = ee_site -> target`
- 动作：`[vx, vy, wz] + 6 关节增量`（tanh 高斯策略）
- 终止条件：成功（距离阈值）或超出最大步数

如需调整奖励、终止条件或目标采样范围，请在 `mobile_robot_env.py` 中修改。

## 视频转换（可选）

将 mp4 转为 README 可预览的 gif：

```bash
python scripts/mp4_to_gif.py --input video/result.mp4 --output video/result.gif --width 720 --fps 10
```
