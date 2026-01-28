# Whole-Body IK RL (简化版)

本仓库是一个精简的移动底盘 + 机械臂 IK 强化学习任务，基于 MuJoCo 与 PPO 训练。
目标：控制底盘与 6 关节机械臂，让 `ee_site` 到达目标点。

任务设定（已简化）：
- 目标点位于底盘前方 ±60° 扇形
- 距离 2–3m
- 目标 z 相对 `ee_site` 在 ±10cm 内
- 观测仅 `state(8)` + `target(3)`
- 动作为 9 维：底盘(3) + 6 关节
- 夹爪固定开，不参与控制

## 目录结构

- `train_ppo.py` 训练入口
- `ppo_main.py` PPO 训练逻辑
- `mobile_robot_env.py` 环境定义（MuJoCo）
- `test_ik_task.py` 测试评估脚本（可视化）
- `model_encoder/state_target_encoder.py` 编码器
- `model_policy/mlp_agent.py` 策略/价值网络
- `model_assets/fw_mini_single_piper/fw_mini_single_piper_v2.xml` MuJoCo 模型
- `car_controller/` 底盘控制器

## 环境配置

建议使用 Python 3.9+，并确保 MuJoCo 可用。

安装依赖（按需增减）：

```bash
pip install torch gymnasium mujoco tyro numpy tqdm
```

需使用 wandb 与 GPU：
```bash
pip install wandb
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

> 如需其他 CUDA 版本，请替换为对应的 PyTorch 官方安装源。

## 启动方式

### 训练

```bash
python train_ppo.py
```

训练参数可以在 ppo_main.py ， 中的 class PPOArgs: 内修改


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

> 训练渲染仅建议 `--ppo.num-envs 1`，否则会禁用可视化。

## 说明

- 观测：`state(8) = base_xy + 6 关节`，`target(3) = ee_site -> target`
- 动作：`[vx, vy, wz] + 6 关节增量`（tanh 高斯策略）
- 终止条件：成功（距离阈值）或超出最大步数

如需调整奖励、终止条件或目标采样范围，请在 `mobile_robot_env.py` 中修改。
