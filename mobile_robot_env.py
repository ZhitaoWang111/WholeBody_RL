"""
Simplified MuJoCo environment for base+arm IK reaching task.
Task: move base and arm to reach a target point 2-5m away from the base.
State-only observation (no images).
"""
import os
import time

import gymnasium as gym
from gymnasium import spaces
import mujoco
import mujoco.viewer
import numpy as np

from car_controller.base_controller import SwerveBaseController


class PiperIKEnv(gym.Env):
    """Piper mobile base + single arm IK reaching task (state-only)."""

    def __init__(
        self,
        visualization: bool = False,
        max_episode_length: int = 40,
        target_min_dist: float = 1.0,
        target_max_dist: float = 2.0,
        sim_steps_per_action: int = 20,
        auto_reset: bool = False,
    ):
        super().__init__()
        self.assets_dir = os.path.join(os.path.dirname(__file__), "model_assets")
        xml_path = os.path.join(self.assets_dir, "fw_mini_single_piper", "fw_mini_single_piper_v2.xml")

        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.model.opt.timestep = 0.002
        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)

        self.visualization = bool(visualization)
        self.auto_reset = bool(auto_reset)
        self._need_reset = False
        if self.visualization:
            self.handle = mujoco.viewer.launch_passive(self.model, self.data)
            self.handle.cam.distance = 3
            self.handle.cam.azimuth = 100
            self.handle.cam.elevation = -60
        else:
            self.handle = None

        # 机械臂关节限位（6 DoF，不含夹爪）
        self.joint_limits = np.array(
            [
                (-2.618, 2.618),
                (0, 3.14),
                (-2.697, 0),
                (-1.832, 1.832),
                (-1.22, 1.22),
                (-3.14, 3.14),
            ],
            dtype=np.float32,
        )

        # 动作：底盘(vx, vy, wz) + 6关节增量
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(9,), dtype=np.float32)

        # 机械臂关节名
        self.arm_joint_names = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
        self.gripper_joint_name = "joint7"
        self.state_joint_names = self.arm_joint_names

        # 末端执行器 site
        self.ee_site_name = "ee_site"

        # 观测：state + target(ee->target)
        self.observation_space = spaces.Dict(
            {
                "state": spaces.Box(low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32),
                "target": spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32),
            }
        )

        # 目标采样参数
        self.target_min_dist = float(target_min_dist)
        self.target_max_dist = float(target_max_dist)
        self.target_pos = np.zeros(3, dtype=np.float32)
        self.target_z_amp = 0.10

        # 回合参数
        self.episode_len = int(max_episode_length)
        self.sim_steps_per_action = int(sim_steps_per_action)

        # 底盘速度缩放 (m/s, m/s, rad/s)
        self.base_cmd_scale = np.array([1.5, 1.5, 2.0], dtype=np.float32)
        self.gripper_fixed = 0.045

        # 奖励系数
        self.success_threshold = 0.04
        self.reach_k = 1.2
        self.base_k = 0.6
        self.reach_scale = 1.0
        self.base_scale = 0.3
        self.progress_scale = 0.6
        self.progress_clip = 0.2
        self.time_penalty = 0.002
        self.success_bonus = 2.0

        self.goal_reached = False
        self.prev_dist_ee = None
        self.prev_dist_base = None

        # 初始关节姿态
        self.init_qpos_left = np.zeros(7, dtype=np.float32)
        self.init_qpos_left[6] = self.gripper_fixed
        self.init_qvel = np.zeros(7, dtype=np.float32)

        self.np_random = np.random.default_rng()

        self._init_base_controller()

    def _init_base_controller(self) -> None:
        params = {
            "wheel_radius": 0.06545,
            "steer_track": 0.25,
            "wheel_base": 0.36,
            "max_steer_angle_parallel": 1.570,
            "min_turn_radius": 0.47644,
        }
        self.base_controller = SwerveBaseController(
            self.model,
            self.data,
            params,
            steer_act_names=["base_sfl", "base_sfr", "base_srl", "base_srr"],
            drive_act_names=["base_dfl", "base_dfr", "base_drl", "base_drr"],
            steer_joint_names=["steer_fl", "steer_fr", "steer_rl", "steer_rr"],
            drive_joint_names=["drive_fl", "drive_fr", "drive_rl", "drive_rr"],
            wheel_body_names=["Wheel3_Link", "Wheel4_Link", "Wheel1_Link", "Wheel2_Link"],
            base_body_name="base_link",
            cmd_vel_swap_xy=True,
            pid_params=None,
            use_pid=True,
        )

    def _get_site_pos_ori(self, site_name: str) -> tuple[np.ndarray, np.ndarray]:
        # 读取 site 的位姿（世界系）
        site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        if site_id == -1:
            raise ValueError(f"Site '{site_name}' not found")

        position = np.asarray(self.data.site(site_id).xpos, dtype=np.float32)
        xmat = np.asarray(self.data.site(site_id).xmat, dtype=np.float64)
        quaternion = np.zeros(4, dtype=np.float64)
        mujoco.mju_mat2Quat(quaternion, xmat)
        return position, quaternion.astype(np.float32)

    def map_action_to_joint_deltas(self, action: np.ndarray) -> np.ndarray:
        # 将归一化动作映射成单步关节增量
        max_delta_per_step = np.array([0.075, 0.045, 0.045, 0.045, 0.045, 0.075], dtype=np.float32)
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (6,):
            raise ValueError(f"Action must be 6D for single arm, got {action.shape}")

        delta_action = action * max_delta_per_step
        return delta_action

    def apply_joint_deltas_with_limits(self, current_qpos: np.ndarray, delta_action: np.ndarray) -> np.ndarray:
        # 应用增量并裁剪到关节限位
        current_qpos = np.asarray(current_qpos, dtype=np.float32)
        delta_action = np.asarray(delta_action, dtype=np.float32)
        new_qpos = current_qpos + delta_action
        lower_bounds = self.joint_limits[:, 0]
        upper_bounds = self.joint_limits[:, 1]
        return np.clip(new_qpos, lower_bounds, upper_bounds)

    def _set_state(
        self,
        qpos_left=None,
        qvel_left=None,
        qpos_base=None,
        qvel_base=None,
    ):
        if qpos_base is not None:
            qpos_base = np.asarray(qpos_base, dtype=np.float32)
            if qpos_base.shape != (7,):
                raise ValueError(f"qpos_base must have shape (7,), got {qpos_base.shape}")
            self.data.qpos[7:14] = np.copy(qpos_base)

        if qvel_base is not None:
            qvel_base = np.asarray(qvel_base, dtype=np.float32)
            if qvel_base.shape != (6,):
                raise ValueError(f"qvel_base must have shape (6,), got {qvel_base.shape}")
            self.data.qvel[6:12] = np.copy(qvel_base)

        if qpos_left is not None:
            qpos_left = np.asarray(qpos_left, dtype=np.float32)
            if qpos_left.shape == (7,):
                self.data.qpos[14:21] = np.copy(qpos_left)
            elif qpos_left.shape == (8,):
                self.data.qpos[14:22] = np.copy(qpos_left)
            else:
                raise ValueError(f"qpos_left must have shape (7,) or (8,), got {qpos_left.shape}")

        if qvel_left is not None:
            qvel_left = np.asarray(qvel_left, dtype=np.float32)
            if qvel_left.shape == (7,):
                self.data.qvel[12:19] = np.copy(qvel_left)
            elif qvel_left.shape == (8,):
                self.data.qvel[12:20] = np.copy(qvel_left)
            else:
                raise ValueError(f"qvel_left must have shape (7,) or (8,), got {qvel_left.shape}")

        mujoco.mj_forward(self.model, self.data)

    def _sample_target(self) -> None:
        # 在底盘前方±60°扇形采样目标点
        base_xy = np.asarray(self.data.body("base_link").xpos[:2], dtype=np.float32)
        ee_pos, _ = self._get_site_pos_ori(self.ee_site_name)

        theta = float(self.np_random.uniform(-np.pi / 3.0, np.pi / 3.0))
        r = float(self.np_random.uniform(self.target_min_dist, self.target_max_dist))
        base_xmat = np.asarray(self.data.body("body_car").xmat, dtype=np.float32).reshape(3, 3)
        # body_car 的朝向：x 轴为前向
        heading_xy = base_xmat[:2, 0]
        heading_xy = heading_xy / (float(np.linalg.norm(heading_xy)) + 1e-6)
        left_xy = np.array([-heading_xy[1], heading_xy[0]], dtype=np.float32)
        dir_xy = np.array([np.cos(theta), np.sin(theta)], dtype=np.float32)
        world_dir = dir_xy[0] * heading_xy + dir_xy[1] * left_xy
        target_xy = base_xy + r * world_dir
        target_z = float(ee_pos[2] + self.np_random.uniform(-self.target_z_amp, self.target_z_amp))
        self.target_pos = np.array([target_xy[0], target_xy[1], target_z], dtype=np.float32)
        self._set_target_body_pose(self.target_pos)

    def _set_target_body_pose(self, pos_xyz: np.ndarray) -> None:
        # 移动可视化目标球体（free joint）
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "target")
        if body_id == -1:
            return
        joint_id = self.model.body_jntadr[body_id]
        qposadr = self.model.jnt_qposadr[joint_id]
        self.data.qpos[qposadr:qposadr + 3] = pos_xyz
        self.data.qpos[qposadr + 3:qposadr + 7] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        qveladr = self.model.jnt_dofadr[joint_id]
        self.data.qvel[qveladr:qveladr + 6] = 0.0


    def reset(self, seed=None, options=None):
        # 重置状态并重新采样目标
        if seed is not None:
            self.seed(seed)

        qpos_base = self.model.qpos0[0:7].copy()
        qvel_base = np.zeros(6, dtype=np.float32)

        qpos_left = self.init_qpos_left.copy()
        qvel_left = self.init_qvel.copy()
        self.data.ctrl[:] = 0.0

        self._set_state(
            qpos_left=qpos_left,
            qvel_left=qvel_left,
            qpos_base=qpos_base,
            qvel_base=qvel_base,
        )

        # print([self.data.joint(n).qpos[0] for n in ["joint1","joint2","joint3","joint4","joint5","joint6"]])

        self.step_number = 0
        self.goal_reached = False
        self._need_reset = False
        self.prev_dist_ee = None
        self.prev_dist_base = None
        self._sample_target()

        obs = self._get_observation()
        return obs, {}

    def _get_state_observation(self):
        # state = base_xy + base_heading_xy + 6关节
        base_xy = np.asarray(self.data.body("base_link").xpos[:2], dtype=np.float32)
        base_xmat = np.asarray(self.data.body("body_car").xmat, dtype=np.float32).reshape(3, 3)
        # body_car 的朝向：x 轴为前向
        heading_xy = base_xmat[:2, 0]
        heading_norm = float(np.linalg.norm(heading_xy))
        if heading_norm < 1e-6:
            heading_xy = np.array([1.0, 0.0], dtype=np.float32)
        else:
            heading_xy = heading_xy / heading_norm
        joint_positions = np.asarray(
            [self.data.joint(name).qpos[0] for name in self.state_joint_names],
            dtype=np.float32,
        )
        return np.concatenate([base_xy, heading_xy, joint_positions], axis=0)

    def _get_observation(self):
        # target 为 ee -> target 的向量
        state_obs = self._get_state_observation()
        ee_pos, _ = self._get_site_pos_ori(self.ee_site_name)
        target = self.target_pos - ee_pos
        return {"state": state_obs, "target": target}

    def _compute_reward(self, action: np.ndarray):
        # 奖励：接近末端 + 底盘距离带 + 进展 - 时间惩罚 + 成功奖励
        ee_pos, _ = self._get_site_pos_ori(self.ee_site_name)
        base_xy = np.asarray(self.data.body("base_link").xpos[:2], dtype=np.float32)

        dist_ee = float(np.linalg.norm(ee_pos - self.target_pos))
        dist_base = float(np.linalg.norm(base_xy - self.target_pos[:2]))

        reach_reward = float(np.exp(-self.reach_k * dist_ee))
        if dist_base < 0.15:
            base_dist_excess = 0.15 - dist_base
        elif dist_base > 0.25:
            base_dist_excess = dist_base - 0.25
        else:
            base_dist_excess = 0.0
        base_reward = float(np.exp(-self.base_k * base_dist_excess))

        progress_ee = 0.0
        progress_base = 0.0
        if self.prev_dist_ee is not None:
            progress_ee = self.prev_dist_ee - dist_ee
        if self.prev_dist_base is not None:
            progress_base = self.prev_dist_base - dist_base

        progress_ee = float(np.clip(progress_ee, -self.progress_clip, self.progress_clip))
        progress_base = float(np.clip(progress_base, -self.progress_clip, self.progress_clip))
        progress_reward = self.progress_scale * (0.7 * progress_ee + 0.3 * progress_base)

        self.prev_dist_ee = dist_ee
        self.prev_dist_base = dist_base

        reward = (
            self.reach_scale * reach_reward
            + self.base_scale * base_reward
            + progress_reward
            - self.time_penalty
        )

        if dist_ee < self.success_threshold:
            reward += self.success_bonus
            self.goal_reached = True

        reward_info = {
            "dist_ee": dist_ee,
            "dist_base": dist_base,
            "reach_reward": reach_reward,
            "base_reward": base_reward,
            "progress_ee": progress_ee,
            "progress_base": progress_base,
            "progress_reward": progress_reward,
            "success": float(self.goal_reached),
        }
        return reward, reward_info

    def step(self, action):
        # 执行动作：底盘速度 + 关节目标，然后推进仿真
        if self._need_reset:
            if self.auto_reset:
                obs, info = self.reset()
                info["auto_reset"] = True
                return obs, 0.0, False, False, info
            raise RuntimeError("Environment needs reset() before calling step() again.")

        action = np.asarray(action, dtype=np.float32)
        if action.shape != (9,):
            raise ValueError(f"Action must have shape (9,), got {action.shape}")

        base_cmd = action[:3] * self.base_cmd_scale
        delta_action = self.map_action_to_joint_deltas(action[3:])

        current_qpos = np.asarray(
            [self.data.joint(name).qpos[0] for name in self.arm_joint_names],
            dtype=np.float32,
        )
        control_indices = slice(8, 15)
        new_qpos = self.apply_joint_deltas_with_limits(current_qpos, delta_action)
        ctrl_target = np.zeros(7, dtype=np.float32)
        ctrl_target[:6] = new_qpos
        ctrl_target[6] = self.gripper_fixed
        self.data.ctrl[control_indices] = ctrl_target

        for _ in range(self.sim_steps_per_action):
            self.base_controller.apply(base_cmd[0], base_cmd[1], base_cmd[2])
            mujoco.mj_step(self.model, self.data)
            if self.visualization and self.handle:
                self.handle.sync()

        self.step_number += 1

        observation = self._get_observation()
        reward, reward_info = self._compute_reward(action)

        terminated = bool(self.goal_reached)
        truncated = self.step_number >= self.episode_len
        self._need_reset = terminated or truncated

        terminated_reasons = []
        if self.goal_reached:
            terminated_reasons.append("success")
        if truncated:
            terminated_reasons.append("time")

        info = {
            "is_success": self.goal_reached,
            "step_number": self.step_number,
            "target_pos": self.target_pos.copy(),
            "terminated_reasons": terminated_reasons,
            "reward_info": reward_info,
            "penalty_info": {
                "time_penalty": float(self.time_penalty),
            },
        }

        return observation, reward, terminated, truncated, info

    def seed(self, seed=None):
        self.np_random = np.random.default_rng(seed)
        return [seed]

    def close(self):
        if hasattr(self, "handle") and self.handle is not None:
            try:
                self.handle.close()
                self.handle = None

                try:
                    import glfw

                    if glfw.get_current_context():
                        glfw.terminate()
                        time.sleep(0.1)
                except ImportError:
                    pass
                except Exception as e:
                    print(f"Warning: GLFW cleanup error: {e}")
            except Exception as e:
                print(f"Warning: Error closing MuJoCo viewer: {e}")

        super().close()

    def __del__(self):
        self.close()


def make_env():
    return PiperIKEnv(visualization=False)


if __name__ == "__main__":
    env = PiperIKEnv(visualization=True)
    obs, info = env.reset()
    print("=== IK Task ===")
    print(f"Action space: {env.action_space}")
    print(f"Observation state shape: {obs['state'].shape}")

    while True:
        action = np.zeros(9, dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            obs, info = env.reset()
