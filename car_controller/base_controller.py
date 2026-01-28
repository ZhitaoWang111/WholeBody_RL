#!/usr/bin/env python3
import argparse
import threading
import time
from typing import Dict, Iterable, List, Optional, Tuple

import mujoco
import mujoco.viewer
import numpy as np

try:
    from .pid import PID
except ImportError:  # pragma: no cover - script usage
    from pid import PID


def get_sensor_data(model, data, name):
    sensor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, name)
    adr = model.sensor_adr[sensor_id]
    dim = model.sensor_dim[sensor_id]
    return data.sensordata[adr:adr + dim].copy()


def damper(value, vmin, vmax):
    if value < vmin:
        return vmin
    if value > vmax:
        return vmax
    return value


def _normalize_names(names: Optional[Iterable[str]], defaults: Iterable[str]) -> List[str]:
    if names is None:
        return list(defaults)
    if isinstance(names, (str, bytes)):
        return [str(names)]
    return list(names)


def ik_base(v_lin_x, v_lin_y, v_yaw, params):
    """
    Calculate the steering angles and wheel velocities for a 4-wheel swerve base.
    Params expects: wheel_radius, steer_track, wheel_base, max_steer_angle_parallel.
    """
    if abs(v_lin_y) < 0.01:
        v_lin_y = 0.0
    if abs(v_lin_x) < 0.01:
        v_lin_x = 0.0
    if abs(v_yaw) < 0.01:
        v_yaw = 0.0
    if abs(v_lin_y) < 0.01 and abs(v_lin_x) < 0.01 and abs(v_yaw) < 0.01:
        return np.zeros(4), np.zeros(4)

    # Straight-line guard: if no lateral or yaw command, lock steering to zero.
    if abs(v_lin_x) < 0.01 and abs(v_yaw) < 0.01:
        steer_ang = np.zeros(4)
        drive_vel = np.array([v_lin_y / params["wheel_radius"]] * 4)
        return steer_ang, drive_vel

    wheel_xy = np.array(params["wheel_xy"], dtype=float)

    steer_ang = np.zeros(4)
    drive_vel = np.zeros(4)
    for i in range(4):
        x_i, y_i = wheel_xy[i]
        v_ix = v_lin_x - v_yaw * y_i
        v_iy = v_lin_y + v_yaw * x_i
        speed = np.hypot(v_ix, v_iy) / params["wheel_radius"]
        steer = -np.arctan2(v_ix, v_iy)

        # Keep steering within [-pi/2, pi/2] by flipping drive direction.
        if steer > np.pi / 2:
            steer -= np.pi
            speed *= -1
        elif steer < -np.pi / 2:
            steer += np.pi
            speed *= -1

        steer = damper(
            steer,
            -params["max_steer_angle_parallel"],
            params["max_steer_angle_parallel"],
        )

        steer_ang[i] = steer
        drive_vel[i] = speed

    return steer_ang, drive_vel


class SwerveBaseController:
    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        params: Dict[str, float],
        steer_act_names: Optional[Iterable[str]] = None,
        drive_act_names: Optional[Iterable[str]] = None,
        steer_act_ids: Optional[Iterable[int]] = None,
        drive_act_ids: Optional[Iterable[int]] = None,
        steer_joint_names: Optional[Iterable[str]] = None,
        drive_joint_names: Optional[Iterable[str]] = None,
        wheel_body_names: Optional[Iterable[str]] = None,
        base_body_name: str = "base_link",
        cmd_vel_swap_xy: bool = True,
        pid_params: Optional[Dict[str, float]] = None,
        use_pid: bool = True,
    ):
        self.model = model
        self.data = data
        self.cmd_vel_swap_xy = cmd_vel_swap_xy
        self.use_pid = use_pid

        self.steer_act_names = _normalize_names(
            steer_act_names, ["base_sfl", "base_sfr", "base_srl", "base_srr"]
        )
        self.drive_act_names = _normalize_names(
            drive_act_names, ["base_dfl", "base_dfr", "base_drl", "base_drr"]
        )
        self.steer_joint_names = _normalize_names(
            steer_joint_names, ["steer_fl", "steer_fr", "steer_rl", "steer_rr"]
        )
        self.drive_joint_names = _normalize_names(
            drive_joint_names, ["drive_fl", "drive_fr", "drive_rl", "drive_rr"]
        )
        self.base_body_name = base_body_name
        self.wheel_body_names = _normalize_names(
            wheel_body_names, ["Wheel3_Link", "Wheel4_Link", "Wheel1_Link", "Wheel2_Link"]
        )

        if steer_act_ids is not None or drive_act_ids is not None:
            if steer_act_ids is None or drive_act_ids is None:
                raise ValueError("Both steer_act_ids and drive_act_ids must be provided together.")
            self.steer_act_ids = list(steer_act_ids)
            self.drive_act_ids = list(drive_act_ids)
        else:
            self.steer_act_ids = self._lookup_actuators(self.steer_act_names)
            self.drive_act_ids = self._lookup_actuators(self.drive_act_names)

        if "wheel_xy" not in params:
            params["wheel_xy"] = self._compute_wheel_xy()
        self.params = params

        pid_params = pid_params or {}
        self.steer_pid = PID(
            "steer",
            pid_params.get("kp_steer", 50.0),
            pid_params.get("ki_steer", 2.5),
            pid_params.get("kd_steer", 7.5),
            dim=4,
            llim=pid_params.get("llim_steer", -50.0),
            ulim=pid_params.get("ulim_steer", 50.0),
            debug=False,
        )
        self.drive_pid = PID(
            "drive",
            pid_params.get("kp_drive", 5.0),
            pid_params.get("ki_drive", 1e-3),
            pid_params.get("kd_drive", 1e-1),
            dim=4,
            llim=pid_params.get("llim_drive", -200.0),
            ulim=pid_params.get("ulim_drive", 200.0),
            debug=False,
        )

    def _clamp_ctrl(self, act_id: int, value: float) -> float:
        if self.model.actuator_ctrllimited[act_id]:
            lo, hi = self.model.actuator_ctrlrange[act_id]
            return float(np.clip(value, lo, hi))
        return float(value)

    def _lookup_actuators(self, names: Iterable[str]) -> List[int]:
        act_ids = []
        for name in names:
            act_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            if act_id < 0:
                raise ValueError(f"Actuator '{name}' not found in model.")
            act_ids.append(act_id)
        return act_ids

    def _compute_wheel_xy(self) -> List[List[float]]:
        mujoco.mj_forward(self.model, self.data)
        base = self.data.body(self.base_body_name)
        rot = base.xmat.reshape(3, 3)
        p0 = base.xpos
        wheel_xy = []
        for name in self.wheel_body_names:
            pw = self.data.body(name).xpos
            pb = rot.T @ (pw - p0)
            wheel_xy.append([float(pb[0]), float(pb[1])])
        return wheel_xy

    def reset(self) -> None:
        self.steer_pid.reset()
        self.drive_pid.reset()

    def update(self, v_lin_x: float, v_lin_y: float, v_yaw: float) -> Dict[int, float]:
        if abs(v_lin_x) < 1e-3 and abs(v_lin_y) < 1e-3 and abs(v_yaw) < 1e-3:
            self.reset()
            return {act_id: 0.0 for act_id in self.steer_act_ids + self.drive_act_ids}

        if self.cmd_vel_swap_xy:
            steer_cmd, drive_cmd = ik_base(v_lin_y, v_lin_x, v_yaw, self.params)
        else:
            steer_cmd, drive_cmd = ik_base(v_lin_x, v_lin_y, v_yaw, self.params)

        if self.use_pid:
            if not self.steer_joint_names or not self.drive_joint_names:
                raise ValueError("steer_joint_names/drive_joint_names are required when use_pid=True.")
            current_steer_pos = np.array(
                [self.data.joint(name).qpos[0] for name in self.steer_joint_names], dtype=float
            )
            current_drive_vel = np.array(
                [self.data.joint(name).qvel[0] for name in self.drive_joint_names], dtype=float
            )

            mv_steer = self.steer_pid.update(steer_cmd, current_steer_pos, self.data.time)
            mv_drive = self.drive_pid.update(drive_cmd, current_drive_vel, self.data.time)
        else:
            # Direct command mapping (no PID)
            mv_steer = steer_cmd
            mv_drive = drive_cmd

        ctrl = {}
        for idx, act_id in enumerate(self.steer_act_ids):
            ctrl[act_id] = self._clamp_ctrl(act_id, mv_steer[idx])
        for idx, act_id in enumerate(self.drive_act_ids):
            ctrl[act_id] = self._clamp_ctrl(act_id, mv_drive[idx])
        return ctrl

    def apply(self, v_lin_x: float, v_lin_y: float, v_yaw: float) -> None:
        ctrl = self.update(v_lin_x, v_lin_y, v_yaw)
        for act_id, value in ctrl.items():
            self.data.ctrl[act_id] = value


def main():
    parser = argparse.ArgumentParser(description="Interactive control for fw_car.xml")
    parser.add_argument("--xml", default="assets/urdf/fw_car.xml", help="Path to MJCF XML")
    parser.add_argument("--viewer", action="store_true", help="Open Mujoco viewer")
    parser.add_argument("--print-rate", type=float, default=1.0, help="Hz for telemetry print, 0 to disable")
    parser.add_argument("--print-wheel-xy", action="store_true", help="Print wheel positions in base_link frame and exit")
    # Base kinematics params (update these if your car model changes)
    parser.add_argument("--wheel-radius", type=float, default=0.07)
    parser.add_argument("--steer-track", type=float, default=0.25)
    parser.add_argument("--wheel-base", type=float, default=0.36)
    parser.add_argument("--max-steer-angle-parallel", type=float, default=1.570)
    parser.add_argument("--min-turn-radius", type=float, default=0.47644)
    # PID params (update if you tune the controller)
    parser.add_argument("--kp-drive", type=float, default=5.0)
    parser.add_argument("--ki-drive", type=float, default=1e-3)
    parser.add_argument("--kd-drive", type=float, default=1e-1)
    parser.add_argument("--llim-drive", type=float, default=-200.0)
    parser.add_argument("--ulim-drive", type=float, default=200.0)
    parser.add_argument("--kp-steer", type=float, default=50.0)
    parser.add_argument("--ki-steer", type=float, default=2.5)
    parser.add_argument("--kd-steer", type=float, default=7.5)
    parser.add_argument("--llim-steer", type=float, default=-50.0)
    parser.add_argument("--ulim-steer", type=float, default=50.0)
    args = parser.parse_args()

    model = mujoco.MjModel.from_xml_path(args.xml)
    data = mujoco.MjData(model)

    if args.print_wheel_xy:
        mujoco.mj_forward(model, data)
        base = data.body("base_link")
        rot = base.xmat.reshape(3, 3)
        p0 = base.xpos
        wheel_names = ["Wheel3_Link", "Wheel4_Link", "Wheel1_Link", "Wheel2_Link"]  # FL, FR, RL, RR
        for name in wheel_names:
            pw = data.body(name).xpos
            pb = rot.T @ (pw - p0)
            print("{} base_frame_xyz = [{:+.4f} {:+.4f} {:+.4f}]".format(name, pb[0], pb[1], pb[2]))
        return

    steer_act_names = ["base_sfl", "base_sfr", "base_srl", "base_srr"]
    drive_act_names = ["base_dfl", "base_dfr", "base_drl", "base_drr"]
    steer_act_ids = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        for name in steer_act_names
    ]
    drive_act_ids = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        for name in drive_act_names
    ]

    steer_pid = PID(
        "steer",
        args.kp_steer,
        args.ki_steer,
        args.kd_steer,
        dim=4,
        llim=args.llim_steer,
        ulim=args.ulim_steer,
        debug=False,
    )
    drive_pid = PID(
        "drive",
        args.kp_drive,
        args.ki_drive,
        args.kd_drive,
        dim=4,
        llim=args.llim_drive,
        ulim=args.ulim_drive,
        debug=False,
    )

    params = {
        "wheel_radius": args.wheel_radius,
        "steer_track": args.steer_track,
        "wheel_base": args.wheel_base,
        "max_steer_angle_parallel": args.max_steer_angle_parallel,
        "min_turn_radius": args.min_turn_radius,
        # Base-frame wheel positions: FL, FR, RL, RR
        "wheel_xy": [
            [-0.1430, 0.1853],
            [0.1430, 0.1853],
            [-0.1428, -0.1848],
            [0.1430, -0.1848],
        ],
    }

    target_vel = np.zeros(3, dtype=float)
    lock = threading.Lock()
    stop_event = threading.Event()

    def input_loop():
        while not stop_event.is_set():
            try:
                line = input("v_lin_x v_lin_y v_yaw (or 'q'): ").strip()
            except EOFError:
                stop_event.set()
                break
            if line.lower() in {"q", "quit", "exit"}:
                stop_event.set()
                break
            if not line:
                continue
            parts = line.split()
            if len(parts) != 3:
                print("Please input exactly 3 numbers, e.g. 0.0 1.0 0.2")
                continue
            try:
                vals = np.array([float(parts[0]), float(parts[1]), float(parts[2])], dtype=float)
            except ValueError:
                print("Invalid numbers.")
                continue
            with lock:
                target_vel[:] = vals

    thread = threading.Thread(target=input_loop, daemon=True)
    thread.start()

    viewer = None
    if args.viewer:
        viewer = mujoco.viewer.launch_passive(model, data)

    last_print = time.time()
    try:
        while not stop_event.is_set():
            with lock:
                v_cmd = target_vel.copy()

            if np.allclose(v_cmd, 0.0):
                for act_id in steer_act_ids + drive_act_ids:
                    data.ctrl[act_id] = 0.0
                steer_pid.reset()
                drive_pid.reset()
                # Hard stop: zero wheel velocities (and base free joint if present).
                for jname in ["drive_fl", "drive_fr", "drive_rl", "drive_rr"]:
                    try:
                        data.joint(jname).qvel[:] = 0.0
                    except Exception:
                        pass
                try:
                    data.joint("dummy_joint").qvel[:] = 0.0
                except Exception:
                    pass
                mujoco.mj_step(model, data)
                if viewer is not None:
                    viewer.sync()
                time.sleep(model.opt.timestep)
                continue

            # Map input so v_lin_x is forward for this model.
            steer_cmd, drive_cmd = ik_base(v_cmd[1], v_cmd[0], v_cmd[2], params)
            current_steer_pos = np.array(
                [
                    data.joint("steer_fl").qpos[0],
                    data.joint("steer_fr").qpos[0],
                    data.joint("steer_rl").qpos[0],
                    data.joint("steer_rr").qpos[0],
                ]
            )
            current_drive_vel = np.array(
                [
                    data.joint("drive_fl").qvel[0],
                    data.joint("drive_fr").qvel[0],
                    data.joint("drive_rl").qvel[0],
                    data.joint("drive_rr").qvel[0],
                ]
            )

            mv_steer = steer_pid.update(steer_cmd, current_steer_pos, data.time)
            mv_drive = drive_pid.update(drive_cmd, current_drive_vel, data.time)

            for idx, act_id in enumerate(steer_act_ids):
                data.ctrl[act_id] = mv_steer[idx]
            for idx, act_id in enumerate(drive_act_ids):
                data.ctrl[act_id] = mv_drive[idx]

            mujoco.mj_step(model, data)
            now = time.time()
            if args.print_rate > 0 and now - last_print >= 1.0 / args.print_rate:
                last_print = now
                base_body = data.body("base_link")
                cvel = base_body.cvel.copy()
                ang_world = cvel[0:3]
                lin_world = cvel[3:6]
                rot = base_body.xmat.reshape(3, 3)
                ang_body = rot.T @ ang_world
                lin_body = rot.T @ lin_world
                gyro = get_sensor_data(model, data, "imu_gyro")
                act_vx = lin_body[1]
                act_vy = lin_body[0]
                print(
                    "cmd[vx vy wz]=[{:+.3f} {:+.3f} {:+.3f}] | "
                    "act_body[vx vy wz]=[{:+.3f} {:+.3f} {:+.3f}] | "
                    "imu_gyro_z={:+.3f}".format(
                        v_cmd[0], v_cmd[1], v_cmd[2],
                        act_vx, act_vy, ang_body[2],
                        gyro[2],
                    ),
                    flush=True,
                )
            if viewer is not None:
                viewer.sync()
            time.sleep(model.opt.timestep)
    finally:
        if viewer is not None:
            viewer.close()


if __name__ == "__main__":
    main()
