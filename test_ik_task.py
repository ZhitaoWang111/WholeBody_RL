import argparse
import numpy as np
import torch

from mobile_robot_env import PiperIKEnv
from model_encoder.state_target_encoder import StateTargetEncoder as Encoder
from model_policy.mlp_agent import MLP_Agent as Agent


def _convert_obs(obs_dict, device):
    converted = {}
    for key in ("state", "target"):
        if key not in obs_dict:
            continue
        value = obs_dict[key]
        if torch.is_tensor(value):
            tensor = value.to(dtype=torch.float32, device=device)
        else:
            tensor = torch.as_tensor(value, dtype=torch.float32, device=device)
        if tensor.dim() == 1:
            tensor = tensor.unsqueeze(0)
        converted[key] = tensor
    return converted


def _load_agent(checkpoint_path, sample_obs, action_space, device):
    encoder = Encoder(sample_obs=sample_obs).to(device)
    agent = Agent(action_space, sample_obs=sample_obs, Encoder=encoder, device=device).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state" in checkpoint:
        agent.load_state_dict(checkpoint["model_state"])
    else:
        agent.load_state_dict(checkpoint)
    agent.eval()
    return agent


def main() -> None:
    parser = argparse.ArgumentParser(description="IK model evaluation (single env, MuJoCo window).")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="runs/piper_ik_try3/100.pt",
    )
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=40)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--target-min", type=float, default=2.0)
    parser.add_argument("--target-max", type=float, default=3.0)
    parser.add_argument("--sim-steps", type=int, default=20)
    parser.add_argument("--cuda", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")

    env = PiperIKEnv(
        visualization=True,
        max_episode_length=args.max_steps,
        target_min_dist=args.target_min,
        target_max_dist=args.target_max,
        sim_steps_per_action=args.sim_steps,
    )

    obs, _ = env.reset(seed=args.seed)
    obs_t = _convert_obs(obs, device)
    agent = _load_agent(args.checkpoint, obs_t, env.action_space, device)

    success_count = 0
    for ep in range(args.episodes):
        obs, _ = env.reset(seed=args.seed + ep)
        done = False

        for step in range(args.max_steps):
            obs_t = _convert_obs(obs, device)
            with torch.no_grad():
                action = agent.get_action(obs_t, deterministic=True)
            action_np = action.squeeze(0).cpu().numpy().reshape(env.action_space.shape)

            obs, reward, terminated, truncated, info = env.step(action_np)
            if terminated or truncated:
                done = True
                if info.get("is_success", False):
                    success_count += 1
                break

        if not done and info.get("is_success", False):
            success_count += 1

        print(
            f"[ep {ep}] success={info.get('is_success', False)} "
            f"steps={info.get('step_number', step + 1)}"
        )

    success_rate = success_count / max(1, args.episodes)
    print(f"Success rate: {success_rate:.3f} ({success_count}/{args.episodes})")

    env.close()


if __name__ == "__main__":
    main()
