from dataclasses import dataclass, field
import json
from typing import Optional

import tyro

from ppo_main import PPOArgs, train


def _load_env_kwargs(path: Optional[str]) -> dict:
    if not path:
        return {}
    with open(path, "r") as f:
        return json.load(f)


@dataclass
class Args:
    env_id: str = "PiperIKEnv"  # 环境 ID
    env_kwargs_json_path: Optional[str] = None  # 环境参数 JSON 路径
    ppo: PPOArgs = field(default_factory=PPOArgs)  # PPO 参数


def main(args: Args) -> None:
    args.ppo.env_id = args.env_id
    env_kwargs = _load_env_kwargs(args.env_kwargs_json_path)
    if env_kwargs:
        args.ppo.env_kwargs = env_kwargs
    train(args=args.ppo)


if __name__ == "__main__":
    main(tyro.cli(Args))
