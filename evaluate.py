from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import numpy as np
import torch as th
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import VecVideoRecorder

from CNN_feature_extractor import CNNFeatureExtractor  # noqa: F401 (ensures registry)
from utils import EnvConfig, build_vec_env, set_global_seeds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate trained CarRacing agents.")
    parser.add_argument("--algo", type=str, choices=["ppo", "sac"], required=True)
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--deterministic", action="store_true", help="Use deterministic policy.")
    parser.add_argument("--discrete-actions", action="store_true")
    parser.add_argument("--continuous-actions", action="store_true")
    parser.add_argument("--domain-randomize", action="store_true")
    parser.add_argument("--use-gentle-shaping", action="store_true")
    parser.add_argument("--random-shift", action="store_true", help="Enable augmentation during eval.")
    parser.add_argument("--video-dir", type=str, default=None, help="Optional directory to record evaluation videos.")
    parser.add_argument("--video-length", type=int, default=0, help="Steps per video (0 = full episode).")
    parser.add_argument("--device", type=str, choices=["cpu", "cuda"], default=None)
    parser.add_argument("--render-mode", type=str, default=None, help="Optional Gym render_mode (e.g., 'human').")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.algo == "sac" and args.discrete_actions:
        raise ValueError("SAC evaluation requires continuous actions.")

    device = (
        th.device(args.device)
        if args.device is not None
        else th.device("cuda" if th.cuda.is_available() else "cpu")
    )

    set_global_seeds(args.seed)

    env_cfg = EnvConfig(
        domain_randomize=args.domain_randomize,
        discrete_actions=args.discrete_actions and not args.continuous_actions,
        use_reward_penalty=True,
        use_gentle_shaping=args.use_gentle_shaping,
        random_shift=args.random_shift,
    )

    env = build_vec_env(
        env_cfg,
        num_envs=1,
        seed=args.seed,
        training=False,
        n_stack=4,
        render_mode=args.render_mode,
    )

    if args.video_dir is not None:
        video_path = Path(args.video_dir)
        video_path.mkdir(parents=True, exist_ok=True)
        env = VecVideoRecorder(
            env,
            video_folder=str(video_path),
            record_video_trigger=lambda step: True,
            video_length=args.video_length,
            name_prefix=f"{args.algo}_eval",
        )

    if args.algo == "ppo":
        model = PPO.load(args.model_path, env=env, device=device)
    else:
        model = SAC.load(args.model_path, env=env, device=device)

    returns = []
    deterministic = args.deterministic or args.algo == "sac"

    for episode in range(args.episodes):
        obs = env.reset()
        done = False
        ep_return = 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, done, _ = env.step(action)
            ep_return += float(reward[0])
        returns.append(ep_return)
        print(f"Episode {episode + 1}/{args.episodes}: return={ep_return:.2f}")

    print(
        f"Average return: {np.mean(returns):.2f} ± {np.std(returns):.2f} "
        f"over {args.episodes} episodes."
    )

    env.close()


if __name__ == "__main__":
    main()
