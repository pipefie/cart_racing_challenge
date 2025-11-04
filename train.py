"""
Train vision-based PPO or SAC agents on Gymnasium CarRacing-v3.

Pipeline (train):
  RecordEpisodeStatistics → RewardPenaltyWrapper → (GentleShapingWrapper) →
  (DiscretizeActionWrapper) → ActionRepeat(k=4) → (Grayscale → Resize) →
  EnsureChannelLast → (RandomShift for training only) → VecFrameStack →
  VecTransposeImage('first') → VecMonitor → policy (CNN encoder divides by 255 once).

Pipeline (eval):
  Identical ordering but RandomShift is disabled and GentleShaping can be toggled off
  for pure performance measurement. DummyVecEnv is used for evaluation.

Usage examples:
  # Memory-aware SAC (continuous actions)
  python train.py --algo sac --total-timesteps 2400000 --buffer-size 300000 --seed 7

  # PPO baseline with discrete wrapper
  python train.py --algo ppo --discrete-actions --num-envs 8 --total-timesteps 2000000

You can also pass a YAML config: python train.py --config configs/sac.yaml
CLI flags override YAML values. Key toggles:
  * --discrete-actions / --continuous-actions
  * --action-repeat
  * --use-gentle-shaping / --gentle-shaping-eval
  * --random-shift-train / --random-shift-eval
"""

from __future__ import annotations

import argparse
import os
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import torch as th
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import VecVideoRecorder
from stable_baselines3.common import logger as sb3_logger

from CNN_feature_extractor import CNNFeatureExtractor
from utils import (
    EnvConfig,
    build_vec_env,
    clone_for_eval,
    linear_schedule,
    load_yaml,
    set_global_seeds,
)


DEFAULTS: Dict[str, Any] = {
    "algo": "sac",
    "total_timesteps": 2_400_000,
    "num_envs": 8,
    "n_stack": 4,
    "seed": 0,
    "discrete_actions": False,
    "domain_randomize": False,
    "use_reward_penalty": True,
    "reward_penalty_value": 1.5,
    "reward_green_ratio": 1.3,
    "speed_reward_scale": 0.0,
    "speed_reward_power": 1.0,
    "reward_scale": 0.1,
    "use_gentle_shaping": False,
    "gentle_shaping_eval": False,
    "gentle_lambda_delta": 1e-3,
    "gentle_lambda_conflict": 1e-3,
    "action_repeat": 4,
    "random_shift_train": True,
    "random_shift_eval": False,
    "total_eval_episodes": 5,
    "eval_freq": 10_000,
    "eval_video_dir": None,
    "eval_video_frequency": None,
    "learning_rate": 3e-4,
    "lr_schedule": "constant",
    "buffer_size": 300_000,
    "batch_size": 256,
    "learning_starts": 50_000,
    "train_freq": 1,
    "gradient_steps": 1,
    "ent_coef": "auto_0.2",
    "gamma": 0.99,
    "tau": 0.005,
    "replay_memory_limit_gb": 6.0,
    "ppo_n_steps": 1024,
    "ppo_batch_size": 256,
    "ppo_epochs": 10,
    "ppo_ent_coef": 0.01,
    "ppo_clip_range": 0.2,
    "tensorboard_dir": "./tb",
    "log_dir": "./logs",
    "checkpoints_dir": "./checkpoints",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PPO/SAC on CarRacing pixels.")
    parser.add_argument("--config", type=str, default=None, help="Optional YAML config path.")
    parser.add_argument("--algo", type=str, choices=["ppo", "sac"], default=None)
    parser.add_argument("--total-timesteps", type=int, default=None)
    parser.add_argument("--num-envs", type=int, default=None)
    parser.add_argument("--n-stack", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--discrete-actions", action="store_true")
    parser.add_argument("--continuous-actions", action="store_true")
    parser.add_argument("--domain-randomize", action="store_true")
    parser.add_argument("--no-domain-randomize", action="store_true")
    parser.add_argument("--use-reward-penalty", action="store_true")
    parser.add_argument("--disable-reward-penalty", action="store_true")
    parser.add_argument("--reward-penalty-value", type=float, default=None)
    parser.add_argument("--reward-green-threshold", type=float, default=None)
    parser.add_argument("--speed-reward-scale", type=float, default=None)
    parser.add_argument("--speed-reward-power", type=float, default=None)
    parser.add_argument("--reward-scale", type=float, default=None)
    parser.add_argument("--use-gentle-shaping", action="store_true")
    parser.add_argument("--gentle-shaping-eval", action="store_true")
    parser.add_argument("--action-repeat", type=int, default=None)
    parser.add_argument("--random-shift-train", action="store_true")
    parser.add_argument("--no-random-shift-train", action="store_true")
    parser.add_argument("--random-shift-eval", action="store_true")
    parser.add_argument("--no-random-shift-eval", action="store_true")
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--lr-schedule", type=str, choices=["constant", "linear"], default=None)
    parser.add_argument("--buffer-size", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--learning-starts", type=int, default=None)
    parser.add_argument("--train-freq", type=int, default=None)
    parser.add_argument("--gradient-steps", type=int, default=None)
    parser.add_argument("--ent-coef", type=str, default=None)
    parser.add_argument("--gamma", type=float, default=None)
    parser.add_argument("--tau", type=float, default=None)
    parser.add_argument("--replay-memory-limit-gb", type=float, default=None)
    parser.add_argument("--ppo-n-steps", type=int, default=None)
    parser.add_argument("--ppo-batch-size", type=int, default=None)
    parser.add_argument("--ppo-epochs", type=int, default=None)
    parser.add_argument("--ppo-ent-coef", type=float, default=None)
    parser.add_argument("--ppo-clip-range", type=float, default=None)
    parser.add_argument("--total-eval-episodes", type=int, default=None)
    parser.add_argument("--eval-freq", type=int, default=None)
    parser.add_argument("--eval-video-dir", type=str, default=None)
    parser.add_argument("--eval-video-frequency", type=int, default=None)
    parser.add_argument("--tensorboard-dir", type=str, default=None)
    parser.add_argument("--log-dir", type=str, default=None)
    parser.add_argument("--checkpoints-dir", type=str, default=None)
    parser.add_argument("--device", type=str, choices=["cpu", "cuda"], default=None)
    return parser.parse_args()


def resolve_run_config(args: argparse.Namespace) -> Dict[str, Any]:
    cfg = deepcopy(DEFAULTS)
    yaml_cfg = load_yaml(args.config)
    cfg.update(yaml_cfg)

    arg_dict = vars(args)

    def maybe_update(key: str, arg_key: str | None = None):
        attr = arg_key or key
        value = arg_dict.get(attr)
        if value is not None:
            cfg[key] = value

    maybe_update("algo")
    maybe_update("total_timesteps", "total_timesteps")
    maybe_update("num_envs", "num_envs")
    maybe_update("n_stack", "n_stack")
    maybe_update("seed")
    maybe_update("action_repeat", "action_repeat")
    maybe_update("learning_rate", "learning_rate")
    maybe_update("lr_schedule", "lr_schedule")
    maybe_update("buffer_size", "buffer_size")
    maybe_update("batch_size", "batch_size")
    maybe_update("learning_starts", "learning_starts")
    maybe_update("train_freq", "train_freq")
    maybe_update("gradient_steps", "gradient_steps")
    maybe_update("ent_coef", "ent_coef")
    maybe_update("gamma", "gamma")
    maybe_update("tau", "tau")
    maybe_update("replay_memory_limit_gb", "replay_memory_limit_gb")
    maybe_update("ppo_n_steps", "ppo_n_steps")
    maybe_update("ppo_batch_size", "ppo_batch_size")
    maybe_update("ppo_epochs", "ppo_epochs")
    maybe_update("ppo_ent_coef", "ppo_ent_coef")
    maybe_update("ppo_clip_range", "ppo_clip_range")
    maybe_update("total_eval_episodes", "total_eval_episodes")
    maybe_update("eval_freq", "eval_freq")
    maybe_update("eval_video_dir", "eval_video_dir")
    maybe_update("eval_video_frequency", "eval_video_frequency")
    maybe_update("tensorboard_dir", "tensorboard_dir")
    maybe_update("log_dir", "log_dir")
    maybe_update("checkpoints_dir", "checkpoints_dir")

    if args.discrete_actions:
        cfg["discrete_actions"] = True
    if args.continuous_actions:
        cfg["discrete_actions"] = False
    if args.domain_randomize:
        cfg["domain_randomize"] = True
    if args.no_domain_randomize:
        cfg["domain_randomize"] = False
    if args.use_reward_penalty:
        cfg["use_reward_penalty"] = True
    if args.disable_reward_penalty:
        cfg["use_reward_penalty"] = False
    maybe_update("reward_penalty_value", "reward_penalty_value")
    maybe_update("reward_green_ratio", "reward_green_threshold")
    maybe_update("speed_reward_scale", "speed_reward_scale")
    maybe_update("speed_reward_power", "speed_reward_power")
    maybe_update("reward_scale", "reward_scale")
    if args.use_gentle_shaping:
        cfg["use_gentle_shaping"] = True
    if args.gentle_shaping_eval:
        cfg["gentle_shaping_eval"] = True
    if args.random_shift_train:
        cfg["random_shift_train"] = True
    if args.no_random_shift_train:
        cfg["random_shift_train"] = False
    if args.random_shift_eval:
        cfg["random_shift_eval"] = True
    if args.no_random_shift_eval:
        cfg["random_shift_eval"] = False
    if args.device is not None:
        cfg["device"] = args.device

    return cfg


def main() -> None:
    args = parse_args()
    cfg = resolve_run_config(args)

    if cfg["algo"] == "sac" and cfg["discrete_actions"]:
        raise ValueError("SAC requires continuous actions. Disable --discrete-actions.")

    device = (
        th.device(cfg.get("device"))
        if cfg.get("device") is not None
        else th.device("cuda" if th.cuda.is_available() else "cpu")
    )

    set_global_seeds(cfg["seed"])

    run_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    algo_dir = Path(cfg["log_dir"]) / cfg["algo"]
    log_dir = algo_dir / run_id
    tensorboard_dir = Path(cfg["tensorboard_dir"]) / cfg["algo"]
    checkpoint_dir = Path(cfg["checkpoints_dir"]) / cfg["algo"]
    monitor_dir = log_dir / "monitor"
    eval_monitor_dir = log_dir / "monitor_eval"

    for path in [log_dir, tensorboard_dir, checkpoint_dir]:
        path.mkdir(parents=True, exist_ok=True)

    sb3_logger.configure(str(log_dir), ["stdout", "log", "csv", "tensorboard"])

    train_env_cfg = EnvConfig(
        domain_randomize=cfg["domain_randomize"],
        gray=True,
        resize_shape=(84, 84),
        discrete_actions=cfg["discrete_actions"],
        use_reward_penalty=cfg["use_reward_penalty"],
        reward_penalty_value=cfg["reward_penalty_value"],
        reward_green_ratio=cfg["reward_green_ratio"],
        speed_reward_scale=cfg["speed_reward_scale"],
        speed_reward_power=cfg["speed_reward_power"],
        reward_scale=cfg["reward_scale"],
        use_gentle_shaping=cfg["use_gentle_shaping"],
        gentle_lambda_delta=cfg["gentle_lambda_delta"],
        gentle_lambda_conflict=cfg["gentle_lambda_conflict"],
        action_repeat=cfg["action_repeat"],
        random_shift=cfg["random_shift_train"],
    )

    eval_env_cfg = clone_for_eval(
        train_env_cfg,
        disable_random_shift=not cfg["random_shift_eval"],
        disable_gentle_shaping=not cfg["gentle_shaping_eval"],
    )

    if cfg["algo"] == "sac":
        height, width = train_env_cfg.resize_shape
        obs_pixels = cfg["n_stack"] * int(height) * int(width)
        bytes_per_transition = obs_pixels  # uint8 => 1 byte each
        estimated_bytes = cfg["buffer_size"] * bytes_per_transition
        limit_bytes = cfg["replay_memory_limit_gb"] * (1024**3)
        if estimated_bytes > limit_bytes:
            adjusted_buffer = int(limit_bytes // bytes_per_transition)
            adjusted_buffer = max(adjusted_buffer, 100_000)
            if adjusted_buffer < cfg["buffer_size"]:
                print(
                    f"[train] Reducing replay buffer from {cfg['buffer_size']:,} to "
                    f"{adjusted_buffer:,} to stay within ~{cfg['replay_memory_limit_gb']} GiB."
                )
                cfg["buffer_size"] = adjusted_buffer

    train_env = build_vec_env(
        train_env_cfg,
        num_envs=cfg["num_envs"],
        seed=cfg["seed"],
        training=True,
        n_stack=cfg["n_stack"],
        monitor_dir=monitor_dir,
    )

    eval_env = build_vec_env(
        eval_env_cfg,
        num_envs=1,
        seed=cfg["seed"] + 10_000,
        training=False,
        n_stack=cfg["n_stack"],
        monitor_dir=eval_monitor_dir,
    )

    if cfg["eval_video_dir"] is not None:
        video_base = Path(cfg["eval_video_dir"]) / cfg["algo"]
        video_base.mkdir(parents=True, exist_ok=True)
        frequency = cfg["eval_video_frequency"]
        if frequency is None or frequency <= 0:
            trigger = lambda step: True  # record each eval call
        else:
            trigger = lambda step: step % frequency == 0
        eval_env = VecVideoRecorder(
            eval_env,
            video_folder=str(video_base),
            record_video_trigger=trigger,
            video_length=0,
            name_prefix=f"{cfg['algo']}_eval",
        )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(checkpoint_dir),
        log_path=str(log_dir / "eval"),
        eval_freq=cfg["eval_freq"],
        n_eval_episodes=cfg["total_eval_episodes"],
        deterministic=True,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=cfg["eval_freq"],
        save_path=str(checkpoint_dir),
        name_prefix=f"{cfg['algo']}_carracing",
    )

    policy_kwargs = dict(
        features_extractor_class=CNNFeatureExtractor,
        features_extractor_kwargs=dict(features_dim=512),
        normalize_images=False,
    )

    if cfg["algo"] == "ppo":
        model = PPO(
            policy="CnnPolicy",
            env=train_env,
            n_steps=cfg["ppo_n_steps"],
            batch_size=cfg["ppo_batch_size"],
            n_epochs=cfg["ppo_epochs"],
            gamma=cfg["gamma"],
            gae_lambda=0.95,
            clip_range=cfg["ppo_clip_range"],
            ent_coef=cfg["ppo_ent_coef"],
            vf_coef=0.5,
            learning_rate=linear_schedule(cfg["learning_rate"])
            if cfg["lr_schedule"] == "linear"
            else cfg["learning_rate"],
            tensorboard_log=str(tensorboard_dir),
            verbose=1,
            device=device,
            policy_kwargs={
                **policy_kwargs,
                "net_arch": dict(pi=[256, 256], vf=[256, 256]),
            },
        )
    else:
        model = SAC(
            policy="CnnPolicy",
            env=train_env,
            learning_rate=cfg["learning_rate"],
            buffer_size=cfg["buffer_size"],
            batch_size=cfg["batch_size"],
            learning_starts=cfg["learning_starts"],
            train_freq=cfg["train_freq"],
            gradient_steps=cfg["gradient_steps"],
            tau=cfg["tau"],
            gamma=cfg["gamma"],
            ent_coef=cfg["ent_coef"],
            target_update_interval=1,
            policy_kwargs={
                **policy_kwargs,
                "net_arch": dict(pi=[256, 256], qf=[256, 256]),
                "share_features_extractor": False,
            },
            optimize_memory_usage=True,
            tensorboard_log=str(tensorboard_dir),
            verbose=1,
            device=device,
            replay_buffer_kwargs=dict(handle_timeout_termination=False),
        )

    model.learn(
        total_timesteps=cfg["total_timesteps"],
        callback=[eval_callback, checkpoint_callback],
    )

    final_path = checkpoint_dir / f"{cfg['algo']}_carracing_final"
    model.save(str(final_path))
    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
