from __future__ import annotations

import copy
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Optional, Sequence

import gymnasium as gym
import numpy as np
try:
    import torch as th
except ImportError:  # pragma: no cover - optional dependency
    th = None
from gymnasium.wrappers import GrayscaleObservation, RecordEpisodeStatistics, ResizeObservation
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    SubprocVecEnv,
    VecFrameStack,
    VecMonitor,
    VecTransposeImage,
)

from wrappers import (
    ActionRepeat,
    DiscretizeActionWrapper,
    EnsureChannelLast,
    GentleShapingWrapper,
    RandomShift,
    RewardPenaltyWrapper,
    RewardScaleWrapper,
)

ENV_ID = "CarRacing-v3"


@dataclass
class EnvConfig:
    """Configuration for the single-environment preprocessing pipeline."""

    env_id: str = ENV_ID
    domain_randomize: bool = False
    gray: bool = True
    resize_shape: Sequence[int] = (84, 84)
    discrete_actions: bool = False
    use_reward_penalty: bool = True
    reward_penalty_value: float = 1.5
    reward_green_ratio: float = 1.3
    reward_scale: float = 1.0
    use_gentle_shaping: bool = False
    gentle_lambda_delta: float = 1e-3
    gentle_lambda_conflict: float = 1e-3
    action_repeat: int = 4
    random_shift: bool = True
    random_shift_pad: int = 4


def set_global_seeds(seed: int) -> None:
    """Seed numpy, random, and torch for reproducibility."""
    np.random.seed(seed)
    random.seed(seed)
    if th is not None:
        th.manual_seed(seed)
        if th.cuda.is_available():
            th.cuda.manual_seed_all(seed)


def make_env(
    cfg: EnvConfig,
    *,
    seed: int,
    env_idx: int,
    training: bool,
    render_mode: Optional[str] = None,
) -> Callable[[], gym.Env]:
    """Creates a thunk that builds a single CarRacing environment with wrappers."""

    def _thunk() -> gym.Env:
        env = gym.make(cfg.env_id, domain_randomize=cfg.domain_randomize, render_mode=render_mode)
        # Gymnasium seeding API: reset(seed=...) seeds RNGs used by wrappers downstream.
        env.reset(seed=seed + env_idx)
        env.action_space.seed(seed + env_idx)

        env = RecordEpisodeStatistics(env)

        if cfg.use_reward_penalty:
            env = RewardPenaltyWrapper(
                env,
                penalty=cfg.reward_penalty_value,
                green_ratio_threshold=cfg.reward_green_ratio,
            )

        if cfg.gray:
            env = GrayscaleObservation(env, keep_dim=True)

        if cfg.resize_shape is not None:
            env = ResizeObservation(env, tuple(cfg.resize_shape))

        if cfg.use_gentle_shaping and not cfg.discrete_actions:
            env = GentleShapingWrapper(
                env,
                lambda_delta_steer=cfg.gentle_lambda_delta,
                lambda_conflict=cfg.gentle_lambda_conflict,
            )

        if cfg.discrete_actions:
            env = DiscretizeActionWrapper(env)

        if cfg.action_repeat > 1:
            env = ActionRepeat(env, repeat=cfg.action_repeat)

        env = EnsureChannelLast(env)

        if cfg.reward_scale != 1.0:
            env = RewardScaleWrapper(env, scale=cfg.reward_scale)

        if cfg.random_shift and training:
            env = RandomShift(env, pad=cfg.random_shift_pad)

        return env

    return _thunk


def build_vec_env(
    cfg: EnvConfig,
    *,
    num_envs: int,
    seed: int,
    training: bool,
    n_stack: int = 4,
    monitor_dir: Optional[os.PathLike] = None,
    start_method: str = "spawn",
    render_mode: Optional[str] = None,
) -> DummyVecEnv | SubprocVecEnv:
    """Creates a vectorized environment with consistent wrapper order."""

    env_fns = [
        make_env(cfg, seed=seed, env_idx=i, training=training, render_mode=render_mode)
        for i in range(num_envs)
    ]

    if training and num_envs > 1:
        vec_env = SubprocVecEnv(env_fns, start_method=start_method)
    else:
        vec_env = DummyVecEnv(env_fns)

    vec_env = VecFrameStack(vec_env, n_stack=n_stack, channels_order="last")
    vec_env = VecTransposeImage(vec_env)

    monitor_path = None
    if monitor_dir is not None:
        Path(monitor_dir).mkdir(parents=True, exist_ok=True)
        monitor_path = str(Path(monitor_dir) / "monitor.csv")

    vec_env = VecMonitor(vec_env, filename=monitor_path)
    return vec_env


def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """Linear schedule for learning rate or other hyper-parameters."""

    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value

    return func


def clone_for_eval(
    cfg: EnvConfig,
    *,
    disable_random_shift: bool = True,
    disable_gentle_shaping: bool = True,
    discrete_override: Optional[bool] = None,
) -> EnvConfig:
    """Create an evaluation configuration mirroring training with optional tweaks."""

    updated = copy.deepcopy(cfg)
    if disable_random_shift:
        updated.random_shift = False
    if disable_gentle_shaping:
        updated.use_gentle_shaping = False
    if discrete_override is not None:
        updated.discrete_actions = discrete_override
    return updated


def load_yaml(path: Optional[str]) -> Dict:
    """Loads a YAML file if provided."""
    if path is None:
        return {}
    try:
        import yaml  # type: ignore
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("PyYAML is required to load configuration files.") from exc

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected a mapping at root of {path}, found {type(data)}")
    return data
