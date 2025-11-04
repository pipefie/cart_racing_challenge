from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Sequence

import gymnasium as gym
import numpy as np
from gymnasium import ObservationWrapper
from gymnasium.spaces import Box


class DiscretizeActionWrapper(gym.ActionWrapper):
    """Maps the continuous CarRacing action space to five discrete primitives."""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        # Steering, throttle, brake tuples chosen to cover basic maneuvers.
        self._actions: Dict[int, np.ndarray] = {
            0: np.array([-1.0, 0.0, 0.0], dtype=np.float32),  # Steer left
            1: np.array([1.0, 0.0, 0.0], dtype=np.float32),   # Steer right
            2: np.array([0.0, 1.0, 0.0], dtype=np.float32),   # Accelerate
            3: np.array([0.0, 0.0, 0.8], dtype=np.float32),   # Brake
            4: np.array([0.0, 0.0, 0.0], dtype=np.float32),   # Coast
        }
        self.action_space = gym.spaces.Discrete(len(self._actions))

    def action(self, act: int) -> np.ndarray:
        return self._actions[int(act)]


class GentleShapingWrapper(gym.Wrapper):
    """Soft penalties for abrupt steering changes and brake/throttle conflicts."""

    def __init__(
        self,
        env: gym.Env,
        lambda_delta_steer: float = 1e-3,
        lambda_conflict: float = 1e-3,
    ):
        super().__init__(env)
        self.lambda_delta_steer = lambda_delta_steer
        self.lambda_conflict = lambda_conflict
        self._prev_action: Optional[np.ndarray] = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._prev_action = None
        return obs, info

    def step(self, action: np.ndarray):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if self._prev_action is not None and action is not None:
            delta = float(np.abs(action[0] - self._prev_action[0]))
            reward -= self.lambda_delta_steer * delta
        if action is not None and action[1] > 0.0 and action[2] > 0.0:
            reward -= self.lambda_conflict
        self._prev_action = np.array(action, copy=True)
        return obs, reward, terminated, truncated, info


class RewardPenaltyWrapper(gym.Wrapper):
    """Adds a penalty when the car sits on grass based on RGB observations."""

    def __init__(
        self,
        env: gym.Env,
        penalty: float = 1.5,
        green_ratio_threshold: float = 1.3,
        patch_rows: Sequence[int] = (48, 72),
        patch_cols: Sequence[int] = (42, 54),
    ):
        super().__init__(env)
        self.penalty = penalty
        self.green_ratio_threshold = green_ratio_threshold
        self.patch_rows = patch_rows
        self.patch_cols = patch_cols

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        if obs.ndim != 3 or obs.shape[2] != 3:
            # Wrapper must be placed before grayscale conversion; fail loudly otherwise.
            raise RuntimeError(
                "RewardPenaltyWrapper expects RGB observations. "
                "Place it before grayscale/resize wrappers."
            )

        h, w, _ = obs.shape
        r0 = min(max(self.patch_rows[0], 0), h - 1)
        r1 = min(max(self.patch_rows[1], r0 + 1), h)
        c0 = min(max(self.patch_cols[0], 0), w - 1)
        c1 = min(max(self.patch_cols[1], c0 + 1), w)
        patch = obs[r0:r1, c0:c1]
        if patch.size == 0:
            return obs, reward, terminated, truncated, info

        mean_green = float(np.mean(patch[..., 1]))
        mean_all = float(np.mean(patch) + 1e-6)
        if mean_all > 0.0:
            green_ratio = mean_green / mean_all
            if green_ratio > self.green_ratio_threshold:
                reward -= self.penalty

        return obs, reward, terminated, truncated, info


class SpeedRewardWrapper(gym.Wrapper):
    """Adds a shaped reward proportional to the car's speed."""

    def __init__(
        self,
        env: gym.Env,
        scale: float = 0.0,
        power: float = 1.0,
        speed_key: str = "speed",
    ):
        super().__init__(env)
        self.scale = float(scale)
        self.power = float(power)
        self.speed_key = speed_key

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        raw_speed = info.get(self.speed_key, 0.0)
        try:
            speed = float(raw_speed)
        except (TypeError, ValueError):
            speed = 0.0
        bonus = self.scale * (speed ** self.power)
        if bonus != 0.0:
            reward += bonus
            info["speed_reward"] = bonus
        return obs, reward, terminated, truncated, info


class RewardScaleWrapper(gym.Wrapper):
    """Uniformly scales rewards to stabilise critic targets."""

    def __init__(self, env: gym.Env, scale: float = 1.0):
        super().__init__(env)
        self.scale = float(scale)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        original_reward = reward
        reward *= self.scale
        if "original_reward" not in info:
            info["original_reward"] = original_reward
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


class EnsureChannelLast(ObservationWrapper):
    """Guarantees observations are shaped (H, W, C)."""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        obs_space = self.observation_space
        if len(obs_space.shape) == 2:
            h, w = obs_space.shape
            low = np.expand_dims(obs_space.low, -1)
            high = np.expand_dims(obs_space.high, -1)
            self.observation_space = Box(low=low, high=high, shape=(h, w, 1), dtype=obs_space.dtype)
        elif len(obs_space.shape) == 3:
            h, w, c = obs_space.shape
            self.observation_space = Box(
                low=obs_space.low,
                high=obs_space.high,
                shape=(h, w, c),
                dtype=obs_space.dtype,
            )
        else:
            raise ValueError("EnsureChannelLast only supports 2D or 3D observation spaces.")

    def observation(self, observation: np.ndarray) -> np.ndarray:
        return observation[..., None] if observation.ndim == 2 else observation


class RandomShift(ObservationWrapper):
    """Pad-and-crop augmentation used by DrQ. Keeps dtype and shape intact."""

    def __init__(self, env: gym.Env, pad: int = 4):
        super().__init__(env)
        self.pad = pad
        h, w, *channels = self.observation_space.shape
        c = channels[0] if channels else 1
        self.observation_space = Box(
            low=self.observation_space.low,
            high=self.observation_space.high,
            shape=(h, w, c),
            dtype=self.observation_space.dtype,
        )

    def observation(self, observation: np.ndarray) -> np.ndarray:
        if observation.ndim == 2:
            observation = observation[..., None]
        h, w, c = observation.shape
        pad = self.pad
        padded = np.pad(observation, ((pad, pad), (pad, pad), (0, 0)), mode="edge")
        dy = int(self.np_random.integers(0, 2 * pad + 1))
        dx = int(self.np_random.integers(0, 2 * pad + 1))
        return padded[dy:dy + h, dx:dx + w]


class ActionRepeat(gym.Wrapper):
    """Repeats the same action for k steps while accumulating rewards."""

    def __init__(self, env: gym.Env, repeat: int = 4):
        super().__init__(env)
        if repeat < 1:
            raise ValueError("repeat must be >= 1")
        self.repeat = repeat

    def step(self, action):
        total_reward = 0.0
        terminated = False
        truncated = False
        info = {}
        obs = None
        for _ in range(self.repeat):
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            if terminated or truncated:
                break
        return obs, total_reward, terminated, truncated, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)
