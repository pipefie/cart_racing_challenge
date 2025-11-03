import gymnasium as gym
from gymnasium.wrappers import (
    RecordEpisodeStatistics,
    ResizeObservation,
    GrayscaleObservation,
)
from gymnasium import ObservationWrapper
from gymnasium.spaces import Box
import numpy as np
import logging
import sys
from datetime import datetime
from typing import Callable

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    SubprocVecEnv,
    VecMonitor,
    VecNormalize,
    VecTransposeImage,
    VecFrameStack,
)
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common import logger as sb3_logger
import os

from CNN_feature_extractor import CNN_feature_extraction

import torch as th

from typing import Optional

ENV_ID = "CarRacing-v3"

# Wrapper to discretize the action space, to simplify the learning task for the agent

class DiscretizeActionWrapper(gym.ActionWrapper):
    """
    Wrapper to discretize the action space of the CarRacing-v2 environment.
    
    The new action space will be:
    0: Steer left
    1: Steer right
    2: Accelerate
    3: Brake
    4: Do nothing
    """
    def __init__(self, env):
        super().__init__(env)
        self._actions = {
            0: (-1.0, 0.0, 0.0),  # Steer left
            1: (1.0, 0.0, 0.0),   # Steer right
            2: (0.0, 1.0, 0.0),   # Accelerate
            3: (0.0, 0.0, 0.8),   # Brake
            4: (0.0, 0.0, 0.0),   # Do nothing
        }
        self.action_space = gym.spaces.Discrete(len(self._actions))

    def action(self, act):
        return self._actions[act]

class GentleShapingStepWrapper(gym.Wrapper):
    def __init__(self, env, lambda_delta_steer=1e-3, lambda_conflict=1e-3):
        super().__init__(env)
        self.prev_action: Optional[np.ndarray] = None
        self.lambda_delta_steer = lambda_delta_steer
        self.lambda_conflict = lambda_conflict
    def reset(self, **kwargs):
        self.prev_action = None
        return self.env.reset(**kwargs)
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if self.prev_action is not None:
            reward -= self.lambda_delta_steer * abs(float(action[0] - self.prev_action[0]))
        if action[1] > 0.0 and action[2] > 0.0:
            reward -= self.lambda_conflict
        self.prev_action = action
        return obs, reward, terminated, truncated, info


# Wrapper to penalize the agent for driving on the grass
# We check the pixels under the car to see if they are mostly green (grass)
class RewardWrapper(gym.RewardWrapper):
    """
    Wrapper to penalize the agent for driving on the grass.
    """
    def __init__(self, env):
        super().__init__(env)

    def reward(self, reward):
        # Get the original observation before grayscaling
        obs = self.env.unwrapped.state
        
        # Check the 12x12 pixel patch under the car
        # A high value in the green channel indicates grass
        is_on_grass = np.mean(obs[84:96, 42:54, 1]) > 180
        
        if is_on_grass:
            # Heavy penalty for being on the grass
            reward -= 5.0
            
        return reward

class EnsureChannelLast(ObservationWrapper):
    """Garantiza que la observación tenga forma (H, W, C)."""
    def __init__(self, env):
        super().__init__(env)
        shp = self.observation_space.shape
        dtype = self.observation_space.dtype
        low, high = self.observation_space.low, self.observation_space.high
        if len(shp) == 2:
            h, w = shp
            if low.ndim == 2: low = low[..., None]
            if high.ndim == 2: high = high[..., None]
            self.observation_space = Box(low=low, high=high, shape=(h, w, 1), dtype=dtype)
        elif len(shp) == 3:
            self.observation_space = Box(low=low, high=high, shape=shp, dtype=dtype)

    def observation(self, obs):
        return obs[..., None] if obs.ndim == 2 else obs


class RandomShift(ObservationWrapper):
    def __init__(self, env, pad=4):
        super().__init__(env)
        self.pad = pad
        h, w, *rest = self.observation_space.shape
        c = rest[0] if rest else 1
        self.observation_space = Box(
            low=self.observation_space.low,
            high=self.observation_space.high,
            shape=(h, w, c),
            dtype=self.observation_space.dtype
        )

    def observation(self, obs):
        # Accept (H, W) or (H, W, C)
        if obs.ndim == 2:
            obs = obs[..., None]
        h, w, c = obs.shape
        p = self.pad
        padded = np.pad(obs, ((p, p), (p, p), (0, 0)), mode="edge")
        dy = np.random.randint(0, 2*p + 1)
        dx = np.random.randint(0, 2*p + 1)
        return padded[dy:dy+h, dx:dx+w, :]

def make_env(
    gray: bool = True,
    resize_shape=(84, 84),
    domain_randomize: bool = False,
    discrete: bool = False,
    use_gentle_shaping: bool = False,
):
    """Pipeline de creación de entorno solo con pre-procesamiento de imagen."""
    def thunk():
        env = gym.make(ENV_ID, domain_randomize=domain_randomize)
        env = RecordEpisodeStatistics(env)
        env = RewardWrapper(env)
        if gray:
            env = GrayscaleObservation(env, keep_dim=True)
        env = ResizeObservation(env, resize_shape)
        
        if use_gentle_shaping and not discrete:
            env = GentleShapingStepWrapper(env, 1e-3, 1e-3)  # optional, continuous only
        if discrete:
            env = DiscretizeActionWrapper(env)    
        env = EnsureChannelLast(env)
        env = RandomShift(env, pad=4)    # (84,84,1)  <-- augmentation (HWC)

        return env
    return thunk


def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """Tasa de aprendizaje que decrece linealmente."""
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value
    return func