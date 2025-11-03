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
    """
    Random 2D shift (pad + crop) on HWC images.
    - Works with grayscale (H,W,1) or RGB (H,W,3) and with FrameStack later.
    - Keeps dtype/shape invariant.
    """
    def __init__(self, env, pad=4):
        super().__init__(env)
        self.pad = pad
        h, w, c = self.observation_space.shape
        self.observation_space = Box(
            low=self.observation_space.low,
            high=self.observation_space.high,
            shape=(h, w, c),
            dtype=self.observation_space.dtype
        )

    def observation(self, obs):
        # obs: (H, W, C), uint8
        h, w, c = obs.shape
        p = self.pad
        # pad with edge pixels (fast & stable)
        padded = np.pad(obs, ((p, p), (p, p), (0, 0)), mode="edge")
        # sample top-left corner of the crop
        dy = np.random.randint(0, 2*p + 1)
        dx = np.random.randint(0, 2*p + 1)
        return padded[dy:dy+h, dx:dx+w, :]

def make_env(gray=True, resize_shape=(84, 84), domain_randomize=False):
    """Pipeline de creación de entorno solo con pre-procesamiento de imagen."""
    def thunk():
        env = gym.make(ENV_ID, domain_randomize=domain_randomize)
        env = RecordEpisodeStatistics(env)
        env = RewardWrapper(env)
        env = DiscretizeActionWrapper(env)
        if gray:
            env = GrayscaleObservation(env, keep_dim=True)
        env = ResizeObservation(env, resize_shape)
        env = RandomShift(env, pad=4)                       # (84,84,1)  <-- augmentation (HWC
        env = EnsureChannelLast(env)

        return env
    return thunk


def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """Tasa de aprendizaje que decrece linealmente."""
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value
    return func



if __name__ == "__main__":
    
    log_filename = f"entrenamiento_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler(sys.stdout)
        ]
    )
    # Configure SB3 logger to persist the console/tabular output to files
    run_id = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    sb3_log_dir = os.path.join("./logs", run_id)
    os.makedirs(sb3_log_dir, exist_ok=True)
    # Save console output (human readable), csv progress and tensorboard data
    sb3_logger.configure(sb3_log_dir, format_strings=["stdout", "log", "csv", "tensorboard"])
    logging.info(f"SB3 logs will be written to: {sb3_log_dir}")
    # Also write Python logging output to the same run folder for convenience
    pylog_path = os.path.join(sb3_log_dir, "python_logging.log")
    fh = logging.FileHandler(pylog_path)
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logging.getLogger().addHandler(fh)
    logging.info(f"Python logging also written to: {pylog_path}")
    
    NUM_ENVS = 8
    N_STACK = 4
    TOTAL_TIMESTEPS = 2_400_000

    venv = SubprocVecEnv([make_env(gray=True, domain_randomize=True) for _ in range(NUM_ENVS)])

    venv = VecFrameStack(venv, n_stack=N_STACK, channels_order="last")
    venv = VecTransposeImage(venv)
    venv = VecMonitor(venv)
    venv = VecNormalize(venv, norm_obs=False, norm_reward=True, clip_reward=10.0)

    eval_env = DummyVecEnv([make_env(gray=True)])
    eval_env = VecFrameStack(eval_env, n_stack=N_STACK, channels_order="last")
    eval_env = VecTransposeImage(eval_env)
    eval_env = VecMonitor(eval_env)
    eval_env = VecNormalize(eval_env, training=False, norm_obs=False, norm_reward=True)

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path="./checkpoints/",
        log_path="./logs/",
        eval_freq=10_000,
        n_eval_episodes=5,
        deterministic=True
    )
    ckpt_cb = CheckpointCallback(
        save_freq=50_000,
        save_path="./checkpoints/",
        name_prefix="ppo_carracing",
    )

    policy_kwargs = dict(
        features_extractor_class=CNN_feature_extraction,
        features_extractor_kwargs=dict(features_dim=512),
        net_arch=dict(pi=[256, 256], vf=[256, 256])
    )

    # Creación del modelo PPO
    # Selección explícita del device: usar CUDA si está disponible, si no CPU
    device_name = "cuda" if th.cuda.is_available() else "cpu"
    logging.info(f"Device seleccionado para entrenamiento: {device_name}")
    model = PPO(
        policy="CnnPolicy", 
        env=venv,
        n_steps=2048,
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,      # Fomentar la exploración en el espacio continuo
        vf_coef=0.5,
        learning_rate=linear_schedule(3e-4),
        verbose=1,
        tensorboard_log="./tb/",
        device=device_name,
        policy_kwargs=policy_kwargs,
    )

    logging.info("==================================================")
    logging.info("       INICIANDO ENTRENAMIENTO                    ")
    logging.info("==================================================")
    logging.info(f"Estrategia: Sin modificar entorno. El agente debe aprender control continuo.")
    logging.info(f"Total timesteps: {TOTAL_TIMESTEPS}")

    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=[eval_cb, ckpt_cb])
    
    logging.info("==================================================")
    logging.info("           ENTRENAMIENTO FINALIZADO               ")
    logging.info("==================================================")
    
    model.save("ppo_carracing_final")