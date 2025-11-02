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

from CNN_feature_extractor import CNN_feature_extraction

ENV_ID = "CarRacing-v3"

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



def make_env(gray=True, resize_shape=(84, 84), domain_randomize=False):
    """Pipeline de creación de entorno solo con pre-procesamiento de imagen."""
    def thunk():
        env = gym.make(ENV_ID, domain_randomize=domain_randomize)
        env = RecordEpisodeStatistics(env)
        
        if gray:
            env = GrayscaleObservation(env, keep_dim=True)
        env = ResizeObservation(env, resize_shape)
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
    
    NUM_ENVS = 8
    N_STACK = 4
    TOTAL_TIMESTEPS = 5_000_000

    venv = SubprocVecEnv([make_env(gray=True, domain_randomize=True) for _ in range(NUM_ENVS)])

    venv = VecFrameStack(venv, n_stack=N_STACK, channels_order="last")
    venv = VecTransposeImage(venv)
    venv = VecMonitor(venv)
    venv = VecNormalize(venv, norm_obs=False, norm_reward=True, clip_reward=10.0)

    eval_env = DummyVecEnv([make_env(gray=True)])
    eval_env = VecFrameStack(eval_env, n_stack=N_STACK, channels_order="last")
    eval_env = VecTransposeImage(eval_env)
    eval_env = VecMonitor(eval_env)
    eval_env = VecNormalize(eval_env, training=False, norm_obs=False, norm_reward=False)

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path="./checkpoints/",
        log_path="./logs/",
        eval_freq=10_000,
        n_eval_episodes=5,
        deterministic=True,
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
    model = PPO(
        policy="CnnPolicy", # SB3 usa la política correcta para espacios continuos automáticamente
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
        device="auto",
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