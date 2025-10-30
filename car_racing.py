import gymnasium as gym
from gymnasium.wrappers import (
    RecordEpisodeStatistics,
    ResizeObservation,
    GrayscaleObservation,
)
from gymnasium import ObservationWrapper
from gymnasium.spaces import Box
import numpy as np

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

# Entorno
ENV_ID = "CarRacing-v3"

class EnsureChannelLast(ObservationWrapper):
    """
    Garantiza que la observación tenga forma (H, W, C).
    Si llega (H, W), añade canal al final -> (H, W, 1).
    No cambia dtype (debe quedar uint8 para pasar el check de VecTransposeImage).
    """
    def __init__(self, env):
        super().__init__(env)
        shp = self.observation_space.shape
        dtype = self.observation_space.dtype
        low = self.observation_space.low
        high = self.observation_space.high

        if len(shp) == 2:
            h, w = shp
            # expandimos low/high de (H, W) a (H, W, 1)
            if low.ndim == 2:
                low = low[..., None]
            if high.ndim == 2:
                high = high[..., None]
            self.observation_space = Box(
                low=low, high=high, shape=(h, w, 1), dtype=dtype
            )
        elif len(shp) == 3:
            # ya es HWC, no tocar
            self.observation_space = Box(
                low=low, high=high, shape=shp, dtype=dtype
            )
        else:
            raise RuntimeError(
                f"EnsureChannelLast: observación con rank inesperado: {shp}"
            )

    def observation(self, obs):
        if obs.ndim == 2:
            return obs[..., None]
        return obs

def make_env(gray=True, resize_shape=(84, 84), domain_randomize=False):
    """
    Pipeline HWC uint8: (opcional gris con keep_dim) -> resize -> ensure channel last
    """
    def thunk():
        env = gym.make(ENV_ID, domain_randomize=domain_randomize)
        env = RecordEpisodeStatistics(env)
        if gray:
            env = GrayscaleObservation(env, keep_dim=True)  # (H, W, 1) o (H, W)
        env = ResizeObservation(env, resize_shape)          # (84, 84, C?) o (84, 84)
        env = EnsureChannelLast(env)                        # fuerza (H, W, C)
        return env
    return thunk

if __name__ == "__main__":
    NUM_ENVS = 8
    N_STACK = 4

    # === Entrenamiento: vectorized env ===
    venv = SubprocVecEnv([make_env(gray=True) for _ in range(NUM_ENVS)])

    # 1) APILAR FRAMES en HWC
    venv = VecFrameStack(venv, n_stack=N_STACK, channels_order="last")   # -> (H, W, C*N_STACK)

    # 2) Transponer a CHW para PyTorch
    venv = VecTransposeImage(venv)                                       # -> (C*N_STACK, H, W)

    # 3) Monitor y normalización de recompensas
    venv = VecMonitor(venv)
    venv = VecNormalize(venv, norm_obs=False, norm_reward=True, clip_reward=10.0)

    # === Evaluación: MISMO pipeline y orden ===
    eval_env = DummyVecEnv([make_env(gray=True)])
    eval_env = VecFrameStack(eval_env, n_stack=N_STACK, channels_order="last")
    eval_env = VecTransposeImage(eval_env)
    eval_env = VecMonitor(eval_env)
    eval_env = VecNormalize(eval_env, training=False, norm_obs=False, norm_reward=False)

    # === Callbacks (eval + checkpoints) ===
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

    # === PPO con extractor CNN personalizado ===
    policy_kwargs = dict(
        features_extractor_class=CNN_feature_extraction,
        features_extractor_kwargs=dict(features_dim=512),
    )

    model = PPO(
        policy="CnnPolicy",
        env=venv,
        n_steps=1024,
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        learning_rate=3e-4,
        verbose=1,
        tensorboard_log="./tb/",
        device="auto",
        policy_kwargs=policy_kwargs,
    )

    model.learn(total_timesteps=2_000_000, callback=[eval_cb, ckpt_cb])
    model.save("ppo_carracing_final")
