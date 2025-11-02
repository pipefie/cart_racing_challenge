import gymnasium as gym
from gymnasium import ObservationWrapper
from gymnasium.spaces import Box
from gymnasium.wrappers import GrayscaleObservation, ResizeObservation
import numpy as np
import time

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    VecMonitor,
    VecTransposeImage,
    VecFrameStack,
)

MODEL_PATH = "checkpoints/ppo_carracing_4800000_steps.zip"
ENV_ID = "CarRacing-v3"
N_STACK = 4

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

def make_eval_env(gray=True, resize_shape=(84, 84)):
    """
    Crea el entorno para evaluación.
    ELIMINAMOS EL WRAPPER de discretización para que coincida con el nuevo entrenamiento.
    """
    def thunk():
        env = gym.make(ENV_ID, render_mode='human')
        
        env = GrayscaleObservation(env, keep_dim=True)
        env = ResizeObservation(env, resize_shape)
        env = EnsureChannelLast(env)
        return env
    return thunk



if __name__ == "__main__":
    print("Creando entorno de evaluación para control continuo...")
    eval_env = DummyVecEnv([make_eval_env(gray=True)])

    eval_env = VecFrameStack(eval_env, n_stack=N_STACK, channels_order="last")
    eval_env = VecTransposeImage(eval_env)
    eval_env = VecMonitor(eval_env)
    print(f"Cargando modelo desde: {MODEL_PATH}")
    try:
        model = PPO.load(MODEL_PATH, env=eval_env)
    except FileNotFoundError:
        print(f"Error: No se encontró el archivo del modelo en '{MODEL_PATH}'")
        exit()

    print("\nIniciando evaluación... (Pulsa Ctrl+C en la terminal para parar)")
    
    episodes = 5
    for ep in range(episodes):
        obs = eval_env.reset()
        done = False
        total_reward = 0
        print(f"\n--- Empezando Episodio {ep + 1}/{episodes} ---")
        try:
            while not done:
                action, _states = model.predict(obs, deterministic=True)
                obs, reward, done, info = eval_env.step(action)
                total_reward += reward[0]
                
        except KeyboardInterrupt:
            print("\nEvaluación interrumpida por el usuario.")
            break
        print(f"Recompensa del episodio: {total_reward:.2f}")

    eval_env.close()
    print("\nEvaluación finalizada.")