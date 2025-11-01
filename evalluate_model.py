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
    VecNormalize,
    VecTransposeImage,
    VecFrameStack,
)

# --- CONFIGURACIÓN ---
# ▼▼▼ ¡CAMBIA ESTA LÍNEA PARA ELEGIR QUÉ CHECKPOINT PROBAR! ▼▼▼
MODEL_PATH = "checkpoints/ppo_carracing_4800000_steps.zip"
# ▲▲▲ ¡CAMBIA ESTA LÍNEA PARA ELEGIR QUÉ CHECKPOINT PROBAR! ▲▲▲

ENV_ID = "CarRacing-v3"
N_STACK = 4


# --- DEFINICIONES DEL ENTORNO (COPIADAS DE TU SCRIPT DE ENTRENAMIENTO) ---
# Se necesita replicar el entorno de entrenamiento EXACTAMENTE.

class DiscretizeActionsWrapper(gym.ActionWrapper):
    """Wrapper para discretizar el espacio de acciones de CarRacing."""
    def __init__(self, env):
        super().__init__(env)
        self._actions = [
            [0.0, 1.0, 0.0],  # 0: Acelerar recto
            [-1.0, 0.3, 0.0], # 1: Girar a la izquierda
            [1.0, 0.3, 0.0],  # 2: Girar a la derecha
            [0.0, 0.0, 0.8],  # 3: Frenar
            [0.0, 0.0, 0.0],  # 4: No hacer nada
        ]
        self.action_space = gym.spaces.Discrete(len(self._actions))

    def action(self, act):
        return np.array(self._actions[act], dtype=np.float32)

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

    def observation(self, obs):
        return obs[..., None] if obs.ndim == 2 else obs

def make_eval_env(gray=True, resize_shape=(84, 84)):
    """
    Crea el entorno para evaluación, AÑADIENDO EL MODO DE RENDERIZADO.
    El pipeline de wrappers debe ser IDÉNTICO al de entrenamiento.
    """
    def thunk():
        env = gym.make(ENV_ID, render_mode='human')
        
        # <-- ¡CORRECCIÓN CLAVE! AÑADIMOS EL WRAPPER DE DISCRETIZACIÓN
        env = DiscretizeActionsWrapper(env)
        
        env = GrayscaleObservation(env, keep_dim=True)
        env = ResizeObservation(env, resize_shape)
        env = EnsureChannelLast(env)
        return env
    return thunk


# --- SCRIPT PRINCIPAL DE EVALUACIÓN ---

if __name__ == "__main__":
    print("Creando entorno de evaluación...")
    # 1. Creamos el entorno base con nuestra función corregida.
    eval_env = DummyVecEnv([make_eval_env(gray=True)])

    # 2. Aplicamos los wrappers vectorizados EXACTAMENTE en el mismo orden que en el entrenamiento.
    eval_env = VecFrameStack(eval_env, n_stack=N_STACK, channels_order="last")
    eval_env = VecTransposeImage(eval_env)
    eval_env = VecMonitor(eval_env)
    
    # NOTA: Aunque el modelo se entrenó con VecNormalize, para la evaluación es más simple
    # usar un wrapper "passthrough" si no guardaste las estadísticas de normalización.
    # Esta configuración es correcta para la visualización.
    eval_env = VecNormalize(eval_env, training=False, norm_obs=False, norm_reward=False)

    # --- Carga y Evaluación del Modelo ---
    print(f"Cargando modelo desde: {MODEL_PATH}")
    try:
        model = PPO.load(MODEL_PATH)
    except FileNotFoundError:
        print(f"Error: No se encontró el archivo del modelo en '{MODEL_PATH}'")
        exit()

    print("\nIniciando evaluación... (Pulsa Ctrl+C en la terminal para parar)")
    
    episodes = 5
    for ep in range(episodes):
        obs = eval_env.reset()
        done = False
        print(f"\n--- Empezando Episodio {ep + 1}/{episodes} ---")
        try:
            while not done:
                action, _states = model.predict(obs, deterministic=True)
                obs, reward, done, info = eval_env.step(action)
                # Opcional: un pequeño retardo para ver mejor la acción
                time.sleep(0.01)
        except KeyboardInterrupt:
            print("\nEvaluación interrumpida por el usuario.")
            break

    # Cierra el entorno y la ventana de Pygame de forma segura.
    eval_env.close()
    print("\nEvaluación finalizada.")