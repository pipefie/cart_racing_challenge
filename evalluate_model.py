import gymnasium as gym
from gymnasium import ObservationWrapper
from gymnasium.spaces import Box
from gymnasium.wrappers import GrayscaleObservation, ResizeObservation
import numpy as np

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
MODEL_PATH = "checkpoints/ppo_carracing_2000000_steps.zip"
# ▲▲▲ ¡CAMBIA ESTA LÍNEA PARA ELEGIR QUÉ CHECKPOINT PROBAR! ▲▲▲

ENV_ID = "CarRacing-v3"  # Usar v2 que es el ID estándar para evitar problemas
N_STACK = 4


# --- DEFINICIONES DEL ENTORNO (COPIADAS DE TU SCRIPT DE ENTRENAMIENTO) ---
# Se copian aquí para que el script sea independiente y podamos añadir el render_mode.

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

def make_eval_env(gray=True, resize_shape=(84, 84), domain_randomize=False):
    """
    Crea el entorno para evaluación, AÑADIENDO EL MODO DE RENDERIZADO.
    """
    def thunk():
        # LA LÍNEA CLAVE: render_mode='human' para que se vea la ventana.
        env = gym.make(ENV_ID, domain_randomize=domain_randomize, render_mode='human')
        env = GrayscaleObservation(env, keep_dim=True)
        env = ResizeObservation(env, resize_shape)
        env = EnsureChannelLast(env)
        return env
    return thunk


# --- SCRIPT PRINCIPAL DE EVALUACIÓN ---

if __name__ == "__main__":
    print("Creando entorno de evaluación...")
    # 1. Usamos nuestra función `make_eval_env` para crear el entorno base visible.
    eval_env = DummyVecEnv([make_eval_env(gray=True)])

    # 2. Aplicamos los wrappers vectorizados EXACTAMENTE en el mismo orden que en el entrenamiento.
    eval_env = VecFrameStack(eval_env, n_stack=N_STACK, channels_order="last")
    eval_env = VecTransposeImage(eval_env)
    eval_env = VecMonitor(eval_env)
    eval_env = VecNormalize(eval_env, training=False, norm_obs=False, norm_reward=False)

    # --- Carga y Evaluación del Modelo ---
    print(f"Cargando modelo desde: {MODEL_PATH}")
    # Comprobamos si el archivo existe para dar un error más claro
    try:
        model = PPO.load(MODEL_PATH)
    except FileNotFoundError:
        print(f"Error: No se encontró el archivo del modelo en '{MODEL_PATH}'")
        print("Asegúrate de que la ruta es correcta y de que has entrenado el modelo.")
        exit()

    print("\nIniciando evaluación... (Pulsa Ctrl+C en la terminal para parar)")
    
    obs = eval_env.reset()
    
    # BUCLE CORREGIDO: se ejecuta hasta que el episodio termine (o lo pares).
    try:
        while True:
            action, _states = model.predict(obs, deterministic=True)
            obs, rewards, dones, info = eval_env.step(action)
            
            # Si el episodio termina (coche se sale, etc.), el bucle para.
            if dones[0]:
                print("Episodio terminado.")
                break
    except KeyboardInterrupt:
        print("\nEvaluación interrumpida por el usuario.")
    finally:
        # Cierra el entorno y la ventana de Pygame de forma segura.
        eval_env.close()
        print("Entorno cerrado. Evaluación finalizada.")