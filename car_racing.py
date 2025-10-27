import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecFrameStack


vec_env = make_vec_env("CarRacing-v3", n_envs=4, env_kwargs={'continuous': True})

vec_env = VecFrameStack(vec_env, n_stack=4)

model = PPO(
    'CnnPolicy',
    vec_env,
    verbose=1,
    tensorboard_log="./ppo_car_racing_tensorboard/"
)

print("Iniciando entrenamiento...")
model.learn(total_timesteps=100000)
print("Entrenamiento finalizado.")

model.save("ppo_car_racing")
print("Modelo guardado como ppo_car_racing.zip")

vec_env.close()