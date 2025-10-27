import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecFrameStack


print("\nIniciando evaluación...")

model = PPO.load("ppo_car_racing")

eval_vec_env = make_vec_env("CarRacing-v3", n_envs=1, env_kwargs={'continuous': True, 'render_mode': 'human'})
eval_vec_env = VecFrameStack(eval_vec_env, n_stack=4)


obs = eval_vec_env.reset()
dones = False
while not dones:
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, dones, info = eval_vec_env.step(action)

eval_vec_env.close()
print("Evaluación terminada.")