import gymnasium as gym
from gymnasium.wrappers import RecordEpisodeStatistics, FrameStack, ResizeObservation, GrayScaleObservation
from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor, VecNormalize, VecTransposeImage
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env

ENV_ID = "CarRacing-v2"   # Gymnasium id

def make_env(gray=True):
    def thunk():
        """
        Create and return a wrapped CarRacing-v2 environment.
        If gray is True, the observations are converted to grayscale.
        This is like a "factory" function for creating environments.
        """
        env = gym.make(ENV_ID, domain_randomize=False)
        # basic stats wrapper
        env = RecordEpisodeStatistics(env)
        # vision pipeline
        if gray:
            env = GrayScaleObservation(env, keep_dim=True)      # (H,W,1)
        env = ResizeObservation(env, (84, 84))                   # (84,84,C)
        env = FrameStack(env, 4)                                 # (84,84,4) or (84,84,12) if RGB
        return env
    return thunk

# Vectorized envs (PPO benefits from more envs)
"""
A vectorized environment (VecEnv) is a wrapper that runs several environments in parallel, collects experiences 
from all of them, and returns their combined batch of observations, rewards, etc.
"""
NUM_ENVS = 8
venv = SubprocVecEnv([make_env(gray=True) for _ in range(NUM_ENVS)])
venv = VecTransposeImage(venv, channel_order='first')  # from (N,H,W,C) to (N,C,H,W) because PyTorch convs uses channels first, also SB3 CNN policies expect channels first
venv = VecMonitor(venv)
# Optional: normalize obs & rewards (often helpful for PPO)
venv = VecNormalize(venv, norm_obs=True, norm_reward=True, clip_obs=10.0)

# Eval env (no normalization updates during eval)
eval_env = DummyVecEnv([make_env(gray=True)])
eval_env = VecMonitor(eval_env)
eval_env = VecNormalize(eval_env, training=False, norm_obs=True, norm_reward=False)

# Callbacks: evaluate & checkpoint
eval_cb = EvalCallback(
    eval_env,
    best_model_save_path="./checkpoints/",
    log_path="./logs/",
    eval_freq=10_000,
    n_eval_episodes=5,
    deterministic=True,
)
ckpt_cb = CheckpointCallback(save_freq=50_000, save_path="./checkpoints/", name_prefix="ppo_carracing")

# PPO with a small CNN policy
model = PPO(
    "CnnPolicy",
    venv,
    n_steps=1024,           # per env → 1024*NUM_ENVS per update
    batch_size=256,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,
    vf_coef=0.5,
    learning_rate=3e-4,
    verbose=1,
    tensorboard_log="./tb/"
)

model.learn(total_timesteps=2_000_000, callback=[eval_cb, ckpt_cb])
model.save("ppo_carracing_final")
