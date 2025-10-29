import gymnasium as gym
from gymnasium.wrappers import RecordEpisodeStatistics, FrameStack, ResizeObservation, GrayScaleObservation
import gymnasium as gym
from gymnasium import ObservationWrapper
from gymnasium.spaces import Box
import numpy as np
from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor, VecNormalize, VecTransposeImage
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from CNN_feature_extractor import CNN_feature_extraction

ENV_ID = "CarRacing-v2"   # Gymnasium id

class ScaleToUnitInterval(ObservationWrapper):
    """
    Map uint8 images in [0, 255] to float32 in [0, 1].
    Works with HWC images and with FrameStack concatenated along C.
    """
    def __init__(self, env):
        super().__init__(env)
        low = np.array(self.observation_space.low, dtype=np.float32) / 255.0
        high = np.array(self.observation_space.high, dtype=np.float32) / 255.0
        self.observation_space = Box(low=low, high=high, shape=self.observation_space.shape, dtype=np.float32)

    def observation(self, obs):
        return (obs.astype(np.float32) / 255.0)

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
        env = FrameStack(env, 4)                                 # (84,84,4) or (84,84,12) if RGB, buffer of 4 frames
        env = ScaleToUnitInterval(env)                       # scale to [0,1] float32
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
# normalize rewards (often helpful for PPO) 
venv = VecNormalize(venv, norm_obs=False, norm_reward=True, clip_reward=10.0)

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
policy_kwargs = dict(
    features_extractor_class=CNN_feature_extraction,
    features_extractor_kwargs=dict(features_dim=512)
)

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
    tensorboard_log="./tb/",
    device="auto",
    policy_kwargs=policy_kwargs
)

model.learn(total_timesteps=2_000_000, callback=[eval_cb, ckpt_cb])
model.save("ppo_carracing_final")
