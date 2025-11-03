# car_racing_SAC.py
import os, sys, logging
from datetime import datetime
import torch as th
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import (
    DummyVecEnv, SubprocVecEnv, VecMonitor, VecNormalize, VecTransposeImage, VecFrameStack
)
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common import logger as sb3_logger

from CNN_feature_extractor import CNN_feature_extraction
from utils import *

if __name__ == "__main__":
    # Logging
    run_id = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    sb3_log_dir = os.path.join("./logs/sac/", run_id)
    os.makedirs(sb3_log_dir, exist_ok=True)
    sb3_logger.configure(sb3_log_dir, format_strings=["stdout","log","csv","tensorboard"])
    logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler(sys.stdout)])

    NUM_ENVS = 8
    N_STACK = 4
    TOTAL_TIMESTEPS = 2_400_000

    # --- TRAIN ENV (continuous) ---
    venv = SubprocVecEnv([make_env(gray=True, domain_randomize=True, discrete=False, use_gentle_shaping=False)
                          for _ in range(NUM_ENVS)])
    venv = VecFrameStack(venv, n_stack=N_STACK, channels_order="last")   # (HWC, stacked)
    venv = VecTransposeImage(venv)                # -> CHW (C=4)
    venv = VecMonitor(venv)
    # Optional reward norm (keep obs norm OFF because we scale inside the CNN extractor):
    # venv = VecNormalize(venv, norm_obs=False, norm_reward=True, clip_reward=10.0)

    # --- EVAL ENV (mirror train) ---
    eval_env = DummyVecEnv([make_env(gray=True, domain_randomize=False, discrete=False, use_gentle_shaping=False)])
    eval_env = VecFrameStack(eval_env, n_stack=N_STACK, channels_order="last")
    eval_env = VecTransposeImage(eval_env)
    eval_env = VecMonitor(eval_env)
    # If you used VecNormalize for rewards in train, load stats here and:
    # eval_env.training = False; eval_env.norm_reward = False

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path="./checkpoints_sac/",
        log_path="./logs/sac/",
        eval_freq=10_000,
        n_eval_episodes=5,
        deterministic=True,   # evaluate mean action
    )
    ckpt_cb = CheckpointCallback(save_freq=50_000, save_path="./checkpoints_sac/", name_prefix="sac_carracing")

    policy_kwargs = dict(
        features_extractor_class=CNN_feature_extraction,
        features_extractor_kwargs=dict(features_dim=512),
        net_arch=dict(pi=[256, 256], qf=[256, 256])  # SAC uses pi and qf
    )

    device_name = "cuda" if th.cuda.is_available() else "cpu"
    logging.info(f"Device: {device_name}")

    model = SAC(
        policy="CnnPolicy",
        env=venv,
        learning_rate=3e-4,
        buffer_size=200_000,
        optimize_memory_usage=True,     
        learning_starts=10_000,
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        train_freq=1,
        gradient_steps=1,
        ent_coef="auto",
        target_update_interval=1,
        policy_kwargs=policy_kwargs,
        replay_buffer_kwargs=dict(
            handle_timeout_termination=False              # set it HERE for your SB3 version
        ),
        tensorboard_log="./tb/sac/",
        verbose=1,
        device=device_name
    )

    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=[eval_cb, ckpt_cb])
    model.save("sac_carracing_final")

