import os, sys, argparse, logging
from datetime import datetime

import numpy as np
import torch as th
import gymnasium as gym
from gymnasium.wrappers import GrayScaleObservation, ResizeObservation

from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import (
    DummyVecEnv, VecMonitor, VecTransposeImage, VecFrameStack, VecNormalize, VecVideoRecorder
)

# ---------- Config you will edit per run ----------
ENV_ID = "CarRacing-v3"
N_STACK = 4
RESIZE_SHAPE = (84, 84)

# If you used VecNormalize(reward) during training, set this to the path where you saved stats (e.g., vecnorm.pkl)
VECNORM_STATS_PATH = None   # or e.g. "./checkpoints/vecnorm.pkl"

# Choose which algo we’re evaluating and where the model is
ALGO = "ppo"  # "ppo" or "sac"
MODEL_PATH = "checkpoints/ppo_carracing_2400000_steps.zip"   # change for SAC model
# --------------------------------------------------


def make_eval_env(gray=True, resize_shape=(84, 84), domain_randomize=False):
    """
    Single-env preprocessing (HWC). We mirror train preprocessing EXCEPT:
    - no RandomShift in eval (measure canonical performance)
    - no discretization (continuous control)
    """
    def thunk():
        env = gym.make(ENV_ID, domain_randomize=domain_randomize, render_mode="human")
        if gray:
            env = GrayScaleObservation(env, keep_dim=True)   # (H,W,1)
        env = ResizeObservation(env, resize_shape)           # (84,84,1)
        return env
    return thunk


def load_model(algo: str, model_path: str, env, device):
    if algo.lower() == "ppo":
        return PPO.load(model_path, env=env, device=device)
    elif algo.lower() == "sac":
        return SAC.load(model_path, env=env, device=device)
    else:
        raise ValueError("ALGO must be 'ppo' or 'sac'")


def main(args):
    # Logging
    run_id = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler(sys.stdout)])
    logging.info(f"Evaluating {ALGO.upper()} model: {MODEL_PATH}")

    # ---- Build eval env (mirror training order on Vec side) ----
    base_env = DummyVecEnv([make_eval_env(gray=True, resize_shape=RESIZE_SHAPE, domain_randomize=False)])
    base_env = VecFrameStack(base_env, n_stack=N_STACK, channels_order="last")
    base_env = VecTransposeImage(base_env)
    base_env = VecMonitor(base_env)

    # Optional: load VecNormalize stats (only if used during training!)
    if VECNORM_STATS_PATH is not None and os.path.exists(VECNORM_STATS_PATH):
        logging.info(f"Loading VecNormalize stats from: {VECNORM_STATS_PATH}")
        base_env = VecNormalize.load(VECNORM_STATS_PATH, base_env)
        base_env.training = False
        base_env.norm_reward = False
    else:
        logging.info("No VecNormalize stats loaded (eval uses raw rewards).")

    # Optional: record video instead of (or in addition to) live render
    env = base_env
    if args.video_dir is not None:
        os.makedirs(args.video_dir, exist_ok=True)
        # record every episode; use a trigger to record only some if you prefer
        env = VecVideoRecorder(
            base_env,
            video_folder=args.video_dir,
            record_video_trigger=lambda step: True,
            video_length=0,  # 0 = full episodes
            name_prefix=f"{ALGO}_eval_{run_id}"
        )
        logging.info(f"Recording evaluation videos to: {args.video_dir}")

    device = th.device("cuda" if (th.cuda.is_available() and not args.cpu) else "cpu")
    logging.info(f"Eval device: {device}")

    model = load_model(ALGO, MODEL_PATH, env, device)

    # --- Deterministic policy for eval ---
    deterministic = True
    episodes = args.episodes

    returns = []
    for ep in range(episodes):
        obs = env.reset()
        done = False
        ep_ret = 0.0
        logging.info(f"--- Episode {ep+1}/{episodes} ---")
        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, done, infos = env.step(action)
            ep_ret += float(reward[0])
        logging.info(f"Episode return: {ep_ret:.2f}")
        returns.append(ep_ret)

    logging.info(f"Mean return over {episodes} eps: {np.mean(returns):.2f} ± {np.std(returns):.2f}")
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=5, help="Number of evaluation episodes")
    parser.add_argument("--cpu", action="store_true", help="Force CPU evaluation")
    parser.add_argument("--video_dir", type=str, default=None, help="Folder to save videos (optional)")
    args = parser.parse_args()
    main(args)
