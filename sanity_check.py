from __future__ import annotations

import numpy as np

from CNN_feature_extractor import CNNFeatureExtractor
from utils import EnvConfig, build_vec_env, make_env, set_global_seeds


def main() -> None:
    seed = 123
    set_global_seeds(seed)

    train_cfg = EnvConfig(
        domain_randomize=False,
        discrete_actions=False,
        use_reward_penalty=True,
        use_gentle_shaping=False,
        action_repeat=4,
        random_shift=True,
    )

    base_env = make_env(train_cfg, seed=seed, env_idx=0, training=True)()
    obs, _ = base_env.reset()
    print(f"Single-env obs (train pipeline, HWC) shape={obs.shape}, dtype={obs.dtype}")
    base_env.close()

    train_vec = build_vec_env(
        train_cfg,
        num_envs=2,
        seed=seed,
        training=True,
        n_stack=4,
    )
    stacked_obs = train_vec.reset()
    print(
        f"Vec env obs after transpose (CHW) shape={stacked_obs.shape}, "
        f"dtype={stacked_obs.dtype}"
    )

    try:
        import torch as th
    except ImportError as exc:  # pragma: no cover - informative error
        raise RuntimeError(
            "Torch is required for the sanity check. Install PyTorch before running."
        ) from exc

    extractor = CNNFeatureExtractor(train_vec.observation_space, features_dim=512)
    with th.no_grad():
        tensor_obs = th.as_tensor(stacked_obs[:1], dtype=th.float32)
        features = extractor(tensor_obs)
    print(f"CNN output shape={tuple(features.shape)} (should be (1, 512))")

    eval_cfg = EnvConfig(
        domain_randomize=False,
        discrete_actions=False,
        use_reward_penalty=True,
        use_gentle_shaping=False,
        action_repeat=4,
        random_shift=False,
    )
    eval_vec = build_vec_env(
        eval_cfg,
        num_envs=1,
        seed=seed + 42,
        training=False,
        n_stack=4,
    )
    eval_obs = eval_vec.reset()
    print(f"Eval env obs shape={eval_obs.shape}, random shift enabled? {eval_cfg.random_shift}")

    train_vec.close()
    eval_vec.close()


if __name__ == "__main__":
    main()
