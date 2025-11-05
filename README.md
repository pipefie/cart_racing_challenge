## CarRacing Pixel Control Toolkit

### Project Goals
- Train vision-based RL agents (SAC & PPO) on Gymnasium `CarRacing-v3` purely from pixels.
- Keep the training pipeline reproducible and memory-aware (<=16 GB RAM).
- Provide a reference feature extractor, environment wrappers, and CLIs for training/evaluation.
- Support discrete and continuous control with optional reward shaping (grass penalty, speed bonus).

### Wrapper Order
- **Train:** `RecordEpisodeStatistics -> RewardPenaltyWrapper -> (SpeedRewardWrapper) -> (TrackEdgePenaltyWrapper) -> (GentleShapingWrapper) -> (DiscretizeActionWrapper) -> ActionRepeat(k=4) -> (Grayscale, Resize) -> EnsureChannelLast -> (RandomShift training-only) -> VecFrameStack -> VecTransposeImage('first') -> VecMonitor`.
- **Eval:** Same order with RandomShift disabled and GentleShaping optional (off by default). `DummyVecEnv` is always used for evaluation.

All policies receive CHW uint8 tensors and the CNN encoder scales observations to `[0, 1]` exactly once. CarRacing rewards are further multiplied by `reward_scale` (default `0.1`) to stabilise critic targets; the original reward is exposed via `info["original_reward"]`.

### Reward Shaping Summary
- **RewardPenaltyWrapper**: detects grass via RGB pixels and subtracts a moderate penalty (default `1.5`), encouraging the agent to stay on asphalt without overwhelming the base reward.
- **SpeedRewardWrapper**: adds `scale * speed^power`, using the simulator-provided `info["speed"]`. Defaults (`scale=0.03`, `power=0.6`) reward sustained pace while capping runaway bonuses; the shaping term is exposed as `info["speed_reward"]`.
- **TrackEdgePenaltyWrapper**: watches the `info["track"]` rangefinder distances and subtracts a speed-weighted penalty when the car hugs the edge (default threshold `0.65`, scale `0.05`). This keeps curves tight without sacrificing aggression.
- **GentleShapingWrapper** (optional): discourages rapid steering oscillations and simultaneous brake/throttle usage in continuous mode.
- **RewardScaleWrapper**: rescales the combined reward so critic targets remain in a narrow range (default x0.1). Disable with `--reward-scale 1.0` if you prefer raw rewards.

### Feature Extractor
- `CNNFeatureExtractor` implements a Nature-CNN-style encoder tailored to stacked grayscale frames (`VecFrameStack + VecTransposeImage`).
- Observations are uint8; we normalise once inside the extractor to keep SB3's built-in normalisation disabled (`normalize_images=False`).

### Training
```bash
# SAC with memory-safe defaults (continuous actions)
python train.py --algo sac --total-timesteps 2400000 --buffer-size 300000 --learning-starts 50000

# PPO baseline (discrete wrapper, 8 envs, 2M steps)
python train.py --algo ppo --discrete-actions --num-envs 8 --total-timesteps 2000000
```
Add `--config path/to/run.yaml` to load settings from YAML. CLI flags override YAML entries.

Key toggles:
- `--discrete-actions` / `--continuous-actions` switch between discrete wrapper and native continuous control.
- `--action-repeat K` adjusts frame skipping (default 4).
- `--use-gentle-shaping` (train) and `--gentle-shaping-eval` (eval) toggle shaping penalties.
- `--speed-reward-scale` / `--speed-reward-power` shape the speed bonus (defaults 0.03 and 0.6).
- `--use-track-penalty` / `--disable-track-penalty` and `--track-edge-threshold`, `--track-edge-scale` control the edge-avoidance shaping.
- `--reward-penalty-value` / `--reward-scale` tune shaping severity and overall reward magnitude (set scale to `1.0` to disable scaling).
- `--random-shift-train` / `--random-shift-eval` control DrQ augmentation (eval defaults to off).
- `--domain-randomize` mirrors Gym's domain randomization flag.

TensorBoard logs live under `./tb/<algo>/` and include policy/value losses, entropy/alpha, and episodic returns.

Implementation detail: we set `normalize_images=False` in SB3 policies so our custom CNN handles the sole `[0, 255] -> [0, 1]` scaling. This avoids the double-normalization that was stalling earlier SAC runs.

### Evaluation
```bash
python evaluate.py --algo sac --model-path checkpoints/sac/best_model.zip --episodes 5 --deterministic --render-mode human
```
Optional:
- `--video-dir videos/` records evaluation rollouts (`VecVideoRecorder`).
- `--random-shift` enables augmentation during eval if explicitly requested.
- `--render-mode human` opens the native CarRacing window so you can watch the policy drive in real time (can be combined with video recording).

### Sanity Check
`python sanity_check.py` instantiates train/eval envs, prints HWC vs CHW observation shapes, and verifies the CNN extractor emits a 512-D latent vector. Use this after changing wrappers or feature extractors.

### Memory-Aware SAC Defaults
- `buffer_size` defaults to 300_000 but is automatically clamped to respect the replay RAM limit (default 6 GiB). Tune with `--replay-memory-limit-gb` if you have more headroom.
- `batch_size=256`
- `learning_starts=50_000`, `train_freq=1`, `gradient_steps=1`
- `ent_coef='auto_0.2'`, `gamma=0.99`, `tau=0.005`
These fit comfortably in 16 GB RAM while retaining robust pixel-based performance.

### Outputs & Logging
- **Checkpoints**: saved to `./checkpoints/<algo>/` (e.g., `sac_carracing_final.zip`). The best model during training is also stored here via `EvalCallback`.
- **Training logs**: SB3 console/TensorBoard data go to `./logs/<algo>/<timestamp>/` and `./tb/<algo>/`.
- **VecMonitor CSVs**: each run keeps `monitor/monitor.csv` for episodic stats.
- **Video recordings**: when enabled via CLI, MP4s are placed in the directory you provide.

### Reproducibility
- `set_global_seeds` seeds Python `random`, NumPy, and (if available) Torch/CUDA.
- Vector envs are seeded per-worker (`seed + env_idx`) so SubprocVecEnv and DummyVecEnv produce deterministic rollouts given the same seed.

### What We're Learning
- Continuous-control SAC from high-dimensional pixels benefits from gentle reward shaping, action repeat, and data augmentation.  
- Speed-based bonuses encourage lap-time optimisation without destabilising learning when scaled modestly (e.g., 0.02-0.05).  
- Edge-aware penalties derived from the track sensors keep the agent centered through high-speed curves.
- Keeping rewards within a narrow numeric range (`reward_scale`) reduces critic loss spikes and avoids entropy collapse.
- PPO serves as a baseline to validate the preprocessing stack--if PPO improves, the wrappers and feature extractor are wired correctly even before SAC converges.
