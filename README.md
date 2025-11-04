## CarRacing Pixel Control Toolkit

### Wrapper Order
- **Train:** `RecordEpisodeStatistics -> RewardPenaltyWrapper -> (GentleShapingWrapper) -> (DiscretizeActionWrapper) -> ActionRepeat(k=4) -> (Grayscale, Resize) -> EnsureChannelLast -> (RandomShift training-only) -> VecFrameStack -> VecTransposeImage('first') -> VecMonitor`.
- **Eval:** Same order with RandomShift disabled and GentleShaping optional (off by default). `DummyVecEnv` is always used for evaluation.

All policies receive CHW uint8 tensors and the CNN encoder scales observations to `[0, 1]` exactly once. CarRacing rewards are further multiplied by `reward_scale` (default `0.1`) to stabilise critic targets; the original reward is exposed via `info["original_reward"]`.

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
- `--reward-penalty-value` / `--reward-scale` tune shaping severity and reward magnitude (set scale to `1.0` to disable).
- `--random-shift-train` / `--random-shift-eval` control DrQ augmentation (eval defaults to off).
- `--domain-randomize` mirrors Gym's domain randomization flag.

TensorBoard logs live under `./tb/<algo>/` and include policy/value losses, entropy/alpha, and episodic returns.

Implementation detail: we set `normalize_images=False` in SB3 policies so our custom CNN handles the sole `[0, 255] -> [0, 1]` scaling. This avoids the double-normalization that was stalling earlier SAC runs.

### Evaluation
```bash
python evaluate.py --algo sac --model-path checkpoints/sac/best_model.zip --deterministic --episodes 10
```
Optional:
- `--video-dir videos/` records evaluation rollouts (`VecVideoRecorder`).
- `--random-shift` enables augmentation during eval if explicitly requested.

### Sanity Check
`python sanity_check.py` instantiates train/eval envs, prints HWC vs CHW observation shapes, and verifies the CNN extractor emits a 512-D latent vector. Use this after changing wrappers or feature extractors.

### Memory-Aware SAC Defaults
- `buffer_size` defaults to 300_000 but is automatically clamped to respect the replay RAM limit (default 6 GiB). Tune with `--replay-memory-limit-gb` if you have more headroom.
- `batch_size=256`
- `learning_starts=50_000`, `train_freq=1`, `gradient_steps=1`
- `ent_coef='auto_0.2'`, `gamma=0.99`, `tau=0.005`
These fit comfortably in 16 GB RAM while retaining robust pixel-based performance.
