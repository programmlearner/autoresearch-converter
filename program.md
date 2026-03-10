# Agent Instructions for autoresearch-converter

You are an AI research agent tasked with improving a surrogate model for
converter electromagnetic transient (EMT) simulation. Your goal is to minimize
`val_nrmse` on the validation set by iterating on `train.py`.

## Setup Phase

1. **Create a branch**: `git checkout -b autoresearch/<descriptive-tag>`
2. **Read the codebase**: Read `README.md`, `prepare.py`, and `train.py` carefully.
3. **Verify data**: Check that CSV files exist in `~/.cache/autoresearch-converter/data/`.
   If not, run `uv run generate_sample_data.py` to create them.
4. **Create results log**: Create `results.tsv` with columns:
   `step	tag	val_nrmse	peak_error	thd_error	power_balance	num_params_K	notes`
5. **Run baseline**: Execute `uv run train.py`, record the baseline `val_nrmse`.

## Experiment Loop

Repeat the following cycle:

1. **Hypothesize**: Based on previous results and domain knowledge, form a
   hypothesis about what change will improve `val_nrmse`.
2. **Modify `train.py`**: Implement the change. Only modify `train.py` —
   never modify `prepare.py`.
3. **Commit**: `git add train.py && git commit -m "<brief description>"`
4. **Train**: Run `uv run train.py` and wait for it to complete (~5 min).
5. **Extract metrics**: Parse the output to get `val_nrmse` and other metrics.
6. **Record**: Append a row to `results.tsv`.
7. **Decide**:
   - If `val_nrmse` improved → keep the change, continue iterating.
   - If `val_nrmse` worsened → `git revert HEAD` and try a different approach.

## Domain Knowledge

Use these insights to guide your experiments:

### Signal Characteristics
- Strong 50 Hz fundamental component in all signals
- ~5 kHz switching transients from PWM — model must capture both fast and slow dynamics
- Three-phase symmetry with 120° phase shifts between phases

### Recommended Architectures (implement with PyTorch primitives only)
- **TCN** (Temporal Convolutional Network): dilated causal convolutions
- **Causal Transformer**: masked self-attention for temporal modeling
- **SSM/Mamba-style**: state-space models with selective scan
- **WaveNet-style**: dilated causal convolutions with gated activations
- Hybrid approaches combining multiple architectures

### Input Engineering
- **dv/dt features**: finite-difference derivative of voltage inputs
- **Phase angle**: sin(2πft) and cos(2πft) as auxiliary inputs
- **dq0 transform**: Park transformation to decouple three-phase signals
- **Multi-scale inputs**: provide both raw and downsampled signals

### Loss Function Ideas
- **Frequency-domain loss**: penalize FFT magnitude/phase errors
- **Multi-scale loss**: MSE at multiple temporal resolutions
- **Derivative loss**: penalize di/dt prediction errors
- **Envelope loss**: penalize peak/envelope tracking errors

### Training Strategies
- Curriculum learning: start with steady-state, add transients later
- Multi-step prediction with teacher forcing ratio decay
- Gradient accumulation for effective larger batch sizes
- Learning rate warmup + cosine decay

### Hard Constraints
- **Causality**: `check_causality()` must PASS. If it fails, the experiment
  is invalid. Use causal convolutions or autoregressive architectures only.
- **Time budget**: Training must complete within 300 seconds.
- **No external packages**: Only use PyTorch primitives (nn.Module, nn.Conv1d,
  nn.Linear, nn.LSTM, etc.). Do not import mamba-ssm or similar.

## Tips
- Start with small changes (hyperparameter tuning) before large architectural changes.
- Keep the model size reasonable — very large models won't train well in 5 minutes.
- Monitor training loss convergence — if loss plateaus early, try a higher learning rate.
- If an experiment fails to run, fix the error before moving on.
- Document your reasoning in commit messages.
