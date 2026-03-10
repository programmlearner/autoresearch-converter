"""
Data infrastructure for autoresearch-converter.

Provides data loading, normalization, dataloader creation, and evaluation.
THIS FILE SHOULD NOT BE MODIFIED by the agent.
"""

import time
import numpy as np
import pandas as pd
import torch
from pathlib import Path

# ── Constants ────────────────────────────────────────────────────────────────
SAMPLE_RATE = 50000
WINDOW_SIZE = 1000       # 20ms = one fundamental cycle at 50Hz
INPUT_CHANNELS = 3       # va, vb, vc
OUTPUT_CHANNELS = 3      # ia, ib, ic
TIME_BUDGET = 300         # 5 minutes

TRAIN_RANGE = (0, 160)   # scenario indices for training
VAL_RANGE = (160, 200)   # scenario indices for validation

DATA_DIR = Path.home() / ".cache" / "autoresearch-converter" / "data"


def load_data(data_dir=None):
    """
    Load CSV scenario files.

    Returns:
        train_scenarios: list of (voltage_array, current_array) tuples
        val_scenarios:   list of (voltage_array, current_array) tuples
        Each array has shape (T, 3).
    """
    if data_dir is None:
        data_dir = DATA_DIR
    data_dir = Path(data_dir)

    train_scenarios = []
    val_scenarios = []

    for idx in range(TRAIN_RANGE[0], TRAIN_RANGE[1]):
        csv_path = data_dir / f"scenario_{idx:04d}.csv"
        df = pd.read_csv(csv_path)
        v = df[["va", "vb", "vc"]].values.astype(np.float32)
        i = df[["ia", "ib", "ic"]].values.astype(np.float32)
        train_scenarios.append((v, i))

    for idx in range(VAL_RANGE[0], VAL_RANGE[1]):
        csv_path = data_dir / f"scenario_{idx:04d}.csv"
        df = pd.read_csv(csv_path)
        v = df[["va", "vb", "vc"]].values.astype(np.float32)
        i = df[["ia", "ib", "ic"]].values.astype(np.float32)
        val_scenarios.append((v, i))

    return train_scenarios, val_scenarios


def normalize_data(scenarios, stats=None):
    """
    Z-score normalization across all scenarios.

    Args:
        scenarios: list of (voltage, current) tuples
        stats: optional dict with 'v_mean', 'v_std', 'i_mean', 'i_std'
               If None, compute from data.

    Returns:
        normalized_scenarios: list of (norm_v, norm_i) tuples
        stats: dict with normalization statistics
    """
    if stats is None:
        all_v = np.concatenate([v for v, i in scenarios], axis=0)
        all_i = np.concatenate([i for v, i in scenarios], axis=0)
        stats = {
            "v_mean": all_v.mean(axis=0),
            "v_std": all_v.std(axis=0) + 1e-8,
            "i_mean": all_i.mean(axis=0),
            "i_std": all_i.std(axis=0) + 1e-8,
        }

    normalized = []
    for v, i in scenarios:
        norm_v = (v - stats["v_mean"]) / stats["v_std"]
        norm_i = (i - stats["i_mean"]) / stats["i_std"]
        normalized.append((norm_v, norm_i))

    return normalized, stats


def make_dataloader(scenarios, batch_size, window_size=WINDOW_SIZE, shuffle=True, device="cpu"):
    """
    Sliding window dataloader generator.

    Yields batches of (voltage_input, current_target) tensors.
    Each tensor has shape (B, T, 3) where T = window_size.
    """
    # Collect all windows
    windows_v = []
    windows_i = []

    for v, i in scenarios:
        num_samples = len(v)
        # Stride = window_size // 2 for overlap
        stride = window_size // 2
        for start in range(0, num_samples - window_size + 1, stride):
            end = start + window_size
            windows_v.append(v[start:end])
            windows_i.append(i[start:end])

    windows_v = np.array(windows_v)  # (N, T, 3)
    windows_i = np.array(windows_i)  # (N, T, 3)
    num_windows = len(windows_v)

    indices = np.arange(num_windows)
    if shuffle:
        np.random.shuffle(indices)

    for start in range(0, num_windows, batch_size):
        batch_idx = indices[start:start + batch_size]
        v_batch = torch.tensor(windows_v[batch_idx], dtype=torch.float32, device=device)
        i_batch = torch.tensor(windows_i[batch_idx], dtype=torch.float32, device=device)
        yield v_batch, i_batch


def evaluate_nrmse(model, val_scenarios, norm_stats, device="cpu"):
    """
    Main evaluation function.

    Args:
        model: PyTorch model with forward(v_input) -> i_pred
        val_scenarios: list of (voltage, current) tuples (unnormalized)
        norm_stats: normalization statistics dict
        device: torch device

    Returns:
        metrics: dict with val_nrmse, peak_error, thd_error, power_balance
    """
    model.eval()

    all_nrmse = []
    all_peak_errors = []
    all_thd_errors = []
    all_power_errors = []

    with torch.no_grad():
        for v_raw, i_raw in val_scenarios:
            # Normalize input
            v_norm = (v_raw - norm_stats["v_mean"]) / norm_stats["v_std"]
            v_tensor = torch.tensor(v_norm, dtype=torch.float32, device=device).unsqueeze(0)  # (1, T, 3)

            # Predict
            i_pred_norm = model(v_tensor).squeeze(0).cpu().numpy()  # (T, 3)

            # Denormalize prediction
            i_pred = i_pred_norm * norm_stats["i_std"] + norm_stats["i_mean"]

            # NRMSE per phase
            for ch in range(3):
                i_true_ch = i_raw[:, ch]
                i_pred_ch = i_pred[:len(i_true_ch), ch]

                rmse = np.sqrt(np.mean((i_true_ch - i_pred_ch) ** 2))
                data_range = i_true_ch.max() - i_true_ch.min()
                if data_range < 1e-6:
                    data_range = 1.0
                nrmse = rmse / data_range
                all_nrmse.append(nrmse)

                # Peak error
                peak_true = np.max(np.abs(i_true_ch))
                peak_pred = np.max(np.abs(i_pred_ch))
                if peak_true > 1e-6:
                    peak_err = abs(peak_true - peak_pred) / peak_true
                else:
                    peak_err = 0.0
                all_peak_errors.append(peak_err)

                # THD error (simplified: ratio of harmonic content)
                fft_true = np.abs(np.fft.rfft(i_true_ch))
                fft_pred = np.abs(np.fft.rfft(i_pred_ch))
                if len(fft_true) > 1 and fft_true[1] > 1e-6:
                    thd_true = np.sqrt(np.sum(fft_true[2:] ** 2)) / fft_true[1]
                    thd_pred = np.sqrt(np.sum(fft_pred[2:] ** 2)) / fft_pred[1]
                    thd_err = abs(thd_true - thd_pred)
                else:
                    thd_err = 0.0
                all_thd_errors.append(thd_err)

            # Power balance error (average instantaneous power)
            v_used = v_raw[:len(i_pred)]
            p_true = np.mean(np.sum(v_used * i_raw[:len(i_pred)], axis=1))
            p_pred = np.mean(np.sum(v_used * i_pred[:len(i_pred)], axis=1))
            if abs(p_true) > 1e-6:
                power_err = abs(p_true - p_pred) / abs(p_true)
            else:
                power_err = 0.0
            all_power_errors.append(power_err)

    metrics = {
        "val_nrmse": float(np.mean(all_nrmse)),
        "peak_error": float(np.mean(all_peak_errors)),
        "thd_error": float(np.mean(all_thd_errors)),
        "power_balance": float(np.mean(all_power_errors)),
    }

    return metrics


def check_causality(model, sample_input, device="cpu"):
    """
    Check if model output at time t depends only on inputs at time <= t.

    Uses gradient-based detection: perturb future inputs and check if
    past outputs change.

    Args:
        sample_input: tensor of shape (1, T, 3)
        device: torch device

    Returns:
        is_causal: bool
        max_violation: float (max absolute gradient from future to past)
    """
    model.eval()
    T = sample_input.shape[1]
    mid = T // 2

    x = sample_input.clone().to(device).requires_grad_(True)
    y = model(x)  # (1, T, 3)

    # Sum outputs at time mid
    target = y[0, mid, :].sum()
    target.backward()

    # Check if gradient w.r.t. future inputs (t > mid) is zero
    grad = x.grad[0]  # (T, 3)
    future_grad = grad[mid + 1:, :]
    max_violation = float(future_grad.abs().max())

    is_causal = max_violation < 1e-5

    return is_causal, max_violation


def print_results(metrics, training_seconds, peak_vram_mb, num_params_k, num_steps):
    """Print results in the standard format."""
    print("---")
    print(f"val_nrmse:        {metrics['val_nrmse']:.6f}")
    print(f"peak_error:       {metrics['peak_error']:.6f}")
    print(f"thd_error:        {metrics['thd_error']:.6f}")
    print(f"power_balance:    {metrics['power_balance']:.6f}")
    print(f"training_seconds: {training_seconds:.1f}")
    print(f"peak_vram_mb:     {peak_vram_mb:.1f}")
    print(f"num_params_K:     {num_params_k:.1f}")
    print(f"num_steps:        {num_steps}")
