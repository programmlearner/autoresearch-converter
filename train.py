"""
Baseline surrogate model for converter EMT simulation.

Architecture: 2-layer LSTM, hidden_dim=128
The agent modifies this file to explore better architectures and training strategies.
"""

import time
import math
import torch
import torch.nn as nn
import numpy as np
from prepare import (
    load_data,
    normalize_data,
    make_dataloader,
    evaluate_nrmse,
    check_causality,
    print_results,
    WINDOW_SIZE,
    INPUT_CHANNELS,
    OUTPUT_CHANNELS,
    TIME_BUDGET,
)


class ConverterSurrogate(nn.Module):
    """Baseline LSTM surrogate model."""

    def __init__(self, input_dim=INPUT_CHANNELS, output_dim=OUTPUT_CHANNELS,
                 hidden_dim=128, num_layers=2, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        """
        Args:
            x: (B, T, input_dim) voltage input
        Returns:
            y: (B, T, output_dim) current prediction
        """
        lstm_out, _ = self.lstm(x)  # (B, T, hidden_dim)
        y = self.fc(lstm_out)       # (B, T, output_dim)
        return y


def count_parameters(model):
    """Count trainable parameters in thousands."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad) / 1000.0


def train():
    # ── Device setup ─────────────────────────────────────────────────────
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # ── Data loading ─────────────────────────────────────────────────────
    print("Loading data...")
    train_scenarios, val_scenarios = load_data()
    train_norm, stats = normalize_data(train_scenarios)
    val_norm, _ = normalize_data(val_scenarios, stats)

    # ── Model ────────────────────────────────────────────────────────────
    model = ConverterSurrogate().to(device)
    num_params_k = count_parameters(model)
    print(f"Model parameters: {num_params_k:.1f}K")

    # ── Causality check ──────────────────────────────────────────────────
    sample_v = train_norm[0][0][:WINDOW_SIZE]
    sample_input = torch.tensor(sample_v, dtype=torch.float32).unsqueeze(0).to(device)
    is_causal, max_violation = check_causality(model, sample_input, device)
    print(f"Causality check: {'PASS' if is_causal else 'FAIL'} (max_violation={max_violation:.2e})")
    if not is_causal:
        print("WARNING: Model is not causal. Results may be invalid.")

    # ── Training setup ───────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.MSELoss()

    batch_size = 32
    num_steps = 0
    best_val_nrmse = float("inf")
    peak_vram_mb = 0.0

    # Estimate total steps for cosine schedule
    # Approximate: num_windows * epochs / batch_size
    num_windows_approx = sum(
        (len(v) - WINDOW_SIZE) // (WINDOW_SIZE // 2) + 1
        for v, i in train_norm
    )
    steps_per_epoch = num_windows_approx // batch_size
    estimated_total_steps = steps_per_epoch * 50  # rough upper bound
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=estimated_total_steps, eta_min=1e-6
    )

    # ── Training loop ────────────────────────────────────────────────────
    print("Starting training...")
    start_time = time.time()
    epoch = 0

    while True:
        elapsed = time.time() - start_time
        if elapsed >= TIME_BUDGET:
            break

        model.train()
        epoch_loss = 0.0
        epoch_steps = 0

        for v_batch, i_batch in make_dataloader(train_norm, batch_size, device=device):
            if time.time() - start_time >= TIME_BUDGET:
                break

            optimizer.zero_grad()
            i_pred = model(v_batch)
            loss = criterion(i_pred, i_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            epoch_steps += 1
            num_steps += 1

            # Track VRAM
            if device == "cuda":
                vram = torch.cuda.max_memory_allocated() / 1024 / 1024
                peak_vram_mb = max(peak_vram_mb, vram)

        epoch += 1
        avg_loss = epoch_loss / max(epoch_steps, 1)

        if epoch % 5 == 0:
            elapsed = time.time() - start_time
            print(f"Epoch {epoch} | Loss: {avg_loss:.6f} | Steps: {num_steps} | Time: {elapsed:.1f}s")

    training_seconds = time.time() - start_time

    # ── Evaluation ───────────────────────────────────────────────────────
    print("\nEvaluating on validation set...")
    metrics = evaluate_nrmse(model, val_scenarios, stats, device)

    # ── Print results ────────────────────────────────────────────────────
    print_results(metrics, training_seconds, peak_vram_mb, num_params_k, num_steps)


if __name__ == "__main__":
    train()
