# -*- coding: utf-8 -*-
r"""
M3 BNN training script

Input
calculation_results.csv

Features
Strain Amplitude (%), Material, Heat Treatment
Wp (kJ/m3), We (kJ/m3)
Tensile Peak Stress (MPa), Compressive Peak Stress (MPa)
Inflection Strain (%), TYS Stress (MPa), Twinning Gradient (GPa)

Target
Relative Cycle (N/Nf)

Notes
Keeps per epoch loss arrays
Plots raw and EMA smoothed loss curves
Saves model state dict, scaler, loss plot
Includes early stopping based on validation objective
To disable early stopping set patience to 0
"""

import os
import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

import torch
import torch.nn as nn
import torch.optim as optim
import torchbnn as bnn


# =========================
# Paths
# =========================
BASE_DIR = r"F:\Paperwork\AZ91+SEN9_lowcyclefatigue"
BNN_DATA_CSV = os.path.join(BASE_DIR, "calculation_results.csv")

BNN_SAVE_ROOT = os.path.join(BASE_DIR, r"ML\BNN models\full_amplitude")
os.makedirs(BNN_SAVE_ROOT, exist_ok=True)


# =========================
# Columns
# =========================
COMMON_COLS = ["Strain Amplitude (%)", "Material", "Heat Treatment"]
BASELINE_COLS = [
    "Wp (kJ/m3)",
    "We (kJ/m3)",
    "Tensile Peak Stress (MPa)",
    "Compressive Peak Stress (MPa)",
]
ENHANCED_COLS = [
    "Inflection Strain (%)",
    "TYS Stress (MPa)",
    "Twinning Gradient (GPa)",
]

INPUT_COLS = COMMON_COLS + BASELINE_COLS + ENHANCED_COLS
TARGET_COL = "Relative Cycle"


# =========================
# Load data
# =========================
print("=== Loading data for BNN ===")
df = pd.read_csv(BNN_DATA_CSV)

missing = [c for c in INPUT_COLS + [TARGET_COL] if c not in df.columns]
if missing:
    raise ValueError(f"Missing columns in {BNN_DATA_CSV}: {missing}")

df = df.dropna(subset=INPUT_COLS + [TARGET_COL]).reset_index(drop=True)

X = df[INPUT_COLS].values.astype(np.float32)
y = df[TARGET_COL].values.astype(np.float32).reshape(-1, 1)

print(f"Total samples: {X.shape[0]}")


# =========================
# Train val split and scaling
# =========================
X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
)

scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32, device=device)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32, device=device)
X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32, device=device)
y_val_tensor = torch.tensor(y_val, dtype=torch.float32, device=device)


# =========================
# Model
# =========================
class TorchBNN(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.bnn = nn.Sequential(
            bnn.BayesLinear(
                prior_mu=0.0,
                prior_sigma=0.1,
                in_features=input_dim,
                out_features=256,
            ),
            nn.ReLU(),
            bnn.BayesLinear(
                prior_mu=0.0,
                prior_sigma=0.1,
                in_features=256,
                out_features=256,
            ),
            nn.ReLU(),
            bnn.BayesLinear(
                prior_mu=0.0,
                prior_sigma=0.1,
                in_features=256,
                out_features=128,
            ),
            nn.ReLU(),
            bnn.BayesLinear(
                prior_mu=0.0,
                prior_sigma=0.1,
                in_features=128,
                out_features=64,
            ),
            nn.ReLU(),
            bnn.BayesLinear(
                prior_mu=0.0,
                prior_sigma=0.1,
                in_features=64,
                out_features=1,
            ),
        )

    def forward(self, x):
        return self.bnn(x)


# =========================
# Smoothing utilities
# =========================
def ema(arr, alpha=0.02):
    arr = np.asarray(arr, dtype=np.float32)
    out = np.empty_like(arr, dtype=np.float32)
    out[0] = arr[0]
    for i in range(1, len(arr)):
        out[i] = alpha * arr[i] + (1 - alpha) * out[i - 1]
    return out


def moving_average(arr, window=200):
    arr = np.asarray(arr, dtype=np.float32)
    if window <= 1:
        return arr.copy()
    kernel = np.ones(window, dtype=np.float32) / window
    return np.convolve(arr, kernel, mode="same").astype(np.float32)


# =========================
# Training
# =========================
def train_bnn(
    n_epochs: int = 50000,
    kl_weight: float = 0.01,
    lr: float = 1e-3,
    patience: int = 3000,
    min_delta: float = 0.0,
    log_every: int = 500,
    n_mc_val: int = 200,
    mc_infer: int = 100,
):
    model = TorchBNN(input_dim=X_train_scaled.shape[1]).to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    mse_loss = nn.MSELoss()
    kl_loss_fn = bnn.BKLLoss(reduction="mean", last_layer_only=False)

    train_losses = []
    val_losses = []

    best_val_loss = float("inf")
    best_state_dict = None
    best_epoch = -1
    no_improve = 0

    print("=== Training BNN (M3) ===")
    for epoch in range(n_epochs):
        model.train()
        optimizer.zero_grad()

        out = model(X_train_tensor)
        mse = mse_loss(out, y_train_tensor)
        kl = kl_loss_fn(model)
        loss = mse + kl_weight * kl
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            out_val = model(X_val_tensor)
            mse_val = mse_loss(out_val, y_val_tensor)
            kl_val = kl_loss_fn(model)
            val_loss = mse_val + kl_weight * kl_val

        train_losses.append(float(loss.item()))
        val_losses.append(float(val_loss.item()))

        improved = val_loss.item() < (best_val_loss - float(min_delta))
        if improved:
            best_val_loss = float(val_loss.item())
            best_epoch = epoch
            best_state_dict = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if log_every and epoch % log_every == 0:
            print(
                f"[{epoch:05d}] "
                f"train_loss={loss.item():.6f} "
                f"val_loss={val_loss.item():.6f} "
                f"mse={mse.item():.6f} "
                f"mse_val={mse_val.item():.6f} "
                f"no_improve={no_improve}"
            )

        if patience and no_improve >= int(patience):
            print(f"Early stopping at epoch {epoch}  no val improvement")
            break

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    train_losses = np.array(train_losses, dtype=np.float32)
    val_losses = np.array(val_losses, dtype=np.float32)
    epochs_ran_bnn = int(len(train_losses))

    train_losses_ema = ema(train_losses, alpha=0.02)
    val_losses_ema = ema(val_losses, alpha=0.02)
    train_losses_ma = moving_average(train_losses, window=200)
    val_losses_ma = moving_average(val_losses, window=200)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(BNN_SAVE_ROOT, timestamp)
    os.makedirs(save_dir, exist_ok=True)

    plt.figure(figsize=(6, 4))
    plt.plot(train_losses, alpha=0.25, label="train (raw)")
    plt.plot(val_losses, alpha=0.25, label="validation (raw)")
    plt.plot(train_losses_ema, label="train (EMA)")
    plt.plot(val_losses_ema, label="validation (EMA)")
    plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("Loss  MSE + KL")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    loss_path = os.path.join(save_dir, "BNN_loss.png")
    plt.savefig(loss_path, dpi=300)
    plt.show()
    print(f"BNN loss curve saved to {loss_path}")

    model_path = os.path.join(save_dir, "enhanced_model.pt")
    torch.save(model.state_dict(), model_path)

    scaler_path = os.path.join(save_dir, "scaler_enhanced.npy")
    np.save(scaler_path, {"min": scaler.min_, "scale": scaler.scale_}, allow_pickle=True)

    print(f"BNN model saved to {model_path}")
    print(f"BNN scaler saved to {scaler_path}")

    # MC sampling based validation metrics
    model.eval()
    with torch.no_grad():
        preds = []
        for _ in range(int(n_mc_val)):
            preds.append(model(X_val_tensor).cpu().numpy())
        preds = np.stack(preds, axis=0)
        mean_pred = preds.mean(axis=0).flatten()
        std_pred = preds.std(axis=0).flatten()

    y_val_np = y_val.flatten()
    rmse_val = np.sqrt(np.mean((y_val_np - mean_pred) ** 2))
    print(f"Validation RMSE on Relative Cycle: {rmse_val:.6f}")

    lower = mean_pred - 1.64 * std_pred
    upper = mean_pred + 1.64 * std_pred
    coverage = np.mean((y_val_np >= lower) & (y_val_np <= upper))
    print(f"Approx. 90 percent interval empirical coverage: {coverage:.3f}")

    print(f"Epochs ran (M3): {epochs_ran_bnn} / n_epochs={n_epochs}")
    print(f"Best val loss epoch: {best_epoch}  best_val_loss={best_val_loss:.6f}")
    print(f"All outputs saved under: {save_dir}")

    # Inference configuration note
    print(f"MC forward passes for inference: {int(mc_infer)}")

    return {
        "model": model,
        "scaler": scaler,
        "train_loss_raw": train_losses,
        "val_loss_raw": val_losses,
        "train_loss_ema": train_losses_ema,
        "val_loss_ema": val_losses_ema,
        "train_loss_ma": train_losses_ma,
        "val_loss_ma": val_losses_ma,
        "epochs_ran": epochs_ran_bnn,
        "best_val_loss": best_val_loss,
        "best_epoch": best_epoch,
        "save_dir": save_dir,
    }


if __name__ == "__main__":
    torch.manual_seed(42)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(42)

    out_m3 = train_bnn(
        n_epochs=50000,
        kl_weight=0.01,
        lr=1e-3,
        patience=3000,
        min_delta=0.0,
        log_every=500,
        n_mc_val=200,
        mc_infer=100,
    )

    train_loss_m3 = out_m3["train_loss_raw"]
    val_loss_m3 = out_m3["val_loss_raw"]
    train_ema_m3 = out_m3["train_loss_ema"]
    val_ema_m3 = out_m3["val_loss_ema"]
