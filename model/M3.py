
"""
M3: Bayesian neural network for low-cycle fatigue life prediction

- Clean paths and CLI flags for distribution
- Train/test split by a user-selected target strain amplitude
- Baseline vs enhanced feature sets
- PP score, parity plot, per-sample scores
"""

import argparse
import os
from datetime import datetime
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torchbnn as bnn
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt


# -----------------------------
# Utilities
# -----------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_columns(df: pd.DataFrame, required_cols: list):
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in CSV: {missing}")


def to_tensor(x, dtype=torch.float32):
    return torch.tensor(x, dtype=dtype)


# -----------------------------
# Model
# -----------------------------
class TorchBNN(nn.Module):
    def __init__(self, input_dim: int, hidden: int = 64, prior_mu: float = 0.0, prior_sigma: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            bnn.BayesLinear(prior_mu=prior_mu, prior_sigma=prior_sigma, in_features=input_dim, out_features=hidden),
            nn.ReLU(),
            bnn.BayesLinear(prior_mu=prior_mu, prior_sigma=prior_sigma, in_features=hidden, out_features=hidden),
            nn.ReLU(),
            bnn.BayesLinear(prior_mu=prior_mu, prior_sigma=prior_sigma, in_features=hidden, out_features=1),
        )

    def forward(self, x):
        return self.net(x)


def train_bnn(model: nn.Module,
              X: np.ndarray,
              y: np.ndarray,
              n_epochs: int = 20000,
              lr: float = 1e-3,
              kl_weight: float = 1e-2,
              log_every: int = 1000):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    mse_loss = nn.MSELoss()
    kl_loss_fn = bnn.BKLLoss(reduction='mean', last_layer_only=False)

    X_tensor = to_tensor(X)
    y_tensor = to_tensor(y)

    model.train()
    for epoch in range(n_epochs):
        output = model(X_tensor)
        mse = mse_loss(output, y_tensor)
        kl = kl_loss_fn(model)
        loss = mse + kl_weight * kl

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if log_every and (epoch % log_every == 0):
            print(f"[{epoch}] MSE {mse.item():.6f} KL {kl.item():.6f} Total {loss.item():.6f}")

    return model


@torch.no_grad()
def predict_bnn(model: nn.Module, X: np.ndarray, n_samples: int = 100):
    model.eval()
    X_tensor = to_tensor(X)
    preds = []
    for _ in range(n_samples):
        preds.append(model(X_tensor).numpy())
    preds = np.stack(preds, axis=0)  # [S, N, 1]
    mean = preds.mean(axis=0)
    std = preds.std(axis=0)
    return mean, std


def pp_score(y_true: np.ndarray, mu: np.ndarray, sigma: np.ndarray, eps: float = 1e-8):
    """Posterior-probability score under a Gaussian likelihood."""
    s = np.clip(sigma, eps, None)
    coeff = 1.0 / (np.sqrt(2.0 * np.pi) * s)
    expo = np.exp(-0.5 * ((y_true - mu) ** 2) / (s ** 2))
    return coeff * expo  # shape [N, 1]


# -----------------------------
# CLI and main
# -----------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Train and evaluate M3 BNN for LCF life prediction")

    # I/O
    p.add_argument("--csv", type=str, required=True,
                   help="Path to the features CSV produced by M2")
    p.add_argument("--outdir", type=str, default="outputs",
                   help="Directory for outputs plots and models")

    # Columns
    p.add_argument("--col_strain_amp", type=str, default="Strain Amplitude (%)")
    p.add_argument("--col_material", type=str, default="Material")
    p.add_argument("--col_heat_treat", type=str, default="Heat Treatment")
    p.add_argument("--col_target", type=str, default="Relative Cycle")

    # Feature sets
    p.add_argument("--use_enhanced", action="store_true",
                   help="Use enhanced feature set that includes microstructural parameters")
    p.add_argument("--mc_samples", type=int, default=100,
                   help="Monte Carlo forward passes for uncertainty")

    # Split
    p.add_argument("--target_strain", type=float, default=1.2,
                   help="Strain amplitude reserved for testing for example 1.2")
    p.add_argument("--test_query", type=str, default="",
                   help="Optional pandas query string for further test filtering for example \"Material == 1 and `Heat Treatment` == 400\"")

    # Training
    p.add_argument("--epochs", type=int, default=20000)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--kl_weight", type=float, default=1e-2)
    p.add_argument("--seed", type=int, default=42)

    # Plots
    p.add_argument("--no_plots", action="store_true",
                   help="Disable plotting")

    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    # Load
    df = pd.read_csv(args.csv)

    # Required columns and feature sets
    common_cols = [args.col_strain_amp, args.col_material, args.col_heat_treat]
    baseline_cols = ["Wp (kJ/m3)", "We (kJ/m3)", "Tensile Peak Stress (MPa)", "Compressive Peak Stress (MPa)"]
    enhanced_cols = ["Inflection Strain (%)", "TYS Stress (MPa)", "Twinning Gradient (GPa)"]

    baseline_input_cols = common_cols + baseline_cols
    enhanced_input_cols = baseline_input_cols + enhanced_cols
    input_cols = enhanced_input_cols if args.use_enhanced else baseline_input_cols

    ensure_columns(df, input_cols + [args.col_target])

    # Split by target strain amplitude
    train_df = df[df[args.col_strain_amp] != args.target_strain].copy()
    test_df = df[df[args.col_strain_amp] == args.target_strain].copy()

    # Optional further restriction on the test set
    if args.test_query:
        try:
            test_df = test_df.query(args.test_query)
        except Exception as e:
            raise ValueError(f"Failed to apply test_query. Error {e}")

    if len(test_df) == 0:
        raise ValueError("Empty test set after filtering. Adjust --target_strain or --test_query")

    # Data arrays
    X_train = train_df[input_cols].values
    X_test = test_df[input_cols].values
    y_train = train_df[[args.col_target]].values
    y_test = test_df[[args.col_target]].values

    # Scaling on train only
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train
    model = TorchBNN(input_dim=X_train_scaled.shape[1])
    model = train_bnn(model,
                      X_train_scaled,
                      y_train,
                      n_epochs=args.epochs,
                      lr=args.lr,
                      kl_weight=args.kl_weight,
                      log_every=1000)

    # Predict with MC sampling
    y_mean, y_std = predict_bnn(model, X_test_scaled, n_samples=args.mc_samples)

    # Scores
    pp = pp_score(y_test, y_mean, y_std)
    total_pp = float(pp.sum())

    # Output folder
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = os.path.join(args.outdir, f"m3_{'enh' if args.use_enhanced else 'base'}_{stamp}")
    os.makedirs(outdir, exist_ok=True)

    # Save model weights locally for the user
    torch.save(model.state_dict(), os.path.join(outdir, "model.pt"))

    # Save predictions
    pred_df = test_df.copy()
    pred_df["y_true"] = y_test.flatten()
    pred_df["y_pred_mean"] = y_mean.flatten()
    pred_df["y_pred_std"] = y_std.flatten()
    pred_df["pp_score"] = pp.flatten()
    pred_path = os.path.join(outdir, "predictions.csv")
    pred_df.to_csv(pred_path, index=False)

    print("\n--- Evaluation ---")
    print(f"Test count {len(test_df)}")
    print(f"Total PP score {total_pp:.6f}")
    print(f"Saved predictions to {pred_path}")
    print(f"Model saved to {os.path.join(outdir, 'model.pt')}")

    # Plots
    if not args.no_plots:
        eps = 1e-6
        y_true = y_test.flatten() + eps
        y_mu = y_mean.flatten() + eps
        y_sigma = y_std.flatten()

        # Parity plot
        plt.figure(figsize=(7, 6))
        plt.errorbar(y_true, y_mu, yerr=y_sigma, fmt="o", capsize=3, alpha=0.8)
        mn = min(y_true.min(), y_mu.min())
        mx = max(y_true.max(), y_mu.max())
        plt.plot([mn, mx], [mn, mx], "k--", linewidth=1)
        plt.xscale("log")
        plt.yscale("log")
        plt.xlabel("True Relative Cycle N/Nf")
        plt.ylabel("Predicted mean ± std")
        plt.title(f"BNN parity plot  {'enhanced' if args.use_enhanced else 'baseline'}")
        plt.grid(True, which="both", ls="--", linewidth=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, "parity_plot.png"), dpi=200)

        # Instance-wise PP
        plt.figure(figsize=(8, 4))
        plt.plot(pp.flatten(), "o-")
        plt.axhline(0, color="gray", linestyle="--", linewidth=0.8)
        plt.xlabel("Test sample index")
        plt.ylabel("PP score")
        plt.title("Posterior probability per sample")
        plt.grid(True, ls="--", linewidth=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, "pp_scores.png"), dpi=200)

        plt.close("all")


if __name__ == "__main__":
    main()
