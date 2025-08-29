
"""
M3 analysis utility
- Loads saved Bayesian models (baseline and/or enhanced)
- Rebuilds the same train/test split by target strain amplitude
- Fits scalers on the train split, applies to test split and grids
- Generates parity plot and 1D/2D uncertainty maps
- Saves outputs into outputs/analysis_<stamp>/
"""

import argparse
import os
from datetime import datetime
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchbnn as bnn
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt


# -----------------------------
# Utils
# -----------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_columns(df: pd.DataFrame, cols: list):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def to_tensor(x, dtype=torch.float32):
    return torch.tensor(x, dtype=dtype)


# -----------------------------
# Model (same as training)
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


@torch.no_grad()
def predict_bnn(model: nn.Module, X: np.ndarray, n_samples: int = 100):
    model.eval()
    X_tensor = to_tensor(X)
    preds = []
    for _ in range(n_samples):
        preds.append(model(X_tensor).numpy())
    arr = np.stack(preds, axis=0)  # [S, N, 1]
    return arr.mean(axis=0), arr.std(axis=0)


# -----------------------------
# Plot helpers
# -----------------------------
def parity_plot(y_true, y_mu, y_sigma, title, outpath):
    eps = 1e-6
    yt = y_true.flatten() + eps
    ym = y_mu.flatten() + eps
    ys = y_sigma.flatten()

    plt.figure(figsize=(7, 6))
    plt.errorbar(yt, ym, yerr=ys, fmt="o", capsize=3, alpha=0.85)
    mn = min(yt.min(), ym.min())
    mx = max(yt.max(), ym.max())
    plt.plot([mn, mx], [mn, mx], "k--", linewidth=1)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("True relative cycle N/Nf")
    plt.ylabel("Predicted mean ± std")
    plt.title(title)
    plt.grid(True, which="both", ls="--", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def one_d_maps(model, scaler, base_point_scaled, cols, grid_points, title_prefix, outdir):
    for i, col in enumerate(cols):
        grid = np.tile(base_point_scaled, (grid_points, 1))
        grid[:, i] = np.linspace(0.0, 1.0, grid_points)
        mu, sd = predict_bnn(model, grid, n_samples=100)
        raw = scaler.inverse_transform(grid)[:, i]

        plt.figure(figsize=(8, 5))
        sc = plt.scatter(raw, mu, c=sd, cmap="viridis", s=45, edgecolors="k", linewidths=0.4)
        cbar = plt.colorbar(sc)
        cbar.set_label("Predictive std")
        plt.xlabel(col)
        plt.ylabel("Predicted mean N/Nf")
        plt.title(f"{title_prefix} vs {col}  color is uncertainty")
        plt.grid(True, ls="--", alpha=0.6)
        plt.tight_layout()
        fname = f"map1d_{sanitize(col)}.png"
        plt.savefig(os.path.join(outdir, fname), dpi=200)
        plt.close()


def two_d_contours(model, scaler, base_point_scaled, cols, grid_points, title_prefix, outdir):
    # all unordered pairs
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            c1, c2 = cols[i], cols[j]
            x1 = np.linspace(0.0, 1.0, grid_points)
            x2 = np.linspace(0.0, 1.0, grid_points)
            X1, X2 = np.meshgrid(x1, x2)
            flat = np.tile(base_point_scaled, (X1.size, 1))
            flat[:, i] = X1.ravel()
            flat[:, j] = X2.ravel()

            mu, sd = predict_bnn(model, flat, n_samples=100)
            MU = mu.reshape(X1.shape)
            SD = sd.reshape(X1.shape)

            raw = scaler.inverse_transform(flat)
            R1 = raw[:, i].reshape(X1.shape)
            R2 = raw[:, j].reshape(X1.shape)

            # Uncertainty
            plt.figure(figsize=(8, 6))
            cp = plt.contourf(R1, R2, SD, levels=20, cmap="viridis")
            cbar = plt.colorbar(cp)
            cbar.set_label("Predictive std")
            plt.xlabel(c1)
            plt.ylabel(c2)
            plt.title(f"{title_prefix} uncertainty  {c1} vs {c2}")
            plt.tight_layout()
            plt.savefig(os.path.join(outdir, f"map2d_unc_{sanitize(c1)}__{sanitize(c2)}.png"), dpi=200)
            plt.close()

            # Mean
            plt.figure(figsize=(8, 6))
            cp2 = plt.contourf(R1, R2, MU, levels=20, cmap="plasma")
            cbar2 = plt.colorbar(cp2)
            cbar2.set_label("Predicted mean N/Nf")
            plt.xlabel(c1)
            plt.ylabel(c2)
            plt.title(f"{title_prefix} mean  {c1} vs {c2}")
            plt.tight_layout()
            plt.savefig(os.path.join(outdir, f"map2d_mean_{sanitize(c1)}__{sanitize(c2)}.png"), dpi=200)
            plt.close()


def sanitize(name: str):
    return name.lower().replace(" ", "_").replace("/", "_").replace("%", "pct").replace("(", "").replace(")", "")


# -----------------------------
# CLI
# -----------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Analyze saved M3 models with parity and uncertainty maps")

    # I/O
    p.add_argument("--csv", type=str, required=True, help="Path to M2 feature CSV")
    p.add_argument("--outdir", type=str, default="outputs", help="Directory for outputs")

    # Columns
    p.add_argument("--col_strain_amp", type=str, default="Strain Amplitude (%)")
    p.add_argument("--col_material", type=str, default="Material")
    p.add_argument("--col_heat_treat", type=str, default="Heat Treatment")
    p.add_argument("--col_target", type=str, default="Relative Cycle")

    # Feature sets
    p.add_argument("--use_enhanced", action="store_true",
                   help="Use enhanced feature set for plotting and model inputs")
    p.add_argument("--plot_cols", type=str, default="",
                   help="Comma separated list of columns to vary for maps. Defaults to all input columns")

    # Split
    p.add_argument("--target_strain", type=float, default=1.2,
                   help="Held out strain amplitude used for test split")
    p.add_argument("--test_query", type=str, default="",
                   help="Optional pandas query to narrow the test set")

    # Models
    p.add_argument("--baseline_model", type=str, default="",
                   help="Path to saved baseline model .pt file")
    p.add_argument("--enhanced_model", type=str, default="",
                   help="Path to saved enhanced model .pt file")

    # Inference
    p.add_argument("--mc_samples", type=int, default=100)
    p.add_argument("--grid_points", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)

    # Plots
    p.add_argument("--no_parity", action="store_true")
    p.add_argument("--no_maps", action="store_true")

    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    # Load data
    df = pd.read_csv(args.csv)

    # Define feature sets
    common = [args.col_strain_amp, args.col_material, args.col_heat_treat]
    baseline = ["Wp (kJ/m3)", "We (kJ/m3)", "Tensile Peak Stress (MPa)", "Compressive Peak Stress (MPa)"]
    enhanced = ["Inflection Strain (%)", "TYS Stress (MPa)", "Twinning Gradient (GPa)"]

    base_cols = common + baseline
    enh_cols = base_cols + enhanced

    # Choose active input columns for plotting
    input_cols = enh_cols if args.use_enhanced else base_cols
    ensure_columns(df, input_cols + [args.col_target])

    # Train/test split consistent with training
    train_df = df[df[args.col_strain_amp] != args.target_strain].copy()
    test_df = df[df[args.col_strain_amp] == args.target_strain].copy()
    if args.test_query:
        test_df = test_df.query(args.test_query)

    if len(test_df) == 0:
        raise ValueError("Empty test set after filtering. Adjust --target_strain or --test_query")

    # Scale on train only
    scaler = MinMaxScaler()
    X_train = train_df[input_cols].values
    X_test = test_df[input_cols].values
    y_train = train_df[[args.col_target]].values
    y_test = test_df[[args.col_target]].values
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # Base point for 1D and 2D maps
    base_point_scaled = X_train_s.mean(axis=0)

    # Output dirs
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    root = os.path.join(args.outdir, f"analysis_{'enh' if args.use_enhanced else 'base'}_{stamp}")
    os.makedirs(root, exist_ok=True)

    # Which models to load
    model_specs = []
    if args.baseline_model:
        model_specs.append(("baseline", args.baseline_model, base_cols))
    if args.enhanced_model:
        model_specs.append(("enhanced", args.enhanced_model, enh_cols))
    if not model_specs:
        raise ValueError("Provide at least one of --baseline_model or --enhanced_model")

    # For plotting columns
    if args.plot_cols:
        cols_for_maps = [c.strip() for c in args.plot_cols.split(",")]
        ensure_columns(pd.DataFrame(columns=input_cols), cols_for_maps)
    else:
        cols_for_maps = input_cols

    # Loop over provided models
    for label, model_path, cols_for_model in model_specs:
        # If plotting with a different feature set than the model was trained on, warn
        if set(cols_for_maps).issubset(set(cols_for_model)):
            plot_cols = cols_for_maps
        else:
            # Fallback to that model's own inputs
            plot_cols = cols_for_model

        # Prepare scaled arrays according to this model's columns
        # Refit scaler for this feature set
        scaler_m = MinMaxScaler()
        Xt_train = train_df[cols_for_model].values
        Xt_test = test_df[cols_for_model].values
        Xt_train_s = scaler_m.fit_transform(Xt_train)
        Xt_test_s = scaler_m.transform(Xt_test)
        base_point_s = Xt_train_s.mean(axis=0)

        # Load model
        model = TorchBNN(input_dim=Xt_train_s.shape[1])
        state = torch.load(model_path, map_location="cpu")
        model.load_state_dict(state)

        # Predict on test
        mu, sd = predict_bnn(model, Xt_test_s, n_samples=args.mc_samples)

        # Save predictions
        pred = test_df.copy()
        pred["y_true"] = y_test.flatten()
        pred["y_pred_mean"] = mu.flatten()
        pred["y_pred_std"] = sd.flatten()
        pred.to_csv(os.path.join(root, f"predictions_{label}.csv"), index=False)

        # Parity
        if not args.no_parity:
            parity_plot(
                y_true=y_test,
                y_mu=mu,
                y_sigma=sd,
                title=f"Parity plot {label}",
                outpath=os.path.join(root, f"parity_{label}.png"),
            )

        # Maps
        if not args.no_maps:
            subdir = os.path.join(root, f"maps_{label}")
            os.makedirs(subdir, exist_ok=True)
            one_d_maps(model, scaler_m, base_point_s, plot_cols, args.grid_points, f"{label}", subdir)
            two_d_contours(model, scaler_m, base_point_s, plot_cols, args.grid_points, f"{label}", subdir)

    print(f"Saved analysis to {root}")


if __name__ == "__main__":
    main()
