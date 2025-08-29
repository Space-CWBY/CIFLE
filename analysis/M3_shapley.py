
"""
M3 SHAP analysis for a trained Bayesian NN

- Loads a saved BNN model and explains its MEAN prediction via SHAP
- Rebuilds the train/test split by a target strain amplitude
- Fits MinMaxScaler on the train split only
- Supports baseline or enhanced feature sets
- Segments test SHAP values into low/mid/high N/Nf by quantiles
- Saves heatmaps and CSV summaries to outputs/
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
import shap
import seaborn as sns
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
    miss = [c for c in cols if c not in df.columns]
    if miss:
        raise ValueError(f"Missing required columns: {miss}")


def to_tensor(x, dtype=torch.float32):
    return torch.tensor(x, dtype=dtype)


def sanitize(name: str):
    return (
        name.lower()
        .replace(" ", "_")
        .replace("/", "_")
        .replace("%", "pct")
        .replace("(", "")
        .replace(")", "")
    )


# -----------------------------
# BNN model same as training
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
def bnn_mean_predict(model: nn.Module, X: np.ndarray, mc_samples: int = 100) -> np.ndarray:
    """
    Deterministic wrapper for SHAP
    Returns the Monte Carlo mean prediction of the BNN
    """
    Xt = to_tensor(X)
    preds = []
    for _ in range(mc_samples):
        preds.append(model(Xt).numpy())  # [N, 1]
    mu = np.stack(preds, axis=0).mean(axis=0)  # [N, 1]
    return mu.ravel()  # shape [N]


# -----------------------------
# SHAP helpers
# -----------------------------
def shap_by_segments(shap_vals: np.ndarray, y_ref: np.ndarray, feature_names: list, k_segments: int = 3):
    """
    Split test samples into k quantile bins by y_ref and aggregate mean |SHAP|
    Returns dict of DataFrames keyed by segment name and a percent-normalized version
    """
    assert shap_vals.shape[0] == y_ref.shape[0]
    # Quantile edges
    qs = np.linspace(0.0, 1.0, k_segments + 1)
    edges = np.quantile(y_ref, qs)
    segments = []
    for i in range(k_segments):
        low, high = edges[i], edges[i + 1]
        if i == k_segments - 1:
            mask = (y_ref >= low) & (y_ref <= high)
        else:
            mask = (y_ref >= low) & (y_ref < high)
        segments.append((f"seg_{i+1}", mask))

    # Mean |SHAP| per segment
    seg_tables = {}
    seg_tables_pct = {}
    for name, mask in segments:
        sv = shap_vals[mask]  # [n_seg, p]
        mean_abs = np.abs(sv).mean(axis=0) if sv.size > 0 else np.zeros(shap_vals.shape[1])
        df_mean = pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs})
        df_mean = df_mean.set_index("feature")
        seg_tables[name] = df_mean

        total_abs = np.abs(sv).sum() if sv.size > 0 else 1.0
        pct = 100.0 * mean_abs / (total_abs / max(1, sv.shape[0]))  # normalize by avg per-sample abs sum
        df_pct = pd.DataFrame({"feature": feature_names, "shap_pct": pct}).set_index("feature")
        seg_tables_pct[name] = df_pct

    # Merge segments into wide format
    wide = pd.concat({k: v["mean_abs_shap"] for k, v in seg_tables.items()}, axis=1)
    wide_pct = pd.concat({k: v["shap_pct"] for k, v in seg_tables_pct.items()}, axis=1)
    return wide, wide_pct


def heatmap(df: pd.DataFrame, title: str, cbar_label: str, outpath: str, fmt: str = ".3f", cmap: str = "Reds"):
    plt.figure(figsize=(8, 5))
    sns.heatmap(df, annot=True, cmap=cmap, fmt=fmt, cbar_kws={"label": cbar_label})
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


# -----------------------------
# CLI
# -----------------------------
def parse_args():
    p = argparse.ArgumentParser(description="SHAP analysis for a trained M3 Bayesian NN")

    # I/O
    p.add_argument("--csv", type=str, required=True, help="Path to M2 features CSV")
    p.add_argument("--model", type=str, required=True, help="Path to saved BNN model .pt")
    p.add_argument("--outdir", type=str, default="outputs", help="Output directory")

    # Columns
    p.add_argument("--col_strain_amp", type=str, default="Strain Amplitude (%)")
    p.add_argument("--col_material", type=str, default="Material")
    p.add_argument("--col_heat_treat", type=str, default="Heat Treatment")
    p.add_argument("--col_target", type=str, default="Relative Cycle")

    # Feature set
    p.add_argument("--use_enhanced", action="store_true",
                   help="Use enhanced feature set including microstructural parameters")
    p.add_argument("--exclude_cols", type=str, default="Strain Amplitude (%),Material,Heat Treatment",
                   help="Comma separated features to exclude in filtered plots")

    # Split
    p.add_argument("--target_strain", type=float, default=1.2,
                   help="Held out strain amplitude to define test split")
    p.add_argument("--test_query", type=str, default="",
                   help="Optional pandas query to further filter the test set")

    # SHAP
    p.add_argument("--background_samples", type=int, default=256,
                   help="Number of train rows used as SHAP background")
    p.add_argument("--mc_samples", type=int, default=100, help="MC samples for BNN mean wrapper")
    p.add_argument("--seed", type=int, default=42)

    return p.parse_args()


# -----------------------------
# Main
# -----------------------------
def main():
    args = parse_args()
    set_seed(args.seed)

    # Load
    df = pd.read_csv(args.csv)

    # Feature sets
    common = [args.col_strain_amp, args.col_material, args.col_heat_treat]
    baseline = ["Wp (kJ/m3)", "We (kJ/m3)", "Tensile Peak Stress (MPa)", "Compressive Peak Stress (MPa)"]
    enhanced = ["Inflection Strain (%)", "TYS Stress (MPa)", "Twinning Gradient (GPa)"]

    input_cols = common + baseline + (enhanced if args.use_enhanced else [])
    ensure_columns(df, input_cols + [args.col_target])

    # Split
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
    y_test = test_df[[args.col_target]].values.ravel()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # Model
    model = TorchBNN(input_dim=X_train_s.shape[1])
    state = torch.load(args.model, map_location="cpu")
    model.load_state_dict(state)

    # SHAP background
    bg_n = min(args.background_samples, X_train_s.shape[0])
    bg_idx = np.random.choice(X_train_s.shape[0], size=bg_n, replace=False)
    X_bg = X_train_s[bg_idx]

    # Deterministic prediction wrapper
    def predict_fn(X):
        return bnn_mean_predict(model, X, mc_samples=args.mc_samples)

    # Explainer
    # Using permutation explainer via the unified API ensures stable runtime
    explainer = shap.Explainer(predict_fn, X_bg, algorithm="permutation", feature_names=input_cols)
    explanation = explainer(X_test_s)  # SHAP for test rows
    shap_vals = explanation.values  # [N_test, P]

    # Segment by quantiles of y_test
    wide, wide_pct = shap_by_segments(shap_vals, y_test, input_cols, k_segments=3)

    # Output dirs
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    root = os.path.join(args.outdir, f"shap_{'enh' if args.use_enhanced else 'base'}_{stamp}")
    os.makedirs(root, exist_ok=True)

    # Save raw SHAP values and summaries
    np.save(os.path.join(root, "shap_values.npy"), shap_vals)
    pd.DataFrame(X_test_s, columns=input_cols).to_csv(os.path.join(root, "X_test_scaled.csv"), index=False)
    pd.DataFrame({"y_test": y_test}).to_csv(os.path.join(root, "y_test.csv"), index=False)
    wide.to_csv(os.path.join(root, "shap_meanabs_by_segment.csv"))
    wide_pct.to_csv(os.path.join(root, "shap_pct_by_segment.csv"))

    # Heatmaps
    heatmap(wide, "Mean |SHAP| by N/Nf segment", "Mean |SHAP|", os.path.join(root, "heatmap_meanabs.png"))
    heatmap(wide_pct, "SHAP contribution percent by segment", "Contribution percent", os.path.join(root, "heatmap_percent.png"), fmt=".1f", cmap="YlGnBu")

    # Filtered features heatmaps
    exclude_cols = [c.strip() for c in args.exclude_cols.split(",")] if args.exclude_cols else []
    keep = [c for c in input_cols if c not in exclude_cols]
    keep_idx = [input_cols.index(c) for c in keep]
    wide_f = wide.iloc[keep_idx]
    wide_pct_f = wide_pct.iloc[keep_idx]
    wide_f.to_csv(os.path.join(root, "shap_meanabs_filtered.csv"))
    wide_pct_f.to_csv(os.path.join(root, "shap_pct_filtered.csv"))
    heatmap(wide_f, "Filtered mean |SHAP|  response features only", "Mean |SHAP|", os.path.join(root, "heatmap_meanabs_filtered.png"))
    heatmap(wide_pct_f, "Filtered SHAP contribution percent", "Contribution percent", os.path.join(root, "heatmap_percent_filtered.png"), fmt=".1f", cmap="YlGnBu")

    print(f"Saved SHAP analysis to {root}")


if __name__ == "__main__":
    main()
