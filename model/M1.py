
"""
M1 preprocessing + ANN training (distribution-ready)

- Recursively reads raw loop CSVs from subdirectories of --data_root
- Selects evenly spaced cycles and samples points within each cycle
- Builds lagged strain feature per cycle
- Splits train/validation, scales FEATURES and TARGET separately
- Trains a Keras ANN to predict stress from features
- Saves model, scalers, logs, and demo predictions to outputs/
"""

import argparse
import json
import os
import re
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt


# -----------------------------
# Utilities
# -----------------------------
def set_seed(seed: int = 42):
    np.random.seed(seed)
    tf.random.set_seed(seed)


def mkdir(path: str):
    os.makedirs(path, exist_ok=True)


def find_subdirs(root: str, include: list[str] | None) -> list[str]:
    if include:
        return [os.path.join(root, sd) for sd in include]
    # default: all immediate subdirectories
    return [os.path.join(root, d) for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]


def infer_material_and_heat(subdir_name: str) -> tuple[int, int]:
    # material: 0 for AZ91, 1 for others by default
    material = 0 if re.search(r"AZ91", subdir_name, re.IGNORECASE) else 1
    # heat treatment: 300 if contains 300 else 400 (fallback)
    heat = 300 if re.search(r"300", subdir_name) else 400
    return material, heat


def parse_filename_tokens(fname_stem: str) -> tuple[float | None, float | None]:
    """
    Extracts strain amplitude in percent and frequency in hz from filename tokens.
    Accepts formats like '4-1 1,2% 0.2hz.csv' or '... 1.2% 0.5Hz ...'
    Returns (strain_amp_percent, frequency_hz) possibly None if not found.
    """
    toks = re.split(r"[ _\-]+", fname_stem)
    strain = None
    freq = None
    for t in toks:
        t_clean = t.replace(",", ".")
        if "%" in t_clean:
            try:
                strain = float(t_clean.replace("%", ""))
            except Exception:
                pass
        if re.search(r"hz$", t_clean, re.IGNORECASE):
            try:
                freq = float(re.sub(r"hz$", "", t_clean, flags=re.IGNORECASE))
            except Exception:
                pass
    return strain, freq


def pick_cycles(max_cycle: int, start: int, n_cycles: int) -> list[int]:
    if max_cycle <= start + 3:
        return list(range(1, max_cycle + 1))
    n = min(n_cycles, max(1, max_cycle - start - 2))
    return list(np.linspace(start, max_cycle - 3, n, dtype=int))


def sample_within_cycle(df_cycle: pd.DataFrame, sampling_rate: float) -> pd.DataFrame:
    n = max(1, int(len(df_cycle) * sampling_rate))
    idx = np.linspace(0, len(df_cycle) - 1, n, dtype=int)
    return df_cycle.iloc[idx]


def add_lagged_strain(df: pd.DataFrame, strain_col: str, cycle_col: str, new_col: str = "Strain - 1") -> pd.DataFrame:
    df[new_col] = df.groupby(cycle_col)[strain_col].shift(1)
    # fill first lag by last value within the same cycle
    df[new_col] = df[new_col].fillna(df.groupby(cycle_col)[strain_col].transform("last"))
    return df


def save_mean_std_csv(path: str, feature_scaler: StandardScaler, target_scaler: StandardScaler, feature_cols: list[str], target_col: str):
    # build a unified table indexed by column name
    d = {}
    for col, m, s in zip(feature_cols, feature_scaler.mean_, feature_scaler.scale_):
        d[col] = {"mean": float(m), "std": float(s)}
    d[target_col] = {"mean": float(target_scaler.mean_[0]), "std": float(target_scaler.scale_[0])}
    df = pd.DataFrame.from_dict(d, orient="index")
    df.to_csv(path)


def fake_dataset(half_cycle: int, strain_amp_percent: float, material: int, heat: int, half_length: int) -> pd.DataFrame:
    """
    Builds a synthetic loop at a specific half-life 'cycle', with symmetric strain path.
    Returns columns: Cycle, Strain 1 %, Strain - 1, Strain Amplitude, Material, Heat Treatment
    """
    cycle = np.full(half_length * 2 - 2, half_cycle)
    amp = strain_amp_percent
    s1 = np.linspace(-amp, amp, half_length)  # percent units
    s_m = s1[::-1][1:-1]  # remove duplicated endpoints
    s = np.concatenate([s1, s_m])
    s_lag = np.concatenate([[s[-1]], s[:-1]])

    df = pd.DataFrame(
        {
            "Cycle": cycle,
            "Strain 1 %": s,
            "Strain - 1": s_lag,
            "Strain Amplitude": np.full_like(s, amp),
            "Material": np.full_like(s, material),
            "Heat Treatment": np.full_like(s, heat),
        }
    )
    return df


# -----------------------------
# Model builders
# -----------------------------
def build_paper_ann(input_dim: int, alpha: float = 0.01) -> keras.Model:
    # 6 hidden layers as described in Methods with LeakyReLU
    inputs = keras.Input(shape=(input_dim,))
    x = layers.Dense(97)(inputs)
    x = layers.LeakyReLU(alpha=alpha)(x)
    x = layers.Dense(102)(x)
    x = layers.LeakyReLU(alpha=alpha)(x)
    x = layers.Dense(70)(x)
    x = layers.LeakyReLU(alpha=alpha)(x)
    x = layers.Dense(47)(x)
    x = layers.LeakyReLU(alpha=alpha)(x)
    x = layers.Dense(67)(x)
    x = layers.LeakyReLU(alpha=alpha)(x)
    x = layers.Dense(70)(x)
    x = layers.LeakyReLU(alpha=alpha)(x)
    outputs = layers.Dense(1, activation="linear")(x)
    model = keras.Model(inputs, outputs, name="m1_ann_paper")
    return model


# -----------------------------
# CLI
# -----------------------------
def parse_args():
    p = argparse.ArgumentParser(description="M1 preprocessing and ANN training")
    # I/O
    p.add_argument("--data_root", type=str, required=True, help="Root folder containing subdirectories of CSV files")
    p.add_argument("--subdirs", type=str, default="", help="Comma separated subfolder names to include. Default uses all immediate subfolders")
    p.add_argument("--outdir", type=str, default="outputs", help="Output root directory")

    # Raw CSV format
    p.add_argument("--cols", type=str, default="3,4,5", help="Zero based indices for columns strain, cycle, stress. Example 3,4,5")
    p.add_argument("--header", type=int, default=0, help="CSV header row index. Use None for no header")
    p.add_argument("--sep", type=str, default=",", help="CSV delimiter")

    # Cycle sampling
    p.add_argument("--cycle_start", type=int, default=30, help="First cycle considered for sampling")
    p.add_argument("--n_cycles", type=int, default=60, help="Number of cycles to sample evenly across the test")
    p.add_argument("--sample_rate", type=float, default=0.30, help="Fraction of points sampled within each cycle")

    # Split and scaling
    p.add_argument("--val_size", type=float, default=0.2, help="Validation fraction")
    p.add_argument("--seed", type=int, default=42, help="Random seed")

    # Training
    p.add_argument("--epochs", type=int, default=20000, help="Training epochs")
    p.add_argument("--batch", type=int, default=64, help="Batch size")
    p.add_argument("--lr", type=float, default=10 ** (-3.74), help="Learning rate. Default approx 1.82e-4")
    p.add_argument("--arch", type=str, default="paper", choices=["paper", "deep"], help="ANN architecture choice")
    p.add_argument("--huber_delta", type=float, default=1.0, help="Huber loss delta")

    # Demo prediction
    p.add_argument("--demo", action="store_true", help="Run a demo prediction on a synthetic loop")
    p.add_argument("--demo_half_cycle", type=int, default=200)
    p.add_argument("--demo_strain_amp", type=float, default=1.2)
    p.add_argument("--demo_material", type=int, default=1)
    p.add_argument("--demo_heat", type=int, default=400)
    p.add_argument("--demo_half_length", type=int, default=15)

    return p.parse_args()


# -----------------------------
# Main
# -----------------------------
def main():
    args = parse_args()
    set_seed(args.seed)

    subdirs = [s.strip() for s in args.subdirs.split(",") if s.strip()] if args.subdirs else None
    col_idx = [int(x) for x in args.cols.split(",")]
    if len(col_idx) != 3:
        raise ValueError("Provide exactly three column indices for strain, cycle, stress")

    # Prepare output folders
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = os.path.join(args.outdir, f"m1_{stamp}")
    mkdir(out_root)

    # ------------------ Preprocessing ------------------
    processed = []
    for folder in find_subdirs(args.data_root, subdirs):
        subname = os.path.basename(folder)
        if not os.path.isdir(folder):
            continue
        material, heat = infer_material_and_heat(subname)

        for fname in os.listdir(folder):
            if not fname.lower().endswith(".csv"):
                continue
            fpath = os.path.join(folder, fname)
            try:
                # filename metadata
                stem = os.path.splitext(fname)[0]
                strain_amp_percent, freq_hz = parse_filename_tokens(stem)

                # load raw
                header = None if str(args.header).lower() == "none" else args.header
                df = pd.read_csv(fpath, header=header, sep=args.sep)

                # select columns by index and rename
                use = df.iloc[:, col_idx].copy()
                use.columns = ["Strain 1 %", "Cycle", "Stress MPa"]

                # choose cycles then subsample within cycle
                max_cycle = int(np.nanmax(use["Cycle"].values))
                cycles = pick_cycles(max_cycle=max_cycle, start=args.cycle_start, n_cycles=args.n_cycles)
                use = use[use["Cycle"].isin(cycles)]
                use = use.groupby("Cycle", group_keys=False).apply(lambda g: sample_within_cycle(g, args.sample_rate))

                # add metadata columns
                if strain_amp_percent is None:
                    # if not found in filename, compute from data or leave NaN
                    strain_amp_percent = float(np.abs(use["Strain 1 %"]).max())
                use["Strain Amplitude"] = strain_amp_percent
                use["Material"] = material
                use["Heat Treatment"] = heat

                # add lagged strain per cycle
                use = add_lagged_strain(use, strain_col="Strain 1 %", cycle_col="Cycle", new_col="Strain - 1")

                processed.append(use)

            except Exception as e:
                print(f"Skip {fpath} due to error: {e}")

    if not processed:
        raise RuntimeError("No CSVs processed. Check --data_root, --subdirs, and --cols.")

    final_df = pd.concat(processed, ignore_index=True)

    # Save raw processed log
    log_csv = os.path.join(out_root, "log.csv")
    final_df.to_csv(log_csv, index=False)

    # ------------------ Split and scaling ------------------
    features = ["Strain 1 %", "Cycle", "Strain Amplitude", "Material", "Heat Treatment", "Strain - 1"]
    target = "Stress MPa"

    train_df, val_df = train_test_split(final_df, test_size=args.val_size, random_state=args.seed)

    X_train = train_df[features].values
    X_val = val_df[features].values
    y_train = train_df[[target]].values
    y_val = val_df[[target]].values

    # scale features and target separately
    x_scaler = StandardScaler()
    y_scaler = StandardScaler()

    X_train_s = x_scaler.fit_transform(X_train)
    X_val_s = x_scaler.transform(X_val)
    y_train_s = y_scaler.fit_transform(y_train)
    y_val_s = y_scaler.transform(y_val)

    # Save scaled CSVs for reproducibility
    train_scaled = pd.DataFrame(np.hstack([X_train_s, y_train_s]), columns=features + [target])
    val_scaled = pd.DataFrame(np.hstack([X_val_s, y_val_s]), columns=features + [target])

    train_scaled.to_csv(os.path.join(out_root, "train_data.csv"), index=False)
    val_scaled.to_csv(os.path.join(out_root, "validation_data.csv"), index=False)
    save_mean_std_csv(os.path.join(out_root, "mean_std.csv"), x_scaler, y_scaler, features, target)

    # ------------------ Model and training ------------------
    input_dim = len(features)
    if args.arch == "paper":
        model = build_paper_ann(input_dim)
    else:
        model = build_deep_ann(input_dim)

    opt = keras.optimizers.Adam(learning_rate=args.lr)
    loss = keras.losses.Huber(delta=args.huber_delta)
    model.compile(optimizer=opt, loss=loss)

    hist = model.fit(
        X_train_s, y_train_s,
        validation_data=(X_val_s, y_val_s),
        epochs=args.epochs,
        batch_size=args.batch,
        verbose=2
    )

    # Save model
    model_path = os.path.join(out_root, "ann_model.keras")
    model.save(model_path)

    # ------------------ Plots ------------------
    # Learning curves
    y_vloss = hist.history.get("val_loss", [])
    y_loss = hist.history.get("loss", [])
    x_len = np.arange(len(y_loss))

    plt.figure(figsize=(7, 6))
    plt.plot(x_len, y_vloss, marker=".", label="Validation loss")
    plt.plot(x_len, y_loss, marker=".", label="Train loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True, ls="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_root, "learning_curves.png"), dpi=200)
    plt.close()

    # ------------------ Demo prediction (optional) ------------------
    if args.demo:
        demo = fake_dataset(
            half_cycle=args.demo_half_cycle,
            strain_amp_percent=args.demo_strain_amp,
            material=args.demo_material,
            heat=args.demo_heat,
            half_length=args.demo_half_length,
        )

        # scale using stored scalers
        X_demo = demo[features].values
        X_demo_s = x_scaler.transform(X_demo)
        y_pred_s = model.predict(X_demo_s, verbose=0)
        # inverse scale
        y_pred = y_scaler.inverse_transform(y_pred_s)

        demo_out = demo.copy()
        demo_out[target] = y_pred.ravel()
        demo_csv = os.path.join(out_root, "demo_predictions.csv")
        demo_out.to_csv(demo_csv, index=False)

        # plot stress vs strain
        plt.figure(figsize=(7, 6))
        plt.plot(demo_out["Strain 1 %"], demo_out[target], label="Predicted stress MPa")
        plt.xlabel("Strain 1 %")
        plt.ylabel("Stress MPa")
        plt.grid(True, ls="--", alpha=0.6)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_root, "demo_loop.png"), dpi=200)
        plt.close()

    # ------------------ Save config ------------------
    cfg = vars(args).copy()
    cfg["outdir"] = out_root
    with open(os.path.join(out_root, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    print(f"\nSaved outputs to {out_root}")
    print(f"- Preprocessed log: {log_csv}")
    print(f"- Train and val CSVs: train_data.csv, validation_data.csv")
    print(f"- Mean and std: mean_std.csv")
    print(f"- Model: ann_model.keras")
    if args.demo:
        print(f"- Demo predictions: demo_predictions.csv")


if __name__ == "__main__":
    main()
