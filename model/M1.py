
# -*- coding: utf-8 -*-
"""
M1 ANN training script (distribution / paper-aligned)

Goal
- Predict stress within a hysteresis loop using 6 inputs:
  1) Material (binary)
  2) Extrusion Temperature (binary or numeric descriptor)
  3) Strain Amplitude (%)
  4) Cycle
  5) Strain within loop (%)
  6) Lagged Strain (%)

Paper alignment
- Activation: LeakyReLU(alpha=0.01)
- Loss: Huber
- Architecture (final): 6 hidden layers with units:
  97, 102, 70, 47, 67, 70
- Learning rate: 10^(-3.74)

Notes
- This script is designed for anonymized distribution.
  It supports flexible folder naming rules and column naming.
- "Material" is kept as a binary categorical code without standardization by default,
  while other continuous inputs are standardized.
"""

import os
import re
import argparse
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# -----------------------------
# Utilities: metadata parsing
# -----------------------------
def extract_metadata_from_folder(folder_name: str):
    """
    Returns (material_code, extrusion_code)

    Supports anonymous tokens:
      - Material: MAT0, MAT1
      - Process temp: TEMP0, TEMP1, LT, HT
    Also tolerates legacy tokens if present.

    If not detected, returns (0, 0) as safe defaults.
    """
    name = folder_name.upper()

    # Material
    if "MAT1" in name:
        material = 1
    elif "MAT0" in name:
        material = 0
    else:
        # Backward-tolerant tokens (optional)
        if "SEN9" in name:
            material = 1
        elif "AZ91" in name:
            material = 0
        else:
            material = 0

    # Extrusion temperature / processing condition
    if "TEMP1" in name or "HT" in name:
        ext = 1
    elif "TEMP0" in name or "LT" in name:
        ext = 0
    else:
        # Try to capture a numeric token e.g., 573, 673
        m = re.search(r"(5\d{2}|6\d{2}|7\d{2})", name)
        if m:
            ext = float(m.group(1))
        else:
            ext = 0

    return material, ext


def parse_filename_tokens(fname_stem: str):
    """
    Extract strain amplitude in percent from filename tokens.
    Accepts tokens like "... 1.2% ..." or "... 1,2% ...".
    """
    toks = re.split(r"[ _\-]+", fname_stem)
    strain = None
    for t in toks:
        t_clean = t.replace(",", ".")
        if "%" in t_clean:
            try:
                strain = float(t_clean.replace("%", ""))
            except Exception:
                pass
    return strain


def pick_cycles(max_cycle: int, start: int, n_cycles: int):
    if max_cycle <= start + 3:
        return list(range(1, max_cycle + 1))
    n = min(n_cycles, max(1, max_cycle - start - 2))
    return list(np.linspace(start, max_cycle - 3, n, dtype=int))


def sample_within_cycle(df_cycle: pd.DataFrame, sampling_rate: float):
    """
    sampling_rate = fraction of points kept within a cycle.
    Default is 0.70 to approximate 'undersampled by 30%' conceptually.
    """
    n = max(1, int(len(df_cycle) * sampling_rate))
    idx = np.linspace(0, len(df_cycle) - 1, n, dtype=int)
    return df_cycle.iloc[idx]


def add_lagged_strain(df: pd.DataFrame, strain_col: str, cycle_col: str, new_col: str = "Strain - 1"):
    df[new_col] = df.groupby(cycle_col)[strain_col].shift(1)
    df[new_col] = df[new_col].fillna(df.groupby(cycle_col)[strain_col].transform("last"))
    return df


def save_feature_mean_std(path: str, scaler: StandardScaler, scale_cols: list[str]):
    d = {}
    for col, m, s in zip(scale_cols, scaler.mean_, scaler.scale_):
        d[col] = {"mean": float(m), "std": float(s)}
    pd.DataFrame.from_dict(d, orient="index").to_csv(path)


def build_paper_ann(input_dim: int, alpha: float = 0.01):
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
    return keras.Model(inputs, outputs, name="m1_ann_paper")


def make_feature_matrix(df: pd.DataFrame, feature_cols: list[str], scaler: StandardScaler, scale_cols: list[str]):
    tmp = df[feature_cols].copy()

    if scale_cols:
        scaled = scaler.transform(tmp[scale_cols].values)
        tmp.loc[:, scale_cols] = scaled

    return tmp.values.astype(np.float32)


# -----------------------------
# CLI
# -----------------------------
def parse_args():
    p = argparse.ArgumentParser(description="M1 preprocessing and ANN training (paper-aligned)")

    # I/O
    p.add_argument("--data_root", type=str, required=True,
                   help="Root folder containing subdirectories of raw CSV files")
    p.add_argument("--subdirs", type=str, default="",
                   help="Comma-separated subfolder names to include. Default uses all immediate subfolders")
    p.add_argument("--outdir", type=str, default="outputs",
                   help="Output root directory")

    # Raw CSV format
    p.add_argument("--cols", type=str, default="3,4,5",
                   help="Zero-based indices for columns: strain, cycle, stress. Example 3,4,5")
    p.add_argument("--header", type=str, default="0",
                   help="CSV header row index. Use 'None' for no header")
    p.add_argument("--sep", type=str, default=",")

    # Cycle sampling
    p.add_argument("--cycle_start", type=int, default=30)
    p.add_argument("--n_cycles", type=int, default=60)
    p.add_argument("--sample_rate", type=float, default=0.70,
                   help="Fraction of points kept within each sampled cycle")

    # Split
    p.add_argument("--val_size", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)

    # Training
    p.add_argument("--epochs", type=int, default=3000)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--lr", type=float, default=10 ** (-3.74))
    p.add_argument("--alpha", type=float, default=0.01)

    # Column names (for post-processing outputs)
    p.add_argument("--col_strain", type=str, default="Strain 1 %")
    p.add_argument("--col_cycle", type=str, default="Cycle")
    p.add_argument("--col_stress", type=str, default="Stress MPa")
    p.add_argument("--col_amp", type=str, default="Strain Amplitude (%)")
    p.add_argument("--col_material", type=str, default="Material")
    p.add_argument("--col_extrusion", type=str, default="Extrusion Temperature")
    p.add_argument("--col_lag", type=str, default="Strain - 1")

    return p.parse_args()


def main():
    args = parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    col_idx = [int(c.strip()) for c in args.cols.split(",")]

    header = None if str(args.header).lower() == "none" else int(args.header)

    # Determine subfolders
    if args.subdirs.strip():
        subfolders = [s.strip() for s in args.subdirs.split(",")]
        subfolders = [os.path.join(args.data_root, s) for s in subfolders]
    else:
        subfolders = [os.path.join(args.data_root, d) for d in os.listdir(args.data_root)
                      if os.path.isdir(os.path.join(args.data_root, d))]

    processed = []

    for sub in subfolders:
        folder_name = os.path.basename(sub)
        material_code, extrusion_code = extract_metadata_from_folder(folder_name)

        for fname in os.listdir(sub):
            if not fname.lower().endswith(".csv"):
                continue

            fpath = os.path.join(sub, fname)
            stem = os.path.splitext(fname)[0]
            strain_amp_percent = parse_filename_tokens(stem)

            try:
                raw = pd.read_csv(fpath, header=header, sep=args.sep)

                use = raw.iloc[:, col_idx].copy()
                use.columns = [args.col_strain, args.col_cycle, args.col_stress]

                # Cycle selection + within-cycle sampling
                max_cycle = int(np.nanmax(use[args.col_cycle].values))
                cycles = pick_cycles(max_cycle=max_cycle, start=args.cycle_start, n_cycles=args.n_cycles)
                use = use[use[args.col_cycle].isin(cycles)]
                use = use.groupby(args.col_cycle, group_keys=False).apply(
                    lambda g: sample_within_cycle(g, args.sample_rate)
                )

                # Strain amplitude
                if strain_amp_percent is None:
                    strain_amp_percent = float(np.abs(use[args.col_strain]).max())

                use[args.col_amp] = strain_amp_percent
                use[args.col_material] = material_code
                use[args.col_extrusion] = extrusion_code

                # Lagged strain
                use = add_lagged_strain(use, strain_col=args.col_strain, cycle_col=args.col_cycle, new_col=args.col_lag)

                processed.append(use)

            except Exception as e:
                print(f"Skip {fpath} due to error: {e}")

    if not processed:
        raise RuntimeError("No CSVs processed. Check --data_root, --subdirs, --cols.")

    final_df = pd.concat(processed, ignore_index=True)

    # Save raw processed log
    log_csv = os.path.join(args.outdir, "m1_processed_log.csv")
    final_df.to_csv(log_csv, index=False)

    # Feature set (paper order)
    FEATURES = [args.col_strain, args.col_cycle, args.col_amp,
                args.col_material, args.col_extrusion, args.col_lag]
    TARGET = args.col_stress

    # Train/val split
    train_df, val_df = train_test_split(final_df, test_size=args.val_size, random_state=args.seed)

    # Selective standardization:
    # keep material as binary code, standardize others
    NON_SCALED = [args.col_material]
    SCALE_COLS = [c for c in FEATURES if c not in NON_SCALED]

    x_scaler = StandardScaler()
    x_scaler.fit(train_df[SCALE_COLS].values)

    X_train = make_feature_matrix(train_df, FEATURES, x_scaler, SCALE_COLS)
    X_val = make_feature_matrix(val_df, FEATURES, x_scaler, SCALE_COLS)
    y_train = train_df[TARGET].values.astype(np.float32)
    y_val = val_df[TARGET].values.astype(np.float32)

    # Save scaler stats
    mean_std_path = os.path.join(args.outdir, "m1_feature_mean_std.csv")
    save_feature_mean_std(mean_std_path, x_scaler, SCALE_COLS)

    # Build model
    model = build_paper_ann(input_dim=len(FEATURES), alpha=args.alpha)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=args.lr),
        loss=tf.keras.losses.Huber(),
        metrics=[keras.metrics.MeanSquaredError(name="mse"),
                 keras.metrics.MeanAbsoluteError(name="mae")]
    )

    callbacks = [
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=30, min_lr=1e-6, verbose=1
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=120, restore_best_weights=True, verbose=1
        ),
    ]

    print("=== Training M1 ANN ===")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch,
        callbacks=callbacks,
        verbose=2
    )

    # Loss curve
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.figure(figsize=(6, 4))
    plt.plot(history.history["loss"], label="train")
    plt.plot(history.history["val_loss"], label="validation")
    plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (Huber)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    loss_path = os.path.join(args.outdir, f"m1_loss_{stamp}.png")
    plt.savefig(loss_path, dpi=300)
    plt.close()

    # Save model
    model_path = os.path.join(args.outdir, f"m1_model_{stamp}.h5")
    model.save(model_path)

    print(f"Processed log saved to: {log_csv}")
    print(f"Feature mean/std saved to: {mean_std_path}")
    print(f"Loss curve saved to: {loss_path}")
    print(f"Model saved to: {model_path}")


if __name__ == "__main__":
    # GPU safe setup
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except Exception:
            pass

    main()

