# -*- coding: utf-8 -*-
"""
Created on Thu Dec 18 15:20:41 2025

@author: USER
"""

# -*- coding: utf-8 -*-
"""
CIFLE.py: Integrated Loop Synthesis & Life Prediction Framework (Deployment Version)

This script integrates:
  - M1 (ANN): Hysteresis loop synthesis
  - M2 (Feature Extraction): Robust loop parameter extraction (imported)
  - M3 (BNN): Probabilistic life scoring (PP score)
  - Modified Morrow: Energy-based life estimation bounds

Usage Example:
  python CIFLE.py \
    --ann_model "models/m1_ann.h5" \
    --ann_stats "models/m1_mean_std.csv" \
    --bnn_model "models/m3_bnn.pt" \
    --bnn_train_data "data/calculation_results.csv" \
    --raw_data_dir "data/raw_loops" \
    --outdir "results"

Prerequisites:
  - M2.py must be in the same directory or Python path.
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import Rbf

# Deep Learning Imports
import tensorflow as tf
from tensorflow.keras.models import load_model
import torch
import torch.nn as nn
import torchbnn as bnn
from sklearn.preprocessing import MinMaxScaler

# Import M2 for robust feature extraction
try:
    import M2
except ImportError:
    print("Error: M2.py not found. Please place M2.py in the same directory.")
    sys.exit(1)


# ==========================================
# 1. Configuration & CLI Parsing
# ==========================================
def parse_args():
    p = argparse.ArgumentParser(description="CIFLE Framework: Synthesis & Prediction")
    
    # Model Paths
    p.add_argument("--ann_model", type=str, required=True, help="Path to M1 ANN .h5 model")
    p.add_argument("--ann_stats", type=str, required=True, help="Path to M1 mean_std.csv")
    p.add_argument("--bnn_model", type=str, required=True, help="Path to M3 BNN .pt model")
    
    # Data Paths
    p.add_argument("--bnn_train_data", type=str, required=True, 
                   help="CSV used for BNN training (to fit Scaler), usually calculation_results.csv")
    p.add_argument("--raw_data_dir", type=str, required=True, 
                   help="Directory containing subfolders of raw loop CSVs (for Morrow fitting)")
    p.add_argument("--outdir", type=str, default="CIFLE_outputs", help="Output directory")

    # Parameters
    p.add_argument("--amp_step", type=float, default=0.01, help="Strain amplitude step for dense grid")
    p.add_argument("--candidates", type=int, default=10, help="Number of Nf candidates per instance")
    p.add_argument("--n_mc", type=int, default=200, help="Number of Monte Carlo samples for BNN")
    
    return p.parse_args()


# ==========================================
# 2. M3 BNN Model Definition (Must match training)
# ==========================================
class TorchBNN(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.bnn = nn.Sequential(
            bnn.BayesLinear(prior_mu=0.0, prior_sigma=0.1, in_features=input_dim, out_features=256),
            nn.ReLU(),
            bnn.BayesLinear(prior_mu=0.0, prior_sigma=0.1, in_features=256, out_features=256),
            nn.ReLU(),
            bnn.BayesLinear(prior_mu=0.0, prior_sigma=0.1, in_features=256, out_features=128),
            nn.ReLU(),
            bnn.BayesLinear(prior_mu=0.0, prior_sigma=0.1, in_features=128, out_features=64),
            nn.ReLU(),
            bnn.BayesLinear(prior_mu=0.0, prior_sigma=0.1, in_features=64, out_features=1),
        )

    def forward(self, x):
        return self.bnn(x)


# ==========================================
# 3. M1 ANN Logic (Synthesis)
# ==========================================
def load_ann(model_path, stats_path):
    model = load_model(model_path)
    stats = pd.read_csv(stats_path, index_col=0).to_dict(orient="index")
    return model, stats

def synthesize_loop(ann_model, ann_stats, half_cycle, strain_amp, material, heat, half_length=25):
    """
    Synthesize a hysteresis loop using M1 ANN.
    """
    # Create synthetic input features based on M1 training structure
    cycle = np.full(half_length * 2 - 2, half_cycle)
    strain1 = np.linspace(-strain_amp, strain_amp, half_length)
    strain_mirr = strain1[::-1][1:-1] # Remove first and last to avoid dupes
    strain2 = np.concatenate((strain1, strain_mirr))
    
    # Lagged strain logic
    strain_last = strain2[-1]
    strain_lag = np.insert(strain2, 0, strain_last)[:-1]
    
    strain_amp_arr = np.full_like(strain2, strain_amp)
    
    # DataFrame construction
    input_df = pd.DataFrame({
        "Strain 1 %": strain2,
        "Cycle": cycle,
        "Strain Amplitude": strain_amp_arr,
        "Material": np.full_like(strain2, material),
        "Heat Treatment": np.full_like(strain2, heat),
        "Strain - 1": strain_lag
    })

    # Normalization
    norm_df = input_df.copy()
    features = ["Strain 1 %", "Cycle", "Strain Amplitude", "Material", "Heat Treatment", "Strain - 1"]
    
    for col in features:
        if col in ann_stats:
            mu = ann_stats[col]["mean"]
            sigma = ann_stats[col]["std"]
            norm_df[col] = (norm_df[col] - mu) / sigma
            
    # Prediction
    preds_norm = ann_model.predict(norm_df[features], verbose=0)
    
    # Inverse transform target
    target_col = "Stress MPa"
    mu_y = ann_stats[target_col]["mean"]
    sigma_y = ann_stats[target_col]["std"]
    stress = preds_norm * sigma_y + mu_y
    
    return pd.DataFrame({
        "Strain": input_df["Strain 1 %"].values,
        "Cycle": input_df["Cycle"].values,
        "Stress": stress.flatten()
    })


# ==========================================
# 4. M3 BNN Scoring Logic
# ==========================================
def load_bnn_resources(model_path, train_csv_path):
    # Load training data to fit scaler
    df = pd.read_csv(train_csv_path)
    
    # Feature columns defined in M3
    cols = [
        "Strain Amplitude (%)", "Material", "Heat Treatment",
        "Wp (kJ/m3)", "We (kJ/m3)", "Tensile Peak Stress (MPa)", "Compressive Peak Stress (MPa)",
        "Inflection Strain (%)", "TYS Stress (MPa)", "Twinning Gradient (GPa)"
    ]
    
    df = df.dropna(subset=cols)
    scaler = MinMaxScaler()
    scaler.fit(df[cols].values)
    
    # Load Model
    model = TorchBNN(input_dim=len(cols))
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    
    return model, scaler, cols

def calculate_pp_score(bnn_model, scaler, feature_row, feature_cols, n_mc=200):
    """
    Calculate Posterior Predictive (PP) score.
    Target is always Relative Cycle = 1.0 (End of Life assumption).
    """
    # Prepare input
    x_vals = np.array([[feature_row[c] for c in feature_cols]])
    x_scaled = scaler.transform(x_vals)
    x_tensor = torch.tensor(x_scaled, dtype=torch.float32)
    
    # MC Inference
    with torch.no_grad():
        preds = np.array([bnn_model(x_tensor).item() for _ in range(n_mc)])
    
    y_mean = preds.mean()
    y_std = preds.std()
    
    # PP Score for target y=1.0
    y_true = 1.0
    epsilon = 1e-6
    score = -((y_true - y_mean) ** 2) / (y_std ** 2 + epsilon)
    
    return score, y_mean, y_std


# ==========================================
# 5. Modified Morrow Logic
# ==========================================
def fit_morrow(energy_life_df):
    """
    Fit log10(Nf) vs log10(DeltaW [J/m3])
    """
    df = energy_life_df.dropna(subset=["Nf", "DeltaW (J/m3)"])
    df = df[(df["Nf"] > 0) & (df["DeltaW (J/m3)"] > 0)]
    
    x = np.log10(df["Nf"].values)
    y = np.log10(df["DeltaW (J/m3)"].values)
    
    slope, intercept = np.polyfit(x, y, 1)
    
    resid = y - (slope * x + intercept)
    s_resid = np.std(resid, ddof=2)
    
    return {
        "m": -slope,
        "C": 10**intercept,
        "slope": slope,
        "intercept": intercept,
        "resid_std": s_resid,
        "x_mean": x.mean(),
        "Sxx": np.sum((x - x.mean())**2),
        "n": len(x)
    }

def morrow_pred(val, m, C, mode="life_from_energy"):
    if mode == "life_from_energy":
        return np.power(C / val, 1.0/m)
    else: # energy_from_life
        return C / np.power(val, m)


# ==========================================
# 6. Data Processing Wrapper
# ==========================================
def process_raw_data(raw_root, subdirs=None):
    """
    Scan raw folders -> Extract features using M2 -> Return DF for Morrow fitting
    Note: Converts M2's kJ/m3 to J/m3 for Morrow compatibility.
    """
    if not subdirs:
        subdirs = [d for d in os.listdir(raw_root) if os.path.isdir(os.path.join(raw_root, d))]
    
    rows = []
    cfg = M2.M2Config(energy_units="kJ/m3") # M2 defaults
    
    for sub in subdirs:
        folder = os.path.join(raw_root, sub)
        # Infer metadata
        mat = 0 if "AZ91" in sub.upper() else 1
        ht = 300 if "300" in sub else 400
        
        for fname in os.listdir(folder):
            if not fname.lower().endswith(".csv"): continue
            
            # Parse Strain Amp from filename (e.g., "... 1.2% ...")
            try:
                # Simple parser, modify if filename format differs
                import re
                toks = re.split(r"[ _\-]+", os.path.splitext(fname)[0])
                amp = None
                for t in toks:
                    if "%" in t: 
                        amp = float(t.replace("%","").replace(",","."))
                        break
                if amp is None: continue
            except:
                continue

            try:
                # Read Loop
                df = pd.read_csv(os.path.join(folder, fname), header=0) # Adjust header if needed
                # Assume columns 3,4,5 are Strain, Cycle, Stress (standard project format)
                df = df.iloc[:, [3,4,5]]
                df.columns = ["Strain", "Cycle", "Stress"]
                
                max_cyc = df["Cycle"].max()
                half_life = int(max_cyc // 2)
                
                # Extract middle loop
                loop_df = df[df["Cycle"] == half_life]
                if loop_df.empty:
                    # Closest if exact match fail
                    c_near = df.iloc[(df["Cycle"] - half_life).abs().argsort()[:1]]["Cycle"].values[0]
                    loop_df = df[df["Cycle"] == c_near]
                
                # M2 Extraction
                feats = M2.extract_loop_features(loop_df[["Strain", "Stress"]], cfg=cfg)
                
                wp_j = feats["Wp (kJ/m3)"] * 1000.0
                we_j = feats["We (kJ/m3)"] * 1000.0
                
                if np.isnan(wp_j) or np.isnan(we_j): continue
                
                rows.append({
                    "Material": mat,
                    "Heat Treatment": ht,
                    "Strain Amplitude (%)": amp,
                    "Nf": max_cyc,
                    "DeltaW (J/m3)": wp_j + we_j
                })
            except Exception as e:
                # print(f"Skipping {fname}: {e}")
                pass
                
    return pd.DataFrame(rows)


# ==========================================
# 7. Main CIFLE Logic
# ==========================================
def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    
    # --- A. Load Models ---
    print("Loading models...")
    ann_model, ann_stats = load_ann(args.ann_model, args.ann_stats)
    bnn_model, bnn_scaler, bnn_cols = load_bnn_resources(args.bnn_model, args.bnn_train_data)
    
    # --- B. Process Raw Data & Fit Morrow ---
    print("Processing raw data for Morrow baseline...")
    raw_df = process_raw_data(args.raw_data_dir)
    if raw_df.empty:
        raise RuntimeError("No valid data found in raw_data_dir")
    
    morrow_params = fit_morrow(raw_df)
    print(f"Morrow Fit: m={morrow_params['m']:.4f}, C={morrow_params['C']:.2e}")
    
    # --- C. Dense Grid Search (CIFLE Protocol) ---
    print("Starting Dense CIFLE Loop Search...")
    results = []
    
    # Determine unique conditions
    conditions = raw_df[["Material", "Heat Treatment"]].drop_duplicates().values
    
    # Global Amplitude Grid
    amps = np.arange(raw_df["Strain Amplitude (%)"].min(), 
                     raw_df["Strain Amplitude (%)"].max() + 1e-9, 
                     args.amp_step)
    
    for mat, ht in conditions:
        # Filter raw data for this condition to get bounds
        cond_df = raw_df[(raw_df["Material"] == mat) & (raw_df["Heat Treatment"] == ht)]
        if cond_df.empty: continue
        
        # Determine Nmin/Nmax bounds for this material
        # (Simple approach: use min/max observed Nf for safety, or local interpolation)
        N_obs_min, N_obs_max = cond_df["Nf"].min(), cond_df["Nf"].max()
        
        for amp in amps:
            # 1. Estimate Energy Bounds using M1 (Synthesize loops at N_min and N_max)
            #    We use global observed N bounds to start, then refine with Morrow
            
            # Synthesize at boundaries
            try:
                loop_lo = synthesize_loop(ann_model, ann_stats, int(N_obs_min/2), amp, mat, ht)
                loop_hi = synthesize_loop(ann_model, ann_stats, int(N_obs_max/2), amp, mat, ht)
                
                feat_lo = M2.extract_loop_features(loop_lo)
                feat_hi = M2.extract_loop_features(loop_hi)
                
                dw_lo = (feat_lo["Wp (kJ/m3)"] + feat_lo["We (kJ/m3)"]) * 1000.0 # J/m3
                dw_hi = (feat_hi["Wp (kJ/m3)"] + feat_hi["We (kJ/m3)"]) * 1000.0
                
                dw_min = min(dw_lo, dw_hi)
                dw_max = max(dw_lo, dw_hi)
                
                # 2. Refine Life Window via Morrow
                n_start = morrow_pred(dw_max, morrow_params['m'], morrow_params['C']) # Max energy -> Min life
                n_end = morrow_pred(dw_min, morrow_params['m'], morrow_params['C'])   # Min energy -> Max life
                
                # Clamp to reasonable physics
                n_start = max(10, n_start)
                n_end = min(1e7, n_end)
                
                if n_end <= n_start: n_end = n_start + 100
                
            except Exception:
                continue

            # 3. Candidate Generation & Scoring
            candidates = np.linspace(n_start, n_end, args.candidates)
            best_cand = None
            best_pp = -np.inf
            
            for nf_cand in candidates:
                # Synthesize "End of Life" loop (Cycle ~ Nf)
                loop_eol = synthesize_loop(ann_model, ann_stats, int(nf_cand), amp, mat, ht)
                
                # Extract features (use M2)
                feats = M2.extract_loop_features(loop_eol)
                
                # Prepare row for BNN
                # M2 keys match what BNN expects, but add metadata
                row = feats.copy()
                row["Strain Amplitude (%)"] = amp
                row["Material"] = mat
                row["Heat Treatment"] = ht
                
                # Check for NaNs
                if any(np.isnan(row[k]) for k in bnn_cols if k in row):
                    continue
                    
                # Score
                pp, y_mean, y_std = calculate_pp_score(bnn_model, bnn_scaler, row, bnn_cols, n_mc=args.n_mc)
                
                if pp > best_pp:
                    best_pp = pp
                    best_cand = {
                        "Material": mat,
                        "Heat Treatment": ht,
                        "Strain Amplitude (%)": amp,
                        "Candidate Nf": nf_cand,
                        "DeltaW (J/m3)": (feats["Wp (kJ/m3)"] + feats["We (kJ/m3)"]) * 1000.0,
                        "PP Score": pp,
                        "Pred Rel Cycle Mean": y_mean,
                        "Pred Rel Cycle Std": y_std
                    }
            
            if best_cand:
                results.append(best_cand)

    res_df = pd.DataFrame(results)
    res_path = os.path.join(args.outdir, "CIFLE_results.csv")
    res_df.to_csv(res_path, index=False)
    print(f"Results saved to {res_path}")
    
    # --- D. Plotting ---
    if not res_df.empty:
        plot_results(raw_df, res_df, morrow_params, args.outdir)


# ==========================================
# 8. Plotting Routines
# ==========================================
def plot_results(raw_df, sim_df, fit_info, outdir):
    # Prepare bounds
    x_grid = np.linspace(2, 4, 100) # log scale
    y_hat = fit_info["slope"] * x_grid + fit_info["intercept"]
    
    # Confidence Interval Calculation
    z = 2.576 # 99%
    se = fit_info["resid_std"] * np.sqrt(1 + 1/fit_info["n"] + (x_grid - fit_info["x_mean"])**2 / fit_info["Sxx"])
    y_lo = y_hat - z * se
    y_hi = y_hat + z * se
    
    N_grid = 10**x_grid
    W_hat = 10**y_hat
    W_lo = 10**y_lo
    W_hi = 10**y_hi

    # --- Plot 1: Morrow Map ---
    plt.figure(figsize=(7, 6))
    plt.scatter(raw_df["Nf"], raw_df["DeltaW (J/m3)"], c='gray', alpha=0.5, label="Experimental")
    plt.plot(N_grid, W_hat, 'k-', label="Modified Morrow Fit")
    plt.fill_between(N_grid, W_lo, W_hi, color='gray', alpha=0.2, label="99% Band")
    
    sc = plt.scatter(sim_df["Candidate Nf"], sim_df["DeltaW (J/m3)"], 
                     c=sim_df["Strain Amplitude (%)"], cmap="coolwarm", marker='x', s=20, label="CIFLE Instances")
    plt.colorbar(sc, label="Strain Amplitude (%)")
    
    plt.xscale("log"); plt.yscale("log")
    plt.xlabel("Fatigue Life ($N_f$)"); plt.ylabel("$\Delta W_p + \Delta W_e^+$ (J/$m^3$)")
    plt.grid(True, which="both", ls=":")
    plt.legend()
    plt.savefig(os.path.join(outdir, "1_Morrow_Map.png"), dpi=300)
    plt.close()

    # --- Plot 2: PP Score Map ---
    # Interpolate PP scores
    try:
        x_vals = np.log10(sim_df["Candidate Nf"])
        y_vals = sim_df["Strain Amplitude (%)"]
        z_vals = sim_df["PP Score"]
        
        # Normalize PP for viz (quantile clip)
        q_lo, q_hi = np.quantile(z_vals, [0.05, 0.95])
        z_norm = np.clip((z_vals - q_lo)/(q_hi - q_lo), 0, 1)
        
        rbf = Rbf(x_vals, y_vals, z_norm, function="multiquadric", smooth=0.1)
        
        xi = np.linspace(2, 4, 200)
        yi = np.linspace(y_vals.min(), y_vals.max(), 200)
        XI, YI = np.meshgrid(xi, yi)
        ZI = rbf(XI, YI)
        ZI = np.clip(ZI, 0, 1) # Clip for plot
        
        plt.figure(figsize=(7, 6))
        pcm = plt.pcolormesh(10**XI, YI, ZI, shading='auto', cmap='viridis')
        plt.colorbar(pcm, label="Normalized PP Score")
        plt.scatter(sim_df["Candidate Nf"], sim_df["Strain Amplitude (%)"], c='k', s=1, alpha=0.3)
        
        plt.xscale("log")
        plt.xlabel("Fatigue Life ($N_f$)"); plt.ylabel("Strain Amplitude (%)")
        plt.title("Probabilistic Prediction Map")
        plt.savefig(os.path.join(outdir, "2_PP_Score_Map.png"), dpi=300)
        plt.close()
    except Exception as e:
        print(f"Skipping heatmap generation due to data sparsity: {e}")


if __name__ == "__main__":
    main()