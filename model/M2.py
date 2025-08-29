
"""
M2: Automated extraction of fatigue parameters

Data expectations
- Raw loop CSV per test, with columns for strain, cycle, stress
- Strain may be in percent or fraction. Specify via --strain_units
- Stress in MPa

Optional
- You may provide a trained M1 ANN model to synthesize additional cycles beyond max measured cycle

Outputs
- calculation_results.csv
- calculation_results_with_scaling.csv  (per-file min-max normalization of selected parameters)
- Optional PNG plots with annotated features

"""

import argparse
import os
from datetime import datetime
import re
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from scipy.optimize import curve_fit
from scipy.integrate import simpson
import matplotlib.pyplot as plt

# Optional Keras for M1-based synthesis
try:
    from tensorflow import keras
except Exception:
    keras = None


# -----------------------------
# Utilities
# -----------------------------
def set_seed(seed: int = 42):
    np.random.seed(seed)


def mkdir(path: str):
    os.makedirs(path, exist_ok=True)


def odd_window(n: int, frac: float = 0.15, minimum: int = 11) -> int:
    """Pick a Savitzky–Golay window length that is odd and < n."""
    w = max(minimum, int(frac * n) | 1)  # make odd
    if w >= n:
        w = (n - 1) if (n % 2 == 0) else (n - 2)
    if w < 5:
        w = 5
    if w % 2 == 0:
        w -= 1
    return max(5, w)


def parse_filename_tokens(fname_stem: str):
    """Extract strain amplitude in percent and frequency in Hz from filename tokens if present."""
    toks = re.split(r"[ _\-]+", fname_stem)
    strain = None
    freq = None
    for t in toks:
        t2 = t.replace(",", ".")
        if "%" in t2:
            try:
                strain = float(t2.replace("%", ""))
            except Exception:
                pass
        if re.search(r"hz$", t2, re.IGNORECASE):
            try:
                freq = float(re.sub(r"hz$", "", t2, flags=re.IGNORECASE))
            except Exception:
                pass
    return strain, freq


def to_fractional_strain(strain_arr: np.ndarray, units: str) -> np.ndarray:
    if units.lower() in ["percent", "pct", "%"]:
        return strain_arr / 100.0
    return strain_arr.astype(float)


def energy_units_j_per_m3_from_mpa(mpa: np.ndarray, strain: np.ndarray) -> float:
    """
    Integrate σ dε with σ in MPa and ε dimensionless to get MPa * strain.
    1 MPa = 1e6 Pa so multiply by 1e6 to convert to J/m³.
    """
    return simpson(mpa, strain) * 1e6


def polygon_loop_area(strain: np.ndarray, stress_mpa: np.ndarray) -> float:
    """
    Approximate loop area by integrating around the closed path once.
    Assumes points correspond to one contiguous cycle in acquisition order.
    """
    s = np.asarray(strain)
    sig = np.asarray(stress_mpa)
    # Ensure closed path
    s_closed = np.concatenate([s, s[:1]])
    sig_closed = np.concatenate([sig, sig[:1]])
    # Polygon area in σ–ε plane equals ∮ σ dε
    return np.trapz(sig_closed, s_closed)


# -----------------------------
# Loop segmentation
# -----------------------------
def identify_upper_lower_loops(df_cycle: pd.DataFrame):
    """
    Split one cycle into UPPER (tension) and LOWER (compression) branches.
    Method: path from min(σ) to max(σ) defines upper branch when sorted by strain,
            complement defines lower branch. This is robust to mean stress.
    """
    idx_min = df_cycle["Stress"].idxmin()
    idx_max = df_cycle["Stress"].idxmax()

    if idx_min < idx_max:
        upper = df_cycle.loc[idx_min:idx_max]
    else:
        upper = pd.concat([df_cycle.loc[idx_min:], df_cycle.loc[:idx_max]])

    upper = upper.sort_values("Strain").reset_index(drop=True)
    lower = df_cycle.drop(upper.index).sort_values("Strain").reset_index(drop=True)
    return upper, lower


# -----------------------------
# Paper-consistent features
# -----------------------------
def inflection_point_upper(upper: pd.DataFrame):
    """
    εIP detection per paper:
      on UPPER branch, find first index where second derivative changes from negative to positive
    """
    if len(upper) < 9:
        return np.nan, np.nan

    s = upper["Strain"].values
    sig = upper["Stress"].values
    win = odd_window(len(upper))
    # Smooth σ(ε)
    sig_s = savgol_filter(sig, window_length=win, polyorder=3, mode="interp")
    # First and second derivatives vs strain
    # Use Savitzky–Golay for derivatives
    d1 = savgol_filter(sig, window_length=win, polyorder=3, deriv=1, delta=np.median(np.diff(s)), mode="interp")
    d2 = savgol_filter(sig, window_length=win, polyorder=3, deriv=2, delta=np.median(np.diff(s)), mode="interp")

    # Find first negative->positive crossing in d2
    for i in range(1, len(d2)):
        if np.isfinite(d2[i - 1]) and np.isfinite(d2[i]) and d2[i - 1] < 0 <= d2[i]:
            return s[i], sig_s[i]

    # Fallback: strongest curvature minimum
    i_min = int(np.nanargmin(d2))
    return s[i_min], sig_s[i_min]


def tangents_and_tys_lower(lower: pd.DataFrame):
    """
    σyT detection per paper:
      Fit two lines to the first and second halves of LOWER branch in strain order,
      take their intersection for twinning yield.
    """
    if len(lower) < 10:
        return np.nan, np.nan, (np.nan, np.nan), (np.nan, np.nan)

    lo = lower.sort_values("Strain").reset_index(drop=True)
    mid = len(lo) // 2
    first = lo.iloc[:mid]
    second = lo.iloc[mid:]

    # Trim 30–70% region in each half to avoid end-effects
    def middle_trim(df):
        n = len(df)
        a = int(0.30 * n)
        b = int(0.70 * n)
        if b <= a:
            a = max(0, n // 4)
            b = max(a + 2, 3 * n // 4)
        return df.iloc[a:b]

    f1 = middle_trim(first)
    f2 = middle_trim(second)

    def linfit(x, y):
        p, _ = curve_fit(lambda x, a, b: a * x + b, x, y, maxfev=2000)
        return p  # a, b

    try:
        a1, b1 = linfit(f1["Strain"].values, f1["Stress"].values)
        a2, b2 = linfit(f2["Strain"].values, f2["Stress"].values)
    except Exception:
        return np.nan, np.nan, (np.nan, np.nan), (np.nan, np.nan)

    if abs(a1 - a2) < 1e-12:
        return np.nan, np.nan, (a1, b1), (a2, b2)

    x_int = (b2 - b1) / (a1 - a2)
    y_int = a1 * x_int + b1
    return x_int, y_int, (a1, b1), (a2, b2)


def twinning_gradient_postyield(lower: pd.DataFrame, tys_strain: float, strain_fractional: bool) -> float:
    """
    σ̇_T: slope on LOWER branch beyond σyT.
    Fit a line on the post-yield window [tys, tys + Δ], where Δ spans middle 30–70% of the remainder.
    Returns GPa.
    """
    if np.isnan(tys_strain):
        return np.nan

    lo = lower.sort_values("Strain").reset_index(drop=True)
    post = lo[lo["Strain"] >= tys_strain].copy()
    if len(post) < 6:
        return np.nan

    n = len(post)
    a = int(0.30 * n)
    b = int(0.70 * n)
    seg = post.iloc[a:b] if b > a else post

    x = seg["Strain"].values
    y = seg["Stress"].values

    # slope in MPa per unit strain
    try:
        a_fit, b_fit = np.polyfit(x, y, 1)
    except Exception:
        return np.nan

    # Units:
    #  if strain is fractional, slope is MPa => convert to GPa by /1000
    #  if strain is in percent, caller converted to fraction beforehand
    return a_fit / 1000.0  # GPa


def zero_stress_crossing_on_upper(upper: pd.DataFrame):
    """Find tensile-side zero-stress crossing by linear interpolation."""
    sig = upper["Stress"].values
    eps = upper["Strain"].values
    for i in range(len(sig) - 1):
        if sig[i] <= 0 < sig[i + 1]:
            # linear interpolate between i and i+1
            x1, x2 = eps[i], eps[i + 1]
            y1, y2 = sig[i], sig[i + 1]
            if y2 != y1:
                return x1 - y1 * (x2 - x1) / (y2 - y1)
    # fallback: smallest |stress| point
    return eps[np.argmin(np.abs(sig))]


def we_plus_tangent(upper: pd.DataFrame, eps_ip: float, sig_ip: float, eps_mt: float, mode: str = "tangent_at_inflection") -> float:
    """
    ΔWe+ per paper:
      area under the TANGENT line to the tension branch.
      We take the tangent at εIP by default, integrate from tensile zero-crossing to ε at σMT.
      Units returned in kJ/m^3.
    """
    if np.isnan(eps_ip) or np.isnan(sig_ip) or np.isnan(eps_mt):
        return np.nan

    # Local tangent slope around εIP
    s = upper["Strain"].values
    sig = upper["Stress"].values
    win = odd_window(len(upper))
    d1 = savgol_filter(sig, window_length=win, polyorder=3, deriv=1, delta=np.median(np.diff(s)), mode="interp")

    # pick index closest to εIP
    i_ip = int(np.argmin(np.abs(s - eps_ip)))
    slope_mpa = d1[i_ip]  # MPa per unit strain

    # Intercept through the point (εIP, σIP): σ = a*(ε - εIP) + σIP
    # Integrate between ε0 (tensile zero) and εMT
    eps0 = zero_stress_crossing_on_upper(upper)
    a = slope_mpa
    b = sig_ip - a * eps_ip  # σ = a ε + b

    # Integral of σ dε = ∫(a ε + b) dε = 0.5 a (ε²) + b ε
    eps1 = float(eps0)
    eps2 = float(eps_mt)
    j_per_m3 = (0.5 * a * (eps2**2 - eps1**2) + b * (eps2 - eps1)) * 1e6  # MPa->Pa
    return j_per_m3 / 1000.0  # kJ/m^3


# -----------------------------
# Optional M1 synthesis
# -----------------------------
def fake_loop_dataset(half_cycle: int, strain_amp_pct: float, material: int, heat: int, half_len: int):
    cycle = np.full(half_len * 2 - 2, half_cycle)
    s1 = np.linspace(-strain_amp_pct, strain_amp_pct, half_len)
    s_m = s1[::-1][1:-1]
    s = np.concatenate([s1, s_m])
    s_lag = np.concatenate([[s[-1]], s[:-1]])
    return pd.DataFrame(
        {
            "Cycle": cycle,
            "Strain": s,
            "Strain - 1": s_lag,
            "Strain Amplitude": np.full_like(s, strain_amp_pct),
            "Material": np.full_like(s, material),
            "Heat Treatment": np.full_like(s, heat),
        }
    )


def maybe_load_m1(model_path: str):
    if not model_path:
        return None
    if keras is None:
        raise RuntimeError("TensorFlow Keras is required to load the M1 model")
    return keras.models.load_model(model_path)


def normalize_with_table(df: pd.DataFrame, mean_std_csv: str) -> pd.DataFrame:
    ms = pd.read_csv(mean_std_csv, index_col=0)
    d = ms.to_dict(orient="index")
    out = df.copy()
    for col in out.columns:
        if col in d:
            out[col] = (out[col] - d[col]["mean"]) / d[col]["std"]
    return out


def inverse_from_table(values: np.ndarray, mean_std_csv: str, target_col: str) -> np.ndarray:
    ms = pd.read_csv(mean_std_csv, index_col=0)
    m = ms.loc[target_col, "mean"]
    s = ms.loc[target_col, "std"]
    return values * s + m


# -----------------------------
# Plotting
# -----------------------------
def plot_results(df_cycle, upper, lower, inflect, tys, lin1, lin2, cycle, save_path):
    plt.figure(figsize=(8, 5.5))
    plt.plot(df_cycle["Strain"], df_cycle["Stress"], color="gray", linewidth=1.2, label="Loop")
    if len(upper):
        plt.scatter(upper["Strain"], upper["Stress"], s=10, color="#1f77b4", label="Upper")
    if len(lower):
        plt.scatter(lower["Strain"], lower["Stress"], s=10, color="#2ca02c", label="Lower")
    if not (np.isnan(inflect[0]) or np.isnan(inflect[1])):
        plt.scatter(inflect[0], inflect[1], color="red", s=30, zorder=5, label="εIP")
    if not (np.isnan(tys[0]) or np.isnan(tys[1])):
        plt.scatter(tys[0], tys[1], color="magenta", s=30, zorder=5, label="σyT")
    # Draw tangent lines
    xs = np.linspace(df_cycle["Strain"].min(), df_cycle["Strain"].max(), 100)
    if not any(np.isnan(lin1)):
        plt.plot(xs, lin1[0] * xs + lin1[1], "--", color="orange", linewidth=1.2, label="Lower tangent 1")
    if not any(np.isnan(lin2)):
        plt.plot(xs, lin2[0] * xs + lin2[1], "--", color="purple", linewidth=1.2, label="Lower tangent 2")

    plt.xlabel("Strain (fraction)")
    plt.ylabel("Stress (MPa)")
    plt.title(f"Cycle {cycle}")
    plt.grid(True, ls="--", alpha=0.5)
    plt.legend(fontsize=9, ncol=2)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


# -----------------------------
# CLI
# -----------------------------
def parse_args():
    p = argparse.ArgumentParser(description="M2 parameter extraction per paper definitions with log-spaced cycle sampling")
    # I/O
    p.add_argument("--raw_root", type=str, required=True, help="Root folder containing subdirectories with raw loop CSVs")
    p.add_argument("--subdirs", type=str, default="", help="Comma-separated subfolders to include. Default all immediate subfolders")
    p.add_argument("--outdir", type=str, default="outputs", help="Output directory")

    # Raw CSV format
    p.add_argument("--cols", type=str, default="3,4,5", help="Zero-based column indices for [strain, cycle, stress]")
    p.add_argument("--header", type=str, default="0", help="Header row index or 'None'")
    p.add_argument("--sep", type=str, default=",", help="CSV delimiter")
    p.add_argument("--strain_units", type=str, default="percent", choices=["percent", "fraction"], help="Units of strain column in raw CSV")

    # Cycle sampling
    p.add_argument("--min_cycle", type=int, default=3)
    p.add_argument("--log_points", type=int, default=15, help="Number of log-spaced cycles between min and max")
    p.add_argument("--include_half_life", action="store_true", help="Also include half-life cycle")

    # Optional M1 synthesis
    p.add_argument("--m1_model", type=str, default="", help="Path to trained M1 ANN model (.keras or .h5)")
    p.add_argument("--m1_mean_std", type=str, default="", help="Path to mean_std.csv used for M1")
    p.add_argument("--synth_points", type=int, default=0, help="If >0, synthesize this many extra cycles in logspace above max")
    p.add_argument("--synth_half_length", type=int, default=25)

    # Plots
    p.add_argument("--save_plots", action="store_true")
    p.add_argument("--plot_every", type=int, default=1, help="Save every Nth plot to limit output size")

    # Misc
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


# -----------------------------
# Main
# -----------------------------
def main():
    args = parse_args()
    set_seed(args.seed)

    # prepare output
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = os.path.join(args.outdir, f"m2_{stamp}")
    mkdir(out_root)

    # subdirs
    if args.subdirs.strip():
        subfolders = [os.path.join(args.raw_root, s.strip()) for s in args.subdirs.split(",")]
    else:
        subfolders = [os.path.join(args.raw_root, d) for d in os.listdir(args.raw_root) if os.path.isdir(os.path.join(args.raw_root, d))]

    # optional M1
    model = None
    if args.m1_model:
        model = maybe_load_m1(args.m1_model)
        if not args.m1_mean_std:
            raise ValueError("--m1_mean_std is required when --m1_model is provided")

    # accumulation
    rows = []

    for folder in subfolders:
        subname = os.path.basename(folder)
        if not os.path.isdir(folder):
            continue

        material = 0 if re.search(r"AZ91", subname, re.IGNORECASE) else 1
        heat = 300 if re.search(r"300", subname) else 400

        for fname in os.listdir(folder):
            if not fname.lower().endswith(".csv"):
                continue

            path = os.path.join(folder, fname)
            stem = os.path.splitext(fname)[0]
            strain_amp_pct, freq_hz = parse_filename_tokens(stem)

            # load raw
            header = None if args.header.lower() == "none" else int(args.header)
            try:
                raw = pd.read_csv(path, header=header, sep=args.sep)
            except Exception as e:
                print(f"Skip {path}  reason {e}")
                continue

            cidx = [int(x) for x in args.cols.split(",")]
            use = raw.iloc[:, cidx].copy()
            use.columns = ["Strain_raw", "Cycle", "Stress"]
            # convert strain to fraction
            use["Strain"] = to_fractional_strain(use["Strain_raw"].astype(float), args.strain_units)
            use.drop(columns=["Strain_raw"], inplace=True)

            # Optionally thin for very large recordings
            # Determine max cycle
            if use["Cycle"].empty:
                continue
            max_cycle = int(np.nanmax(use["Cycle"].values))
            if max_cycle < args.min_cycle + 5:
                # too small sequence
                continue

            # Log-spaced cycles
            life_cycles = np.unique(np.logspace(np.log10(args.min_cycle), np.log10(max_cycle), args.log_points).astype(int))
            if args.include_half_life:
                life_cycles = np.unique(np.append(life_cycles, max_cycle // 2))

            # Optional synthesis beyond max
            synth_cycles = []
            if model is not None and args.synth_points > 0:
                ext = np.unique(np.logspace(np.log10(max_cycle), np.log10(3 * max_cycle), args.synth_points + 1).astype(int))[1:]
                synth_cycles = ext.tolist()

            # process measured cycles
            all_cycles = list(life_cycles) + synth_cycles
            for idx_c, cyc in enumerate(all_cycles, 1):
                if cyc <= max_cycle:
                    cyc_df = use[use["Cycle"] == cyc].copy()
                    if cyc_df.empty:
                        continue
                    cyc_df = cyc_df.sort_values("Strain").reset_index(drop=True)
                else:
                    # synthesize with M1
                    if model is None:
                        continue
                    half_len = args.synth_half_length
                    fd = fake_loop_dataset(cyc, strain_amp_pct if strain_amp_pct is not None else 1.0, material, heat, half_len)
                    # Normalize using mean_std
                    fd_norm = normalize_with_table(
                        fd.rename(columns={"Strain": "Strain 1 %"}),  # mean_std came from M1 training schema
                        args.m1_mean_std,
                    )
                    # Columns for M1
                    feat = ["Strain 1 %", "Cycle", "Strain Amplitude", "Material", "Heat Treatment", "Strain - 1"]
                    y_pred = model.predict(fd_norm[feat], verbose=0).ravel()
                    # Inverse to original stress units
                    y_pred = inverse_from_table(y_pred.reshape(-1, 1), args.m1_mean_std, target_col="Stress MPa").ravel()
                    cyc_df = pd.DataFrame({"Strain": fd["Strain"].values / 100.0,  # convert % to fraction
                                           "Cycle": fd["Cycle"].values,
                                           "Stress": y_pred})

                # split branches
                upper, lower = identify_upper_lower_loops(cyc_df)

                # εIP from upper
                eps_ip, sig_ip = inflection_point_upper(upper)

                # σyT from lower
                tys_x, tys_y, lin1, lin2 = tangents_and_tys_lower(lower)

                # σ̇_T from lower after σyT
                sigma_dot_GPa = twinning_gradient_postyield(lower, tys_x, strain_fractional=True)

                # extremes
                sig_mt = float(np.nanmax(cyc_df["Stress"].values))  # MPa
                sig_mc = float(np.nanmin(cyc_df["Stress"].values))  # MPa
                # ε at σMT on upper
                if len(upper):
                    i_mt = int(np.nanargmax(upper["Stress"].values))
                    eps_mt = float(upper["Strain"].values[i_mt])
                else:
                    eps_mt = np.nan

                # Energies
                # Loop area ΔWp with proper units
                loop_area_mpa = polygon_loop_area(cyc_df["Strain"].values, cyc_df["Stress"].values)
                dWp_kj_per_m3 = abs(loop_area_mpa) * 1e6 / 1000.0

                dWe_plus_kj_per_m3 = we_plus_tangent(upper, eps_ip, sig_ip, eps_mt, mode="tangent_at_inflection")

                # Relative cycle
                rel = float(cyc / max_cycle)

                rows.append(
                    {
                        "Filename": fname,
                        "Strain Amplitude (%)": strain_amp_pct if strain_amp_pct is not None else np.nan,
                        "Frequency (Hz)": freq_hz if freq_hz is not None else np.nan,
                        "Material": material,
                        "Heat Treatment": heat,
                        "Cycle": int(cyc),
                        "Relative Cycle": rel,
                        "Inflection Strain (%)": eps_ip * 100.0 if np.isfinite(eps_ip) else np.nan,
                        "Inflection Stress (MPa)": sig_ip if np.isfinite(sig_ip) else np.nan,
                        "TYS Strain (%)": tys_x * 100.0 if np.isfinite(tys_x) else np.nan,
                        "TYS Stress (MPa)": tys_y if np.isfinite(tys_y) else np.nan,
                        "Twinning Gradient (GPa)": sigma_dot_GPa if np.isfinite(sigma_dot_GPa) else np.nan,
                        "Wp (kJ/m3)": dWp_kj_per_m3 if np.isfinite(dWp_kj_per_m3) else np.nan,
                        "We (kJ/m3)": dWe_plus_kj_per_m3 if np.isfinite(dWe_plus_kj_per_m3) else np.nan,
                        "Mean Stress (MPa)": 0.5 * (sig_mt + sig_mc),
                        "Tensile Peak Stress (MPa)": sig_mt,
                        "Compressive Peak Stress (MPa)": sig_mc,
                    }
                )

                if args.save_plots and (idx_c % max(1, args.plot_every) == 0):
                    inf_pair = (eps_ip, sig_ip)
                    tys_pair = (tys_x, tys_y)
                    plot_path = os.path.join(out_root, f"{os.path.splitext(fname)[0]}__cycle_{int(cyc)}.png")
                    plot_results(cyc_df, upper, lower, inf_pair, tys_pair, lin1, lin2, cyc, plot_path)

            print(f"Processed {fname} in {subname}")

    # Save master results
    res = pd.DataFrame(rows)
    csv_main = os.path.join(out_root, "calculation_results.csv")
    res.to_csv(csv_main, index=False)
    print(f"Saved: {csv_main}")

    # Per-file min-max scaling for selected columns
    cols_scale = [
        "Inflection Strain (%)",
        "Inflection Stress (MPa)",
        "TYS Strain (%)",
        "TYS Stress (MPa)",
        "Twinning Gradient (GPa)",
    ]
    scaled_blocks = []
    for file_id, grp in res.groupby("Filename"):
        g = grp.copy()
        for c in cols_scale:
            if c in g.columns:
                vmin = np.nanmin(g[c].values)
                vmax = np.nanmax(g[c].values)
                if np.isfinite(vmin) and np.isfinite(vmax) and vmax > vmin:
                    g[c] = (g[c] - vmin) / (vmax - vmin)
        scaled_blocks.append(g)
    res_scaled = pd.concat(scaled_blocks, ignore_index=True)
    csv_scaled = os.path.join(out_root, "calculation_results_with_scaling.csv")
    res_scaled.to_csv(csv_scaled, index=False)
    print(f"Saved: {csv_scaled}")

    print(f"Outputs stored in: {out_root}")


if __name__ == "__main__":
    main()
