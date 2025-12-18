# =========================
# File: M2.py
# =========================
# -*- coding: utf-8 -*-
"""
M2: Automated extraction of loop-based fatigue parameters (deployment)

Implements deterministic, rule-based extraction used in CIFLE.

Key definitions
- Upper/lower branches split using max/min stress locations in acquisition order,
  then each branch is strain-sorted.
- Twinning yield strength (TYS, σyT):
  1) take lower branch (strain-sorted)
  2) split into two equal segments
  3) fit a straight line on the central 30–70% subrange of each segment
  4) take intersection (ε*, σ*)
  5) report σyT as the stress at the discrete loop point closest to ε*
- Inflection point (εIP, σIP): sign change of smoothed second derivative on upper branch
- ΔWp: absolute loop area (plastic strain energy density per cycle)
- ΔWe+: area under tangent line at εIP, integrated from tensile zero-stress crossing
         to ε at σMT (max tensile stress)

Units
- Stress: MPa
- Strain input: percent or fraction
- Energies returned for compatibility: kJ/m3 (Wp, We)

CLI supports batch-processing raw CSVs (project-compatible defaults).
"""

from __future__ import annotations

import argparse
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

Number = Union[int, float, np.number]


# -----------------------------
# Configuration
# -----------------------------
@dataclass(frozen=True)
class M2Config:
    strain_units: str = "percent"   # "percent" or "fraction"
    energy_units: str = "kJ/m3"     # internal computation; outputs still kJ/m3
    smooth_frac: float = 0.15
    smooth_min_window: int = 11
    smooth_polyorder: int = 3
    tys_trim_lo: float = 0.30
    tys_trim_hi: float = 0.70
    grad_trim_lo: float = 0.30
    grad_trim_hi: float = 0.70


# -----------------------------
# Utilities
# -----------------------------
def _as_float_array(x: Sequence[Number]) -> np.ndarray:
    return np.asarray(x, dtype=float)


def _odd_window(n: int, frac: float, minimum: int) -> int:
    if n <= 1:
        return 1
    w = int(round(frac * n))
    w = max(minimum, w)
    if w % 2 == 0:
        w += 1
    if w >= n:
        w = n - 1 if (n - 1) % 2 == 1 else n - 2
    return max(3, w)


def _to_fractional_strain(strain: np.ndarray, strain_units: str) -> np.ndarray:
    u = strain_units.lower()
    if u in ["percent", "%", "pct"]:
        return strain / 100.0
    if u in ["fraction", "frac"]:
        return strain
    raise ValueError(f"Unsupported strain_units='{strain_units}'. Use 'percent' or 'fraction'.")


def loop_area_mpa(strain_frac: np.ndarray, stress_mpa: np.ndarray) -> float:
    x = _as_float_array(strain_frac)
    y = _as_float_array(stress_mpa)
    if len(x) < 3:
        return float("nan")
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def energy_from_area(area_mpa: float, units: str = "kJ/m3") -> float:
    if not np.isfinite(area_mpa):
        return float("nan")
    j_per_m3 = abs(area_mpa) * 1e6
    if units == "J/m3":
        return float(j_per_m3)
    if units == "kJ/m3":
        return float(j_per_m3 / 1000.0)
    raise ValueError(f"Unsupported energy units '{units}'.")


def _cyclic_slice(df: pd.DataFrame, start: int, end: int) -> pd.DataFrame:
    n = len(df)
    if n == 0:
        return df.copy()
    start = int(start) % n
    end = int(end) % n
    if start <= end:
        return df.iloc[start : end + 1].copy()
    return pd.concat([df.iloc[start:], df.iloc[: end + 1]], axis=0).copy()


def split_upper_lower_branches(loop_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    loop_df must contain columns ['Strain','Stress'] in acquisition order.
    Returns (upper, lower) as strain-sorted, duplicate-strain removed DataFrames.
    """
    if loop_df.empty:
        return loop_df.copy(), loop_df.copy()

    df = loop_df[["Strain", "Stress"]].copy().reset_index(drop=True)
    if df["Stress"].isna().all():
        return df.copy(), df.copy()

    idx_min = int(df["Stress"].idxmin())
    idx_max = int(df["Stress"].idxmax())

    upper_raw = _cyclic_slice(df, idx_min, idx_max)
    lower_raw = _cyclic_slice(df, idx_max, idx_min)

    upper = upper_raw.sort_values("Strain").drop_duplicates(subset=["Strain"]).reset_index(drop=True)
    lower = lower_raw.sort_values("Strain").drop_duplicates(subset=["Strain"]).reset_index(drop=True)
    return upper, lower


def inflection_point_upper(upper: pd.DataFrame, cfg: M2Config) -> Tuple[float, float]:
    if upper is None or len(upper) < 9:
        return float("nan"), float("nan")

    s = _as_float_array(upper["Strain"].values)
    sig = _as_float_array(upper["Stress"].values)

    order = np.argsort(s)
    s = s[order]
    sig = sig[order]

    ds = np.diff(s)
    if not np.all(np.isfinite(ds)) or np.nanmedian(np.abs(ds)) <= 0:
        return float("nan"), float("nan")

    win = _odd_window(len(s), cfg.smooth_frac, cfg.smooth_min_window)
    try:
        d2 = savgol_filter(
            sig,
            window_length=win,
            polyorder=cfg.smooth_polyorder,
            deriv=2,
            delta=float(np.median(ds)),
            mode="interp",
        )
    except Exception:
        return float("nan"), float("nan")

    sign = np.sign(d2)
    valid = np.isfinite(sign)
    if not np.any(valid):
        return float("nan"), float("nan")

    idx_change = None
    for i in range(1, len(sign)):
        if not (valid[i - 1] and valid[i]):
            continue
        if sign[i - 1] < 0 and sign[i] > 0:
            idx_change = i
            break

    if idx_change is None:
        for i in range(1, len(sign)):
            if not (valid[i - 1] and valid[i]):
                continue
            if sign[i - 1] == 0 or sign[i] == 0:
                continue
            if sign[i - 1] != sign[i]:
                idx_change = i
                break

    if idx_change is None:
        return float("nan"), float("nan")

    lo = max(0, idx_change - 2)
    hi = min(len(s), idx_change + 3)
    j = lo + int(np.nanargmin(np.abs(d2[lo:hi])))

    return float(s[j]), float(sig[j])


def _trim_middle(df: pd.DataFrame, lo: float, hi: float) -> pd.DataFrame:
    n = len(df)
    if n < 3:
        return df.copy()
    a = int(np.floor(lo * n))
    b = int(np.ceil(hi * n))
    a = max(0, min(n - 2, a))
    b = max(a + 2, min(n, b))
    return df.iloc[a:b].copy()


def twinning_yield_from_lower(lower: pd.DataFrame, cfg: M2Config) -> Dict[str, float]:
    """
    Returns:
      - TYS Strain (%) and TYS Stress (MPa): discrete loop point closest to intersection strain
      - plus diagnostic line/intersection values
    """
    out: Dict[str, float] = {
        "TYS Strain (%)": float("nan"),
        "TYS Stress (MPa)": float("nan"),
        "TYS Intersection Strain (%)": float("nan"),
        "TYS Intersection Stress (MPa)": float("nan"),
        "TYS Tangent1 Slope (MPa)": float("nan"),
        "TYS Tangent1 Intercept (MPa)": float("nan"),
        "TYS Tangent2 Slope (MPa)": float("nan"),
        "TYS Tangent2 Intercept (MPa)": float("nan"),
    }
    if lower is None or len(lower) < 10:
        return out

    lo = lower.sort_values("Strain").reset_index(drop=True)
    mid = len(lo) // 2
    seg1 = lo.iloc[:mid]
    seg2 = lo.iloc[mid:]

    seg1m = _trim_middle(seg1, cfg.tys_trim_lo, cfg.tys_trim_hi)
    seg2m = _trim_middle(seg2, cfg.tys_trim_lo, cfg.tys_trim_hi)

    def fit_line(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
        a, b = np.polyfit(x.astype(float), y.astype(float), deg=1)
        return float(a), float(b)

    try:
        a1, b1 = fit_line(seg1m["Strain"].values, seg1m["Stress"].values)
        a2, b2 = fit_line(seg2m["Strain"].values, seg2m["Stress"].values)
    except Exception:
        return out

    out["TYS Tangent1 Slope (MPa)"] = a1
    out["TYS Tangent1 Intercept (MPa)"] = b1
    out["TYS Tangent2 Slope (MPa)"] = a2
    out["TYS Tangent2 Intercept (MPa)"] = b2

    if abs(a1 - a2) < 1e-12:
        return out

    x_int = (b2 - b1) / (a1 - a2)
    y_int = a1 * x_int + b1

    out["TYS Intersection Strain (%)"] = float(x_int) * 100.0
    out["TYS Intersection Stress (MPa)"] = float(y_int)

    strains = _as_float_array(lo["Strain"].values)
    stresses = _as_float_array(lo["Stress"].values)
    if len(strains) == 0:
        return out

    k = int(np.nanargmin(np.abs(strains - x_int)))
    out["TYS Strain (%)"] = float(strains[k]) * 100.0
    out["TYS Stress (MPa)"] = float(stresses[k])
    return out


def twinning_gradient_postyield(lower: pd.DataFrame, tys_strain_frac: float, cfg: M2Config) -> float:
    if lower is None or len(lower) < 10 or not np.isfinite(tys_strain_frac):
        return float("nan")

    lo = lower.sort_values("Strain").reset_index(drop=True)
    post = lo[lo["Strain"] >= tys_strain_frac].copy()
    if len(post) < 6:
        return float("nan")

    seg = _trim_middle(post, cfg.grad_trim_lo, cfg.grad_trim_hi)
    if len(seg) < 3:
        seg = post

    x = _as_float_array(seg["Strain"].values)
    y = _as_float_array(seg["Stress"].values)
    if len(x) < 2 or np.nanstd(x) == 0:
        return float("nan")

    a, _ = np.polyfit(x, y, deg=1)  # MPa per strain fraction
    return float(a) / 1000.0  # GPa


def tensile_zero_crossing(upper: pd.DataFrame) -> float:
    if upper is None or len(upper) < 2:
        return float("nan")

    s = _as_float_array(upper["Strain"].values)
    sig = _as_float_array(upper["Stress"].values)

    order = np.argsort(s)
    s = s[order]
    sig = sig[order]

    for i in range(1, len(s)):
        if not (np.isfinite(sig[i - 1]) and np.isfinite(sig[i])):
            continue
        if (sig[i - 1] <= 0.0 and sig[i] >= 0.0) or (sig[i - 1] >= 0.0 and sig[i] <= 0.0):
            if sig[i] == sig[i - 1]:
                return float(s[i])
            t = (0.0 - sig[i - 1]) / (sig[i] - sig[i - 1])
            return float(s[i - 1] + t * (s[i] - s[i - 1]))
    return float("nan")


def we_plus_tangent(
    upper: pd.DataFrame,
    eps_ip: float,
    sig_ip: float,
    eps_at_sig_mt: float,
    cfg: M2Config,
) -> float:
    """
    ΔWe+ (kJ/m3): integrate tangent at εIP from ε(σ=0) to ε(σMT).
    """
    if upper is None or len(upper) < 9:
        return float("nan")
    if not (np.isfinite(eps_ip) and np.isfinite(sig_ip) and np.isfinite(eps_at_sig_mt)):
        return float("nan")

    s = _as_float_array(upper["Strain"].values)
    sig = _as_float_array(upper["Stress"].values)
    order = np.argsort(s)
    s = s[order]
    sig = sig[order]

    ds = np.diff(s)
    if np.nanmedian(np.abs(ds)) <= 0:
        return float("nan")

    win = _odd_window(len(s), cfg.smooth_frac, cfg.smooth_min_window)
    try:
        d1 = savgol_filter(
            sig,
            window_length=win,
            polyorder=cfg.smooth_polyorder,
            deriv=1,
            delta=float(np.median(ds)),
            mode="interp",
        )
    except Exception:
        return float("nan")

    j = int(np.nanargmin(np.abs(s - eps_ip)))
    a = float(d1[j])
    b = float(sig_ip) - a * float(eps_ip)  # σ = a ε + b

    eps0 = tensile_zero_crossing(pd.DataFrame({"Strain": s, "Stress": sig}))
    if not np.isfinite(eps0):
        return float("nan")

    eps1 = float(eps0)
    eps2 = float(eps_at_sig_mt)
    if eps2 <= eps1:
        return float("nan")

    area_mpa = 0.5 * a * (eps2 ** 2 - eps1 ** 2) + b * (eps2 - eps1)  # MPa
    area_mpa = max(0.0, float(area_mpa))

    return energy_from_area(area_mpa, units=cfg.energy_units)  # kJ/m3 if cfg.energy_units="kJ/m3"


def extract_loop_features(
    loop_df: pd.DataFrame,
    cfg: Optional[M2Config] = None,
    strain_units: Optional[str] = None,
) -> Dict[str, float]:
    """
    Extract M2 parameters from a single hysteresis loop.

    Returns keys compatible with M3:
      - Wp (kJ/m3)
      - We (kJ/m3)
      - Tensile Peak Stress (MPa)
      - Compressive Peak Stress (MPa)
      - Inflection Strain (%)
      - TYS Stress (MPa)
      - Twinning Gradient (GPa)

    Extra keys also included:
      - Inflection Stress (MPa)
      - TYS Strain (%)
    """
    if cfg is None:
        cfg = M2Config()
    if strain_units is None:
        strain_units = cfg.strain_units

    if loop_df is None or loop_df.empty:
        return {
            "Wp (kJ/m3)": float("nan"),
            "We (kJ/m3)": float("nan"),
            "Inflection Strain (%)": float("nan"),
            "Inflection Stress (MPa)": float("nan"),
            "TYS Stress (MPa)": float("nan"),
            "TYS Strain (%)": float("nan"),
            "Twinning Gradient (GPa)": float("nan"),
            "Tensile Peak Stress (MPa)": float("nan"),
            "Compressive Peak Stress (MPa)": float("nan"),
        }

    df = loop_df.copy()
    if "Strain" not in df.columns or "Stress" not in df.columns:
        raise ValueError("loop_df must have columns 'Strain' and 'Stress'.")

    strain_in = _as_float_array(df["Strain"].values)
    stress_mpa = _as_float_array(df["Stress"].values)

    strain_frac = _to_fractional_strain(strain_in, strain_units=strain_units)

    loop_frac = pd.DataFrame({"Strain": strain_frac, "Stress": stress_mpa})
    upper, lower = split_upper_lower_branches(loop_frac)

    eps_ip_f, sig_ip = inflection_point_upper(upper, cfg)
    tys_info = twinning_yield_from_lower(lower, cfg)

    tys_strain_f = float(tys_info["TYS Strain (%)"]) / 100.0 if np.isfinite(tys_info["TYS Strain (%)"]) else float("nan")
    sigma_dot_gpa = twinning_gradient_postyield(lower, tys_strain_f, cfg)

    sig_mt = float(np.nanmax(stress_mpa)) if np.any(np.isfinite(stress_mpa)) else float("nan")
    sig_mc = float(np.nanmin(stress_mpa)) if np.any(np.isfinite(stress_mpa)) else float("nan")

    eps_at_sig_mt_f = float("nan")
    if upper is not None and len(upper) > 0 and np.any(np.isfinite(upper["Stress"].values)):
        i_mt = int(np.nanargmax(upper["Stress"].values))
        eps_at_sig_mt_f = float(upper["Strain"].values[i_mt])

    area_mpa = loop_area_mpa(strain_frac, stress_mpa)
    wp_kj = energy_from_area(area_mpa, units=cfg.energy_units)

    we_kj = we_plus_tangent(upper, eps_ip_f, sig_ip, eps_at_sig_mt_f, cfg)

    # If someone sets cfg.energy_units="J/m3", still output kJ/m3 for compatibility
    if cfg.energy_units == "J/m3":
        wp_kj = wp_kj / 1000.0 if np.isfinite(wp_kj) else float("nan")
        we_kj = we_kj / 1000.0 if np.isfinite(we_kj) else float("nan")

    return {
        "Wp (kJ/m3)": float(wp_kj),
        "We (kJ/m3)": float(we_kj),
        "Inflection Strain (%)": float(eps_ip_f) * 100.0 if np.isfinite(eps_ip_f) else float("nan"),
        "Inflection Stress (MPa)": float(sig_ip) if np.isfinite(sig_ip) else float("nan"),
        "TYS Strain (%)": float(tys_info["TYS Strain (%)"]),
        "TYS Stress (MPa)": float(tys_info["TYS Stress (MPa)"]),
        "Twinning Gradient (GPa)": float(sigma_dot_gpa),
        "Tensile Peak Stress (MPa)": float(sig_mt),
        "Compressive Peak Stress (MPa)": float(sig_mc),
    }


# -----------------------------
# CLI (batch)
# -----------------------------
def _parse_cols(cols: str) -> Tuple[int, int, int]:
    parts = [c.strip() for c in cols.split(",")]
    if len(parts) != 3:
        raise ValueError("--cols must be three comma-separated integers like '3,4,5'.")
    return int(parts[0]), int(parts[1]), int(parts[2])


def _log_spaced_cycles(min_cycle: int, max_cycle: int, n_points: int) -> List[int]:
    if max_cycle <= min_cycle:
        return [max_cycle]
    grid = np.logspace(np.log10(min_cycle), np.log10(max_cycle), int(n_points))
    cyc = np.unique(np.clip(np.round(grid).astype(int), min_cycle, max_cycle))
    return [int(c) for c in cyc.tolist()]


def _material_heat_from_folder(folder_name: str) -> Tuple[int, int]:
    mat = 0 if re.search(r"AZ91", folder_name, re.IGNORECASE) else 1
    ht = 300 if re.search(r"300", folder_name) else 400
    return mat, ht


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="M2 loop feature extraction (CIFLE)")
    p.add_argument("--raw_root", type=str, required=True, help="Root folder containing subdirectories of raw loop CSVs")
    p.add_argument("--subdirs", type=str, default="", help="Comma-separated subfolder names. Default scans all immediate subfolders")
    p.add_argument("--outdir", type=str, default="outputs", help="Output directory")

    p.add_argument("--cols", type=str, default="3,4,5", help="Zero-based indices for [strain, cycle, stress] columns")
    p.add_argument("--header", type=str, default="0", help="Header row index, or 'None'")
    p.add_argument("--sep", type=str, default=",", help="CSV delimiter")
    p.add_argument("--strain_units", type=str, default="percent", choices=["percent", "fraction"])

    p.add_argument("--min_cycle", type=int, default=3)
    p.add_argument("--log_points", type=int, default=15)
    p.add_argument("--failure_offset", type=int, default=0)

    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    np.random.seed(int(args.seed))

    out_root = os.path.abspath(args.outdir)
    os.makedirs(out_root, exist_ok=True)

    if args.subdirs.strip():
        subfolders = [os.path.join(args.raw_root, s.strip()) for s in args.subdirs.split(",") if s.strip()]
    else:
        subfolders = [os.path.join(args.raw_root, d) for d in os.listdir(args.raw_root)]

    col_s, col_c, col_sig = _parse_cols(args.cols)
    header = None if str(args.header).lower() == "none" else int(args.header)

    cfg = M2Config(strain_units=args.strain_units, energy_units="kJ/m3")

    rows: List[Dict[str, float]] = []

    for folder in subfolders:
        if not os.path.isdir(folder):
            continue

        subname = os.path.basename(folder)
        material, heat = _material_heat_from_folder(subname)

        for fname in os.listdir(folder):
            if not fname.lower().endswith(".csv"):
                continue

            fpath = os.path.join(folder, fname)
            try:
                df = pd.read_csv(fpath, header=header, sep=args.sep)
            except Exception as e:
                print(f"[M2] Skip {fpath}  read error: {e}")
                continue

            try:
                df = df.iloc[:, [col_s, col_c, col_sig]].copy()
                df.columns = ["Strain", "Cycle", "Stress"]
            except Exception as e:
                print(f"[M2] Skip {fpath}  column error: {e}")
                continue

            df = df.dropna(subset=["Strain", "Cycle", "Stress"]).reset_index(drop=True)
            if df.empty:
                continue

            max_cycle = int(np.nanmax(df["Cycle"].values))
            nf = max(1, max_cycle - int(args.failure_offset))

            cycles = _log_spaced_cycles(args.min_cycle, max_cycle, args.log_points)

            for cyc in cycles:
                sub_df = df[df["Cycle"] == cyc].copy()
                if sub_df.empty:
                    idx_near = int((df["Cycle"] - cyc).abs().idxmin())
                    cyc_near = int(df.loc[idx_near, "Cycle"])
                    sub_df = df[df["Cycle"] == cyc_near].copy()
                    cyc = cyc_near

                feat = extract_loop_features(sub_df[["Strain", "Stress"]], cfg=cfg, strain_units=args.strain_units)

                row = {
                    "Filename": fname,
                    "Folder": subname,
                    "Material": int(material),
                    "Heat Treatment": int(heat),
                    "Cycle": int(cyc),
                    "Nf": float(nf),
                    "Relative Cycle": float(cyc) / float(nf) if nf > 0 else float("nan"),
                }
                row.update(feat)
                rows.append(row)

            print(f"[M2] Processed {fname} in {subname}")

    if not rows:
        raise RuntimeError("[M2] No results were produced. Check input paths and CSV format.")

    res = pd.DataFrame(rows)
    out_csv = os.path.join(out_root, "calculation_results.csv")
    res.to_csv(out_csv, index=False)
    print(f"[M2] Saved: {out_csv}")


if __name__ == "__main__":
    main()

