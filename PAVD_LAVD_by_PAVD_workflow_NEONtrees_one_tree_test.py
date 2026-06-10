# -*- coding: utf-8 -*-
"""
For one tree:
1. Read all *_height_profiles.csv files
2. Remove rows with negative height using tolerance
3. Force tiny remaining negative heights to 0
4. Pick the scan with the highest LAI_cum at the tallest non-negative height
5. Use that scan to compute a WIDE one-row summary for top 10,20,...,100%
   of the LAVD-defined crown portion
6. Save outputs

Output wide file has one row for the tree / selected scan and columns like:
    LAI_top10, eLAI_top10, cutoff_h_top10, PAVD_mean_top10, PAVD_sd_top10, ...
"""

import os
import glob
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------
# USER SETTINGS
# ---------------------------------------------------------------------
TREE_DIR  = r"E:\TLS\2025_summer\2025_SERC\Tree_scan\ForestGEO_SERC\L1_segmented_split\SERC_usb1_job28_LITU\PAVD_height_profiles_from_bincount"
OUT_DIR = os.path.join(TREE_DIR, "best_scan_topFrac_LAVD_nonNegH_wide0.3")
os.makedirs(OUT_DIR, exist_ok=True)

FILE_PATTERN = "SERC_usb1_job28_LITU_Scan*_height_profiles*.csv"
TOP_FRACS = np.arange(0.1, 1.01, 0.1)   # 0.1 ... 1.0

# tolerance for tiny negative heights from floating-point noise
H_TOL = 1e-8

# ---------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------
def clean_height_profile(df):
    df = df.copy().sort_values("h").reset_index(drop=True)

    # keep rows not meaningfully below zero
    df = df[df["h"] >= -H_TOL].copy().reset_index(drop=True)

    if df.empty:
        return df

    # hard-clip tiny residual negatives
    df.loc[df["h"] < 0, "h"] = 0.0
    return df


def load_scan_profiles(tree_dir, pattern):
    files = sorted(glob.glob(os.path.join(tree_dir, pattern)))
    if len(files) == 0:
        raise FileNotFoundError(f"No files matched: {os.path.join(tree_dir, pattern)}")

    scans = []
    for f in files:
        df = pd.read_csv(f)

        required = {
            "scan", "z_mid", "h",
            "pavd_h", "pavde_h", "lavd_h", "lavde_h",
            "LAI_cum", "LAIe_cum"
        }
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"{os.path.basename(f)} missing columns: {missing}")

        df = clean_height_profile(df)
        if df.empty:
            raise ValueError(f"{os.path.basename(f)} has no non-negative height rows after filtering.")

        tallest_idx = df["h"].idxmax()
        tallest_h = float(df.loc[tallest_idx, "h"])
        lai_at_tallest = float(df.loc[tallest_idx, "LAI_cum"])
        elai_at_tallest = float(df.loc[tallest_idx, "LAIe_cum"])

        scans.append({
            "file": f,
            "scan_name": str(df["scan"].iloc[0]),
            "df": df,
            "tallest_h": tallest_h,
            "LAI_cum_at_tallest": lai_at_tallest,
            "LAIe_cum_at_tallest": elai_at_tallest,
        })

    return scans


def pick_best_scan(scans):
    """
    Pick scan with highest LAI_cum at tallest height.
    Tie-breakers:
      1) higher LAIe_cum at tallest height
      2) taller height
      3) alphabetical scan name
    """
    scans_sorted = sorted(
        scans,
        key=lambda x: (
            x["LAI_cum_at_tallest"],
            x["LAIe_cum_at_tallest"],
            x["tallest_h"],
            x["scan_name"],
        ),
        reverse=True,
    )
    return scans_sorted[0]


def find_crown_window(df):
    """
    Use rows where LAVD or LAVDe is positive to define crown support.
    """
    mask = (df["lavd_h"].to_numpy() > 0) | (df["lavde_h"].to_numpy() > 0)
    idx = np.where(mask)[0]
    if idx.size == 0:
        raise ValueError("No positive LAVD/LAVDe rows found in chosen scan.")

    i0 = int(idx[0])
    i1 = int(idx[-1]) + 1
    return i0, i1


def infer_z_res(df):
    z = df["z_mid"].to_numpy()
    dz = np.diff(z)
    dz = dz[np.isfinite(dz) & (np.abs(dz) > 0)]
    if dz.size == 0:
        raise ValueError("Could not infer z resolution from z_mid.")
    return float(np.median(np.abs(dz)))

import os, sys
import yaml
REPO_DIR  = r"C:\Users\Ricy\PERS-master\TLS\TLSLeAF2.0_2025April"
# repo imports
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)
os.chdir(REPO_DIR)

# config
with open(os.path.join(REPO_DIR, "config.yaml"), "r") as f:
    config = yaml.safe_load(f)
config_pp     = config["preprocessing"]
bincount_para = config_pp["bincount_para"]
z_res = bincount_para["z_res"]

def calc_topfrac_from_lavd(df, top_fracs):
    """
    Compute long-form top-fraction summary using LAVD-defined top crown fractions.
    Also includes PAVD / PAVDe / LAVD / LAVDe means and SDs.

    Fractions are defined by cumulative LAVD-derived LAI from canopy top downward.
    """
    df = clean_height_profile(df)
    if df.empty:
        raise ValueError("No non-negative height rows remain for selected scan.")

    i0, i1 = find_crown_window(df)
    #z_res = infer_z_res(df)
    # config
    

    crown = df.iloc[i0:i1].copy().reset_index(drop=True)

    crown["PAI_bin"]  = crown["pavd_h"]  * z_res
    crown["PAIe_bin"] = crown["pavde_h"] * z_res
    crown["LAI_bin"]  = crown["lavd_h"]  * z_res
    crown["LAIe_bin"] = crown["lavde_h"] * z_res

    total_pai  = float(crown["PAI_bin"].sum())
    total_paie = float(crown["PAIe_bin"].sum())
    total_lai  = float(crown["LAI_bin"].sum())
    total_laie = float(crown["LAIe_bin"].sum())

    # define top fractions using LAVD-derived LAI
    lai_bin_topdown = crown["LAI_bin"].to_numpy()[::-1]
    cum_lai_topdown = np.cumsum(lai_bin_topdown)

    rows = []
    for frac in top_fracs:
        target = frac * total_lai

        idx_rev = np.where(cum_lai_topdown >= target)[0]
        if idx_rev.size == 0:
            k = len(crown) - 1
        else:
            k = int(idx_rev[0])

        n_keep = k + 1
        i_start_local = len(crown) - n_keep
        sub = crown.iloc[i_start_local:].copy()

        rows.append({
            "top_fraction_label": f"top{int(frac * 100):02d}",
            "top_fraction": frac,
            "cutoff_h_m": float(sub["h"].iloc[0]),
            "cutoff_z_mid": float(sub["z_mid"].iloc[0]),
            "n_bins_kept": int(len(sub)),

            # profile means
            "PAVD_mean": float(sub["pavd_h"].mean()),
            "PAVDe_mean": float(sub["pavde_h"].mean()),
            "LAVD_mean": float(sub["lavd_h"].mean()),
            "LAVDe_mean": float(sub["lavde_h"].mean()),

            # profile SDs across bins
            "PAVD_sd": float(sub["pavd_h"].std(ddof=1)) if len(sub) > 1 else 0.0,
            "PAVDe_sd": float(sub["pavde_h"].std(ddof=1)) if len(sub) > 1 else 0.0,
            "LAVD_sd": float(sub["lavd_h"].std(ddof=1)) if len(sub) > 1 else 0.0,
            "LAVDe_sd": float(sub["lavde_h"].std(ddof=1)) if len(sub) > 1 else 0.0,

            # integrated totals over that selected top crown portion
            "PAI": float(sub["PAI_bin"].sum()),
            "PAIe": float(sub["PAIe_bin"].sum()),
            "LAI": float(sub["LAI_bin"].sum()),
            "LAIe": float(sub["LAIe_bin"].sum()),

            # totals for whole selected crown support
            "PAI_total": total_pai,
            "PAIe_total": total_paie,
            "LAI_total": total_lai,
            "LAIe_total": total_laie,
        })

    out = pd.DataFrame(rows)
    return out, crown, z_res


def make_wide_summary(best, crown_df, topfrac_df, z_res):
    """
    Convert long-form top fraction table to one-row wide output.
    """
    out = {
        "tree_id": os.path.basename(os.path.dirname(TREE_DIR)),
        "selected_scan": best["scan_name"],
        "source_file": best["file"],
        "selected_scan_tallest_h": best["tallest_h"],
        "selected_scan_LAI_cum_at_tallest": best["LAI_cum_at_tallest"],
        "selected_scan_LAIe_cum_at_tallest": best["LAIe_cum_at_tallest"],
        "z_res_inferred": z_res,

        "PAI_total_selected_scan": float(crown_df["PAI_bin"].sum()),
        "PAIe_total_selected_scan": float(crown_df["PAIe_bin"].sum()),
        "LAI_total_selected_scan": float(crown_df["LAI_bin"].sum()),
        "LAIe_total_selected_scan": float(crown_df["LAIe_bin"].sum()),

        "PAVD_mean_selected_scan": float(crown_df["pavd_h"].mean()),
        "PAVDe_mean_selected_scan": float(crown_df["pavde_h"].mean()),
        "LAVD_mean_selected_scan": float(crown_df["lavd_h"].mean()),
        "LAVDe_mean_selected_scan": float(crown_df["lavde_h"].mean()),

        "PAVD_sd_selected_scan": float(crown_df["pavd_h"].std(ddof=1)) if len(crown_df) > 1 else 0.0,
        "PAVDe_sd_selected_scan": float(crown_df["pavde_h"].std(ddof=1)) if len(crown_df) > 1 else 0.0,
        "LAVD_sd_selected_scan": float(crown_df["lavd_h"].std(ddof=1)) if len(crown_df) > 1 else 0.0,
        "LAVDe_sd_selected_scan": float(crown_df["lavde_h"].std(ddof=1)) if len(crown_df) > 1 else 0.0,

        "crown_bottom_h": float(crown_df["h"].iloc[0]),
        "crown_top_h": float(crown_df["h"].iloc[-1]),
        "n_crown_bins": int(len(crown_df)),
    }

    for _, row in topfrac_df.iterrows():
        label = row["top_fraction_label"].replace("top", "")   # e.g. "10", "20", ...

        out[f"cutoff_h_top{label}"] = row["cutoff_h_m"]
        out[f"cutoff_zmid_top{label}"] = row["cutoff_z_mid"]
        out[f"nBins_top{label}"] = row["n_bins_kept"]

        out[f"PAVD_top{label}"] = row["PAVD_mean"]
        out[f"PAVDe_top{label}"] = row["PAVDe_mean"]
        out[f"LAVD_top{label}"] = row["LAVD_mean"]
        out[f"LAVDe_top{label}"] = row["LAVDe_mean"]

        out[f"PAVD_sd_top{label}"] = row["PAVD_sd"]
        out[f"PAVDe_sd_top{label}"] = row["PAVDe_sd"]
        out[f"LAVD_sd_top{label}"] = row["LAVD_sd"]
        out[f"LAVDe_sd_top{label}"] = row["LAVDe_sd"]

        out[f"PAI_top{label}"] = row["PAI"]
        out[f"ePAI_top{label}"] = row["PAIe"]
        out[f"LAI_top{label}"] = row["LAI"]
        out[f"eLAI_top{label}"] = row["LAIe"]

    return pd.DataFrame([out])


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------
scans = load_scan_profiles(TREE_DIR, FILE_PATTERN)
best = pick_best_scan(scans)

best_df = clean_height_profile(best["df"].copy())
topfrac_df, crown_df, z_res = calc_topfrac_from_lavd(best_df, TOP_FRACS)
wide_df = make_wide_summary(best, crown_df, topfrac_df, z_res)

# save outputs
best_df.to_csv(
    os.path.join(OUT_DIR, f"{best['scan_name']}_selected_scan_full_profile_nonNegH.csv"),
    index=False
)

crown_df.to_csv(
    os.path.join(OUT_DIR, f"{best['scan_name']}_selected_scan_crown_only_nonNegH.csv"),
    index=False
)

topfrac_df.to_csv(
    os.path.join(OUT_DIR, f"{best['scan_name']}_topFrac_long_PAVD_LAVD_PAI_LAI_by_LAVD_nonNegH.csv"),
    index=False
)

wide_df.to_csv(
    os.path.join(OUT_DIR, f"{best['scan_name']}_topFrac_wide_oneRow_summary.csv"),
    index=False
)

print("Selected scan:", best["scan_name"])
print("LAI_cum at tallest non-negative height:", best["LAI_cum_at_tallest"])
print("LAIe_cum at tallest non-negative height:", best["LAIe_cum_at_tallest"])
print("Outputs written to:", OUT_DIR)
print(wide_df.T)