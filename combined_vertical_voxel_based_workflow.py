# -*- coding: utf-8 -*-
"""
Combined one-tree vertical-slicing workflow: eLAI/eLAVD + leaf angle + CI + true LAI/LAVD + 4-panel PDF.

This script DOES NOT require pre-existing Li/Flynn profile CSVs or CI CSVs.
It rebuilds the effective vertical profile and the vertical voxel CI from the
same inputs used by your previous separate scripts:

  1) batch_voxel_LiFlynn_dynamicK_1mStrata.py
  2) batch_voxel_based_Clumping_trueLAI.py
  3) plot_EVERYTHING_vertical_profiles_all_trees.py

Current version is set up for ONE TREE testing first.
After the one-tree result looks correct, set TEST_ONE_TREE = False for batch.

Key behavior
------------
- Uses 5-cm fine voxels and 1-m vertical strata.
- Calculates effective LAI/eLAVD from projected XY contact frequency with dynamic k.
- Calculates CI from the previous vertical voxel method:
      CI = ln(Pgap_obs) / ln(Pgap_rand)
- Removes CI values >= 1 before plotting, summaries, true LAI/LAVD, and CSV output.
- Removes trailing zero eLAVD rows at the top of the crown before summaries/plotting.
- Removes WAI/WAVD from outputs and plots.
- Fourth panel is a black placeholder box for point cloud visualization.

Author notes
------------
You can change SITE and TEST_TREE_ID below.
"""

import os
import glob
import time
import traceback
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

try:
    from scipy.stats import beta as beta_dist
except Exception as e:
    raise ImportError("This script needs scipy. Please install scipy first.") from e


# =============================================================================
# USER SETTINGS
# =============================================================================

SITE = "SCBI"

MERGED_ROOT = rf"E:\TLS\2025_summer\2025_{SITE}\Tree_scan\ForestGEO_{SITE}\L1_segmented"
SPLIT_ROOT  = rf"E:\TLS\2025_summer\2025_{SITE}\Tree_scan\ForestGEO_{SITE}\L1_segmented_split"

OUT_DIR = os.path.join(
    SPLIT_ROOT,
    f"L2_combined_voxel_based_{SITE}"
)
os.makedirs(OUT_DIR, exist_ok=True)

OUT_PROFILE_LONG_CSV = os.path.join(
    OUT_DIR,
    f"{SITE}_vertical_profiles_LONG.csv"
)

OUT_TOPFRAC_LONG_CSV = os.path.join(
    OUT_DIR,
    f"{SITE}_vertical_topFrac_LONG.csv"
)

OUT_TOPFRAC_WIDE_CSV = os.path.join(
    OUT_DIR,
    f"{SITE}_vertical_topFrac_WIDE.csv"
)

OUT_STATUS_CSV = os.path.join(
    OUT_DIR,
    f"{SITE}_vertical_STATUS.csv"
)

OUT_PDF = os.path.join(
    OUT_DIR,
    f"{SITE}_4panel_vertical_profiles.pdf"
)

# Start with one tree first.
# For batch processing: set TEST_ONE_TREE = False & TEST_TREE_ID = None
TEST_ONE_TREE = False 
TEST_TREE_ID = None
TREE_ID_CONTAINS = None
MAX_TREES = None

# Fine voxel size for 3D occupancy.
VOXEL_SIZE = 0.05

# Coarse vertical stratum thickness.
STRATUM_THICKNESS = 1.0

# Top fractions of cumulative eLAI/eLAVD to summarize.
TOP_FRACS = np.arange(0.1, 1.01, 0.1)
PLOT_TOP_LABEL = "top30"

# Leaf-angle columns in *_angles_class_voxels_global.asc.
ANGLE_COL_PREFERENCE = ["zen_mean", "zen_mode"]
HEIGHT_COL = "HeightAboveGround"
WEIGHT_COL = "counts"
USE_COUNTS_AS_WEIGHTS = True

# G(theta) numerical integration.
N_X = 101
N_PHI = 101
THETA_MIN_DEG = 5.0
THETA_MAX_DEG = 85.0
G_CACHE = {}

# Numerical safety.
EPS = 1e-6
H_TOL = 1e-8
LAVD_ZERO_TOL = 1e-12

# CI validity.
# Requirement: remove CI >= 1 before graphing and outputting csv.
CI_MIN_VALID = 0.05
CI_MAX_VALID_EXCLUSIVE = 1.0
REMOVE_CI_GE_ONE = True

# Plot settings.
FIGSIZE = (17.5, 8.0)
DPI = 300
ANGLE_XLIM = (0, 90)
CI_XLIM = (0, 1.0)

FIXED_LAVD_XMAX = None
GLOBAL_LAVD_PERCENTILE = 99.5
LAVD_X_BUFFER = 1.15
MIN_LAVD_XMAX = 0.10
LAVD_XMAX_ROUND_TO = 0.01

FIXED_YMAX = None
Y_AXIS_BUFFER = 1.05
YMAX_ROUND_TO = 1.0

BAR_ALPHA = 0.35
BAR_HEIGHT_FRACTION = 0.65
SMOOTH_WINDOW = 3
SMOOTH_MIN_PERIODS = 1
CURVE_LINEWIDTH = 2.8

ANGLE_BAR_COLOR = "#9370DB"
ANGLE_LINE_COLOR = "#4B0082"

ELAVD_COLOR = "#006400"
TRUE_LAVD_COLOR = "#404040"

CI_BAR_COLOR = "#6BAED6"
CI_LINE_COLOR = "#1F4E79"


# =============================================================================
# GENERAL HELPERS
# =============================================================================

def pct_to_label(frac):
    return f"top{int(round(frac * 100)):02d}"


def label_to_fraction(label):
    digits = "".join(ch for ch in str(label) if ch.isdigit())
    if not digits:
        raise ValueError(f"Cannot parse top label: {label}")
    return float(digits) / 100.0


def round_up_to(x, step):
    if step is None or step <= 0:
        return x
    return np.ceil(x / step) * step


def smooth_profile(x, window=SMOOTH_WINDOW):
    x = pd.Series(x, dtype="float64")
    if window is None or window <= 1:
        return x.to_numpy()
    if window % 2 == 0:
        window += 1
    return (
        x.rolling(window=window, center=True, min_periods=SMOOTH_MIN_PERIODS)
        .mean()
        .to_numpy()
    )


def safe_gap(p):
    if not np.isfinite(p):
        return np.nan
    return float(np.clip(p, EPS, 1.0 - EPS))


def clean_ci(ci):
    """
    CI used for all outputs/summaries/true LAVD.
    Removes CI <= 0, CI < CI_MIN_VALID, and CI >= 1.
    """
    ci = np.asarray(ci, dtype=float).copy()
    ci[~np.isfinite(ci)] = np.nan
    ci[ci < CI_MIN_VALID] = np.nan
    if REMOVE_CI_GE_ONE:
        ci[ci >= CI_MAX_VALID_EXCLUSIVE] = np.nan
    return ci


def weighted_mean(x, w):
    x = np.asarray(x, dtype=float)
    w = np.asarray(w, dtype=float)
    m = np.isfinite(x) & np.isfinite(w) & (w > 0)
    if not np.any(m):
        return np.nan
    return float(np.sum(x[m] * w[m]) / np.sum(w[m]))


def weighted_sd(x, w):
    x = np.asarray(x, dtype=float)
    w = np.asarray(w, dtype=float)
    m = np.isfinite(x) & np.isfinite(w) & (w > 0)
    x = x[m]
    w = w[m]
    if len(x) <= 1:
        return np.nan
    mu = np.sum(x * w) / np.sum(w)
    var = np.sum(w * (x - mu) ** 2) / np.sum(w)
    return float(np.sqrt(var))


def style_axis(ax):
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.0)
    ax.tick_params(axis="both", labelsize=10)


def add_cutoff_line(ax, cutoff_h):
    if np.isfinite(cutoff_h):
        ax.axhline(cutoff_h, color="black", linestyle="--", linewidth=1.5)


def fmt_val(x, digits=2):
    return f"{x:.{digits}f}" if np.isfinite(x) else "NA"


# =============================================================================
# FILE FINDING AND READING
# =============================================================================

def list_tree_ids(split_root):
    if TEST_ONE_TREE:
        return [TEST_TREE_ID]

    tree_dirs = [
        d for d in glob.glob(os.path.join(split_root, "*"))
        if os.path.isdir(d) and os.path.isdir(os.path.join(d, "cc_woodcl"))
    ]
    tree_ids = sorted([os.path.basename(d) for d in tree_dirs])

    if TREE_ID_CONTAINS:
        tree_ids = [t for t in tree_ids if TREE_ID_CONTAINS in t]

    if MAX_TREES is not None:
        tree_ids = tree_ids[:MAX_TREES]

    return tree_ids


def find_merged_tree_file(merged_root, tree_id):
    patterns = [
        os.path.join(merged_root, f"**/{tree_id}*.ptx"),
        os.path.join(merged_root, f"**/{tree_id}*.asc"),
        os.path.join(merged_root, f"**/{tree_id}*.txt"),
        os.path.join(merged_root, f"**/{tree_id}*.csv"),
        os.path.join(merged_root, f"**/{tree_id}*.xyz"),
    ]
    hits = []
    for pat in patterns:
        hits.extend(glob.glob(pat, recursive=True))

    hits = sorted([h for h in hits if os.path.isfile(h)])

    if not hits:
        raise FileNotFoundError(
            f"Could not find merged tree file for {tree_id} under:\n{merged_root}"
        )

    return hits[0]


def find_leaf_voxel_files(split_root, tree_id):
    angle_dir = os.path.join(split_root, tree_id, "cc_woodcl")
    files = sorted(glob.glob(os.path.join(angle_dir, "*_angles_class_voxels_global.asc")))
    if not files:
        raise FileNotFoundError(f"No leaf voxel files found under:\n{angle_dir}")
    return files


def read_ptx_xyz(path):
    chunks = []
    for chunk in pd.read_csv(
        path,
        sep=r"\s+",
        engine="python",
        skiprows=10,
        header=None,
        usecols=[0, 1, 2],
        names=["x", "y", "z"],
        chunksize=1_000_000,
    ):
        m = (np.abs(chunk[["x", "y", "z"]].to_numpy()) > 1e-8).any(axis=1)
        chunk = chunk.loc[m].copy()
        if not chunk.empty:
            chunks.append(chunk)

    if not chunks:
        raise ValueError(f"No XYZ data found in {path}")

    return pd.concat(chunks, ignore_index=True)


def read_ascii_xyz(path):
    try:
        df = pd.read_csv(path)
        if df.shape[1] == 1:
            raise ValueError
    except Exception:
        df = pd.read_csv(path, sep=r"\s+", engine="python")

    df.columns = [str(c).strip().lower() for c in df.columns]

    if {"x", "y", "z"}.issubset(df.columns):
        out = df[["x", "y", "z"]].copy()
    else:
        num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        if len(num_cols) < 3:
            raise ValueError(f"Could not detect x,y,z columns in {path}")
        out = df[num_cols[:3]].copy()
        out.columns = ["x", "y", "z"]

    for c in ["x", "y", "z"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")

    out = out[np.isfinite(out["x"]) & np.isfinite(out["y"]) & np.isfinite(out["z"])].copy()
    return out


def read_merged_xyz(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".ptx":
        return read_ptx_xyz(path)
    if ext in (".asc", ".txt", ".csv", ".xyz"):
        return read_ascii_xyz(path)
    raise ValueError(f"Unsupported merged point-cloud type: {path}")


def read_one_leaf_voxel_file(path):
    df = pd.read_csv(path, sep=r"\s+", engine="python")
    cols = {c.lower(): c for c in df.columns}

    needed = ["x", "y", "z", "heightaboveground", "counts"]
    for k in needed:
        if k not in cols:
            raise ValueError(f"{os.path.basename(path)} missing column: {k}")

    angle_col = None
    for c in ANGLE_COL_PREFERENCE:
        if c.lower() in cols:
            angle_col = cols[c.lower()]
            break

    if angle_col is None:
        raise ValueError(f"{os.path.basename(path)} missing zen_mean / zen_mode")

    out = pd.DataFrame({
        "x": pd.to_numeric(df[cols["x"]], errors="coerce"),
        "y": pd.to_numeric(df[cols["y"]], errors="coerce"),
        "z": pd.to_numeric(df[cols["z"]], errors="coerce"),
        "h": pd.to_numeric(df[cols["heightaboveground"]], errors="coerce"),
        "angle_deg": pd.to_numeric(df[angle_col], errors="coerce"),
        "counts": pd.to_numeric(df[cols["counts"]], errors="coerce"),
    })

    out = out[
        np.isfinite(out["x"]) &
        np.isfinite(out["y"]) &
        np.isfinite(out["z"]) &
        np.isfinite(out["h"]) &
        np.isfinite(out["angle_deg"]) &
        (out["h"] >= 0) &
        (out["angle_deg"] >= 0) &
        (out["angle_deg"] <= 90)
    ].copy()

    out["counts"] = out["counts"].fillna(1.0)
    if USE_COUNTS_AS_WEIGHTS:
        out = out[out["counts"] > 0].copy()
    else:
        out["counts"] = 1.0

    out["source_file"] = os.path.basename(path)
    return out


def read_all_leaf_voxels(paths):
    frames = []
    for p in paths:
        frames.append(read_one_leaf_voxel_file(p))
    return pd.concat(frames, ignore_index=True)


# =============================================================================
# HEIGHT AND VOXELIZATION
# =============================================================================

def estimate_z_ground_from_leaf_voxels(leaf_df):
    z0 = np.nanmedian(leaf_df["z"].to_numpy() - leaf_df["h"].to_numpy())
    return float(z0)


def clean_heights(df):
    df = df.copy()
    df = df[df["h"] >= -H_TOL].copy().reset_index(drop=True)
    if len(df):
        df.loc[df["h"] < 0, "h"] = 0.0
    return df


def voxelize_xyz(df_xyz, voxel_size, z_ground):
    df = df_xyz.copy()
    df["h"] = df["z"] - z_ground
    df = clean_heights(df)
    if df.empty:
        return df
    df["ix"] = np.floor(df["x"] / voxel_size).astype(int)
    df["iy"] = np.floor(df["y"] / voxel_size).astype(int)
    df["iz"] = np.floor(df["h"] / voxel_size).astype(int)
    df["stratum_id"] = np.floor(df["h"] / STRATUM_THICKNESS).astype(int)
    return df


def voxelize_leaf_voxels(leaf_df, voxel_size):
    df = leaf_df.copy()
    df = clean_heights(df)
    if df.empty:
        return df
    df["ix"] = np.floor(df["x"] / voxel_size).astype(int)
    df["iy"] = np.floor(df["y"] / voxel_size).astype(int)
    df["iz"] = np.floor(df["h"] / voxel_size).astype(int)
    df["stratum_id"] = np.floor(df["h"] / STRATUM_THICKNESS).astype(int)
    return df


def dedup_leaf_for_ci(leaf_vox):
    return (
        leaf_vox
        .drop_duplicates(["ix", "iy", "iz"])
        [["ix", "iy", "iz", "stratum_id"]]
        .copy()
    )


# =============================================================================
# LAD, MU/NU, AND G(THETA)
# =============================================================================

def beta_weighted_params(angles_deg, weights):
    a = np.asarray(angles_deg, dtype=float)
    w = np.asarray(weights, dtype=float)
    m = np.isfinite(a) & np.isfinite(w) & (w > 0) & (a >= 0) & (a <= 90)
    a = a[m]
    w = w[m]

    if len(a) < 5 or w.sum() <= 0:
        return np.nan, np.nan

    x = np.clip(a / 90.0, EPS, 1 - EPS)
    W = w.sum()
    mu = np.sum(w * x) / W
    var = np.sum(w * (x - mu) ** 2) / W
    max_var = mu * (1 - mu)

    if not (0 < mu < 1) or not (var > 0) or var >= max_var:
        return np.nan, np.nan

    nu = max_var / var - 1.0
    if nu <= 0:
        return np.nan, np.nan

    return float(mu), float(nu)


def g_function_from_beta(mu, nu, theta_deg, n_x=N_X, n_phi=N_PHI):
    if not np.isfinite(mu) or not np.isfinite(nu) or not np.isfinite(theta_deg):
        return np.nan
    if nu <= 0:
        return np.nan

    cache_key = (
        round(float(mu), 3),
        round(float(nu), 2),
        round(float(theta_deg), 1),
        n_x,
        n_phi,
    )
    if cache_key in G_CACHE:
        return G_CACHE[cache_key]

    a = mu * nu
    b = (1 - mu) * nu
    if a <= 0 or b <= 0:
        return np.nan

    x = np.linspace(EPS, 1 - EPS, n_x)
    alpha = x * (np.pi / 2.0)
    pdf_x = beta_dist.pdf(x, a=a, b=b)

    theta = np.deg2rad(theta_deg)
    beam = np.array([np.sin(theta), 0.0, np.cos(theta)])
    phi = np.linspace(0.0, 2 * np.pi, n_phi, endpoint=False)

    Abar = np.empty_like(alpha)
    for i, al in enumerate(alpha):
        nx = np.sin(al) * np.cos(phi)
        ny = np.sin(al) * np.sin(phi)
        nz = np.cos(al) * np.ones_like(phi)
        dots = beam[0] * nx + beam[1] * ny + beam[2] * nz
        Abar[i] = np.mean(np.abs(dots))

    num = np.trapz(Abar * pdf_x, x)
    den = np.trapz(pdf_x, x)
    if den <= 0:
        return np.nan

    out = float(num / den)
    G_CACHE[cache_key] = out
    return out


# =============================================================================
# EFFECTIVE LAI/LAVD PROFILE
# =============================================================================

def count_unique_xy_by_stratum(df):
    if df.empty:
        return pd.Series(dtype=int)
    return df.drop_duplicates(["stratum_id", "ix", "iy"]).groupby("stratum_id").size()


def build_effective_profile(all_vox, leaf_vox):
    """
    Rebuilds your previous Li/Flynn-style effective profile.
    Uses contact frequency of unique projected XY cells by stratum.
    Does NOT include WAI/WAVD in the returned profile.
    """
    xy_all = np.unique(all_vox[["ix", "iy"]].to_numpy(), axis=0)
    n_xy_total = len(xy_all)
    if n_xy_total == 0:
        raise ValueError("Projected crown area is empty.")

    min_stratum = int(min(all_vox["stratum_id"].min(), leaf_vox["stratum_id"].min()))
    max_stratum = int(max(all_vox["stratum_id"].max(), leaf_vox["stratum_id"].max()))

    filled_all_by_stratum = count_unique_xy_by_stratum(all_vox)
    filled_leaf_by_stratum = count_unique_xy_by_stratum(leaf_vox)

    mu_tree, nu_tree = beta_weighted_params(leaf_vox["angle_deg"], leaf_vox["counts"])
    theta_tree = weighted_mean(leaf_vox["angle_deg"], leaf_vox["counts"])
    if np.isfinite(theta_tree):
        theta_tree = float(np.clip(theta_tree, THETA_MIN_DEG, THETA_MAX_DEG))

    G_tree = g_function_from_beta(mu_tree, nu_tree, theta_tree) if np.isfinite(theta_tree) else np.nan
    k_tree = (
        np.cos(np.deg2rad(theta_tree)) / G_tree
        if np.isfinite(G_tree) and G_tree > 0
        else np.nan
    )

    rows = []
    pai_cum = 0.0
    elai_cum = 0.0

    for sid in range(min_stratum, max_stratum + 1):
        z_min = sid * STRATUM_THICKNESS
        z_mid = z_min + STRATUM_THICKNESS / 2.0
        z_max = z_min + STRATUM_THICKNESS

        filled_xy_all = int(filled_all_by_stratum.get(sid, 0))
        filled_xy_leaf = int(filled_leaf_by_stratum.get(sid, 0))

        cf_all = filled_xy_all / n_xy_total
        cf_leaf = filled_xy_leaf / n_xy_total

        leaf_slice = leaf_vox[leaf_vox["stratum_id"] == sid]

        theta_layer = weighted_mean(leaf_slice["angle_deg"], leaf_slice["counts"])
        mu_layer, nu_layer = beta_weighted_params(leaf_slice["angle_deg"], leaf_slice["counts"])

        if not np.isfinite(theta_layer):
            theta_layer = theta_tree

        if not np.isfinite(mu_layer) or not np.isfinite(nu_layer):
            mu_layer, nu_layer = mu_tree, nu_tree

        if np.isfinite(theta_layer):
            theta_layer = float(np.clip(theta_layer, THETA_MIN_DEG, THETA_MAX_DEG))

        G_layer = (
            g_function_from_beta(mu_layer, nu_layer, theta_layer)
            if np.isfinite(theta_layer) and np.isfinite(mu_layer) and np.isfinite(nu_layer)
            else np.nan
        )

        k_layer = (
            np.cos(np.deg2rad(theta_layer)) / G_layer
            if np.isfinite(theta_layer) and np.isfinite(G_layer) and G_layer > 0
            else k_tree
        )

        if not np.isfinite(k_layer) or k_layer <= 0:
            PAI_layer = np.nan
            eLAI_layer = np.nan
            PAVD_h = np.nan
            eLAVD_h = np.nan
        else:
            PAI_layer = k_layer * cf_all
            eLAI_layer = k_layer * cf_leaf
            PAVD_h = PAI_layer / STRATUM_THICKNESS
            eLAVD_h = eLAI_layer / STRATUM_THICKNESS

        if np.isfinite(PAI_layer):
            pai_cum += PAI_layer
        if np.isfinite(eLAI_layer):
            elai_cum += eLAI_layer

        rows.append({
            "stratum_id": sid,
            "z_min": z_min,
            "z_mid": z_mid,
            "z_max": z_max,
            "h": z_mid,
            "height_m": z_mid,
            "voxel_size_m": VOXEL_SIZE,
            "stratum_thickness_m": STRATUM_THICKNESS,
            "n_xy_total": n_xy_total,
            "filled_xy_all": filled_xy_all,
            "filled_xy_leaf": filled_xy_leaf,
            "contact_freq_all": cf_all,
            "contact_freq_leaf": cf_leaf,
            "theta_layer_deg": theta_layer,
            "mu": mu_layer,
            "nu": nu_layer,
            "G_theta": G_layer,
            "k_corr": k_layer,
            "PAI_layer": PAI_layer,
            "eLAI_layer": eLAI_layer,
            "PAVD_h": PAVD_h,
            "eLAVD_h": eLAVD_h,
            "PAI_cum": pai_cum,
            "eLAI_cum": elai_cum,
        })

    prof = pd.DataFrame(rows).sort_values("height_m").reset_index(drop=True)
    prof, trim_info = trim_top_zero_elavd_rows(prof)
    return prof, mu_tree, nu_tree, theta_tree, G_tree, k_tree, trim_info


def trim_top_zero_elavd_rows(profile):
    df = profile.sort_values("height_m").reset_index(drop=True).copy()
    x = pd.to_numeric(df["eLAVD_h"], errors="coerce").to_numpy(dtype=float)
    nonzero = np.isfinite(x) & (x > LAVD_ZERO_TOL)

    if not np.any(nonzero):
        raise RuntimeError("All eLAVD layers are zero or invalid; cannot define crown top.")

    last_nonzero_pos = np.where(nonzero)[0].max()
    original_n = len(df)
    original_max_h = float(np.nanmax(df["height_m"]))

    clean = df.iloc[:last_nonzero_pos + 1].copy().reset_index(drop=True)
    clean["profile_row_id"] = np.arange(len(clean))

    clean_max_h = float(np.nanmax(clean["height_m"]))

    trim_info = {
        "original_n_profile_rows": int(original_n),
        "clean_n_profile_rows": int(len(clean)),
        "n_top_zero_eLAVD_rows_removed": int(original_n - len(clean)),
        "original_max_h_m": original_max_h,
        "clean_max_h_m": clean_max_h,
        "clean_canopy_top_edge_m": clean_max_h + STRATUM_THICKNESS / 2.0,
    }

    return clean, trim_info


# =============================================================================
# LEAF ANGLE PROFILE
# =============================================================================

def build_leaf_angle_profile(leaf_vox, profile_clean):
    df = leaf_vox.copy()
    valid_sids = set(profile_clean["stratum_id"].dropna().astype(int).unique())
    df = df[df["stratum_id"].isin(valid_sids)].copy()

    rows = []
    for sid, g in df.groupby("stratum_id"):
        mean_angle = weighted_mean(g["angle_deg"], g["counts"])
        sd_angle = weighted_sd(g["angle_deg"], g["counts"])
        n = int(len(g))
        se_angle = sd_angle / np.sqrt(n) if n > 1 and np.isfinite(sd_angle) else np.nan
        mu, nu = beta_weighted_params(g["angle_deg"], g["counts"])

        rows.append({
            "stratum_id": int(sid),
            "leaf_angle_mean": mean_angle,
            "leaf_angle_sd": sd_angle,
            "leaf_angle_se": se_angle,
            "n_angle_voxels": n,
            "angle_weight_sum": float(g["counts"].sum()),
            "mu_angle_profile": mu,
            "nu_angle_profile": nu,
        })

    angle_prof = pd.DataFrame(rows)
    if angle_prof.empty:
        return angle_prof

    return angle_prof.sort_values("stratum_id").reset_index(drop=True)


def leaf_angle_stats_above_cutoff(leaf_vox, cutoff_h):
    empty = {
        "leaf_angle_mean_top": np.nan,
        "leaf_angle_sd_top": np.nan,
        "leaf_angle_se_top": np.nan,
        "leaf_angle_n_top": 0,
        "leaf_angle_weight_sum_top": 0.0,
        "mu_top": np.nan,
        "nu_top": np.nan,
    }

    if not np.isfinite(cutoff_h):
        return empty

    sub = leaf_vox[leaf_vox["h"] >= cutoff_h].copy()
    if sub.empty:
        return empty

    mean_angle = weighted_mean(sub["angle_deg"], sub["counts"])
    sd_angle = weighted_sd(sub["angle_deg"], sub["counts"])
    n = int(len(sub))
    se_angle = sd_angle / np.sqrt(n) if n > 1 and np.isfinite(sd_angle) else np.nan
    mu, nu = beta_weighted_params(sub["angle_deg"], sub["counts"])

    return {
        "leaf_angle_mean_top": mean_angle,
        "leaf_angle_sd_top": sd_angle,
        "leaf_angle_se_top": se_angle,
        "leaf_angle_n_top": n,
        "leaf_angle_weight_sum_top": float(sub["counts"].sum()),
        "mu_top": mu,
        "nu_top": nu,
    }


# =============================================================================
# CI: CURRENT VERTICAL VOXEL METHOD
# =============================================================================

def compute_ci_vert_for_profile(profile, leaf_vox_for_ci):
    rows = []

    for _, r in profile.iterrows():
        sid = int(r["stratum_id"])
        n_xy_total = r["n_xy_total"]
        filled_xy_leaf = r["filled_xy_leaf"]
        z_min = r["z_min"]
        z_max = r["z_max"]

        sub = leaf_vox_for_ci[leaf_vox_for_ci["stratum_id"] == sid].copy()
        n_leaf_voxels = int(len(sub))

        if not np.isfinite(n_xy_total) or n_xy_total <= 0:
            ci_raw = np.nan
            Pgap_obs = np.nan
            Pgap_rand = np.nan
            p_vox = np.nan
            n_fine_z_layers = np.nan
        else:
            if np.isfinite(z_min) and np.isfinite(z_max) and z_max > z_min:
                n_fine_z_layers = int(np.ceil((z_max - z_min) / VOXEL_SIZE))
            else:
                n_fine_z_layers = int(np.ceil(STRATUM_THICKNESS / VOXEL_SIZE))

            n_fine_z_layers = max(n_fine_z_layers, 1)
            n_possible_voxels = float(n_xy_total * n_fine_z_layers)
            p_vox = float(np.clip(n_leaf_voxels / n_possible_voxels, 0.0, 1.0))

            Pgap_obs = safe_gap(1.0 - (filled_xy_leaf / n_xy_total))
            Pgap_rand = safe_gap((1.0 - p_vox) ** n_fine_z_layers)

            if (
                np.isfinite(Pgap_obs) and
                np.isfinite(Pgap_rand) and
                0 < Pgap_obs < 1 and
                0 < Pgap_rand < 1
            ):
                ci_raw = float(np.log(Pgap_obs) / np.log(Pgap_rand))
            else:
                ci_raw = np.nan

        ci_clean = clean_ci([ci_raw])[0]

        rows.append({
            "stratum_id": sid,
            "n_leaf_voxels_CI": n_leaf_voxels,
            "n_fine_z_layers_CI": n_fine_z_layers,
            "p_voxel_leaf_CI": p_vox,
            "Pgap_obs_CI": Pgap_obs,
            "Pgap_rand_CI": Pgap_rand,
            # Output only cleaned CI. Raw CI >=1 is not exported.
            "CI": ci_clean,
        })

    return pd.DataFrame(rows)


# =============================================================================
# TOP FRACTION SUMMARIES
# =============================================================================

def compute_top_layer_fractions(profile, top_fraction):
    df = profile.copy().sort_values("height_m", ascending=False).reset_index(drop=True)

    basis = pd.to_numeric(df["eLAI_layer"], errors="coerce").to_numpy(dtype=float)
    basis[~np.isfinite(basis)] = 0
    basis[basis < 0] = 0

    total = float(np.nansum(basis))
    if total <= 0:
        raise RuntimeError("Total eLAI is zero; cannot calculate top fractions.")

    target = top_fraction * total
    cum_before = np.concatenate([[0.0], np.cumsum(basis)[:-1]])
    cum_after = np.cumsum(basis)

    frac_layer = np.zeros(len(df), dtype=float)

    for i in range(len(df)):
        layer_area = basis[i]
        if cum_before[i] >= target:
            frac_layer[i] = 0.0
        elif cum_after[i] <= target:
            frac_layer[i] = 1.0
        else:
            frac_layer[i] = (target - cum_before[i]) / layer_area if layer_area > 0 else 0.0

    frac_layer = np.clip(frac_layer, 0, 1)

    crossing = np.where((cum_after >= target) & (basis > 0))[0]
    if len(crossing) == 0:
        cutoff_h = np.nan
    else:
        cross_i = int(crossing[0])
        cross_mid = float(df.loc[cross_i, "height_m"])
        cross_top_edge = cross_mid + STRATUM_THICKNESS / 2.0
        cutoff_h = cross_top_edge - float(frac_layer[cross_i]) * STRATUM_THICKNESS

    out = df[["profile_row_id", "stratum_id", "height_m"]].copy()
    out["top_layer_fraction"] = frac_layer
    out = out.sort_values("height_m").reset_index(drop=True)

    return out, cutoff_h, total, target


def summarize_density_in_top(profile_desc, density_col, top_layer_fraction):
    x = pd.to_numeric(profile_desc[density_col], errors="coerce").to_numpy(dtype=float)
    w = top_layer_fraction * STRATUM_THICKNESS
    mean_x = weighted_mean(x, w)
    sd_x = weighted_sd(x, w)
    n = int(np.sum(np.isfinite(x) & np.isfinite(w) & (w > 0)))
    se_x = sd_x / np.sqrt(n) if n > 1 and np.isfinite(sd_x) else np.nan
    return mean_x, sd_x, se_x, n


def summarize_top_fraction(profile, leaf_vox, top_fraction):
    label = pct_to_label(top_fraction)
    top_layer_df, cutoff_h, total_eLAI, target_eLAI = compute_top_layer_fractions(profile, top_fraction)

    df = pd.merge(
        profile.copy(),
        top_layer_df[["profile_row_id", "top_layer_fraction"]],
        on="profile_row_id",
        how="left",
    )
    df["top_layer_fraction"] = df["top_layer_fraction"].fillna(0.0)

    desc = df.sort_values("height_m", ascending=False).reset_index(drop=True)
    f = desc["top_layer_fraction"].to_numpy(dtype=float)

    w_lai = desc["eLAI_layer"].to_numpy(dtype=float) * f
    eLAI_top = float(np.nansum(w_lai))
    PAI_top = float(np.nansum(desc["PAI_layer"].to_numpy(dtype=float) * f))

    CI = desc["CI"].to_numpy(dtype=float)
    CI_mean = weighted_mean(CI, w_lai)
    CI_sd = weighted_sd(CI, w_lai)
    CI_n = int(np.sum(np.isfinite(CI) & np.isfinite(w_lai) & (w_lai > 0)))

    if np.isfinite(CI_mean) and CI_mean > 0:
        LAIvert_top = eLAI_top / CI_mean
    else:
        LAIvert_top = np.nan

    LAIvert_top_sum_layers = float(np.nansum(desc["LAIvert_layer"].to_numpy(dtype=float) * f))

    eLAVD_mean, eLAVD_sd, eLAVD_se, eLAVD_n = summarize_density_in_top(desc, "eLAVD_h", f)
    LAVDvert_mean, LAVDvert_sd, LAVDvert_se, LAVDvert_n = summarize_density_in_top(desc, "LAVDvert_h", f)

    angle_stats = leaf_angle_stats_above_cutoff(leaf_vox, cutoff_h)

    summary = {
        "top_fraction_label": label,
        "top_fraction": float(top_fraction),
        "cutoff_h_m": cutoff_h,
        "top_fraction_basis": "eLAI/eLAVD",
        "eLAI_total_clean": total_eLAI,
        "eLAI_target": target_eLAI,
        "eLAI_top": eLAI_top,
        "PAI_top": PAI_top,

        "eLAVD_mean_top": eLAVD_mean,
        "eLAVD_sd_top": eLAVD_sd,
        "eLAVD_se_top": eLAVD_se,
        "eLAVD_n_layers_top": eLAVD_n,

        "CI_mean_top": CI_mean,
        "CI_sd_top": CI_sd,
        "CI_n_layers_valid_top": CI_n,

        "LAIvert_top": LAIvert_top,
        "LAIvert_top_sum_layers": LAIvert_top_sum_layers,

        "LAVDvert_mean_top": LAVDvert_mean,
        "LAVDvert_sd_top": LAVDvert_sd,
        "LAVDvert_se_top": LAVDvert_se,
        "LAVDvert_n_layers_top": LAVDvert_n,

        "n_top_layers_any": int(np.sum(f > 0)),
        "n_top_layers_full": int(np.sum(np.isclose(f, 1.0))),
        "n_top_layers_partial": int(np.sum((f > 0) & (f < 1))),
        **angle_stats,
    }

    layer_frac_cols = top_layer_df[["profile_row_id", "top_layer_fraction"]].copy()
    layer_frac_cols = layer_frac_cols.rename(columns={"top_layer_fraction": f"{label}_layer_fraction"})

    return summary, layer_frac_cols


def make_wide_row(tree_id, profile, top_summaries, static_info):
    row = {"tree_id": tree_id}
    row.update(static_info)

    row["eLAI_total_clean"] = float(profile["eLAI_layer"].sum(skipna=True))
    row["LAIvert_total_clean"] = float(profile["LAIvert_layer"].sum(skipna=True))
    row["eLAVD_mean_clean"] = float(profile["eLAVD_h"].mean(skipna=True))
    row["LAVDvert_mean_clean"] = float(profile["LAVDvert_h"].mean(skipna=True))
    row["CI_mean_clean"] = float(profile["CI"].mean(skipna=True))

    keys_to_widen = [
        "cutoff_h_m",
        "eLAI_top",
        "PAI_top",

        "leaf_angle_mean_top",
        "leaf_angle_sd_top",
        "leaf_angle_se_top",
        "leaf_angle_n_top",
        "leaf_angle_weight_sum_top",
        "mu_top",
        "nu_top",

        "eLAVD_mean_top",
        "eLAVD_sd_top",
        "eLAVD_se_top",
        "eLAVD_n_layers_top",

        "CI_mean_top",
        "CI_sd_top",
        "CI_n_layers_valid_top",
        "LAIvert_top",
        "LAIvert_top_sum_layers",
        "LAVDvert_mean_top",
        "LAVDvert_sd_top",
        "LAVDvert_se_top",

        "n_top_layers_any",
        "n_top_layers_full",
        "n_top_layers_partial",
    ]

    for s in top_summaries:
        lbl = s["top_fraction_label"]
        for k in keys_to_widen:
            row[f"{lbl}_{k}"] = s.get(k, np.nan)

    return row


# =============================================================================
# AXIS LIMITS AND PLOTTING
# =============================================================================

def calculate_sitewide_lavd_xlim(profile_dfs):
    if FIXED_LAVD_XMAX is not None:
        return (0, float(FIXED_LAVD_XMAX))

    values = []
    for df in profile_dfs:
        for c in ["eLAVD_h", "LAVDvert_h"]:
            x = pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=float)
            x = x[np.isfinite(x)]
            x = x[x >= 0]
            values.extend(x.tolist())

    if len(values) == 0:
        xmax = MIN_LAVD_XMAX
    else:
        xmax = np.nanpercentile(values, GLOBAL_LAVD_PERCENTILE) * LAVD_X_BUFFER

    xmax = max(xmax, MIN_LAVD_XMAX)
    xmax = round_up_to(xmax, LAVD_XMAX_ROUND_TO)
    return (0, float(xmax))


def calculate_sitewide_ylim(profile_dfs):
    if FIXED_YMAX is not None:
        return (0, float(FIXED_YMAX))

    ymax_values = []
    for df in profile_dfs:
        y = pd.to_numeric(df["height_m"], errors="coerce").to_numpy(dtype=float)
        y = y[np.isfinite(y)]
        if len(y):
            ymax_values.append(np.nanmax(y))

    ymax = np.nanmax(ymax_values) * Y_AXIS_BUFFER
    ymax = round_up_to(ymax, YMAX_ROUND_TO)
    return (0, float(ymax))


def plot_black_placeholder(ax, ylim):
    ax.set_facecolor("black")
    ax.set_xlim(0, 1)
    ax.set_ylim(ylim)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_title("Point cloud placeholder", fontsize=13, fontweight="bold")
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(1.0)


def plot_one_tree(axes, df, tree_id, plot_summary, lavd_xlim, ylim):
    ax1, ax2, ax3, ax4 = axes

    y = df["height_m"].to_numpy(dtype=float)
    bar_height = STRATUM_THICKNESS * BAR_HEIGHT_FRACTION

    angle = df["leaf_angle_mean"].to_numpy(dtype=float)
    elavd = df["eLAVD_h"].to_numpy(dtype=float)
    lavdvert = df["LAVDvert_h"].to_numpy(dtype=float)
    civert = df["CI"].to_numpy(dtype=float)

    cutoff_h = plot_summary.get("cutoff_h_m", np.nan)
    top_pct_text = f"Top {plot_summary.get('top_fraction', np.nan) * 100:.0f}%"

    # Panel 1: leaf angle
    ax1.barh(
        y, angle, height=bar_height, color=ANGLE_BAR_COLOR,
        alpha=BAR_ALPHA, edgecolor="none", label="_nolegend_"
    )
    ax1.plot(
        smooth_profile(angle), y, color=ANGLE_LINE_COLOR,
        linewidth=CURVE_LINEWIDTH, label="MLA"
    )
    add_cutoff_line(ax1, cutoff_h)

    if np.isfinite(cutoff_h):
        ax1.text(
            ANGLE_XLIM[1] * 0.97,
            cutoff_h + (ylim[1] - ylim[0]) * 0.02,
            f"{top_pct_text} MLA = {fmt_val(plot_summary.get('leaf_angle_mean_top', np.nan), 1)}°\n"
            f"μ = {fmt_val(plot_summary.get('mu_top', np.nan), 2)}",
            ha="right", va="bottom", fontsize=9,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.78, boxstyle="round,pad=0.25")
        )

    ax1.set_xlim(ANGLE_XLIM)
    ax1.set_ylim(ylim)
    ax1.set_xlabel("Mean leaf angle (°)", fontsize=13)
    ax1.set_ylabel("Canopy height (m)", fontsize=14)
    ax1.set_title("Leaf angle", fontsize=13, fontweight="bold")
    ax1.legend(frameon=False, loc="upper right", fontsize=10)
    style_axis(ax1)

    # Panel 2: eLAVD vs true LAVD from CI
    ax2.barh(
        y, elavd, height=bar_height, color=ELAVD_COLOR,
        alpha=BAR_ALPHA, edgecolor="none", label="_nolegend_"
    )
    ax2.barh(
        y, lavdvert, height=bar_height, color=TRUE_LAVD_COLOR,
        alpha=BAR_ALPHA, edgecolor="none", label="_nolegend_"
    )
    ax2.plot(smooth_profile(elavd), y, color=ELAVD_COLOR,
             linewidth=CURVE_LINEWIDTH, label="eLAVD")
    ax2.plot(smooth_profile(lavdvert), y, color=TRUE_LAVD_COLOR,
             linewidth=CURVE_LINEWIDTH, label="true LAVD")

    add_cutoff_line(ax2, cutoff_h)

    if np.isfinite(cutoff_h):
        ax2.text(
            lavd_xlim[1] * 0.97,
            cutoff_h + (ylim[1] - ylim[0]) * 0.02,
            f"{top_pct_text}\neLAI = {fmt_val(plot_summary.get('eLAI_top', np.nan))}\n"
            f"true LAI = {fmt_val(plot_summary.get('LAIvert_top', np.nan))}",
            ha="right", va="bottom", fontsize=9,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.78, boxstyle="round,pad=0.25")
        )

    ax2.set_xlim(lavd_xlim)
    ax2.set_ylim(ylim)
    ax2.set_xlabel(r"Area density (m$^2$ m$^{-3}$)", fontsize=13)
    ax2.set_title("eLAVD vs true LAVD", fontsize=13, fontweight="bold")
    ax2.legend(frameon=False, loc="upper right", fontsize=10)
    style_axis(ax2)

    # Panel 3: CI
    ax3.barh(
        y, civert, height=bar_height, color=CI_BAR_COLOR,
        alpha=BAR_ALPHA, edgecolor="none", label="_nolegend_"
    )
    ax3.plot(smooth_profile(civert), y, color=CI_LINE_COLOR,
             linewidth=CURVE_LINEWIDTH, label="CI")
    add_cutoff_line(ax3, cutoff_h)

    if np.isfinite(cutoff_h):
        ax3.text(
            CI_XLIM[1] * 0.97,
            cutoff_h + (ylim[1] - ylim[0]) * 0.02,
            f"{top_pct_text} CI = {fmt_val(plot_summary.get('CI_mean_top', np.nan))}",
            ha="right", va="bottom", fontsize=9,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.78, boxstyle="round,pad=0.25")
        )

    ax3.set_xlim(CI_XLIM)
    ax3.set_ylim(ylim)
    ax3.set_xlabel("Clumping index", fontsize=13)
    ax3.set_title("Clumping index", fontsize=13, fontweight="bold")
    ax3.legend(frameon=False, loc="upper right", fontsize=10)
    style_axis(ax3)

    # Panel 4: black placeholder
    plot_black_placeholder(ax4, ylim)

    for ax in [ax2, ax3, ax4]:
        ax.tick_params(labelleft=False)

    fig = ax1.figure
    fig.suptitle(tree_id, fontsize=15, fontweight="bold", y=0.995)


# =============================================================================
# PROCESS ONE TREE
# =============================================================================

def process_one_tree(tree_id):
    start = time.time()

    merged_file = find_merged_tree_file(MERGED_ROOT, tree_id)
    leaf_files = find_leaf_voxel_files(SPLIT_ROOT, tree_id)

    xyz_all = read_merged_xyz(merged_file)
    leaf_df = read_all_leaf_voxels(leaf_files)

    z_ground = estimate_z_ground_from_leaf_voxels(leaf_df)

    all_vox = voxelize_xyz(xyz_all, VOXEL_SIZE, z_ground)
    leaf_vox = voxelize_leaf_voxels(leaf_df, VOXEL_SIZE)

    if all_vox.empty:
        raise RuntimeError("All-point voxel table is empty after height filtering.")
    if leaf_vox.empty:
        raise RuntimeError("Leaf voxel table is empty after height filtering.")

    all_vox = all_vox[["ix", "iy", "iz", "stratum_id", "x", "y", "z", "h"]].copy()
    leaf_vox = leaf_vox[["ix", "iy", "iz", "stratum_id", "x", "y", "z", "h", "angle_deg", "counts", "source_file"]].copy()

    # 1. Effective eLAI/eLAVD profile.
    prof, mu_tree, nu_tree, theta_tree, G_tree, k_tree, trim_info = build_effective_profile(all_vox, leaf_vox)

    # 2. Leaf angle profile.
    angle_prof = build_leaf_angle_profile(leaf_vox, prof)
    prof = pd.merge(prof, angle_prof, on="stratum_id", how="left")

    # 3. CI.
    leaf_for_ci = dedup_leaf_for_ci(leaf_vox)
    civert_df = compute_ci_vert_for_profile(prof, leaf_for_ci)
    prof = pd.merge(prof, civert_df, on="stratum_id", how="left")

    # 4. True LAI/LAVD using cleaned CI only.
    ci = prof["CI"].to_numpy(dtype=float)
    eLAI = prof["eLAI_layer"].to_numpy(dtype=float)
    eLAVD = prof["eLAVD_h"].to_numpy(dtype=float)

    prof["LAIvert_layer"] = np.where(np.isfinite(ci) & (ci > 0), eLAI / ci, np.nan)
    prof["LAVDvert_h"] = np.where(np.isfinite(ci) & (ci > 0), eLAVD / ci, np.nan)

    # 5. Top-fraction summaries.
    top_summaries = []
    for frac in TOP_FRACS:
        summary, layer_frac_df = summarize_top_fraction(prof, leaf_vox, float(frac))
        top_summaries.append(summary)
        prof = pd.merge(prof, layer_frac_df, on="profile_row_id", how="left")
        col = f"{summary['top_fraction_label']}_layer_fraction"
        prof[col] = prof[col].fillna(0.0)

    plot_matches = [s for s in top_summaries if s["top_fraction_label"] == PLOT_TOP_LABEL]
    plot_summary = plot_matches[0] if plot_matches else top_summaries[0]

    static_info = {
        "merged_file": merged_file,
        "n_leaf_files": len(leaf_files),
        "n_all_points": int(len(xyz_all)),
        "n_leaf_rows_input": int(len(leaf_df)),
        "n_leaf_vox_rows_all_scans": int(len(leaf_vox)),
        "n_leaf_vox_rows_dedup_for_CI": int(len(leaf_for_ci)),
        "z_ground_est_m": z_ground,
        "mu_tree": mu_tree,
        "nu_tree": nu_tree,
        "theta_tree_deg": theta_tree,
        "G_tree": G_tree,
        "k_tree": k_tree,
        **trim_info,
    }

    wide_row = make_wide_row(tree_id, prof, top_summaries, static_info)

    status = {
        "tree_id": tree_id,
        "status": "OK",
        "message": "",
        "elapsed_sec": time.time() - start,
        **static_info,
        "plot_top_label": plot_summary["top_fraction_label"],
        "plot_cutoff_h_m": plot_summary["cutoff_h_m"],
        "plot_CI_mean_top": plot_summary["CI_mean_top"],
        "plot_eLAI_top": plot_summary["eLAI_top"],
        "plot_LAIvert_top": plot_summary["LAIvert_top"],
    }

    prof.insert(0, "tree_id", tree_id)
    top_long = pd.DataFrame(top_summaries)
    top_long.insert(0, "tree_id", tree_id)

    return prof, top_long, wide_row, plot_summary, status


# =============================================================================
# MAIN
# =============================================================================

def main():
    tree_ids = list_tree_ids(SPLIT_ROOT)
    if not tree_ids:
        raise RuntimeError("No tree IDs found.")

    print(f"Site: {SITE}")
    print(f"Tree count to process: {len(tree_ids)}")
    print(f"Output directory:\n{OUT_DIR}")
    print(f"VOXEL_SIZE = {VOXEL_SIZE} m")
    print(f"STRATUM_THICKNESS = {STRATUM_THICKNESS} m")
    print(f"Top fractions = {[pct_to_label(f) for f in TOP_FRACS]}")
    print("Angular CI removed. WAI/WAVD removed. CI >= 1 removed before all summaries/plots/outputs.")

    profile_dfs = []
    top_long_dfs = []
    wide_rows = []
    status_rows = []
    plot_info = []

    for i, tree_id in enumerate(tree_ids, start=1):
        print(f"\n[{i}/{len(tree_ids)}] Processing {tree_id}")

        try:
            prof, top_long, wide_row, plot_summary, status = process_one_tree(tree_id)
            profile_dfs.append(prof)
            top_long_dfs.append(top_long)
            wide_rows.append(wide_row)
            status_rows.append(status)
            plot_info.append((prof, plot_summary))

            print(
                f"    OK | {PLOT_TOP_LABEL} eLAI={plot_summary['eLAI_top']:.3f}, "
                f"CI={plot_summary['CI_mean_top']:.3f}, "
                f"LAIvert={plot_summary['LAIvert_top']:.3f}, "
                f"removed top rows={status['n_top_zero_eLAVD_rows_removed']}"
            )

        except Exception as e:
            msg = str(e)
            status_rows.append({
                "tree_id": tree_id,
                "status": "FAIL",
                "message": msg,
                "elapsed_sec": np.nan,
            })
            print(f"    [WARN] {tree_id}: {msg}")
            print(traceback.format_exc())

        # Save progress after each tree.
        pd.DataFrame(status_rows).to_csv(OUT_STATUS_CSV, index=False)
        if profile_dfs:
            pd.concat(profile_dfs, ignore_index=True).to_csv(OUT_PROFILE_LONG_CSV, index=False)
        if top_long_dfs:
            pd.concat(top_long_dfs, ignore_index=True).to_csv(OUT_TOPFRAC_LONG_CSV, index=False)
        if wide_rows:
            pd.DataFrame(wide_rows).to_csv(OUT_TOPFRAC_WIDE_CSV, index=False)

    if not profile_dfs:
        raise RuntimeError("No trees processed successfully.")

    lavd_xlim = calculate_sitewide_lavd_xlim(profile_dfs)
    site_ylim = calculate_sitewide_ylim(profile_dfs)

    print(f"\nWriting PDF:\n{OUT_PDF}")
    print(f"Shared LAVD x-axis: {lavd_xlim}")
    print(f"Shared y-axis: {site_ylim}")

    with PdfPages(OUT_PDF) as pdf:
        for i, (prof, plot_summary) in enumerate(plot_info, start=1):
            tree_id = prof["tree_id"].iloc[0]
            print(f"[{i}/{len(plot_info)}] Plotting {tree_id}")

            fig, axes = plt.subplots(
                ncols=4,
                nrows=1,
                figsize=FIGSIZE,
                dpi=DPI,
                sharey=True,
            )

            plot_one_tree(
                axes=axes,
                df=prof,
                tree_id=tree_id,
                plot_summary=plot_summary,
                lavd_xlim=lavd_xlim,
                ylim=site_ylim,
            )

            fig.tight_layout(rect=[0, 0, 1, 0.97])
            pdf.savefig(fig)
            plt.close(fig)

    print("\nDone.")
    print("Profile long CSV:")
    print(OUT_PROFILE_LONG_CSV)
    print("Top-fraction long CSV:")
    print(OUT_TOPFRAC_LONG_CSV)
    print("Top-fraction wide CSV:")
    print(OUT_TOPFRAC_WIDE_CSV)
    print("Status CSV:")
    print(OUT_STATUS_CSV)
    print("PDF:")
    print(OUT_PDF)


if __name__ == "__main__":
    main()
