# -*- coding: utf-8 -*-
"""
Compute voxel-based clumping index (CI) and infer true LAI.

Purpose
-------
For each tree:
1) Read current Li/Flynn dynamic-k 1 m profile CSV.
2) Read leaf voxel files from cc_woodcl:
      *_angles_class_voxels_global.asc
3) Reconstruct fine 5 cm leaf voxel occupancy.
4) Compute voxel-based clumping index by stratum:

      Pgap_obs  = 1 - observed occupied projected XY fraction
      Pgap_rand = expected gap fraction if leaf voxels were randomly distributed
                  across vertical sublayers within the same stratum

      CI = ln(Pgap_obs) / ln(Pgap_rand)

   This gives:
      CI < 1  : clumped
      CI ~ 1  : random
      CI > 1  : more regular / over-dispersed

5) For each top n% of cumulative effective LAI/LAVD:
      summarize CI mean and SD
      compute LAIe_top from previous profile
      compute LAItrue_top = LAIe_top / CI_mean_top

Important
---------
- The input LAI_layer from the current Li/Flynn workflow is treated as LAIe.
- This is not the same as the old TLSLeAF ray-bincount CI, but it preserves
  the same conceptual relationship:
      CI = LAIe / LAItrue
      LAItrue = LAIe / CI
- Top fractions are defined using cumulative LAIe/LAVD from canopy top downward.
"""

import os
import glob
import numpy as np
import pandas as pd


# =============================================================================
# USER SETTINGS
# =============================================================================

SITE = "HARV"

SPLIT_ROOT = rf"E:\TLS\2025_summer\2025_{SITE}\Tree_scan\ForestGEO_{SITE}\L1_segmented_split"

PROFILE_SUBDIR = os.path.join("L2", "voxel_LiFlynn_dynamicK_1mStrata")

SITE_OUT_DIR = os.path.join(
    SPLIT_ROOT,
    f"L2_voxel_clumping_CI_LAItrue_{SITE}"
)
os.makedirs(SITE_OUT_DIR, exist_ok=True)

OUT_STRATUM_CSV = os.path.join(
    SITE_OUT_DIR,
    f"{SITE}_voxel_CI_by_stratum_LONG.csv"
)

OUT_TOPFRAC_LONG_CSV = os.path.join(
    SITE_OUT_DIR,
    f"{SITE}_voxel_CI_LAItrue_byTopFrac_LONG.csv"
)

OUT_TOPFRAC_WIDE_CSV = os.path.join(
    SITE_OUT_DIR,
    f"{SITE}_voxel_CI_LAItrue_byTopFrac_WIDE.csv"
)

OUT_STATUS_CSV = os.path.join(
    SITE_OUT_DIR,
    f"{SITE}_voxel_CI_LAItrue_STATUS.csv"
)

# Fine voxel size used in the Li/Flynn workflow.
VOXEL_SIZE = 0.05

# Vertical stratum thickness used in the Li/Flynn workflow.
STRATUM_THICKNESS = 1.0

# Top fractions of cumulative LAVD / LAIe.
TOP_FRACS = np.arange(0.1, 1.01, 0.1)

# Optional tree filter.
# Example: "FAGR", "QUAL", "job25". Set to None for all trees.
TREE_ID_CONTAINS = None

# Numerical safety limits for gap probability.
EPS = 1e-6

# CI constraints.
# Values outside this range are kept in the diagnostic stratum table,
# but excluded from top-fraction mean CI if CLIP_CI_FOR_SUMMARY=True.
CI_MIN_VALID = 0.05
CI_MAX_VALID = 2.00
CLIP_CI_FOR_SUMMARY = True


# =============================================================================
# FILE HELPERS
# =============================================================================

def list_tree_ids(split_root):
    tree_dirs = [
        d for d in glob.glob(os.path.join(split_root, "*"))
        if os.path.isdir(d) and os.path.isdir(os.path.join(d, "cc_woodcl"))
    ]

    tree_ids = sorted([os.path.basename(d) for d in tree_dirs])

    if TREE_ID_CONTAINS:
        tree_ids = [t for t in tree_ids if TREE_ID_CONTAINS in t]

    return tree_ids


def find_profile_file(split_root, tree_id):
    pattern = os.path.join(
        split_root,
        tree_id,
        PROFILE_SUBDIR,
        "*_profile.csv"
    )

    hits = sorted(glob.glob(pattern))

    if not hits:
        raise FileNotFoundError(
            f"No Li/Flynn profile CSV found for {tree_id} under:\n"
            f"{os.path.join(split_root, tree_id, PROFILE_SUBDIR)}"
        )

    return hits[0]


def find_leaf_voxel_files(split_root, tree_id):
    angle_dir = os.path.join(split_root, tree_id, "cc_woodcl")

    files = sorted(
        glob.glob(os.path.join(angle_dir, "*_angles_class_voxels_global.asc"))
    )

    if not files:
        raise FileNotFoundError(f"No leaf voxel files found under:\n{angle_dir}")

    return files


# =============================================================================
# READ PROFILE
# =============================================================================

def read_profile(path):
    df = pd.read_csv(path)

    required = [
        "stratum_id",
        "z_min",
        "z_mid",
        "z_max",
        "h",
        "n_xy_total",
        "filled_xy_leaf",
        "contact_freq_leaf",
        "LAI_layer",
        "LAVD_h"
    ]

    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(f"Profile file missing columns: {missing}")

    for c in required:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df[np.isfinite(df["stratum_id"])].copy()
    df["stratum_id"] = df["stratum_id"].astype(int)

    df = df.sort_values("stratum_id").reset_index(drop=True)

    return df


# =============================================================================
# READ LEAF VOXELS
# =============================================================================

def read_one_leaf_voxel_file(path):
    df = pd.read_csv(path, sep=r"\s+", engine="python")
    cols = {c.lower(): c for c in df.columns}

    needed = ["x", "y", "z", "heightaboveground"]

    for c in needed:
        if c not in cols:
            raise ValueError(f"{os.path.basename(path)} missing column: {c}")

    out = pd.DataFrame({
        "x": pd.to_numeric(df[cols["x"]], errors="coerce"),
        "y": pd.to_numeric(df[cols["y"]], errors="coerce"),
        "z": pd.to_numeric(df[cols["z"]], errors="coerce"),
        "h": pd.to_numeric(df[cols["heightaboveground"]], errors="coerce"),
    })

    if "counts" in cols:
        out["counts"] = pd.to_numeric(df[cols["counts"]], errors="coerce")
    else:
        out["counts"] = 1.0

    out = out[
        np.isfinite(out["x"]) &
        np.isfinite(out["y"]) &
        np.isfinite(out["z"]) &
        np.isfinite(out["h"]) &
        (out["h"] >= 0)
    ].copy()

    out["counts"] = out["counts"].fillna(1.0)

    return out


def read_all_leaf_voxels(paths):
    frames = []

    for p in paths:
        d = read_one_leaf_voxel_file(p)
        d["source_file"] = os.path.basename(p)
        frames.append(d)

    if not frames:
        return pd.DataFrame(columns=["x", "y", "z", "h", "counts"])

    return pd.concat(frames, ignore_index=True)


def voxelize_leaf_df(leaf_df):
    df = leaf_df.copy()

    df["ix"] = np.floor(df["x"] / VOXEL_SIZE).astype(int)
    df["iy"] = np.floor(df["y"] / VOXEL_SIZE).astype(int)
    df["iz"] = np.floor(df["h"] / VOXEL_SIZE).astype(int)
    df["stratum_id"] = np.floor(df["h"] / STRATUM_THICKNESS).astype(int)

    # One occupied leaf voxel per ix/iy/iz.
    vox = (
        df
        .drop_duplicates(["ix", "iy", "iz"])
        [["ix", "iy", "iz", "stratum_id"]]
        .copy()
    )

    return vox


# =============================================================================
# CLUMPING CALCULATION
# =============================================================================

def safe_gap(p):
    """
    Keep gap probability away from 0 and 1 to avoid log(0) / log(1).
    """
    if not np.isfinite(p):
        return np.nan

    return float(np.clip(p, EPS, 1.0 - EPS))


def compute_ci_for_stratum(row, leaf_vox):
    """
    Compute voxel-based CI for one stratum.

    Observed gap:
        Pgap_obs = 1 - (# projected xy cells with at least one leaf voxel) / n_xy_total

    Random gap:
        p_vox = (# occupied leaf voxels in stratum) /
                (n_xy_total * number of fine vertical voxel layers in stratum)

        Pgap_rand = (1 - p_vox) ** n_layers

    CI:
        CI = ln(Pgap_obs) / ln(Pgap_rand)

    Interpretation:
        CI < 1 means clumped.
        CI ~ 1 means random.
        CI > 1 means regular / over-dispersed.
    """

    sid = int(row["stratum_id"])

    n_xy_total = row["n_xy_total"]
    filled_xy_leaf = row["filled_xy_leaf"]

    z_min = row["z_min"]
    z_max = row["z_max"]

    if not np.isfinite(n_xy_total) or n_xy_total <= 0:
        return {
            "n_leaf_voxels": 0,
            "n_fine_z_layers": np.nan,
            "p_voxel_leaf": np.nan,
            "Pgap_obs": np.nan,
            "Pgap_rand": np.nan,
            "CI_voxel": np.nan,
        }

    sub = leaf_vox[leaf_vox["stratum_id"] == sid].copy()

    n_leaf_voxels = int(len(sub))

    # number of 5 cm vertical layers inside the 1 m stratum
    if np.isfinite(z_min) and np.isfinite(z_max) and z_max > z_min:
        n_fine_z_layers = int(np.ceil((z_max - z_min) / VOXEL_SIZE))
    else:
        n_fine_z_layers = int(np.ceil(STRATUM_THICKNESS / VOXEL_SIZE))

    n_fine_z_layers = max(n_fine_z_layers, 1)

    n_possible_voxels = float(n_xy_total * n_fine_z_layers)

    p_voxel_leaf = n_leaf_voxels / n_possible_voxels

    p_voxel_leaf = float(np.clip(p_voxel_leaf, 0.0, 1.0))

    Pgap_obs = 1.0 - (filled_xy_leaf / n_xy_total)
    Pgap_obs = safe_gap(Pgap_obs)

    Pgap_rand = (1.0 - p_voxel_leaf) ** n_fine_z_layers
    Pgap_rand = safe_gap(Pgap_rand)

    if (
        np.isfinite(Pgap_obs) and
        np.isfinite(Pgap_rand) and
        Pgap_obs > 0 and
        Pgap_rand > 0 and
        Pgap_obs < 1 and
        Pgap_rand < 1
    ):
        CI_voxel = np.log(Pgap_obs) / np.log(Pgap_rand)
    else:
        CI_voxel = np.nan

    return {
        "n_leaf_voxels": n_leaf_voxels,
        "n_fine_z_layers": n_fine_z_layers,
        "p_voxel_leaf": p_voxel_leaf,
        "Pgap_obs": Pgap_obs,
        "Pgap_rand": Pgap_rand,
        "CI_voxel": float(CI_voxel) if np.isfinite(CI_voxel) else np.nan,
    }


def compute_ci_profile(profile_df, leaf_vox):
    rows = []

    for _, r in profile_df.iterrows():
        ci_info = compute_ci_for_stratum(r, leaf_vox)

        out = r.to_dict()
        out.update(ci_info)

        # Treat current LAI_layer as effective LAI.
        laie = out.get("LAI_layer", np.nan)
        ci = out.get("CI_voxel", np.nan)

        if (
            np.isfinite(laie) and
            np.isfinite(ci) and
            ci > 0
        ):
            out["LAItrue_layer"] = laie / ci
        else:
            out["LAItrue_layer"] = np.nan

        if np.isfinite(out["LAItrue_layer"]):
            out["LAVDtrue_h"] = out["LAItrue_layer"] / STRATUM_THICKNESS
        else:
            out["LAVDtrue_h"] = np.nan

        rows.append(out)

    out_df = pd.DataFrame(rows)

    return out_df


# =============================================================================
# TOP-FRACTION SUMMARIES
# =============================================================================

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


def ci_for_summary(ci_values):
    ci = np.asarray(ci_values, dtype=float)

    if CLIP_CI_FOR_SUMMARY:
        ci = np.where(
            (ci >= CI_MIN_VALID) & (ci <= CI_MAX_VALID),
            ci,
            np.nan
        )

    return ci


def summarize_top_fractions(ci_profile):
    """
    Top fractions are selected from canopy top downward using cumulative LAIe
    from LAI_layer. This matches the current top cumulative LAVD logic.
    """

    df = ci_profile.copy()

    df = df[
        np.isfinite(df["LAI_layer"]) &
        (df["LAI_layer"] > 0)
    ].copy()

    if df.empty:
        raise ValueError("No positive LAI_layer values found.")

    df = df.sort_values("h").reset_index(drop=True)

    total_laie = float(df["LAI_layer"].sum())

    if total_laie <= 0:
        raise ValueError("Total LAIe is zero.")

    laie_topdown = df["LAI_layer"].to_numpy()[::-1]
    cum_topdown = np.cumsum(laie_topdown)

    rows = []

    for frac in TOP_FRACS:
        target = frac * total_laie

        idx_rev = np.where(cum_topdown >= target)[0]

        if len(idx_rev):
            k = int(idx_rev[0])
        else:
            k = len(df) - 1

        n_keep = k + 1
        i_start = len(df) - n_keep

        sub = df.iloc[i_start:].copy()

        ci_raw = sub["CI_voxel"].to_numpy(dtype=float)
        ci_valid = ci_for_summary(ci_raw)

        # LAIe-weighted CI is better for deriving top-fraction true LAI
        # because layers with more LAIe contribute more to the top fraction.
        weights = sub["LAI_layer"].to_numpy(dtype=float)

        ci_mean = float(np.nanmean(ci_valid)) if np.any(np.isfinite(ci_valid)) else np.nan
        ci_sd = float(np.nanstd(ci_valid, ddof=1)) if np.sum(np.isfinite(ci_valid)) > 1 else np.nan

        ci_wmean = weighted_mean(ci_valid, weights)
        ci_wsd = weighted_sd(ci_valid, weights)

        laie_top = float(sub["LAI_layer"].sum())

        if np.isfinite(ci_wmean) and ci_wmean > 0:
            laitrue_top_from_meanCI = laie_top / ci_wmean
        else:
            laitrue_top_from_meanCI = np.nan

        # Also calculate direct sum of layer-level LAIe/CI.
        laitrue_layer_sum = float(sub["LAItrue_layer"].sum(skipna=True))

        rows.append({
            "top_fraction_label": f"top{int(frac * 100):02d}",
            "top_fraction": float(frac),
            "cutoff_h_m": float(sub["z_min"].iloc[0]) if "z_min" in sub.columns else float(sub["h"].iloc[0]),
            "n_strata_kept": int(len(sub)),

            "LAIe_top": laie_top,

            "CI_mean_top": ci_mean,
            "CI_sd_top": ci_sd,
            "CI_wmean_LAIe_top": ci_wmean,
            "CI_wsd_LAIe_top": ci_wsd,
            "CI_n_valid_top": int(np.sum(np.isfinite(ci_valid))),

            "LAItrue_top_from_CIwmean": laitrue_top_from_meanCI,
            "LAItrue_top_sum_layers": laitrue_layer_sum,

            "LAVDe_mean_top": float(sub["LAVD_h"].mean(skipna=True)),
            "LAVDtrue_mean_top": float(sub["LAVDtrue_h"].mean(skipna=True)),

            "LAIe_total_tree": total_laie,
        })

    return pd.DataFrame(rows)


def topfrac_long_to_wide(long_df):
    rows = []

    for tree_id, g in long_df.groupby("tree_id"):
        row = {"tree_id": tree_id}

        for _, r in g.iterrows():
            lbl = r["top_fraction_label"]

            row[f"cutoff_h_{lbl}"] = r["cutoff_h_m"]
            row[f"LAIe_{lbl}"] = r["LAIe_top"]

            row[f"CI_mean_{lbl}"] = r["CI_mean_top"]
            row[f"CI_sd_{lbl}"] = r["CI_sd_top"]
            row[f"CI_wmean_LAIe_{lbl}"] = r["CI_wmean_LAIe_top"]
            row[f"CI_wsd_LAIe_{lbl}"] = r["CI_wsd_LAIe_top"]
            row[f"CI_n_valid_{lbl}"] = r["CI_n_valid_top"]

            row[f"LAItrue_from_CIwmean_{lbl}"] = r["LAItrue_top_from_CIwmean"]
            row[f"LAItrue_sum_layers_{lbl}"] = r["LAItrue_top_sum_layers"]

            row[f"LAVDe_mean_{lbl}"] = r["LAVDe_mean_top"]
            row[f"LAVDtrue_mean_{lbl}"] = r["LAVDtrue_mean_top"]

        rows.append(row)

    return pd.DataFrame(rows)


# =============================================================================
# PROCESS ONE TREE
# =============================================================================

def process_one_tree(tree_id):
    profile_path = find_profile_file(SPLIT_ROOT, tree_id)
    leaf_files = find_leaf_voxel_files(SPLIT_ROOT, tree_id)

    profile = read_profile(profile_path)

    leaf_df = read_all_leaf_voxels(leaf_files)
    leaf_vox = voxelize_leaf_df(leaf_df)

    ci_profile = compute_ci_profile(profile, leaf_vox)
    ci_profile["tree_id"] = tree_id
    ci_profile["profile_file"] = profile_path
    ci_profile["n_leaf_angle_files"] = len(leaf_files)

    top_long = summarize_top_fractions(ci_profile)
    top_long["tree_id"] = tree_id
    top_long["profile_file"] = profile_path
    top_long["n_leaf_angle_files"] = len(leaf_files)

    return ci_profile, top_long


# =============================================================================
# MAIN
# =============================================================================

def main():
    tree_ids = list_tree_ids(SPLIT_ROOT)

    print(f"Site: {SITE}")
    print(f"Found {len(tree_ids)} tree folders.")

    all_ci_profiles = []
    all_top_long = []
    status_rows = []

    for i, tree_id in enumerate(tree_ids, start=1):
        print(f"[{i}/{len(tree_ids)}] Processing {tree_id}")

        try:
            ci_profile, top_long = process_one_tree(tree_id)

            all_ci_profiles.append(ci_profile)
            all_top_long.append(top_long)

            status_rows.append({
                "tree_id": tree_id,
                "status": "OK",
                "message": "",
                "n_strata": len(ci_profile),
                "n_topfrac_rows": len(top_long),
            })

            top30 = top_long[top_long["top_fraction_label"] == "top30"]

            if not top30.empty:
                r = top30.iloc[0]
                print(
                    f"    OK | top30 CI={r['CI_wmean_LAIe_top']:.3f}, "
                    f"LAIe={r['LAIe_top']:.3f}, "
                    f"LAItrue={r['LAItrue_top_from_CIwmean']:.3f}"
                )
            else:
                print("    OK")

        except Exception as e:
            status_rows.append({
                "tree_id": tree_id,
                "status": "FAIL",
                "message": str(e),
                "n_strata": np.nan,
                "n_topfrac_rows": np.nan,
            })

            print(f"    [WARN] {e}")

    status_df = pd.DataFrame(status_rows)
    status_df.to_csv(OUT_STATUS_CSV, index=False)

    if not all_ci_profiles:
        raise RuntimeError("No trees processed successfully.")

    ci_profile_all = pd.concat(all_ci_profiles, ignore_index=True)
    top_long_all = pd.concat(all_top_long, ignore_index=True)
    top_wide_all = topfrac_long_to_wide(top_long_all)

    ci_profile_all.to_csv(OUT_STRATUM_CSV, index=False)
    top_long_all.to_csv(OUT_TOPFRAC_LONG_CSV, index=False)
    top_wide_all.to_csv(OUT_TOPFRAC_WIDE_CSV, index=False)

    print("\nDone.")
    print("Stratum CI CSV:")
    print(OUT_STRATUM_CSV)
    print("Top-fraction long CSV:")
    print(OUT_TOPFRAC_LONG_CSV)
    print("Top-fraction wide CSV:")
    print(OUT_TOPFRAC_WIDE_CSV)
    print("Status CSV:")
    print(OUT_STATUS_CSV)


if __name__ == "__main__":
    main()