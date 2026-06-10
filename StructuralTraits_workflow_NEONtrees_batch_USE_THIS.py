# -*- coding: utf-8 -*-
"""
Batch process TLS .ptx files for each tree folder under a mother folder.

Each subfolder under L1_segmented_split is treated as a single tree folder,
and all .ptx files inside it will be processed using the same TLSLeAF2.0_2025April pipeline.
"""

import sys, os, importlib

# Explicitly point to your April 2025 repo folder
# repo_dir = r"I:\TLS\My code\TLSLeAF2.0_2025April"
repo_dir = r"C:\Users\Ricy\PERS-master\TLS\TLSLeAF2.0_2025April"

# Remove any cached version of the module
if "Preprocessing_functions_CC" in sys.modules:
    del sys.modules["Preprocessing_functions_CC"]

# Force Python to import from your intended folder only
if repo_dir not in sys.path:
    sys.path.insert(0, repo_dir)
os.chdir(repo_dir)

# Import cleanly
import Preprocessing_functions_CC as ccPP
importlib.reload(ccPP)

print("Loaded Preprocessing_functions_CC from:", ccPP.__file__)
print("get_filenames found:", hasattr(ccPP, "get_filenames"))



import sys
import os
import glob
import shutil
import time
import yaml
import numpy as np
import pandas as pd
import logging
import importlib

# =============================================================================
# PATH SETUP
# =============================================================================
#sys.path.append(r"I:\TLS\My code\TLSLeAF2.0_2025April")
sys.path.append(r"C:\Users\Ricy\PERS-master\TLS\TLSLeAF2.0_2025April")

# add CloudComPy path, assuming it is in the same parent directory as the working directory
sys.path.insert(0, 'D:\\CloudComPy310\\CloudCompare\\')
sys.path.insert(0, 'D:\\CloudComPy310\\')
sys.path.insert(0, 'D:\\CloudComPy310\\CloudCompare\\cloudComPy\\')

from utils.logs import setup_logging
import cloudComPy as cc  # noqa
import Preprocessing_functions_CC as ccPP
importlib.reload(ccPP)

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
def write_class_ss_dummy(pcd_ss, out_file, label_value=1):
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    coords = pcd_ss.toNpArrayCopy()
    if coords is None or coords.size == 0:
        raise RuntimeError("Subsampled cloud has no points; cannot write dummy class.")
    with open(out_file, "w") as f:
        f.write("X Y Z label\n")
        for x, y, z in coords:
            f.write(f"{x:.6f} {y:.6f} {z:.6f} {int(label_value)}\n")


def ensure_angle_and_ss(pcd_fname, filetype, filenames, config_pp):
    os.makedirs(os.path.dirname(filenames['ss']), exist_ok=True)
    if os.path.exists(filenames['angles']):
        pcd_angle_cc = ccPP.load_ascii_pcd(filenames['angles'], out_type='cc', header=0)
    else:
        if filetype == '.ptx':
            transmat, _, pcd_cc = ccPP.read_ptx(
                pcd_fname, radius=config_pp['radius'], fields=[],
                write_cropped=config_pp['write_cropped'], cropped_file=filenames['cropped'])
        else:
            pcd_cc = ccPP.cc_load(
                pcd_fname, radius=config_pp['radius'], fields=[],
                write_cropped=config_pp['write_cropped'], cropped_file=filenames['cropped'])

        if config_pp.get('filter_ground') or config_pp.get('get_aboveground_height'):
            pcd_cc = ccPP.ground_filtering(
                pcd_cc, filenames['topo'], clothResolution=config_pp['CSFclothResolution'],
                filter_ground=config_pp['filter_ground'], get_aboveground_height=config_pp['get_aboveground_height'],
                write_mesh=config_pp['write_mesh'], write_ground=config_pp['write_ground'],
                write_aboveground=config_pp['write_aboveground'],
                mesh_file=filenames['ground_mesh'], ground_file=filenames['ground'],
                aboveground_file=filenames['aboveground']
            )

        pcd_angle_cc = ccPP.angles_calc(
            pcd_cc, transmat=None, max_scatter=config_pp['max_scatter'],
            write_angles=True, angles_file=filenames['angles'])

    if os.path.exists(filenames['ss']):
        pcd_ss = ccPP.load_ascii_pcd(filenames['ss'], out_type='cc', header=0)
    else:
        pcd_ss = ccPP.subsample(
            pcd_angle_cc, SS_size=config_pp['SS_size'],
            write_ss=True, ss_file=filenames['ss'])
        if not os.path.exists(filenames['ss']):
            raise RuntimeError(f"Failed to create subsample file: {filenames['ss']}")
    return pcd_angle_cc, pcd_ss


# =============================================================================
# LOAD CONFIG
# =============================================================================
with open(r"C:\Users\Ricy\PERS-master\TLS\TLSLeAF2.0_2025April\config.yaml", "r") as file:
    config = yaml.safe_load(file)

config_pp = config['preprocessing']
overwrite = config['overwrite']
classification_method = config_pp['classification_method']
class_sel = config_pp[classification_method]['class_sel']
config_pp['write_angles'] = True
config_pp['write_ss'] = True
output_subfolder = config_pp[classification_method]['output_subfolder']

# =============================================================================
# MAIN BATCH WRAPPER
# =============================================================================
# mother_dir = r"I:\TLS\2025_summer\2025_SERC\Tree_scan\ForestGEO_SERC\L1_segmented_split"
mother_dir = r"E:\TLS\2025_summer\2025_BART\Tree_scan\ForestGEO_BART\L1_segmented_split"
tree_dirs = [os.path.join(mother_dir, d) for d in os.listdir(mother_dir)
             if os.path.isdir(os.path.join(mother_dir, d))]

print(f"\nFound {len(tree_dirs)} tree folders under {mother_dir}\n")

for tree_dir in tree_dirs:
    print(f"=== Processing tree folder: {tree_dir} ===")

    pcd_fnames = glob.glob(os.path.join(tree_dir, "*.ptx"))
    if not pcd_fnames:
        print(f"No .ptx files found in {tree_dir}, skipping.")
        continue

    out_path = tree_dir
    os.makedirs(os.path.join(out_path, output_subfolder), exist_ok=True)

    logfile = os.path.join(out_path, 'log.txt')
    logger = setup_logging(logfile)

    with open(r"C:\Users\Ricy\PERS-master\TLS\TLSLeAF2.0_2025April\config.yaml", "r") as src, open(logfile, "a") as dst:
        dst.write("\n# Configuration:\n")
        shutil.copyfileobj(src, dst)

    # -------------------------------------------------------------------------
    # PER-FILE PROCESSING LOOP
    # -------------------------------------------------------------------------
    for pcd_fname in pcd_fnames:
        time_file0 = time.time()
        base_name = os.path.basename(pcd_fname)[:-4]
        logging.info(f"Processing {pcd_fname}")

        filenames = ccPP.get_filenames(out_path, output_subfolder, base_name,
                                       resstr=config_pp['bincount_para']['resstr'])
        filetype = os.path.splitext(pcd_fname)[1]

        # ---------- ANGLE & FILTERING STAGE ----------
        if (overwrite or (not (os.path.exists(filenames['angles']) or os.path.exists(filenames['class'])))):
            if filetype == '.ptx':
                transmat, _, pcd_cc = ccPP.read_ptx(
                    pcd_fname, radius=config_pp['radius'], fields=[],
                    write_cropped=config_pp['write_cropped'], cropped_file=filenames['cropped'])
            else:
                pcd_cc = ccPP.cc_load(
                    pcd_fname, radius=config_pp['radius'], fields=[],
                    write_cropped=config_pp['write_cropped'], cropped_file=filenames['cropped'])

            if config_pp['filter_ground'] or config_pp['get_aboveground_height']:
                pcd_cc = ccPP.ground_filtering(
                    pcd_cc, filenames['topo'], clothResolution=config_pp['CSFclothResolution'],
                    filter_ground=config_pp['filter_ground'], get_aboveground_height=config_pp['get_aboveground_height'],
                    write_mesh=config_pp['write_mesh'], write_ground=config_pp['write_ground'],
                    write_aboveground=config_pp['write_aboveground'],
                    mesh_file=filenames['ground_mesh'], ground_file=filenames['ground'],
                    aboveground_file=filenames['aboveground'])
                pcd_cc_raw_aboveground = pcd_cc.cloneThis()

            pcd_angle_cc = ccPP.angles_calc(
                pcd_cc, transmat=None, max_scatter=config_pp['max_scatter'], write_angles=False)

            if config_pp['filter_noise']:
                pcd_cc = ccPP.noise_filtering(
                    pcd_cc, SOR_KNN=config_pp['SOR_KNN'], SOR_std=config_pp['SOR_std'],
                    write_filtered=config_pp['write_filtered'], filtered_file=filenames['filtered'])

            pcd_angle_cc = ccPP.angles_calc(
                pcd_cc, transmat=None, max_scatter=config_pp['max_scatter'],
                write_angles=config_pp['write_angles'], angles_file=filenames['angles'])
        else:
            logging.info("Skipped angle calculation, file already exists")

        # ---------- SUBSAMPLE + CLASSIFICATION + PROJECTION ----------
        if config_pp['subsample']:
            if ((not os.path.exists(filenames['class'])) or overwrite):
                pcd_angle_cc, pcd_ss = ensure_angle_and_ss(pcd_fname, filetype, filenames, config_pp)
                try:
                    n_ss = pcd_ss.size()
                except Exception:
                    n_ss = (pcd_ss.toNpArrayCopy().shape[0] if hasattr(pcd_ss, "toNpArrayCopy") else 0)
                if n_ss == 0:
                    logging.warning(f"Empty subsample for {base_name}; skipping.")
                    continue

                if config_pp['classification']:
                    made_class_ss = False
                    if ((not os.path.exists(filenames['class_ss'])) or overwrite):
                        try:
                            if classification_method == 'woodcl':
                                ccPP.classification_woodcl(
                                    pcd_ss, config_pp[classification_method],
                                    write_class_ss=True, class_ss_file=filenames['class_ss'])
                            elif classification_method == 'FSCT':
                                ccPP.classification_FSCT(filenames['ss'], config_pp[classification_method])
                                if os.path.exists(filenames['semantic_ss']):
                                    os.rename(filenames['semantic_ss'], filenames['class_ss'])
                            made_class_ss = os.path.exists(filenames['class_ss'])
                        except Exception as e:
                            logging.exception(f"classification failed for {base_name}: {e}")
                            made_class_ss = False

                        if not made_class_ss:
                            try:
                                default_lbl = (config_pp.get(classification_method, {}).get('class_sel') or [1])[0]
                            except Exception:
                                default_lbl = 1
                            logging.warning(f"Classification failed. Writing dummy: {filenames['class_ss']}")
                            write_class_ss_dummy(pcd_ss, filenames['class_ss'], label_value=default_lbl)

                    if not os.path.exists(filenames['class_ss']):
                        logging.warning(f"Missing class_ss for {base_name}; skipping projection.")
                        continue

                    pcd_class_ss = ccPP.load_ascii_pcd(filenames['class_ss'], out_type='cc', header=0)
                    ccPP.SFprojection(
                        pcd_angle_cc, pcd_class_ss, fields='label',
                        write_projected=config_pp['write_class'], projected_file=filenames['class'])
            else:
                logging.info("Skipped classification/projection; already exists.")

        # ---------- VOXELIZATION ----------
        if config_pp['voxelization']:
            if ((not os.path.exists(filenames['voxel'])) 
                or (config_pp['write_voxel_global'] and not os.path.exists(filenames['voxel_global']))
                or overwrite):
                logging.info("Voxelizing pointcloud")
                if 'pcd_class_df' not in locals():
                    pcd_class_df = ccPP.load_ascii_pcd(filenames['class'], out_type='df', header=0, fields='all')

                voxelized_df = ccPP.voxelize_pcd(
                    pcd_class_df, config_pp['voxel_size'], config_pp['voxel_fields_methods'],
                    class_sel=class_sel, scatter_filter=config_pp['max_scatter_voxel'],
                    counts_thresh=config_pp['counts_thresh'], write_voxel=config_pp['write_voxel'],
                    voxel_file=filenames['voxel'])

                if config_pp['write_voxel_global']:
                    if ((not os.path.exists(filenames['voxel_global'])) or overwrite):
                        if filetype == '.ptx':
                            if 'transmat' not in locals():
                                transmat = ccPP.ptxheader2transmat(filenames['headers'])
                        else:
                            transmat = np.loadtxt(filenames['transmat'], delimiter=" ")
                        voxelized_df_global = ccPP.pcd_transform(voxelized_df, transmat)
                        voxelized_df_global.to_csv(filenames['voxel_global'], sep=' ', index=False)
                        logging.info("Wrote voxelized file in global coordinate.")

        # ---------- BINCOUNTS ----------
        if config_pp['countbins']:
            if ((not os.path.exists(filenames['classfreq'])) or overwrite):
                logging.info("calculating bincount for classified point cloud")
                if 'pcd_class_df' not in locals():
                    pcd_class_df = ccPP.load_ascii_pcd(filenames['class'], out_type='df', header=0, fields='all')

                if filetype == '.ptx':
                    tiltmat = np.loadtxt(filenames['tilt'], delimiter=" ")
                else:
                    tiltmat = None

                pcd_class_df = pcd_class_df[['X', 'Y', 'Z', 'label', 'scat']]
                ccPP.bincount(
                    pcd_class_df, config_pp['bincount_para'], filenames['classfreq'], processed=True,
                    class_sel=config_pp[classification_method]['class_sel'], transmat=tiltmat)

            if ((not os.path.exists(filenames['ptxfreq'])) or overwrite):
                logging.info("calculating bincount for raw PTX")
                if 'pcd_cc_raw_aboveground' not in locals():
                    if os.path.isfile(filenames['aboveground']):
                        pcd_df_raw_aboveground = ccPP.load_ascii_pcd(
                            filenames['aboveground'], out_type='df', header=0)
                    else:
                        if filetype == '.ptx':
                            transmat, _, pcd_cc = ccPP.read_ptx(
                                pcd_fname, radius=config_pp['radius'], fields=[],
                                write_cropped=config_pp['write_cropped'], cropped_file=filenames['cropped'])
                        else:
                            pcd_cc = ccPP.cc_load(
                                pcd_fname, radius=config_pp['radius'], fields=[],
                                write_cropped=config_pp['write_cropped'], cropped_file=filenames['cropped'])
                        pcd_cc_raw_aboveground = ccPP.ground_filtering(
                            pcd_cc, filenames['topo'], clothResolution=config_pp['CSFclothResolution'],
                            filter_ground=config_pp['filter_ground'], get_aboveground_height=config_pp['get_aboveground_height'],
                            write_mesh=config_pp['write_mesh'], write_ground=config_pp['write_ground'],
                            write_aboveground=config_pp['write_aboveground'],
                            mesh_file=filenames['ground_mesh'], ground_file=filenames['ground'],
                            aboveground_file=filenames['aboveground'])
                        coordinates = pcd_cc_raw_aboveground.toNpArrayCopy()
                        pcd_df_raw_aboveground = pd.DataFrame(data=coordinates, columns=['X', 'Y', 'Z'])
                else:
                    coordinates = pcd_cc_raw_aboveground.toNpArrayCopy()
                    pcd_df_raw_aboveground = pd.DataFrame(data=coordinates, columns=['X', 'Y', 'Z'])

                if filetype == '.ptx':
                    tiltmat = np.loadtxt(filenames['tilt'], delimiter=" ")
                else:
                    tiltmat = None

                ccPP.bincount(
                    pcd_df_raw_aboveground, config_pp['bincount_para'], filenames['ptxfreq'], processed=False,
                    transmat=tiltmat)

            if config_pp['calc_PAVD']:
                logging.info("calculating PAVD/LAVD/PAVDe/LAVDe")

        # ---------- CLEANUP ----------
        for var in ['pcd_raw_df', 'pcd_class_df', 'pcd_angle_cc', 'pcd_class_ss', 'pcd_ss', 'pcd_cc']:
            if var in locals():
                del locals()[var]

        time_file1 = time.time()
        logging.info(f"Finished {pcd_fname}, took {time_file1 - time_file0:.1f} s\n")

    print(f"--- Finished tree folder: {tree_dir} ---\n")

print("All tree folders processed successfully!")
