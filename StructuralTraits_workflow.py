# -*- coding: utf-8 -*-
"""
Created on Sun Sep 22 10:44:03 2024

@author: lirong

This is the main file for processing TLS pointcloud to obtain plant canopy structural traits.





File requirements: under data_dir, there should be .ptx point cloud [pointcloudname].ptx

For PAVD related calculation, corresponding [pointcloudname]_tiltmat.asc is also required.
This is used for correcting the tilt of the scanner from the nadir direction. Because in 
Leica RTC360 pointclouds, the Z axis is nadir, not the rotating axis. However, the scanning 
rotational axis is required for gap-probability based PAVD estimations. 
The tiltmat files can be obtained from raw Leica scan folder using script "Tools/tiltmat_from_raw.py".
If Z axis in pointcloud is the scanning axis instead of nadir, just put a 
3*3 unit matrix in the tiltmat file. 

TODO: binary option (maybe .sbf?) for intermediate large pointcloud: https://www.cloudcompare.org/doc/wiki/index.php/FILE_I/O
NOTE: use cloneThis if need to copy pointcloud


"""
import sys
sys.path.append(r"C:\Users\Ricy\PERS-master\TLS\TLSLeAF2.0_2025April")
from utils.logs import setup_logging

import numpy as np
import os
import glob
import pandas as pd
import time
import sys
from utils.logs import setup_logging
import logging
import shutil

# Change the working directory to the directory of this file
current_file_path = os.path.abspath(r"C:\Users\Ricy\PERS-master\TLS\TLSLeAF2.0_2025April")
current_dir = os.path.dirname(current_file_path)
os.chdir(current_dir)
# add CloudCompPy path, assuming it is in the same parent directory as the working directory
sys.path.insert(0, '..\\CloudComPy310\\CloudCompare\\')
sys.path.insert(0, '..\\CloudComPy310\\')
sys.path.insert(0, '..\\CloudComPy310\\CloudCompare\\cloudComPy\\')

import cloudComPy as cc 
import yaml
import importlib

sys.path.insert(0, r"C:\Users\Ricy\PERS-master\TLS\TLSLeAF2.0_2025April")
import Preprocessing_functions_CC as ccPP
importlib.reload(ccPP)

print(ccPP.__file__)
print('get_filenames' in dir(ccPP))




# import torch

# cuda_available = torch.cuda.is_available()
# print(torch.version.cuda)
# %% Settings
# Path parameters
# data_dir = "K:\\TLS\\PR_coffee2\\"
# data_dir = "E:\\TLS\\SingleScan\\Across-site\\L1\\Milton2\\"
# data_dir = "E:\\Xulab TLS\\MTP_AS1_20230522\\"
# data_dir = "C:\\TLS\\test\\small\\test2\\"
# data_dir = "D:\\TLS\\SingleScan\\Across-site\\L1\\PR_OMARlower\\"
# data_dir = "C:\\TLS\\test\\Riegl\\"
data_dir = r"E:\TLS\2025_summer\2025_SCBI\Tree_scan\ForestGEO_SCBI\processed_temp\SCBI_job020_2QUAL_1"
pcd_fnames = glob.glob(os.path.join(data_dir, "*.ptx"))
# pcd_fnames = glob.glob(os.path.join(data_dir, "*.laz"))
n_ext = 4 # filename extension length, used to get the file base name
out_path = data_dir


#%% load configuration
with open(r"TLSLeAF2.0_2025April\config.yaml", "r") as file:
    config = yaml.safe_load(file)


config_pp = config['preprocessing']
overwrite = config['overwrite']
classification_method = config_pp['classification_method']
class_sel = config_pp[classification_method]['class_sel']

output_subfolder = config_pp[classification_method]['output_subfolder']
if not os.path.exists(os.path.join(out_path,output_subfolder)):
    os.makedirs(os.path.join(out_path,output_subfolder))

logfile = out_path+'log.txt'
logger = setup_logging(logfile)

with open(r"TLSLeAF2.0_2025April\config.yaml", "r") as src, open(logfile, "a") as dst:
    dst.write("\n# Configuration:\n")
    shutil.copyfileobj(src, dst)

# %% Main workflow
if not os.path.exists(out_path):
    os.mkdir(out_path)

for pcd_fname in pcd_fnames[:]:
    time_file0 = time.time() 
    
    # The center is (0,0,0) for single scan ptx, and all output files except for voxel_global_file!
    base_name = os.path.basename(pcd_fname)[:-n_ext]
    logging.info(f"Processing {pcd_fname}")
    filenames = ccPP.get_filenames(out_path, output_subfolder, base_name, resstr=config_pp['bincount_para']['resstr'])
    filetype = os.path.splitext(pcd_fname)[1]
    # noise and ground filtering, angle calculation
    if (overwrite or (not (os.path.exists(filenames['angles']) or 
                           os.path.exists(filenames['class'])))):
        
        if filetype == '.ptx':
            transmat,_,pcd_cc = ccPP.read_ptx(
                pcd_fname,radius=config_pp['radius'],fields=[], # does not import scalar fields
                write_cropped=config_pp['write_cropped'],cropped_file=filenames['cropped'])
        else:
            pcd_cc = ccPP.cc_load(
                pcd_fname,radius=config_pp['radius'],fields=[],
                write_cropped=config_pp['write_cropped'],cropped_file=filenames['cropped'])
            
            
        # filter ground points
        if config_pp['filter_ground'] or config_pp['get_aboveground_height']:
            pcd_cc = ccPP.ground_filtering(
                pcd_cc,filenames['topo'],clothResolution = config_pp['CSFclothResolution'],
                filter_ground=config_pp['filter_ground'], get_aboveground_height=config_pp['get_aboveground_height'],
                write_mesh=config_pp['write_mesh'],write_ground=config_pp['write_ground'],write_aboveground=config_pp['write_aboveground'],
                mesh_file=filenames['ground_mesh'],ground_file=filenames['ground'],aboveground_file=filenames['aboveground'])
            pcd_cc_raw_aboveground = pcd_cc.cloneThis()

        # NOTE: scattering angle filtering both before and after noise filtering to improve the filtering. TODO: test the improvement 
        pcd_angle_cc = ccPP.angles_calc(
            pcd_cc,transmat=None,max_scatter=config_pp['max_scatter'],
            write_angles=False) # for single scan ptx, transmat should be None
        
        # noise filtering with Statistical Outlier Removal
        if config_pp['filter_noise']:
            pcd_cc = ccPP.noise_filtering(
                pcd_cc,SOR_KNN=config_pp['SOR_KNN'],SOR_std=config_pp['SOR_std'],
                write_filtered=config_pp['write_filtered'],filtered_file=filenames['filtered'])
        
        # angle calculations        
        pcd_angle_cc = ccPP.angles_calc(
            pcd_cc,transmat=None,max_scatter=config_pp['max_scatter'],
            write_angles=config_pp['write_angles'],angles_file=filenames['angles']) # for single scan ptx, transmat should be None
          
    else:
        logging.info("Skipped angle calculation, file already exist")
            
    if config_pp['subsample']: # Use subsampled file for classification
        if ((not os.path.exists(filenames['ss'])) or overwrite):
            # Read angle file if needed
            if 'pcd_angle_cc' not in locals():
                pcd_angle_cc=ccPP.load_ascii_pcd(filenames['angles'], out_type='cc',header = 0,fields=None)

        else:
            logging.info("Skipped subsampling, file already exist") 
    
        if config_pp['classification']: # Use subsampled file for classification
            if((not os.path.exists(filenames['class'])) or overwrite):
                if((not os.path.exists(filenames['class_ss'])) or overwrite):
                    # Read angle file if needed
                    if 'pcd_ss' not in locals():
                        pcd_ss=ccPP.load_ascii_pcd(filenames['ss'], out_type='cc', header = 0,fields=None)
                    # Classify subsampled pointcloud
                    if classification_method == 'woodcl':
                        ccPP.classification_woodcl(
                            pcd_ss,config_pp[classification_method],
                            write_class_ss=config_pp['write_class_ss'],
                            class_ss_file=filenames['class_ss'])
                    elif classification_method == 'FSCT':
                        ccPP.classification_FSCT(filenames['ss'],config_pp[classification_method])
                        os.rename(filenames['semantic_ss'], filenames['class_ss'])                            
                    else:
                        raise Exception("classification method not recognized, accepting woodcl and FSCT")
                else:
                    # Import pointcloud
                    logging.info(f"Importing subsampled classified pointcloud: {filenames['class_ss']}")
                    pcd_df = pd.read_csv(filenames['class_ss'],delimiter=" ",header=0,engine='pyarrow')
                    logging.info(f"Done importing {filenames['class_ss']}")
                
                # Project the subsampled classified pointcloud back to the original resolution pointcloud
                # Read angle file if needed
                if 'pcd_angle_cc' not in locals():
                    pcd_angle_cc=ccPP.load_ascii_pcd(filenames['angles'], out_type='cc', header = 0)
                if 'pcd_class_ss' not in locals():
                    pcd_class_ss=ccPP.load_ascii_pcd(filenames['class_ss'], out_type='cc', header = 0)
                
                pcd_angle_class_cc = ccPP.SFprojection(pcd_angle_cc,pcd_class_ss,fields='label',
                    write_projected=config_pp['write_class'],projected_file=filenames['class'])    
            
            else:
                logging.info("Skipped classification, file already exist")
        
        else:
            raise Exception("Code not implemented for classification at origional resolution. This is not recommended")
        
    # get voxelized fields including leaf angle
    if config_pp['voxelization']:
        if((not os.path.exists(filenames['voxel'])) 
           or (config_pp['write_voxel_global'] and not os.path.exists(filenames['voxel_global']))
           or overwrite):
            logging.info("Voxelizing pointcloud")
            if 'pcd_class_df' not in locals():
                pcd_class_df = ccPP.load_ascii_pcd(filenames['class'], out_type='df', header = 0,fields='all')          
            
            voxelized_df = ccPP.voxelize_pcd(pcd_class_df,config_pp['voxel_size'],config_pp['voxel_fields_methods'],
                class_sel=class_sel,scatter_filter=config_pp['max_scatter_voxel'],
                counts_thresh = config_pp['counts_thresh'],write_voxel=config_pp['write_voxel'],
                voxel_file=filenames['voxel'])

            if config_pp['write_voxel_global']:
                if((not os.path.exists(filenames['voxel_global'])) or overwrite):
                    if filetype == '.ptx':
                        if 'transmat' not in locals():
                            transmat = ccPP.ptxheader2transmat(filenames['headers'])                     
                    else:
                        transmat = np.loadtxt(filenames['transmat'], delimiter=" ")
                    voxelized_df_global = ccPP.pcd_transform(voxelized_df,transmat)
                    voxelized_df_global.to_csv(filenames['voxel_global'],sep=' ',index=False)
                    logging.info("Done writing voxelized file in global coordinate.")
                
    if config_pp['countbins']:
        # bincount for classified pointcloud with angle information 
        if((not os.path.exists(filenames['classfreq'])) or overwrite):
            logging.info("calculating bincount for classified point cloud")
            if 'pcd_class_df' not in locals(): 
                pcd_class_df = ccPP.load_ascii_pcd(filenames['class'], out_type='df', header = 0,fields='all')
                
            if filetype == '.ptx': 
                tiltmat = np.loadtxt(filenames['tilt'], delimiter=" ")
            else: # TODO: deal with the cases where tiltmat is provided
                tiltmat = None
                logging.info("tiltmat not found, assuming no tilt correction")
            
            pcd_class_df = pcd_class_df[['X','Y','Z','label','scat']]
            ccPP.bincount(pcd_class_df, config_pp['bincount_para'],filenames['classfreq'],processed = True,
                          class_sel=config_pp[classification_method]['class_sel'],transmat=tiltmat)
                
        # bincount for raw ptx file without any noise filtering
        # TODO: This part of the code can be cleaner
        if((not os.path.exists(filenames['ptxfreq'])) or overwrite):
            logging.info("calculating bincount for ptx")
            if 'pcd_cc_raw_aboveground' not in locals():
                if os.path.isfile(filenames['aboveground']):
                    pcd_df_raw_aboveground = ccPP.load_ascii_pcd(filenames['aboveground'], out_type='df', header = 0,fields=[])
                else:
                    if filetype == '.ptx':
                        transmat,_,pcd_cc = ccPP.read_ptx(
                            pcd_fname,radius=config_pp['radius'],fields=[], # does not import scalar fields
                            write_cropped=config_pp['write_cropped'],cropped_file=filenames['cropped'])
                    else:
                        pcd_cc = ccPP.cc_load(
                            pcd_fname,radius=config_pp['radius'],fields=[],
                            write_cropped=config_pp['write_cropped'],cropped_file=filenames['cropped'])
                    # filter ground points
                    pcd_cc_raw_aboveground = ccPP.ground_filtering(
                        pcd_cc,filenames['topo'],clothResolution = config_pp['CSFclothResolution'],
                        filter_ground=config_pp['filter_ground'], get_aboveground_height=config_pp['get_aboveground_height'],
                        write_mesh=config_pp['write_mesh'],write_ground=config_pp['write_ground'],write_aboveground=config_pp['write_aboveground'],
                        mesh_file=filenames['ground_mesh'],ground_file=filenames['ground'],aboveground_file=filenames['aboveground'])
                    coordinates = pcd_cc_raw_aboveground.toNpArrayCopy()
                    pcd_df_raw_aboveground = pd.DataFrame(data=coordinates,columns=['X','Y','Z']) # TODO no need to convert to df
            else:
                coordinates = pcd_cc_raw_aboveground.toNpArrayCopy()
                pcd_df_raw_aboveground = pd.DataFrame(data=coordinates,columns=['X','Y','Z']) # TODO no need to convert to df
            if filetype == '.ptx': 
                tiltmat = np.loadtxt(filenames['tilt'], delimiter=" ")
            else:
                tiltmat = None
                logging.info("tiltmat not found, assuming no tilt correction")
            
            ccPP.bincount(pcd_df_raw_aboveground, config_pp['bincount_para'],filenames['ptxfreq'],processed = False,
                          transmat=tiltmat)
        
        if(config_pp['calc_PAVD']):
            logging.info("calculating PAVD, LAVD, PAVDe, and LAVDe\n")
            
    # clean large variables
    variables_to_clean = ['pcd_raw_df', 'pcd_class_df', 'pcd_angle_cc','pcd_class_ss','pcd_ss','pcd_cc','pcd_df']
    for var in variables_to_clean:
        if var in globals():
            del globals()[var]  # Deletes the variable from the global scope
        elif var in locals():
            del locals()[var]  # Deletes the variable from the local scope

    time_file1 = time.time()
    logging.info(f"Finished processing {pcd_fname}, took {time_file1-time_file0} seconds \n\n")

# %%
