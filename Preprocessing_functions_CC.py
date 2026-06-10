# -*- coding: utf-8 -*-
"""
Created on Sun Sep 22 13:02:44 2024

@author: lirong
"""
import pandas as pd
import numpy as np
from numpy.typing import NDArray, ArrayLike
from typing import List, Dict, Tuple, Callable, Any, Union
import os
import sys
import time
import logging


import pyarrow as pa
import pyarrow.csv as csv
import cloudComPy as cc 
import cloudComPy.CSF # This is needed even though it might be marked as unused

from utils.logs import timer
from scipy.interpolate import griddata


# TODO: put this to config
def get_filenames(out_path: str, output_subfolder: str, base_name: str, resstr: str='_05') -> dict:
    """get a dictionary of filenames to be used in the workflow

    Parameters
    ----------
    out_path : str
        output path
    output_subfolder : str
        output subfolder
    base_name : str
        prefix of the filename
    resstr : str, optional
        suffix for bincount files indicating bin resolution used, the default is '_05'.
    Returns
    -------
    filenames: dict
        dictionary of filenames
    """
    file_templates = {
        "headers": "_headers.asc",
        "cropped": "_cropped.asc",
        "filtered": "_filtered.asc",
        "angles": "_angles.asc",
        "ground": "_ground.asc",
        "aboveground": "_aboveground.asc",
        "topo": "_topo.asc",
        "ground_mesh": "_groundmesh.asc",
        "ss": "_ss.asc",
        "class_ss": "_ss_class.asc",
        "semantic_ss": "_ss_segmented.asc",
        "class": "_angles_class.asc",
        "voxel": "_angles_class_voxels.asc",
        "voxel_global": "_angles_class_voxels_global.asc",
        "tilt": "_tiltmat.asc",
        "classfreq": "_bincount" + resstr + ".asc",
        "ptxfreq": "_ptxbincount" + resstr + ".asc",
        "transmat": ".DAT",
    }

    filenames = {
        key: os.path.join(out_path, output_subfolder, base_name + suffix) if (key != "tilt" and key != "headers" and key != "transmat")
        else os.path.join(out_path, base_name + suffix)
        for key, suffix in file_templates.items()
    }
    
    return filenames

@timer("importing ptx")
def read_ptx(filename: str,radius: float=np.nan,fields:Union[None,list]=None,
             header_file: Union[None,str]=None,
             write_cropped: bool=False,cropped_file: Union[None,str]=None
             ) -> tuple[NDArray, pd.DataFrame,cc.ccPointCloud]:
    """ Read single-scan ptx file into a dataframe
    TODO: support multi-scan ptx/check if the file is multi-scan
    NOTE: the cloudcompare Importfile method does not support parallel, 
    so it is slower than the pd.read_csv with pyarrow. So we use pd.readcsv 
    for reading pointclouds

    Parameters
    ----------
    filename : str
        ptx pointcloud filename.
    radius : float, optional
        only keep data within this horizontal radius to scanner. The default is np.nan.
    fields : Union[None, list], optional
        list of fields to keep. Keep all fields if fields = None.
        use fields = [] if you don't want to keep any scalar fields
        The default is None.
    header_file : Union[None, str], optional
        filename for the header file. The default is None.
        If None, name it as filename + "_headers.asc".
    write_cropped : bool, optional
        option to write cropped pointcloud. The default is False.
    cropped_file : str
        filename for the cropped pointcloud. The default is None.

    Raises
    ------
    Exception
        Filetype not ptx
        The number of column of input file does not match known format.

    Returns
    -------
    transmat : NDArray
        ptx header text.
    pcd_df : Dataframe
        Pointcloud data.
    pcd_cc: ccPointcloud
        Pointcloud data.
    """
    filetype = os.path.splitext(filename)[1]
    if(filetype!='.ptx'):
        raise Exception("Filetype is not ptx for function read_ptx")
    headers = []

    # Assuming only one header
    logging.info(f"Importing {filename}")
    with open(filename) as infile:
        headers = [next(infile) for _ in range(10)]
    if(header_file == None):
        header_file = filename[0:-4] + "_headers.asc"
    with open(header_file,'w') as outfile:
        outfile.writelines(headers)
    transstr = headers[6:10] 
    transmat = np.array([list(map(float, row.split())) for row in transstr]) 
    logging.info("Done reading header")

    pcd_df = pd.read_csv(filename,delimiter=" ",header=None,skiprows=10,engine='pyarrow')
    if(pcd_df.shape[1]==7):
        pcd_df.columns = ['X', 'Y', 'Z', 'I', 'R', 'G', 'B']
    elif(pcd_df.shape[1]==4):
        pcd_df.columns = ['X', 'Y', 'Z', 'I']
    else:
        raise Exception("The number of column of input file does not match known format.")
    logging.info("Done importing")

    # remove rows if all elements are zero
    mask = (np.abs(pcd_df.to_numpy()) >= 0.00001).any(axis=1)
    pcd_df = pcd_df[mask]

    # Only keep points within a horizontal radius to scanner
    if(~np.isnan(radius)):
        # The center is (0,0,0) for single scan ptx
        center = np.array([[0,0]])
        distance = np.linalg.norm(pcd_df.iloc[:, :2].to_numpy() - center, axis=1)
        pcd_df = pcd_df[distance <= radius]

    pcd_cc = df2cc(pcd_df,fields,filename)

    if(write_cropped):
        if(cropped_file==None):
            cropped_file = filename[:-4] + "_cropped.asc"
            logging.warning(f"filename for cropped pointcloud not provided, use default:{cropped_file}")
        ccsave(pcd_cc,cropped_file)

    return transmat,pcd_df,pcd_cc

@timer("importing pointcloud")
def cc_load(filename: str,radius: float=np.nan,fields:Union[None,list]=None,
            write_cropped: bool=False,cropped_file: Union[None,str]=None
            ) -> cc.ccPointCloud:
    """ Read single-scan ptx file into a dataframe
    TODO: check if the file is multi-scan
    Parameters
    ----------
    filename : str
        ptx pointcloud filename.
    radius : float, optional
        only keep data within this horizontal radius to scanner. The default is np.nan.
    fields : Union[None, list], optional
        list of fields to keep. Keep all fields if fields = None.
        use fields = [] if you don't want to keep any scalar fields
        The default is None.
    write_cropped : bool, optional
        option to write cropped pointcloud. The default is False.
    cropped_file : str
        filename for the cropped pointcloud. The default is None.

    Raises
    ------
    Exception
        Filetype not ptx
        The number of column of input file does not match known format.

    Returns
    -------
    pcd_df : Dataframe
        Pointcloud data.
    pcd_cc: ccPointcloud
        Pointcloud data.
    """
    ## TODO:  add a filetype check:


    # Assuming only one header
    logging.info(f"Importing {filename}")
    pcd_cc = cc.loadPointCloud(filename,mode=cc.CC_SHIFT_MODE.NO_GLOBAL_SHIFT) # Assume no shift is the global coordinate
    logging.info("Done importing")

    coord = pcd_cc.toNpArrayCopy()

    # Only keep points within a horizontal radius to scanner
    if(~np.isnan(radius)):
        # Assumes the center is (0,0,0)
        center = np.array([[0,0]])
        distance = np.linalg.norm(coord[:, :2] - center, axis=1)
        pcd_cc.addScalarField('distance')
        dic = pcd_cc.getScalarFieldDic()
        sf = pcd_cc.getScalarField(dic['distance'])
        sf.fromNpArrayCopy(distance)
        pcd_cc.setCurrentOutScalarField(dic['distance'])
        pcd_cc = cc.filterBySFValue(0,radius,pcd_cc)
    
    dic = pcd_cc.getScalarFieldDic()
    for sf in dic:
        if sf not in fields:
            pcd_cc.deleteScalarField(dic[sf])

    if(write_cropped):
        if(cropped_file==None):
            cropped_file = filename[:-4] + "_cropped.asc"
            logging.warning(f"filename for cropped pointcloud not provided, use default:{cropped_file}")
        ccsave(pcd_cc,cropped_file)

    return pcd_cc

def load_ascii_pcd(filename: str, out_type: str='cc',header:Union[None, int] = 0,
             engine='pyarrow',fields:Union[None,list]=None) -> Union[pd.DataFrame,cc.ccPointCloud]:
    """ read pointcloud from file (currently only supports asc)
    TODO: support other input format
    NOTE: the cloudcompare Importfile method does not support parallel, 
    so it is slower than the pd.read_csv with pyarrow. So we use pd.readcsv 
    for reading pointclouds

    Parameters
    ----------
    filename : str
        File name.
    out_type : str, optional
        output pointcloud datatype, 'cc' or 'df'
        by default 'cc'
    header : int or None
        row index for data file header
    engine : str, optional
        engine for pd.read_csv, by default 'pyarrow'
    fields : Union[None, list], optional
        fields to keep. The default is None.

    Raises
    ------
    Exception
        If the file doesn't exist or if the file type is not accepted..

    Returns
    -------
    pcd_cc : cc.ccPointCloud
        loaded ccPointCloud.

    """

    # Check if the file exists
    if not os.path.exists(filename):
        raise Exception(f"File {filename} not found")
    
    # Determine the file type
    filetype = os.path.splitext(filename)[1]
    if filetype == '.asc':
        # Import point cloud
        pcd_df = pd.read_csv(filename, delimiter=" ", header=header, engine=engine)     
        if out_type == 'df':
            return pcd_df
        elif out_type == 'cc': # convert to ccPointCloud object
            return df2cc(pcd_df,fields,filename)
    else:
        raise Exception("File type not accepted")
    
def ptxheader2transmat(filename:str) -> NDArray:
    """ Read ptx header (or ptx) file and return the transformation matrix
    Assuming only one scan

    Parameters
    ----------
    filename : str
        ptx pointcloud filename.
    
    Returns
    -------
    transmat : NDArray
        3x3 transformation matrix
        
    """
    with open(filename) as infile:
        headers = [next(infile) for _ in range(10)]
    transstr = headers[6:10]    
    transmat = np.array([list(map(float, row.split())) for row in transstr]) 
    return transmat
def ccsave(pcd_cc:cc.ccPointCloud,filename:str,method:str = "parallel"):
    """ Write pointcloud to file
    

    Parameters
    ----------
    pcd_cc : cc.ccPointCloud
        Point cloud to save.
    filename : str
        File name.
    method : str, optional
        Save method, cc: CloudComPy method; parallel:Pyarrow parallel save. The default is "parallel".

    Returns
    -------
    pcd_df : dataframe
        return pointcloud dataframe.

    """
    @timer(f"writing {filename}")
    def save():
        if(method=="cc"):
            cc.SavePointCloud(pcd_cc,filename)             #save the point cloud to a file
        elif(method=="parallel"):
            coord = pcd_cc.toNpArrayCopy()
            pcd_df = pd.DataFrame(coord)
            pcd_df.columns = ['X','Y','Z']
            
            dic = pcd_cc.getScalarFieldDic()
            # nsf = pcd_cc.getNumberOfScalarFields()
            for sf in dic:
                pcd_df[sf.replace(" ","_").replace('"', '')]=pcd_cc.getScalarField(dic[sf]).toNpArrayCopy()
            
            temp = pa.Table.from_pandas(pcd_df,preserve_index = False)
            write_options = csv.WriteOptions(include_header=True, delimiter=' ')
            csv.write_csv(temp, filename, write_options=write_options)
            return pcd_df
        else:
            raise Exception("method not recognized, supported methods are 'cc' or 'parallel'")
    save()

def df2cc(df: pd.DataFrame, fields:Union[None,list]=None,
          pcd_name:str='')-> cc.ccPointCloud:
    """convert pointcloud in dataframe format to ccPointCloud

    Parameters
    ----------
    df : pd.DataFrame
        pointcloud dataframe
        
    fields : Union[None,list], optional
        list of fields to keep. Keep all fields if fields = None.
        use fields = [] if you don't want to keep any scalar fields
        The default is None.
    pcd_name : str, optional
       name for ccPointCloud , by default ''

    Returns
    -------
    pcd_cc : cc.ccPointCloud
        cc pointcloud
    """
    pcd_cc = cc.ccPointCloud(pcd_name)
    pcd_cc.coordsFromNPArray_copy(df.iloc[:, :3].to_numpy())
    
    # add scalar fields
    if(fields==None):
        sel_fields = df.columns[3:].tolist()
    else:
        sel_fields = fields
    
    for field in sel_fields:
        if(field in df.columns):
            pcd_cc.addScalarField(field)
            dic = pcd_cc.getScalarFieldDic()
            sf = pcd_cc.getScalarField(dic[field])
            sf.fromNpArrayCopy(df[field].to_numpy())
        else:
            logging.warning(f"Importing scalar field {field} failed, do not exist in dataframe")    
    
    return pcd_cc


@timer("noise filtering")
def noise_filtering(pcd_cc:cc.ccPointCloud,SOR_KNN: int = 6, SOR_std: float = 1.0,
                    write_filtered:bool=False,filtered_file:Union[None,str]=None
                    ) -> cc.ccPointCloud:
    """noise filtering

    Parameters
    ----------
    pcd_cc : cc.ccPointCloud
        input pointcloud
    SOR_KNN : int, optional
        number of neighbors,parameter for SOR noise filtering, by default 6
    SOR_std : float, optional
        standard deviation, parameter for SOR noise filtering, by default 1.0
    write_filtered : bool, optional
        wrilte filtered point cloud if True, by default False
    filtered_file : Union[None,str], optional
        file name for filtered point cloud, by default None

    Returns
    -------
    cc.ccPointCloud
        filtered point cloud

    Raises
    ------
    Exception
        filtered_file not provided
    """
    logging.info("Start noise filtering")
    refCloud = cc.CloudSamplingTools.sorFilter(pcd_cc,knn=SOR_KNN,nSigma=SOR_std)
    (pcd_cc, res) = pcd_cc.partialClone(refCloud)
    
    if(write_filtered):
        if(filtered_file==None):
            raise Exception("filename for filtered pointcloud needs to be provided")
        ccsave(pcd_cc,filtered_file)
    return pcd_cc

@timer("ground filtering")
def ground_filtering(pcd_cc:cc.ccPointCloud,topo_file:str,clothResolution:float = 0.5,filter_ground:bool=True,
                     get_aboveground_height:bool=True,write_mesh:bool=True,write_ground:bool=False,write_aboveground:bool=True,
                     mesh_file:Union[None,str]=None,ground_file:Union[None,str]=None,aboveground_file:Union[None,str]=None) -> cc.ccPointCloud:
    """ ground filtering using cloudcompare CSF method
    also writes the ground mesh, and calculate abovegournd height

    Parameters
    ----------
    pcd_cc : cc.ccPointCloud
        input pointcloud
    topo_file : str
        filename for topography
    clothResolution : float, optional
        parameter for cc.CSF, by default 0.5
    filter_ground : bool, optional
        filter ground points if true, by default True
    get_aboveground_height : bool, optional
        calculate aboveground height and add to scalar field if true, by default True
    write_mesh : bool, optional
        write the mesh coordinates of ground if true, by default True
    write_ground : bool, optional
        write ground points to file if true, by default False
    write_aboveground : bool, optional
        write aboveground points to file if true, by default True
    mesh_file : Union[None,str], optional
        file for mesh coordinates, by default None
    ground_file : Union[None,str], optional
        file for ground points, by default None
    aboveground_file : Union[None,str], optional
        file for aboveground points, by default None
    
    Returns
    -------
    cc.ccPointCloud
        filtered non-ground point cloud
    """
    logging.info("Filtering ground points")
    clist=cc.CSF.computeCSF(pcd_cc,clothResolution = clothResolution,computeMesh=True) 
    # clist[0]: ground point cloud, clist[1]: non-ground point cloud,[2]optional cloth mesh
    if filter_ground:
        if(type(clist[1])==cc.ccPointCloud):
            pcd_cc = clist[1] # non-ground points
        else:
            logging.warning("Ground classification seem failed")
            pcd_cc = clist[-2] # uses -2 instead of 1 here, because there could be only one pointcloud object if the pointcloud is a small subset of points
    
    if write_ground:
        if ground_file==None:
            ground_file = topo_file[:-9] + "_cropped.asc"
            logging.warning(f"filename for ground pointcloud not provided, use default:{ground_file}")
        ccsave(clist[0],ground_file)

                
    # Write the ground mean and max file
    ground_coord = clist[0].toNpArrayCopy()
    groundz_mean = np.mean(ground_coord[:,2])
    groundz_median = np.median(ground_coord[:,2])
    groundz_max = np.max(ground_coord[:,2])
    with open(topo_file, 'w') as f:
        f.write(f"{groundz_mean} {groundz_median} {groundz_max}\n")

    mesh_cloud = clist[-1].getAssociatedCloud()
    mesh_coord = mesh_cloud.toNpArrayCopy()
    # mesh_coord = clist[-1].IndexesToNpArray_copy()
    if write_mesh:
        if mesh_file==None:
            mesh_file = topo_file[:-9] + "_groundmesh.asc"
            logging.warning(f"filename for mesh not provided, use default:{mesh_file}")
        np.savetxt(mesh_file, mesh_coord, delimiter=',', fmt='%.6f')
    
    if get_aboveground_height:
        aboveground_coord = pcd_cc.toNpArrayCopy()
        # Extract ground coordinates
        ground_xy = mesh_coord[:, :2]  # or ground_df[['X', 'Y']].values
        ground_z = mesh_coord[:, 2]

        # Points to compute above-ground height
        points_xy = aboveground_coord[:, :2]
        points_z = aboveground_coord[:, 2]

        # Interpolate ground Z at point XY locations
        ground_z_interp = griddata(ground_xy, ground_z, points_xy, method='linear')

        # Compute height above ground
        height_above_ground = points_z - ground_z_interp
        pcd_cc.addScalarField('HeightAboveGround')
        dic = pcd_cc.getScalarFieldDic()
        sf = pcd_cc.getScalarField(dic['HeightAboveGround'])
        sf.fromNpArrayCopy(height_above_ground)
    
    if filter_ground and write_aboveground:
        if aboveground_file==None:
            aboveground_file = topo_file[:-9] + "_aboveground.asc"
            logging.warning(f"filename for aboveground pointcloud not provided, use default:{aboveground_file}")
        ccsave(pcd_cc,aboveground_file)

    return pcd_cc 

@timer("angle calculation")
def angles_calc(pcd_cc: cc.ccPointCloud,transmat: Union[None, NDArray]=None,
                max_scatter: float=85,filter_scatter: bool=True,write_angles:bool=True,
                angles_file:Union[None,str]=None) -> cc.ccPointCloud:
    """ Calculate the zenith (Dip), azimuth (Dipdir), and scattering angles.
    

    Parameters
    ----------
    pcd_cc : cc.ccPointCloud
        Point cloud.
    transmat : NDArray, optional
        Transformation matrix. The default is None.
    max_scatter : float, optional
        Maximum scattering angle. The default is 85.
    filter_scatter : bool, optional
        Filter points with scattering angle above threshold. The default is True.

    Returns
    -------
    pcd_cc : cc.ccPointCloud
        Output Point cloud.

    """
    # Estimate the normals
    logging.info("Start normal estimation")
    time0 = time.time()
    cc.computeNormals([pcd_cc],useScanGridsForComputation=False,defaultRadius=0.01,useScanGridsForOrientation=False,preferredOrientation=cc.Orientation.PLUS_Z)
    coordinates = pcd_cc.toNpArrayCopy()    # coordinates as a numpy array
    normals = pcd_cc.normalsToNpArrayCopy()
    time1 = time.time()
    logging.info(f"Done normal estimation, took {time1-time0}")
    
    # Calculate scatter angle
    scat = scatter(coordinates, normals, transmat)
    pcd_cc.addScalarField("scat")
    dic = pcd_cc.getScalarFieldDic()
    sf_scat = pcd_cc.getScalarField(dic['scat'])
    sf_scat.fromNpArrayCopy(scat)

     # filter out points with large scattering angle
    if(filter_scatter):
        dic = pcd_cc.getScalarFieldDic()
        pcd_cc.setCurrentOutScalarField(dic['scat'])
        pcd_cc = cc.filterBySFValue(0,max_scatter,pcd_cc)
    
    # Convert normals to zenith and azimuth
    pcd_cc.convertNormalToDipDirSFs() 
    
    pcd_cc.unallocateNorms() # remove normal fields
    if(write_angles):
        if(angles_file==None):
            raise Exception("angle_file should be provided")
        ccsave(pcd_cc,angles_file)
    return pcd_cc

# Function to calculate the scattering angle
def scatter(coordinates: NDArray, normals: NDArray, 
            transmat: Union[None, NDArray]=None) -> NDArray:
    """ Calculate scattering angle
    

    Parameters
    ----------
    coordinates : NDArray
        Point cloud coordinates.
    normals : NDArray
        Point cloud normals.
    transmat : NDArray, optional
        None or 4*4 nparray. 
        If the coordinates for 'coordinates' and 'normals' are not the scanner coordination (scanner at [0,0,0]),
        use transmat to provide transformation matrix. None means no transformation needed.
        The default is None.

    Returns
    -------
    scatter_deg : NDArray
        Scattering angle in degree.

    """
    coord_unit = coordinates / np.repeat(np.sqrt(np.sum(coordinates**2, axis=1)).reshape(-1, 1),3,axis=1)
    
    if transmat is not None: ### update needed
        # transinv4 = np.linalg.inv(np.transpose(transmat))
        # transinv3 = np.linalg.inv(np.transpose(transmat[:3, :3]))
        transinv4 = np.linalg.inv(transmat) ## double check if inversion is needed
        transinv3 = np.linalg.inv(transmat[:3, :3])
        
        coord_vec = np.hstack((coord_unit, np.ones((coord_unit.shape[0], 1))))
        
        trans_scat = np.matmul(coord_vec,transinv4)
        trans_scat = trans_scat[:,:3] / np.matmul(np.sqrt(np.sum(trans_scat[:,:3] ** 2, axis=1)).reshape(-1, 1),
                                                  np.ones((1,3)))
        trans_normal = np.matmul(normals,transinv3)
        dot = (trans_scat * trans_normal).sum(axis=1)
        raise Exception("Need to check whether inversion is needed based on what transformation matrix is provided")
        
    else: # scanner is at (0,0,0), coordination is the scanning vector
        dot = (coord_unit*normals).sum(axis=1)
    
    dot = np.abs(dot)
    dot = np.where(dot <= 1, dot, 1)
    scatter_deg = np.degrees(np.arccos(dot))
    return scatter_deg

# Function to calculate zenith and azimuth angle from scanner (0,0,0)
def RZARAA(dat: pd.DataFrame, deg:bool =False) -> pd.DataFrame:
    """calculate zenith and azimuth angle from scanner (0,0,0)

    Parameters
    ----------
    dat : pd.DataFrame
        input dataframes, with columns X,Y,Z
    deg : bool, optional
        if true, return angles in degrees. Otherwise, return angles in radian.
        by default False

    Returns
    -------
    dat : pd.DataFrame
        dataframe with columns inc and incazi for zenith and azimuth angles
    """
    dat = dat.copy()
    r = np.sqrt(dat['X']**2 + dat['Y']**2 + dat['Z']**2)
    inc = np.arccos(dat['Z'] / r)
    l = np.sqrt(dat['X']**2 + dat['Y']**2)
    incazi = np.arccos(dat['X'] / l)
    incazi[dat['Y'] < 0] = 2 * np.pi - incazi[dat['Y'] < 0]
    if deg:
        inc = np.degrees(inc)
        incazi = np.degrees(incazi)
    dat['inc'] = inc
    dat['incazi'] = incazi
    return dat

def shift_pcd(pcd_cc: cc.ccPointCloud, shift: NDArray) -> cc.ccPointCloud:
    """shift pointcloud coordinates

    Parameters
    ----------
    pcd_cc : cc.ccPointCloud
        pointcloud at original resolution
    shift : NDArray
        shift vector

    Returns cc.ccPointCloud
        shifted pointcloud
    """
    coordinates = pcd_cc.toNpArrayCopy()
    coordinates[:,0] = coordinates[:,0] + shift[0]
    coordinates[:,1] = coordinates[:,1] + shift[1]
    coordinates[:,2] = coordinates[:,2] + shift[2]
    pcd_cc.setCoordinates(coordinates)
    pcd_cc.shift(shift)
    return pcd_cc

@timer("subsample")
def subsample(pcd_cc: cc.ccPointCloud,SS_size: float=0.01,write_ss: bool=True,
              ss_file: Union[None,str]=None) -> cc.ccPointCloud:
    """subsample pointcloud using the resampleCloudSpatially method: 
    constrain the minimal distance between points to SS_size

    Parameters
    ----------
    pcd_cc : cc.ccPointCloud
        pointcloud at original resolution
    SS_size : float, optional
        minimal distance between points, by default 0.01
    write_ss : bool, optional
        if true, write subsampled pointcloud, by default True
    ss_file : Union[None,str], optional
        filename for subsampled pointcloud, by default None

    Raises
    ------
    Exception
        filename (ss_file) not provided, but write_ss is True

    Returns
    -------
    pcd_ss : cc.ccPointCloud
        Subsampled pointcloudS.
    """
    logging.info("Subsampling pointcloud")
    refCloud = cc.CloudSamplingTools.resampleCloudSpatially(pcd_cc,SS_size)
    (pcd_ss, res) = pcd_cc.partialClone(refCloud)
    pcd_ss.setName("pcd_ss")
    if(write_ss):
        if(ss_file==None):
            raise Exception("file name for subsampled pointcloud not provided")
        ccsave(pcd_ss,ss_file)  
    return pcd_ss

@timer("leaf-wood classification")
def classification_woodcl(pcd_ss: cc.ccPointCloud,config: dict,
                          write_class_ss: bool=True, class_ss_file:Union[None,str]=None,
                          label: str='label') -> cc.ccPointCloud:
    """Leaf-wood classification using the 3dSegFormer method
    Adapted from: https://github.com/truebelief/cc-TreeAIBox-plugin

    The new pointcloud has a scalar field 'label'. 0 for branch points, 1 for leaf points

    Parameters
    ----------
    pcd_ss : cc.ccPointCloud
        input pointcloud, recommend using a subsampled pointcloud
    config : dict
        configuration for the woodcl leaf-wood classification
    write_class_ss : bool, optional
        if true, write classified pointcloud, by default True
    class_ss_file : Union[None,str], optional
        output filename for classified pointcloud, by default None

    Returns
    -------
    pcd_class_ss: cc.ccPointCloud
        classified pointcloud
    """
    logging.info("Start leaf-wood classification using the 3dSegFormer method")

    # setting up the woodcl method
    sys_path_add = config['sys_path_add']
    sys.path.insert(0, sys_path_add)
    from woodCls import apply_wood_cls # 3dSegFormer leaf-wood classification function
    config_file = config['config_file']
    model_path = config['model_path']
    use_cuda = config['use_cuda']
    progress_bar = config['progress_bar']

    # Convert the CloudComPy point cloud to numpy first
    pcd_ss_np = np.array(pcd_ss.toNpArrayCopy())

    # Then call woodCls correctly
    pcd_pred = apply_wood_cls(
           config_file,
           pcd_ss_np,
           model_path,
           use_cuda=use_cuda,
           progress_bar=progress_bar
    )
    logging.info("Done classification\n")

    logging.info("Done classification\n")


    # add classification result as scalar field
    label = config['class_sel']['label']
    pcd_class_ss = pcd_ss
    pcd_class_ss.addScalarField(label)
    dic = pcd_class_ss.getScalarFieldDic()
    sf_class = pcd_class_ss.getScalarField(dic[label])


    print("pcd_pred type:", type(pcd_pred))
    print("pcd_pred shape:", getattr(pcd_pred, "shape", "no shape"))


    sf_class.fromNpArrayCopy(pcd_pred)    
    if write_class_ss:
        if(class_ss_file==None):
            raise Exception("file name for classified pointcloud not provided")
        else:
            ccsave(pcd_class_ss,class_ss_file)     
    return pcd_class_ss

@timer("leaf-wood classification")
def classification_FSCT(ss_file: str, config: Dict):
    """Leaf-wood classification using the FSCT method

    Adapted from: https://github.com/SKrisanski/FSCT
    Go to ./FSCT/scripts/run_leaf_wood.py for further settings
    TODO: some steps in FSCT preprocessing can be skipped

    The new pointcloud has a scalar field 'label'. 0 for ground points, 1 for leaf points, 2 for leaf points 

    Parameters
    ----------
    ss_file : str
        Pointcloud filename. Recommend using a subsampled pointcloud
    config : dict
        configuration for the FSCT leaf-wood classification
    """
    logging.info("Start leaf-wood classification using the FSCT method")
    sys_path_add = config['sys_path_add']
    sys.path.insert(0, sys_path_add)
    from run_leaf_wood import FSCT_semantic
    
    FSCT_semantic(ss_file, config['model_path'], config['use_CPU_only'])

@timer("projecting classification results to original resolution pointcloud")
def SFprojection(pcd_target:cc.ccPointCloud,pcd_source:cc.ccPointCloud,fields: str='label',
                 write_projected:bool=True,projected_file:Union[None,str]=None
                 ) -> cc.ccPointCloud:
    """Project a scalar field from the one pointcloud (pcd_ss) to another pointcloud (pcd_cc)

    Parameters
    ----------
    pcd_target : cc.ccPointCloud
        Target pointcloud
    pcd_source : cc.ccPointCloud
        Pointcloud with scalar field to be projected
    fields : str, optional
        name of the scalar field, by default 'label'
    write_projected : bool
        if true, write projected pointcloud
    projected_file : Union[None,str]
        output filename
    

    Returns
    -------
    pcd_target : cc.ccPointCloud
        Target pointcloud with projected scalar field
    """
    dic = pcd_source.getScalarFieldDic()
    cc_interp_para = cc.interpolatorParameters()
    cc_interp_para.method = cc.INTERPOL_METHOD.NEAREST_NEIGHBOR
    cc.interpolateScalarFieldsFrom(pcd_target, pcd_source, [dic[fields]],cc_interp_para)
    if write_projected:
        if(projected_file==None):
            raise Exception("file name for classified pointcloud not provided")
        else:
            ccsave(pcd_target,projected_file)
    return pcd_target                          

@timer("voxelization")
def voxelize_pcd(pcd_df: pd.DataFrame, voxel_size: float,fields_method:dict,
                 class_sel: dict={'label': 'label','sel': [1],},scatter_filter:float=85,
                 counts_thresh = 5, write_voxel:bool=False,voxel_file:Union[None,str]=None) -> pd.DataFrame:
    """Voxelize pointcloud

    Parameters
    ----------
    pcd_df : pd.DataFrame
        pointcloud dataframe    
    voxel_size : float
        voxel size
    fields_method : dict
        dictionary of fields and their aggregation method
    class_sel : dict, optional
        dictionary of classification parameters, by default {'label': 'label','sel': [1],}
    scatter_filter : float, optional
        scatter angle filter, by default 85
    counts_thresh : int, optional
        counts threshold, by default 5
    write_voxel : bool, optional
        if true, write voxelized pointcloud, by default False
    voxel_file : Union[None,str], optional
        output filename, by default None

    Returns
    -------
    voxelized_df : pd.DataFrame
        voxelized pointcloud dataframe
    """

    pcd_df = pcd_df.copy() # makesure it is unchanged outside functoin
    # filter based on classification result
    pcd_df = pcd_df[pcd_df[class_sel['label']].isin(class_sel['sel'])] 
    
    ## scatter angle filter 
    pcd_df = pcd_df[pcd_df['scat']<scatter_filter]
    pcd_df = pcd_df[pcd_df['Dip_(degrees)'].notna()]
    
    voxel_indices = np.floor(pcd_df[['X', 'Y', 'Z']].values / voxel_size).astype(np.int32)
    pcd_df.loc[:, ['X_voxel', 'Y_voxel', 'Z_voxel']] = voxel_indices

    # Aggregation
    voxelized_df = aggregate_voxel_data(pcd_df,['X_voxel', 'Y_voxel', 'Z_voxel'],fields_method)

    voxelized_df = voxelized_df.drop(columns=['X_voxel', 'Y_voxel', 'Z_voxel'])
    voxelized_df = voxelized_df[voxelized_df['counts']>=counts_thresh]

    if write_voxel:
        if(voxel_file==None):
            raise Exception("file name for voxelized pointcloud not provided")
        else:
            voxelized_df.to_csv(voxel_file,sep=' ',index=False)
            logging.info("Done writing voxelized file")
    return voxelized_df

def pcd_transform(pcd:pd.DataFrame,transmat:NDArray) -> pd.DataFrame:
    """Apply transformation matrix transmat to pointcloud
    TODO: for cc.Pointcloud 
    Parameters
    ----------
    pcd : pd.DataFrame
        pointcloud to be transformed
    transmat : NDArray
        transformation matrix

    Returns
    -------
    pcd_trans: pd.DataFrame
        transformed pointcloud
    """
    pcd_trans = pcd.copy()
    coord_vec = np.hstack((pcd_trans.iloc[:,:3].to_numpy(), np.ones((pcd_trans.shape[0], 1))))
    global_vec = np.matmul(coord_vec,transmat)
    pcd_trans.iloc[:,:3] = global_vec[:,:3]
    return pcd_trans

def get_highest_frequency_bin(values:ArrayLike, bin_size:int =10):
    """ Get the midpoint of the highest frequency bin of data
    

    Parameters
    ----------
    values : ArrayLike
        data column.
    bin_size : int, optional
        bin size. The default is 10.

    Returns
    -------
    float
        Midpoint of the highest frequency bin, or NaN if input is empty.

    """
    if len(values) == 0:
        return np.nan

    bin_edges = np.arange(0, values.max() + bin_size, bin_size)
    
    bin_indices = np.digitize(values, bins=bin_edges, right=False) - 1

    bincounts = np.bincount(bin_indices, minlength=len(bin_edges) - 1)

    # Find the bin(s) with the maximum count
    max_count = bincounts.max()
    if max_count == 0:
        return np.nan
    max_bins = np.where(bincounts == max_count)[0]

    # Midpoints of the max bins
    midpoints = (bin_edges[max_bins] + bin_edges[max_bins + 1]) / 2

    # if there are multiple bins with max frequency, return their mean
    return np.mean(midpoints)

def aggregate_voxel_data(pcd_df:pd.DataFrame, voxel_columns:List[str],
                         agg_fields: Dict[str, Union[str, Callable]]):
    """ Voxelize point cloud data based on provided field and method
    

    Parameters
    ----------
    pcd_df : pd.DataFrame
        Point cloud.
    voxel_columns : List[str]
        List of variables to gruop point cloud data. For example, ['X_voxel', 'Y_voxel', 'Z_voxel'].
    agg_fields : Dict[str, Union[str, Callable]
        output column name: [column to be aggregated, method] .

    Returns
    -------
    voxelized_df : pd.DataFrame
        Voxelized dataframe.

    """

    # if the method is a string, check if there is corresponding function in the global namespace
    # if there is, replace it with the actual function
    for key, (value, func_name) in agg_fields.items():
        func = globals().get(func_name)  # Retrieve function by name
        if callable(func):
            agg_fields[key][1] = func  # Replace the string with the function reference

    agg_dict = {
        key: (value[0], value[1]) for key, value in agg_fields.items()
    }

    voxelized_df = pcd_df.groupby(voxel_columns).agg(**agg_dict).reset_index()


    return voxelized_df


@timer("bincount for pointcloud with angle information")
def bincount(pcd_df:pd.DataFrame, config:dict, filename: str, processed: bool = False,
             class_sel:dict = {'label':'label','sel': [1]},
             transmat: Union[NDArray, None]=None):
    """
    Count the points in each bin of zenith angle, azimuth angle, and Z.

    If processed is true, then also calculate the mean cosine of scattering angle (for G-function 
    calculation), and count the points in selected and unselected class.
    If processed is true, output column 'class' = 1: counts of points in selected class, 
    'class' = 0: counts of other points

    TODO: check nunpy.ufunc.at() for aggregation (FLIP)

    Parameters
    ----------
    pcd_df : DataFrame
        DataFrame containing the point cloud data with zenith angle (inc), azimuth angle (incazi), and Z.
    config : dict
        Dictionary containing the configuration parameters.
        z_min (float): Minimum Z value for counting (m).
        z_max (float): Maximum Z value for counting.
        z_res (float): Resolution of the Z bins.
        a_bin (float): Size of the bin for zenith angle (degrees).
        azi_bin (float): Size of the bin for azimuth angle (degrees).
        inc_range (float): Zenith angle range of scanner, this is 150 for Leica RTC360 .
    filename : str
        Name of the output file.
    processed : bool, optional
        Whether the pointcloud is processed or not, default is False.
    class_sel : dict, optional
        Dictionary containing the column name and selection criteria for the classification result.
        label (str): Column name of the classification result.
        sel (list): List of values to select (e.g., considered as leaf).
    transmat : Union[NDArray, None], optional
        For RTC360 pointcloud, this shoud be the tilt transformation matrix derived from the raw data,
        default is None.

    Returns
    -------
    DataFrame
        A DataFrame with the counts of points in each combined bin.
    """
    z_min = config['z_min']
    z_max = config['z_max']
    z_res = config['z_res']
    inc_range = config['inc_range']
    a_bin = config['a_bin']
    azi_bin = config['azi_bin']

    pcd_df = pcd_df.copy()
    if transmat is not None:
        coord = pcd_df.iloc[:,:3].to_numpy()
        pcd_df.iloc[:,:3] = np.matmul(coord,transmat)

    #calculate inclination angle of each point (XY)
    pcd_df=RZARAA(pcd_df, deg=True)
      
    # Bin the zenith, azimuth, and Z
    pcd_df.loc[:,'inc_bin'] = pd.cut(pcd_df['inc'], bins=np.arange(0, inc_range + a_bin, a_bin), labels=False)#.astype(int)
    pcd_df.loc[:,'incazi_bin'] = pd.cut(pcd_df['incazi'], bins=np.arange(0, 360 + azi_bin, azi_bin), labels=False)#.astype(int)
    pcd_df.loc[:,'z_bin'] = pd.cut(pcd_df['Z'], bins=np.arange(z_min, z_max + z_res, z_res), labels=False)#.astype(int)
    
    # calculate cosine of scattering angle, this is used for gap-probablity estimations
    if processed:
        pcd_df.loc[:,'cos_scat'] = np.cos(pcd_df['scat']/180*np.pi)
        pcd_df.loc[:,'class'] = pcd_df[class_sel['label']].isin(class_sel['sel']).astype(int)

        # Count rows in each combined bin
        combined_counts = pcd_df.groupby(['inc_bin', 'incazi_bin', 'z_bin','class']).agg(
            scatter=('scat', 'mean'),    # Mean of I for each voxel
            cos_scat=('cos_scat', 'mean'),    # Mean of I for each voxel
            counts=('X','count')
        ).reset_index()
    else:
        combined_counts = pcd_df.groupby(['inc_bin', 'incazi_bin', 'z_bin']).agg(
            counts=('X','count')
        ).reset_index()
    
    combined_counts.to_csv(filename,index=False)

    return combined_counts
    

# # Function to convert normals to leaf orientation and leaf angle (not used)
# def normals2angles(normals_df):
#     normals_df.columns = ['nX', 'nY', 'nZ']
#     Nsign = np.ones(len(normals_df))
#     Nsign[normals_df['nZ'] < 0] = -1.0
#     azi_rad = np.arctan2(Nsign * normals_df['nX'], Nsign * normals_df['nY'])
#     azi_rad[azi_rad < 0] += 2 * np.pi
#     zen_rad = np.arccos(np.abs(normals_df['nZ']))
#     azi_deg = np.degrees(azi_rad)
#     zen_deg = np.degrees(zen_rad)
#     return pd.DataFrame({'zen_deg': zen_deg,'azi_deg': azi_deg})

# # Function to get PTX header
# def getptxheader(input_file, overwrite, fileext):
#     ptxheader_file = input_file.replace(f".{fileext}", "_headers.asc")
#     if not os.path.exists(ptxheader_file) or overwrite:
#         os.system(f'python ptx_header.py {input_file}')
#     headers = read_ptxheaderfile(ptxheader_file)
#     return headers

# # Function to read PTX header file
# def read_ptxheaderfile(ptxheader_file):
#     headers = []
#     i = 0
#     with open(ptxheader_file, 'r') as f:
#         for line in f:
#             line_sp = line.split(' ')
#             if len(line_sp) == 1:
#                 i += 1
#                 headers.append([line.strip()] + [f.readline().strip() for _ in range(9)])
#     return headers

# # Function to calculate mean azimuth angle
# def mean_Azimuth(azimuth):
#     north_proj = np.cos(np.radians(azimuth))
#     east_proj = np.sin(np.radians(azimuth))
#     north_sum = np.sum(north_proj)
#     east_sum = np.sum(east_proj)
#     az_mean = np.arctan2(east_sum, north_sum)
#     az_mean[az_mean < 0] += 2 * np.pi
#     return np.degrees(az_mean)

# # Function to calculate standard deviation of azimuth angle
# def sd_Azimuth(azimuth):
#     az_mean = mean_Azimuth(azimuth)
#     az_dif = azimuth - az_mean
#     az_dif[az_dif < -180] += 360
#     az_dif[az_dif > 180] -= 360
#     return np.std(az_dif)



# # Function to calculate relative zenith angle
# def RZA(dat, deg=False):
#     dat.columns = ['X', 'Y', 'Z']
#     r = np.sqrt(dat['X']**2 + dat['Y']**2 + dat['Z']**2)
#     inc = np.arccos(dat['Z'] / r)
#     if deg:
#         inc = np.degrees(inc)
#     return inc