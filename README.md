# Table of content
- [Table of content](#table-of-content)
- [Program description](#program-description)
- [Environment setup](#environment-setup)
  - [conda](#conda)
  - [python environment](#python-environment)
    - [CloudComPy Test](#cloudcompy-test)
    - [Systems other than windows](#systems-other-than-windows)
- [Running the program](#running-the-program)
  - [File requirements:](#file-requirements)
  - [Output files:](#output-files)
- [Major changes from TLSLeAF by Stovall et al:](#major-changes-from-tlsleaf-by-stovall-et-al)
- [Caveats](#caveats)
- [License](#license)
- [Citation](#citation)

# Program description
Go to [Environment-setup](#Environment-setup) for setting up the environment for the program.

This program takes TLS scan data and estimates leaf-related canopy structural traits including canopy height, leaf angle, leaf area index, plant area index, clumping, etc. It also provides the vertical profiles of plant area volume density, leaf area volume density, effective plant (or leaf) area volume density, clumping index (calculated based on leaf or plant area), and leaf zenith angle (mean and standard deviation).  

![profiles](readme_plots/profile.png)

Below is a workflow of the algorithm. It calls functions from CloudCompare for noise filtering, ground filtering, and normal estimation. So if you want to use different parameters, you can test them in CloudCompare and directly provide them to the algorithm.
![alt text](readme_plots/flowchart.png)

# Environment setup
##  conda
Install miniconda (command line, without graphical interface) or anaconda (with graphical interface). As you will need and only need to use the command line for conda, the light-weight miniconda is recommended unless you will need anaconda for other project.

## python environment
The TLS processing relies on the CloudComPy (python wrapper for CloudCompare) and many python libraries, when not installed properly, there will be incompatability issues preventing the code to execute. Below is the environment setup tested with windows 10 and 11 computers with dedicated GPU and with the [CloudComPy windows 20240927 binary release](https://www.simulation.openfields.fr/index.php/cloudcompy-downloads/3-cloudcompy-binaries/5-windows-cloudcompy-binaries/113-cloudcompy310-20240927). Other systems will require different packages (refer to the [CloudComPy binary releases](https://www.simulation.openfields.fr/index.php/cloudcompy-downloads/3-cloudcompy-binaries) and [pytorch](https://pytorch.org/get-started/previous-versions/)).

Open anaconda prompt, and execuate the lines below one by one:
```
conda activate
conda update -y -n base -c defaults conda
```

If you are using **Windows**, clone or download this repository. Navigate to its directory in anaconda prompt: 

```
cd [directory]
```

and create the python environment: **Note** this will create a conda environment named CloudComPy310, if a environment with such name already exists, it might overwrite the existing environment 
```
conda env create -f environment.yml
```
If the above command does not work, see [conda command list.txt](<conda command list.txt>) for a list of command successfully used to set up the environment

Download the [CloudComPy windows 20240927 binary release](https://www.simulation.openfields.fr/index.php/cloudcompy-downloads/3-cloudcompy-binaries/5-windows-cloudcompy-binaries/113-cloudcompy310-20240927) and extract it to the desired directory
Navigate to the CloudComPy directory in Anaconda prompt
```
cd [directory]
```
execute:
```
envCloudComPy.bat
```
You should see: 
>Checking environment, Python test: import cloudComPy  
>Environment OK!  

If you see error, you might need to install visual studio (no need to add anything except for cmake), see
https://github.com/CloudCompare/CloudComPy/blob/master/doc/BuildWindowsConda.md

CloudComPy paths may need to be added to the IDE. You can add the code below before importing CloudComPy
```
# add CloudCompPy path, assuming it is in the same parent directory as the working directory
sys.path.insert(0, '..\\CloudComPy310\\CloudCompare\\')
sys.path.insert(0, '..\\CloudComPy310\\')
sys.path.insert(0, '..\\CloudComPy310\\CloudCompare\\cloudComPy\\')
```
If you would like to use spyder for developement, you can add paths to the paths PYTHONPATH in spyder, refer to:
https://github.com/CloudCompare/CloudComPy/blob/master/doc/UseWindowsCondaBinary.md

### CloudComPy Test
For addtional test for CloudComPy, see https://github.com/CloudCompare/CloudComPy/blob/master/doc/UseWindowsCondaBinary.md

### Systems other than windows
For other systems (e.g., linux), download the respective binaries https://www.simulation.openfields.fr/index.php/cloudcompy-downloads/3-cloudcompy-binaries, and follow the instructions on the conda environment for the specific binary. 
Install [pytorch](https://pytorch.org/get-started/previous-versions/) for cuda 12.1 for your system (This is for compatability with the [TreeAI](https://github.com/RongLi29/cc-TreeAIBox-plugin) leaf-wood separation).
Then check packages in [environment.yml](environment.yml) that does not exist in your conda environment (conda list), and install them

# Running the program
You need to run the program from the proper environment set up accoding to [Environment setup](#environment-setup). For example, if you use spyder, open anaconda prompt, start spyder with the proper environment by:
```
conda activate CloudComPy310
spyder
```
The main program is [PlantTriats_workflow.py](PlantTraits_workflow.py). You just need to change the data directory to run it. Parameters are all in [config.yaml](config.yaml). This gives the voxelized leaf angle file, and bincounts files needed for PAVD estimation.

Traits estimation is currently done in [site_level_analyses](site_level_analyses.py). This will be merged to [PlantTriats_workflow.py](PlantTraits_workflow.py).

## File requirements: 
Under data_dir, there should be .ptx point cloud [pointcloudname].ptx files
For multiscan projects, current code requires a separate .ptx files for each scan.
TODO: It should be very easy to add a check and deal with .ptx that has multiple scans in it.

For PAVD related calculation, corresponding [pointcloudname]_tiltmat.asc is also required.
This is used for correcting the tilt of the scanner from the nadir direction. Because in 
Leica RTC360 pointclouds, the Z axis is nadir, not the rotating axis. However, the scanning 
rotational axis is required for gap-probability based PAVD estimations. 
The tiltmat files can be obtained from raw Leica scan folder using script "Tools/tiltmat_from_raw.py".
If Z axis in pointcloud is the scanning axis instead of nadir, just put a 
3*3 unit matrix in the tiltmat file. 

## Output files:
Output files will be generated in the 'out_path' specified in [PlantTriats_workflow.py](PlantTraits_workflow.py):

    cropped_file = os.path.join(out_path,output_subfolder, base_name + "_cropped.asc") # point cloud cropped by the defined radius
    filtered_file = os.path.join(out_path,output_subfolder, base_name + "_filtered.asc") # point cloud after noise filtering
    angles_file = os.path.join(out_path,output_subfolder, base_name + "_angles.asc") # point cloud with normals and scattering angles as additional columns
    ground_file = os.path.join(out_path,output_subfolder, base_name + "_ground.asc") # ground points
    topo_file = os.path.join(out_path,output_subfolder, base_name + "_topo.asc") # topography
    ss_file = os.path.join(out_path,output_subfolder, base_name + "_ss.asc") # Subsampled pointcloud (XYZ only)
    class_ss_file = os.path.join(out_path,output_subfolder, base_name + "_ss_class.asc") # Subsampled pointcloud with leaf-wood classification lables
    class_file = os.path.join(out_path,output_subfolder, base_name + "_angles_class.asc") # Pointcloud at original resolution with estimated surface angles, scattering angles, and leaf-wood classification lables
    voxel_file = os.path.join(out_path,output_subfolder, base_name + "_angles_class_voxels.asc") # Voxelized pointcloud with estimated leaf angles
    voxel_global_file = os.path.join(out_path,output_subfolder, base_name + "_angles_class_voxels_global.asc") # Voxelized pointcloud with estimated leaf angles at the global coordinate
    classfreq_file = os.path.join(out_path,output_subfolder, base_name + "_bincount_05.asc") # point counts for PAVDe estimation
    ptxfreq_file = os.path.join(out_path,output_subfolder, base_name + "_ptxbincount_05.asc") # point counts for PAVD estimation


# Major changes from TLSLeAF by Stovall et al:
- Include a noise filtering step, use cloudcompare CSF for ground filtering
- Improve leaf-wood-separation by using the woodcl or FSCT algorithms
- Voxelization at the original resolution of point cloud: As the density of noise points is lower than the density of plant points in the point cloud, subsampled point clouds consist of a higher portion of noise points than the original resolution point cloud. Performing voxelization at the original resolution thus reduces the impact of noise on leaf angle estimations.
- Provide voxelized leaf angle base on mean and mode. Noise has a larger impact on the mean approach than the mode approach. So the mode approach is recommended.
- The algorithm is now in python, and processes except for the leaf-wood separation are faster.


# Caveats
- GPU is recommended if using the leaf-wood separation,
- The estimation of traits related to *leaf area, plant area, and clumping* **REQUIRES**
    - RAW data WITHOUT noise filtering
    - Each scan can be separated, and scan locations are availble
    - Scanner settings (how many rows and columns, the scaning angle range, whether z axis pointing to nadir)  

    This requirement is for gap-probablity-based plant area and clumping estimates that is most consistent with definitions in radiative transfer scheme. We might consider adding alternative leaf/plant area estimates in the future that does not have these requirements. **Leaf angle estimates are fine with filtered data.**


# License
# Citation