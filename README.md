# Synthetic Data Generation

Contains the code for generating data required for the PredictingLocalWindFields in 3D project. 

Generates the following data:

- 3D Wind flow field using Phiflow
- 3D Cityscape using quadtrees for intersection (csv and mat files)
- PointCloud Data from a simulated LiDAR on a UAV using the MATLAB UAB ToolBox

### Usage

Install the required packages using the following command:

```bash
pip install -r requirements.txt
```

Run the following command to generate the cityscapes:

```bash
python main.py cityscapes
```

Run the following command to generate drone positions:

```bash
python main.py drone
```

The MATLAB files present in this repo must be run from the MATLAB editor application.

## Notes

The 3D LiDAR simulation is conducted using the MATLAB UAV toolbox. MATLAB requires 4 corners of a rect to create a mesh. The CSV file contains the start and end coordinates for a rect. The .mat files contain the 4 corners of the rect.

### Output File Structure

```
data/
├─ cityscapes/
│  ├─ city_0.csv
│  ├─ city_1.csv
│  ├─ city_(...).csv
│  ├─ city_60.csv
├─ drone_positions/
│  ├─ city_0.csv
│  ├─ city_1.csv
│  ├─ city_(...).csv
│  ├─ city_60.csv
├─ matlab_meshes/
│  ├─ city_0.mat
│  ├─ city_1.mat
│  ├─ city_(...).mat
│  ├─ city_60.mat
├─ pointclouds/
│  ├─ city_0/
│  │  ├─ pointcloud_1.csv
│  │  ├─ pointcloud_(...).csv
│  │  ├─ pointcloud_10.csv
│  ├─ city_1/
│  │  ├─ pointcloud_1.csv
│  │  ├─ pointcloud_(...).csv
│  │  ├─ pointcloud_10.csv
│  ├─ city_60/
│  │  ├─ pointcloud_1.csv
│  │  ├─ pointcloud_(...).csv
│  │  ├─ pointcloud_10.csv
```


## MATLAB Toolbox requirements
- UAV Toolbox
- Aerospace Blockset
- Aerospace Toolbox
- Control System Toolbox
- Navigation Toolbox
- Stateflow
- Simulink
- Simulink 3D Animation
