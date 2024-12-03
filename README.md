# Synthetic Data Generation

Contains the code for generating data required for the PredictingLocalWindFields in 3D project. 

Generates the following data:

- 3D Wind flow field using Phiflow
- 3D Cityscape using quadtrees for intersection (csv and mat files)

### Usage

Install the required packages using the following command:

```
pip install -r requirements.txt
```

Run the following command to generate the data:

```
python generate_data.py
```

## Notes

The 3D LiDAR simulation is conducted using the MATLAB UAV toolbox. MATLAB requires 4 corners of a rect to create a mesh. The CSV file contains the start and end coordinates for a rect. The .mat files contain the 4 corners of the rect.