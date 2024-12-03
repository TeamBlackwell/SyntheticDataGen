# Synthetic Data Generation

Contains the code for generating data required for the PredictingLocalWindFields in 3D project. 

Generates the following data:

- 3D Wind flow field using Phiflow
- 3D Urban Map using quadtrees for intersection

### Usage

Install the required packages using the following command:

```
pip install -r requirements.txt
```

Run the following command to generate the data:

```
python generate_data.py
```