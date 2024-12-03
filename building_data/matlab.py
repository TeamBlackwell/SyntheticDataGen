"""
Code to translate cityscape csv files into the required format for MATLAB
"""

"""
MATLAB requires a a 2D array of shape 4x2 for each building.
The first column of the array should be the x coordinates of the building's corners,
and the second column should be the y coordinates of the building's corners.

The 4 corners must be in clockwise order starting from the top left corner.
"""

from pathlib import Path
import pandas as pd


def convert_csv_to_matlab(csv_file_path: Path):
    csv_file = pd.read_csv(csv_file_path)
