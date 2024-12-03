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
import numpy as np
from scipy.io import savemat
from tqdm import tqdm


def convert_csv_to_matlab(csv_df: pd.DataFrame, output_path: Path):

    # iterate over each row in the csv file and create the 4 corners of the building
    buildingCoords = []
    buildingHeights = []
    for _, row in csv_df.iterrows():
        x1, y1, x2, y2 = row[["x1", "y1", "x2", "y2"]]
        height = row["height"]

        building = np.array(
            [
                [x1, y1],
                [x2, y1],
                [x2, y2],
                [x1, y2],
            ]
        )

        buildingCoords.append(building)
        buildingHeights.append(np.array([0, height]))
    savemat(
        output_path.with_suffix(".mat"),
        {"buildingCoords": buildingCoords, "buildingHeights": buildingHeights},
    )


def process_csv_to_matlab(cityscape_dir: Path, output_dir: Path):
    """
    Convert all csv files in the cityscape_dir to the required format for MATLAB to be saved in the output_dir.
    MATLAB files will contain buildingCoords and buildingHeights.
    """

    csv_files = [
        csv_file for csv_file in cityscape_dir.iterdir() if csv_file.suffix == ".csv"
    ]
    for csv_file in tqdm(csv_files):
        csv_df = pd.read_csv(csv_file)
        convert_csv_to_matlab(csv_df, output_dir / csv_file.stem)
