"""
Weird name yes but its also 5 AM in the morning and I'm tired. I'll fix it later.
"""

import pandas as pd
import numpy as np
from pathlib import Path

# from .wind_sim_two_d import run_flow2d
from .phiflow_runner import run_flow


# @TODO: Need to find a better technique to resolve this
SPEED_X = 5
SPEED_Y = -5


# @TODO: Need to accept parameters for timing window and etc
def generate_windflow(cityscapes_dir: Path, output_dir: Path):

    cityscape_files = [f for f in cityscapes_dir.iterdir() if f.suffix == ".csv"]

    for cityscape_file in cityscape_files:
        cityscape = pd.read_csv(cityscape_file)

        # convert to numpy array
        cityscape = cityscape.to_numpy()
        cityscape = cityscape[:, 0:4]

        flow = run_flow(cityscape, 1, 2, 250, SPEED_X, SPEED_Y)

        output_path = output_dir / cityscape_file.name
        np.save(output_path.with_suffix(".npy"), flow)
        break
