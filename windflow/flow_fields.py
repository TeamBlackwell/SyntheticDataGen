"""
Weird name yes but its also 5 AM in the morning and I'm tired. I'll fix it later.
"""

import pandas as pd
import numpy as np
from pathlib import Path

from tqdm import tqdm

from .wind_sim import run_flow

def batch_generate_windflow(cityscapes_dir: Path, output_dir: Path,  winds_csv: Path, *, speed_x=None, speed_y=None, speed_candidate_list=[], pre_time=268, post_time=1, map_size=100):
    
    list_mode = False

    if not speed_x and not speed_y:
        if not speed_candidate_list:
            raise ValueError("Either speed_x and speed_y must be provided or speed_candidate_list must be provided")
        else:
            list_mode = True
    elif not speed_x or not speed_y:
        raise ValueError("Both speed_x and speed_y must be provided if one is provided, else provide speed_candidate_list")
    else:
        # both are provided
        list_mode = False

    cityscape_files = [f for f in cityscapes_dir.glob("*.csv")]

    gen_data_df = pd.DataFrame(columns=["cityscape", "speed_x", "speed_y", "pre_time", "post_time", "out_file"])
    for cityscape_file in tqdm(cityscape_files, desc="Cityscape"):
        cityscape = pd.read_csv(cityscape_file)

        # convert to numpy array
        cityscape = cityscape.to_numpy()
        cityscape = cityscape[:, 0:4]

        speed_x, speed_y = speed_candidate_list[np.random.choice(len(speed_candidate_list))]

        try:
            flow = run_flow(cityscape, pre_time, post_time, map_size, speed_x, speed_y)
        except ValueError:
            tqdm.write(f"Cityscape {cityscape_file} failed")
            continue

        output_path = output_dir / cityscape_file.name
        np.save(output_path.with_suffix(".npy"), flow)
        
        gen_data_df.loc[len(gen_data_df)] = [str(cityscape_file), speed_x, speed_y, pre_time, post_time, output_path.with_suffix(".npy")]

        gen_data_df.to_csv(winds_csv, index=False)


