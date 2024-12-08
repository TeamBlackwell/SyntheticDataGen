import pandas as pd
from pathlib import Path
import random
from tqdm import tqdm

MAX_X = 100
MAX_Y = 100


# @TODO: Need to optimise the method for finding empty position
# @TODO: Need to select an appropriate z value for the drone
def find_empty_position(cityscape: pd.DataFrame, max_height: int, buffer=10):
    drone_position = (random.randint(0, MAX_X), random.randint(0, MAX_Y))
    while not is_empty_position(
        cityscape, drone_position[0], drone_position[1], buffer
    ):
        drone_position = (random.randint(0, MAX_X), random.randint(0, MAX_Y))
    return drone_position


def is_empty_position(cityscape: pd.DataFrame, x: int, y: int, buffer: int):
    """
    Check if the position is empty ie not intersecting with any building in the cityscape
    """
    dx1, dx2, dy1, dy2 = x - buffer, x + buffer, y - buffer, y + buffer
    buildings = cityscape[
        (cityscape["x1"] > dx1)
        & (cityscape["x2"] < dx2)
        & (cityscape["y1"] > dy1)
        & (cityscape["y2"] < dy2)
    ]
    if len(buildings) == 0:
        return True
    else:
        return False


def generate_positions_for_cityscape(
    cityscape: pd.DataFrame, num_positions: int, max_height: int = 10
):
    """
    Generate drone positions without duplicating positions
    """
    positions = []
    for _ in range(num_positions):
        pos = find_empty_position(cityscape, 10)
        while pos in positions:
            pos = find_empty_position(cityscape, 10)
        positions.append([pos[0], pos[1], max_height])

    df = pd.DataFrame(positions, columns=["x", "y", "z"])
    return df


def generate_positions(cityscapes_dir: Path, num_positions: int, out_dir: Path):
    """
    Generate positions for all cityscapes in the cityscapes_dir creates a csv file for each cityscape with the positions
    containing the x,y,z coordinates of the drone
    """

    cityscapes = [
        f for f in cityscapes_dir.iterdir() if f.is_file() and f.suffix == ".csv"
    ]
    for cityscape_file in tqdm(cityscapes):
        cityscape = pd.read_csv(cityscape_file)
        # set cityscape data to be float
        cityscape = cityscape.astype(float)
        positions_df = generate_positions_for_cityscape(cityscape, num_positions)

        out_file = out_dir / cityscape_file.name
        positions_df.to_csv(out_file, index=False)
