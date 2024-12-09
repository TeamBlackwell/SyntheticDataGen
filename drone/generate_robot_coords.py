import pandas as pd
import numpy as np
from typing import List
from glob import glob
import os
from tqdm import tqdm


def find_robot_coordinates(
    buildings: List[np.ndarray],
    min_distance: float = 1,
    n_robots: int = 5,
    radius: float = 40.0,
    z_height: float = 15.0,
) -> pd.DataFrame:
    """
    Generate robot coordinates that avoid building locations and ensure
    minimum distance between robots.

    Parameters:
    - buildings: List of building coordinates [x1, y1, x2, y2, height]
    - min_distance: Minimum distance between robots and from buildings
    - n_robots: Number of robots to place
    - radius: Radius of the circular area
    - z_height: Height of robots

    Returns:
    pandas.DataFrame with robot coordinates
    """
    center = (
        np.mean(buildings[:, 0:3:2], axis=0) + np.mean(buildings[:, 1:4:2], axis=0)
    ) / 2

    def is_valid_point(point: np.ndarray, placed_points: List[np.ndarray]) -> bool:
        """
        Check if a point is valid (not too close to buildings or other robots)
        """
        # Check distance from buildings
        for building in buildings:
            # Building bounds
            x1, y1, x2, y2, _ = building
            # Check if point is inside or too close to building
            if (
                x1 - min_distance <= point[0] <= x2 + min_distance
                and y1 - min_distance <= point[1] <= y2 + min_distance
            ):
                return False

        # Check distance from other placed points
        for placed in placed_points:
            if np.linalg.norm(point[:2] - placed[:2]) < min_distance:
                return False

        return True

    # Prepare list to store robot coordinates
    robot_coords = []
    x_center, y_center = center

    # Maximum attempts to place robots
    max_attempts = 1000
    attempts = 0

    while len(robot_coords) < n_robots:
        # Generate a random point within the circular area
        angle = np.random.uniform(0, 2 * np.pi)
        r = (
            np.sqrt(np.random.uniform(0, 1)) * radius
        )  # Uniform distribution within circle
        x = x_center + r * np.cos(angle)
        y = y_center + r * np.sin(angle)

        candidate_point = np.array([x, y, z_height])

        # Check if point is valid
        if is_valid_point(candidate_point, robot_coords):
            robot_coords.append(candidate_point)
            attempts = 0  # Reset attempts after successful placement
        else:
            attempts += 1

        # Prevent infinite loop
        if attempts > max_attempts:
            raise ValueError(
                f"Could not place {n_robots} robots after {max_attempts * n_robots} attempts"
            )

    # make the points so that they are integers
    robot_coords = np.array(robot_coords)
    robot_coords = np.round(robot_coords, 0)

    # Convert to DataFrame
    df = pd.DataFrame(robot_coords, columns=["xr", "yr", "zr"])
    return df


def generate_robot_coordinates(
    robot_dir, buildings_file, n_positions, min_distance, center_radius
):
    df = pd.read_csv(buildings_file)
    robot_df = find_robot_coordinates(
        df.to_numpy(),
        n_robots=n_positions,
        min_distance=min_distance,
        radius=center_radius,
    )
    basename = os.path.basename(buildings_file)
    robot_df.to_csv(
        os.path.join(robot_dir, f"{basename.split('.csv')[0]}.csv"), index=False
    )


def batch_export_robot(
    robot_dir, data_directory, n_positions, min_distance, center_radius_choices
):
    for filename in tqdm(glob(os.path.join(data_directory, "*"))):
        center_radius = np.random.choice(center_radius_choices)
        generate_robot_coordinates(
            robot_dir, filename, n_positions, min_distance, center_radius
        )


if __name__ == "__main__":
    batch_export_robot("./robot", "data")
