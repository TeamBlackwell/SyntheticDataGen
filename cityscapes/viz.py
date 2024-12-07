"""
Visualise the cityscape data
"""

import matplotlib.pyplot as plt
import pandas as pd
from .generator import CityScapeGenerator
from .sampling import sample_poisson_disk, Tag
from matplotlib.patches import Rectangle


def main_cityscape_visualization(map_size=100, buildings=40, density=15):
    """
    Visualize the cityscape with buildings and drone positions.

    Parameters
    - map_size: Size of the cityscape grid
    - buildings: Number of buildings to generate
    - density: Density of building placement
    """  # Create the cityscape generator
    city_gen = CityScapeGenerator(
        2,
        sampling_fncs=[
            (
                sample_poisson_disk,
                Tag.HOUSE,
                {"density": density, "n_buildings": buildings},
            ),
        ],
        map_size=map_size,
        debug=False,
    )

    # Generate the sample cityscape
    city_gen.generate_sample()

    # Create the main plot
    plt.figure(figsize=(10, 10))

    # Plot buildings
    buildings_df = pd.DataFrame(city_gen.buildings)
    buildings_df.columns = ["x1", "y1", "x2", "y2", "height"]

    # Plot each building as a rectangle
    for _, building in buildings_df.iterrows():
        plt.gca().add_patch(
            Rectangle(
                (int(building["x1"]), int(building["y1"])),
                building["x2"] - building["x1"],
                building["y2"] - building["y1"],
                fill=False,
                edgecolor="gray",
                linewidth=1,
            )
        )

    # Plot drone positions
    if hasattr(city_gen, "robot_coords"):
        plt.scatter(
            city_gen.robot_coords["x_r"],
            city_gen.robot_coords["y_r"],
            color="red",
            s=100,
            label="Drone Positions",
        )

    # Plot building centers
    building_centers_x = (buildings_df["x1"] + buildings_df["x2"]) / 2
    building_centers_y = (buildings_df["y1"] + buildings_df["y2"]) / 2
    plt.scatter(
        building_centers_x,
        building_centers_y,
        color="blue",
        alpha=0.5,
        s=20,
        label="Building Centers",
    )

    # Set plot properties
    plt.title("Cityscape Visualization")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.xlim(0, map_size)
    plt.ylim(0, map_size)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend()
    plt.axis("equal")

    # Show the plot
    plt.tight_layout()
    plt.show()
