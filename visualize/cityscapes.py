import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Rectangle
from pathlib import Path


def cityscape_visualization(cityscape_path: Path, map_size: int, world_size: int, fig_size=(5, 5)):
    """
    Visualize the cityscape with buildings

    Parameters
    - cityscape_path: path to cityscape csv file.
    - map_size: side length of the cityscape in meters.
    """

    # Create the main plot
    plt.figure(figsize=fig_size)

    # Plot buildings
    buildings_df = pd.read_csv(cityscape_path)
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

    # Plot building centers
    # building_centers_x = (buildings_df["x1"] + buildings_df["x2"]) / 2
    # building_centers_y = (buildings_df["y1"] + buildings_df["y2"]) / 2
    # plt.scatter(
    #     building_centers_x,
    #     building_centers_y,
    #     color="blue",
    #     alpha=0.5,
    #     s=20,
    #     label="Building Centers",
    # )

    # Set plot properties
    plt.title("Cityscape Visualization")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.xlim((world_size / 2) - (map_size / 2), (world_size / 2) - (map_size / 2) + map_size)
    plt.ylim((world_size / 2) - (map_size / 2), (world_size / 2) - (map_size / 2) + map_size)
    plt.grid(True, linestyle="--", alpha=0.7)

    # Show the plot
    plt.tight_layout()
    plt.show()
