"""
Visualize cityscape, drone lidar and generated windflow data
"""

from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle
from pathlib import Path
import seaborn as sns


def windflow_visualization(
    cityscape_path: Path,
    windflow_path: Path,
    map_size: int,
    fig_size=(5, 5),
    export=None,
):
    """
    Visualize the cityscape with buildings and windflow data

    Parameters
    - cityscape_path: path to cityscape csv file.
    - windflow_path: path to windflow csv file.
    - map_size: side length of the cityscape in meters.
    """

    # Create the main plot
    print(windflow_path)
    print(cityscape_path)

    plt.figure(figsize=fig_size)

    arr = np.load(windflow_path)

    # above is the shape N, N, 2
    # 2 represents X and Y speed
    # calculate magnitude and get a N x N array
    mag_array = np.linalg.norm(arr, axis=2)

    mag_array = np.rot90(mag_array, 1)

    def make_pastel_colormap(base_cmap_name, blend_factor=0.5):
        """
        Create a pastel version of a given base colormap by blending it with white.

        Parameters:
            base_cmap_name (str): Name of the base colormap (e.g., 'jet').
            blend_factor (float): Blending factor with white (0 = no change, 1 = fully white).

        Returns:
            LinearSegmentedColormap: A pastel colormap.
        """
        base_cmap = plt.cm.get_cmap(base_cmap_name)
        colors = base_cmap(np.linspace(0, 1, 256))
        white = np.array([1, 1, 1, 1])  # RGBA for white
        pastel_colors = (1 - blend_factor) * colors + blend_factor * white
        pastel_cmap = LinearSegmentedColormap.from_list(
            f"{base_cmap_name}_pastel", pastel_colors
        )
        return pastel_cmap

    # Create a pastel version of the 'jet' colormap
    pastel_jet = make_pastel_colormap("jet", blend_factor=0.5)

    sns.heatmap(mag_array, cmap=pastel_jet)
    # plot the magnitude array
    # plt.imshow(mag_array, cmap=pastel_jet, interpolation="bicubic")

    # Plot buildings
    buildings_df = pd.read_csv(cityscape_path)
    buildings_df.columns = ["x1", "y1", "x2", "y2", "height"]

    # Plot each building as a rectangle
    for _, building in buildings_df.iterrows():
        building["x1"], building["y1"] = building["y1"], building["x1"]
        building["x2"], building["y2"] = building["y2"], building["x2"]

        building["x1"], building["y1"] = building["y1"], 100 - building["x1"] - 6
        building["x2"], building["y2"] = building["y2"], 100 - building["x2"] - 6

        plt.gca().add_patch(
            Rectangle(
                (int(building["x1"]), int(building["y1"])),
                abs(building["x2"] - building["x1"]),
                abs(building["y2"] - building["y1"]),
                fill=True,
                color="black",
                linewidth=1,
            )
        )

    # # Plot building centers
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

    # add a quiver arrow showing wind direction, from the bottom left (0, 0).
    # it should be a red arrow, the text should say "speed: x, y" of the wind

    # Plot windflow vectors
    for i in range(0, arr.shape[0], 2):
        for j in range(0, arr.shape[1], 2):
            plt.quiver(
                j,
                map_size - i,
                arr[i, j, 0],
                -arr[i, j, 1],
                color="red",
                alpha=0.5,
                angles="xy",
            )

    # Set plot properties
    plt.title("Windflow Visualization")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.xlim(0, map_size)
    plt.ylim(0, map_size)
    # plt.grid(True, linestyle="--", alpha=0.7)
    # add color bar and name the color bar
    plt.axis("equal")

    # Show the plot
    plt.tight_layout()

    if export:
        # remove axes, ticks and labels
        plt.axis("off")
        plt.xticks([])
        plt.yticks([])
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        # make it 800x800
        plt.gcf().set_size_inches(8, 8)
        plt.savefig(export)
    else:
        # plt.colorbar()
        plt.show()

    # # Plot windflow data
    # windflow_df = pd.read_csv(windflow_path)
    # windflow_df.columns = ["x", "y", "u", "v"]

    # # Plot windflow vectors
    # for _, windflow in windflow_df.iterrows():
    #     plt.quiver(
    #         windflow["x"],
    #         windflow["y"],
    #         windflow["u"],
    #         windflow["v"],
    #         color="red",
    #         alpha=0.5,
    #         scale=50,
    #         scale_units="xy",
    #         angles="xy",
    #     )

    # # Set plot properties
    # plt.title("Windflow Visualization")
    # plt.xlabel("X Coordinate")
    # plt.ylabel("Y Coordinate")
    # plt.xlim(0, map_size)
    # plt.ylim(0, map_size)
    # plt.grid(True, linestyle="--", alpha=0.7)
    # plt
