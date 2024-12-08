"""
Visualize cityscape, drone lidar and generated windflow data
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle
from pathlib import Path

from utils import make_pastel_colormap

def windflow_visualization(
    cityscape_path: Path,
    windflow_path: Path,
    map_size: int,
    fig_size=(5, 5),
    export=None,
    plot_vector=False,
    transparent=False,
):
    """
    Visualize the cityscape with buildings and windflow data

    Parameters
    - cityscape_path: path to cityscape csv file.
    - windflow_path: path to windflow csv file.
    - map_size: side length of the cityscape in meters.
    """

    plt.figure(figsize=fig_size)

    arr = np.load(windflow_path)

    # above is the shape N, N, 2
    # 2 represents X and Y speed
    # calculate magnitude and get a N x N array
    mag_array = np.linalg.norm(arr, axis=2)

    mag_array = np.rot90(mag_array, 1)

    # Create a pastel version of the 'jet' colormap
    pastel_jet = make_pastel_colormap("jet", blend_factor=0.5)

    # sns.heatmap(mag_array, cmap=pastel_jet)
    # plot the magnitude array
    if not transparent:
        plt.imshow(mag_array, cmap=pastel_jet, interpolation="bicubic", aspect="equal")
    else:
        plt.imshow(mag_array, cmap=pastel_jet, interpolation="bicubic", aspect="equal", alpha=0.0)
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


    #@TODO: add a quiver arrow showing wind direction, from the bottom left (0, 0).
    # it should be a red arrow, the text should say "speed: x, y" of the wind

    # Plot windflow vectors

    if plot_vector:
        for i in range(0, arr.shape[0], 3):
            for j in range(0, arr.shape[1], 3):
                # the scale should be the magnitude of the vector
                mag = np.linalg.norm(arr[i, j])
                # scale mag to be between 0 and 150
                mag = (mag / np.max(mag_array)) * 150

                plt.quiver(
                    j,
                    map_size - i,
                    arr[i, j, 0],
                    -arr[i, j, 1],
                    color="red",
                    alpha=0.5,
                    scale=mag,
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

        if transparent:
            plt.savefig(export, transparent=True)
        else:
            plt.savefig(export)
    else:
        plt.colorbar()
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
