import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import pandas as pd

# plot the drone_pos, 

def plot_lidar(index):
    """
    Plot the lidar data for a particular index

    Parameters
    - index: index of the lidar data to plot
    """
    base = Path("./data")
    lidars = list((base / "lidar").glob(f"city_{index}_pos*.npy"))
    map_path = base / "cityscapes" / f"city_{index}.csv"
    basemap = base / "demoviz" / f"city_{index}.png"
    drone_pos = base / "drone_positions" / f"city_{index}.csv"

    # Load the city map and resize it to 150 x 150
    city_map = plt.imread(basemap)
    plt.imshow(city_map, extent=[0, 150, 0, 150])


    buildings_df = pd.read_csv(map_path)
    buildings_df.columns = ["x1", "y1", "x2", "y2", "height"]

    world_size = 150
    map_size = 100

    # Plot each building as a rectangle
    for _, building in buildings_df.iterrows():
        building["x1"], building["y1"] = building["y1"], building["x1"]
        building["x2"], building["y2"] = building["y2"], building["x2"]

        building["x1"], building["y1"] = building["y1"], world_size - building["x1"] - 6
        building["x2"], building["y2"] = building["y2"], world_size - building["x2"] - 6
        
        plt.gca().add_patch(
            plt.Rectangle(
                (building["x1"], building["y1"]),
                building["x2"] - building["x1"],
                building["y2"] - building["y1"],
                fill=False,
                edgecolor="gray",
                linewidth=1,
            )
        )

    drone_pos_df = pd.read_csv(drone_pos)

    for lidar in lidars:
        lidar_data = np.load(lidar)
        pos_id = int(lidar.stem.split("_pos")[-1])
        drone_pos_df_row = drone_pos_df.iloc[pos_id]
        drone_x = drone_pos_df_row["xr"]
        drone_y = drone_pos_df_row["yr"]

        position = (drone_x - ((world_size / 2) - (map_size / 2)), drone_y - ((world_size / 2) - (map_size / 2)))
        # position = (position[0], position[1] + 25)
        # plot the lidar data, 360 degrees around robot at distance of lidar_data * 25
        xs = []
        ys = []
        angle_space = np.linspace(-np.pi, np.pi, 360)
        for i, distance in enumerate(lidar_data):
            angle = angle_space[i]
            distance = distance * 25

            x = position[0] + distance * np.cos(angle)
            y = position[1] + distance * np.sin(angle)
            xs += [x]
            ys += [y]
            plt.plot([position[0], x], [position[1], y], color="blue", alpha=0.5)

        # plot drone_x, drone_y
        plt.scatter(position[0], position[1], color="red")
        break

    plt.scatter(position[0], position[1], color="red")

    plt.axis("square")
    plt.show()

