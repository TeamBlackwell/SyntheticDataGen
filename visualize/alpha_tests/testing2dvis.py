# cmap
from pathlib import Path
import pandas as pd
from matplotlib.patches import Rectangle
import numpy as np
import matplotlib.pyplot as plt

yy = np.load("data/windflow/city_0.npy")
yy.shape

mag = np.linalg.norm(yy, axis=2)
mag = np.rot90(mag, 1)
# ensure corner is at bottom left
plt.imshow(mag, cmap='flag', interpolation='bicubic')
plt.colorbar()

cityscape_path = Path("../../data/cityscapes/city_0.csv")
buildings_df = pd.read_csv(cityscape_path)
buildings_df.columns = ["x1", "y1", "x2", "y2", "height"]

# Plot each building as a rectangle
for _, building in buildings_df.iterrows():
    
    building["x1"], building["y1"] = building["y1"], building["x1"]
    building["x2"], building["y2"] = building["y2"], building["x2"]

    building["x1"], building["y1"] = building["y1"], 100 - building["x1"] - 6
    building["x2"], building["y2"] = building["y2"], 100 - building["x2"] - 6

    rect_xy = (int(building["x1"]), int(building["y1"])) # swapped x and y
    rect_width = abs(building["x2"] - building["x1"]) # swapped x and y
    rect_height = abs(building["y2"] - building["y1"]) # swapped x and y

    plt.gca().add_patch(
        Rectangle(
            rect_xy,
            rect_width,
            rect_height,
            fill=True,
            facecolor="black",
            edgecolor="gray",
            linewidth=1,
        )
    )

plt.show()