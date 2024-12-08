from matplotlib import pyplot as plt
import pandas as pd
import pygame
import math
import numpy as np
from PIL import Image
from tqdm import tqdm


def uncertainty_add(distance, angle, sigma):
    # mean = np.array([distance, angle])
    # covariance = np.diag(sigma**2)
    # distance, angle = np.random.multivariate_normal(mean, covariance)
    distance = max(0, distance)
    angle = max(0, angle)
    return [distance, angle]


class Lidar:
    def __init__(self, Range, map, uncertainty):
        self.Range = Range
        self.speed = 4
        self.sigma = np.array([uncertainty[0], uncertainty[1]])
        self.position = (0, 0)
        self.map = map
        self.W, self.H = pygame.display.get_surface().get_size()
        self.sensedObstacles = []

    def distance(self, obstaclePosition):
        px = (obstaclePosition[0] - self.position[0]) ** 2
        py = (obstaclePosition[1] - self.position[1]) ** 2
        return math.sqrt(px + py)

    def sense_obstacles(self):
        data = []
        lidar_data = []
        x1, y1 = self.position[0], self.position[1]
        for angle in np.linspace(0, 2 * math.pi, 360, False):
            x2, y2 = x1 + self.Range * math.cos(angle), y1 - self.Range * math.sin(
                angle
            )
            added = False
            for i in range(0, 100):
                u = i / 100
                x = int(x2 * u + x1 * (1 - u))
                y = int(y2 * u + y1 * (1 - u))
                if 0 < x < self.W and 0 < y < self.H:
                    color = self.map.get_at((x, y))
                    if color[0] < 10 and color[1] < 10 and color[2] < 10:
                        distance = self.distance((x, y))
                        output = uncertainty_add(distance, angle, self.sigma)
                        lidar_data.append(distance)
                        output.append(self.position)
                        data.append(output)
                        added = True
                        break
                    if i == 99:
                        distance = self.Range
                        output = uncertainty_add(distance, angle, self.sigma)
                        lidar_data.append(distance)
                        output.append(self.position)
                        data.append(output)
                        added = True
            if not added:
                distance = self.Range
                lidar_data.append(distance)
        lidar_data = np.array(lidar_data) / self.Range
        # angles = np.linspace(-180, 180, 360, False)
        # plt.fill_between(angles, lidar_data, 0, alpha=0.2, color="r")
        # plt.plot(angles, lidar_data, color="r")
        # plt.ylim(0, 1.5)
        # plt.xlabel("Angle (deg)")
        # plt.ylabel("D/D_max")
        # plt.title("Model Input")
        # plt.show()

        if len(data) > 0:
            return data
        else:
            return False


def run_lidar_only(range_, uncertainty, binary_map_mask, position):

    sigma = np.array([uncertainty[0], uncertainty[1]])

    def calc_distance(obstaclePosition):
        px = (obstaclePosition[0] - position[0]) ** 2
        py = (obstaclePosition[1] - position[1]) ** 2
        return math.sqrt(px + py)

    width = binary_map_mask.shape[1]
    height = binary_map_mask.shape[0]

    data = []
    lidar_data = []
    x1, y1 = position[0], position[1]

    for angle in np.linspace(0, 2 * math.pi, 360, False):
        x2, y2 = x1 + range_ * math.cos(angle), y1 - range_ * math.sin(angle)
        added = False
        for i in range(0, 100):
            u = i / 100
            x = int(x2 * u + x1 * (1 - u))
            y = int(y2 * u + y1 * (1 - u))
            if 0 < x < width and 0 < y < height:
                color = binary_map_mask[x, y]
                if color == 1:
                    # obstacle
                    distance = calc_distance((x, y))
                    output = uncertainty_add(distance, angle, sigma)
                    lidar_data.append(distance)
                    added = 1
                    output.append(position)
                    data.append(output)
                    break
                if i == 99:
                    distance = range_
                    output = uncertainty_add(distance, angle, sigma)
                    lidar_data.append(distance)
                    output.append(position)
                    data.append(output)
                    added = True
        if not added:
            distance = range_
            lidar_data.append(distance)

    lidar_data = np.array(lidar_data) / range_

    return lidar_data


def binarize_citymap_image(rgb_image):
    # black is obstacle, any other color is free space
    # so we make a binary map where 1 is obstacle and 0 is free space
    binary_map = np.zeros(rgb_image.shape[:2])
    for i in range(rgb_image.shape[0]):
        for j in range(rgb_image.shape[1]):

            if np.all(rgb_image[i, j] == [0, 0, 0]):
                binary_map[i, j] = 1

    return binary_map


def gen_iterative_lidar(citymaps_dir, positions_dir, output_dir):

    for city in tqdm(list(citymaps_dir.glob("*.png"))):
        # open image as np array, without pygame
        city_map = Image.open(city)
        # remove alpha
        city_map = city_map.convert("RGB")
        city_map = np.array(city_map)
        binary_map = binarize_citymap_image(city_map)

        corresponding_positions = positions_dir / f"{city.stem}.csv"
        if not corresponding_positions.exists():
            continue
        positions = pd.read_csv(corresponding_positions)

        for i in range(len(positions)):
            position = (positions.iloc[i]["xr"], positions.iloc[i]["yr"])
            # scale to 800x800 map by multiplying above by 800/100
            position = (position[0] * 8, position[1] * 8)

            lidar_output = run_lidar_only(
                200,
                uncertainty=(0.5, 0.01),
                binary_map_mask=binary_map,
                position=position,
            )

            np.save(output_dir / f"{city.stem}_pos{i}.npy", lidar_output)
