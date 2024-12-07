import pygame
import math
import numpy as np


def uncertainty_add(distance, angle, sigma):
    mean = np.array([distance, angle])
    covariance = np.diag(sigma**2)
    distance, angle = np.random.multivariate_normal(mean, covariance)
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
        for angle in np.linspace(-math.pi, math.pi, 360, False):
            x2, y2 = x1 + self.Range * math.cos(angle), y1 - self.Range * math.sin(
                angle
            )
            added = 0
            for i in range(0, 100):
                u = i / 100
                x = int(x2 * u + x1 * (1 - u))
                y = int(y2 * u + y1 * (1 - u))
                if 0 < x < self.W and 0 < y < self.H:
                    color = self.map.get_at((x, y))
                    if color[0] < 150 and color[1] < 150 and color[2] < 150:
                        distance = self.distance((x, y))
                        output = uncertainty_add(distance, angle, self.sigma)
                        lidar_data.append(distance)
                        added = 1
                        output.append(self.position)
                        data.append(output)
                        break
            if added == 0:
                lidar_data.append(np.nan)
        # plot lidar_data x axis is angle, y axis is distance
        import matplotlib.pyplot as plt

        lidar_data = np.nan_to_num(lidar_data, nan=np.nanmax(lidar_data))
        lidar_data /= np.nanmax(lidar_data)
        angles = np.linspace(-180, 180, 360, False)
        plt.fill_between(angles, lidar_data, 0, alpha=0.2, color="r")
        plt.plot(angles, lidar_data, color="r")
        plt.ylim(0, 1.5)
        plt.xlabel("Angle (deg)")
        plt.ylabel("D/D_max")
        plt.title("Model Input")
        plt.show()

        if len(data) > 0:
            return data
        else:
            return False

    def sense_and_return(self):
        data = []
        lidar_data = []
        x1, y1 = self.position[0], self.position[1]
        for angle in np.linspace(-math.pi, math.pi, 360, False):
            x2, y2 = x1 + self.Range * math.cos(angle), y1 - self.Range * math.sin(
                angle
            )
            added = 0
            for i in range(0, 100):
                u = i / 100
                x = int(x2 * u + x1 * (1 - u))
                y = int(y2 * u + y1 * (1 - u))
                if 0 < x < self.W and 0 < y < self.H:
                    color = self.map.get_at((x, y))
                    if color[0] < 150 and color[1] < 150 and color[2] < 150:
                        distance = self.distance((x, y))
                        output = uncertainty_add(distance, angle, self.sigma)
                        lidar_data.append(distance)
                        added = 1
                        output.append(self.position)
                        data.append(output)
                        break
            if added == 0:
                lidar_data.append(np.nan)

        lidar_data = np.nan_to_num(lidar_data, nan=np.nanmax(lidar_data))
        lidar_data /= np.nanmax(lidar_data)
        
        print(lidar_data.shape)


def gen_iterative_lidar(cityscapes_dir, positions_dir, output_dir):
    pass
