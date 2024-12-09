import math
import pygame


class buildEnvironment:
    def __init__(self, map_dimensions: tuple[int, int], path_to_map_file: str):
        pygame.init()
        self.base_map_image = pygame.image.load(path_to_map_file)

        self.maph, self.mapw = map_dimensions

        self.MapWindowName = "2D Lidar Simulation"

        pygame.display.set_caption(self.MapWindowName)
        self.map = pygame.display.set_mode((self.mapw, self.maph))
        self.map.blit(self.base_map_image, (0, 0))

        self.red_color = (255, 0, 0)

    def AD2pos(self, distance, angle, robotPosition):
        x = distance * math.cos(angle) + robotPosition[0]
        y = -distance * math.sin(angle) + robotPosition[1]
        return (int(x), int(y))

    def dataStorage(self, data):
        self.map.blit(self.base_map_image, (0, 0))
        pointCloud = []
        if data != False:
            for element in data:
                point = self.AD2pos(element[0], element[1], element[2])
                if point not in pointCloud:
                    pointCloud.append(point)
        # self.infomap = self.map.copy()
        for point in pointCloud:
            self.map.set_at((int(point[0]), int(point[1])), (0, 255, 0))
            pygame.draw.circle(self.map, self.red_color, point, 3)
