import math
import pygame


class buildEnvironment:
    def __init__(self, MapDimensions, path):
        pygame.init()
        # self.pointCloud = []
        self.externalMap = pygame.image.load(path)
        self.maph, self.mapw = MapDimensions
        self.MapWindowName = "2d Lidar simulation"

        pygame.display.set_caption(self.MapWindowName)
        self.map = pygame.display.set_mode((self.mapw, self.maph))
        self.map.blit(self.externalMap, (0, 0))

        self.black = (0, 0, 0)
        self.grey = (128, 128, 128)
        self.Blue = (0, 0, 255)
        self.Green = (0, 255, 0)
        self.Red = (255, 0, 0)
        self.White = (255, 255, 255)

    def AD2pos(self, distance, angle, robotPosition):
        x = distance * math.cos(angle) + robotPosition[0]
        y = -distance * math.sin(angle) + robotPosition[1]
        return (int(x), int(y))

    def dataStorage(self, data):
        self.map.blit(self.externalMap, (0, 0))
        pointCloud = []
        if data != False:
            for element in data:
                point = self.AD2pos(element[0], element[1], element[2])
                if point not in pointCloud:
                    pointCloud.append(point)
        self.infomap = self.map.copy()
        for point in pointCloud:
            self.infomap.set_at((int(point[0]), int(point[1])), (0, 255, 0))
            pygame.draw.circle(self.infomap, self.Red, point, 3)
