import numpy as np
import env
import lidar_funcs as lidar
import pygame
import math

environment = env.buildEnvironment((800, 800))
environment.originalMap = environment.map.copy()
laser = lidar.Lidar(200, environment.originalMap, uncertainty=(0.5, 0.01))
# environment.map.fill((0, 0, 0))
environment.infomap = environment.map.copy()
pygame.init()
running = True


# create rectangle

while running:
    sensorON = False

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        # elif not pygame.mouse.get_focused():
        # sensorON = False
        if event.type == pygame.MOUSEBUTTONDOWN:
            # if pygame.mouse.get_focused():
            # if input_rect.collidepoint(event.pos):
            sensorON = True
        else:
            sensorON = False

    if sensorON:
        position = pygame.mouse.get_pos()
        # position = (200, 200)
        laser.position = position
        pygame.draw.circle(environment.map, (255, 0, 0), laser.position, 5)
        sensor_data = laser.sense_obstacles()
        environment.dataStorage(sensor_data)
        environment.show_sensordata()

    environment.map.blit(environment.infomap, (0, 0))
    pygame.display.update()

pygame.quit()
