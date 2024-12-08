from pathlib import Path
import numpy as np
import pygame
import math
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns


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


def draw_arrow(surface, color, start, end, width=5, head_length=15, head_width=10):
    """
    Draws an arrow on the Pygame surface.

    Parameters:
        surface: The Pygame surface to draw on.
        color: The color of the arrow.
        start: A tuple (x, y) representing the starting point of the arrow.
        end: A tuple (x, y) representing the end point of the arrow.
        width: The width of the arrow shaft.
        head_length: The length of the arrowhead.
        head_width: The width of the arrowhead at its widest point.
    """
    pygame.draw.line(surface, color, start, end, width)
    angle = math.atan2(end[1] - start[1], end[0] - start[0])

    head_end1 = (
        end[0] - head_length * math.cos(angle - math.pi / 6),
        end[1] - head_length * math.sin(angle - math.pi / 6),
    )
    head_end2 = (
        end[0] - head_length * math.cos(angle + math.pi / 6),
        end[1] - head_length * math.sin(angle + math.pi / 6),
    )

    pygame.draw.polygon(surface, color, [end, head_end1, head_end2])


def draw_prediction(surface, prediction, drone_pos, alpha=0.5):
    plt.figure(figsize=(5, 5))
    pastel_jet = make_pastel_colormap("jet", blend_factor=0.5)
    prediction = np.linalg.norm(prediction, axis=2)
    sns.heatmap(prediction, cmap=pastel_jet, cbar=False)
    plt.axis("off")
    plt.savefig("prediction.png", bbox_inches="tight", pad_inches=0)
    prediction_img = pygame.image.load("prediction.png")
    prediction_img = pygame.transform.scale(
        prediction_img, (prediction.shape[0] * 8, prediction.shape[1] * 8)
    )
    prediction_img.set_alpha(alpha * 255)
    surface.blit(
        prediction_img,
        (
            drone_pos[0] - (prediction.shape[0] * 4),
            drone_pos[1] - (prediction.shape[0] * 4),
        ),
    )


def run_with_index(data_dir, index,debug=False):
    previous_position = (0, 0)
    from . import env
    from . import lidar_funcs as lidar

    cityimage_path = data_dir / "exportviz" / f"city_{index}.png"
    windflow_path = data_dir / "windflow" / f"city_{index}.npy"

    environment = env.buildEnvironment((800, 800), str(cityimage_path))
    environment.originalMap = environment.map.copy()
    laser = lidar.Lidar(150, environment.originalMap, uncertainty=(0.5, 0.01))
    # environment.map.fill((0, 0, 0))
    environment.infomap = environment.map.copy()
    pygame.init()
    running = True

    while running:
        sensorON = False

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            # elif not pygame.mouse.get_focused():
            # sensorON = False
            # if event.type == pygame.MOUSEBUTTONDOWN:
            if pygame.mouse.get_focused():
                sensorON = True
            else:
                sensorON = False

        if sensorON:
            position = pygame.mouse.get_pos()
            laser.position = position
            sensor_data = laser.sense_obstacles()
            environment.dataStorage(sensor_data)
        environment.map.blit(environment.infomap, (0, 0))
        pygame.draw.circle(environment.map, (255, 0, 0), laser.position, 5)
        robot_position = (laser.position[0] // 8, laser.position[1] // 8)

        windflow = np.load(str(windflow_path))

        import matplotlib.pyplot as plt

        # fig, ax = plt.subplots()
        # X, Y = np.meshgrid(np.arange(windflow.shape[1]), np.arange(windflow.shape[0]))
        # U = windflow[:, :, 0]
        # V = windflow[:, :, 1]
        # ax.quiver(X, Y, U, V)
        # plt.show()

        wind_robot = windflow[robot_position[0]][robot_position[1]]
        magnitude = math.sqrt(wind_robot[0] ** 2 + wind_robot[1] ** 2)
        magnitude = math.floor(25 * magnitude)
        angle = math.atan2(wind_robot[1], wind_robot[0])
        direction = (math.cos(angle), math.sin(angle))
        direction_sign = (1 if direction[0] > 0 else -1, 1 if direction[1] > 0 else -1)
        wind_robot = (
            laser.position[0]
            + direction_sign[0] * max(min(abs(magnitude * direction[0]), 200), 15),
            laser.position[1]
            + direction_sign[1] * max(min(abs(magnitude * direction[1]), 200), 15),
        )
        if debug:
            print(f"Angle: {math.degrees(angle)}, Magnitude: {magnitude}")
            print(f"robot_coords and Wind Robot: {laser.position}, {wind_robot}")
        draw_arrow(
            environment.map,
            (0, 0, 0),
            laser.position,
            (
                wind_robot[0],
                wind_robot[1],
            ),
        )
        # run model to get the prediction of the particular index and particular position of the robot
        #prediction = np.random.rand(21, 21, 2)  # 21x21 grid
        #draw_prediction(environment.map, prediction, laser.position, alpha=0.5)
        previous_position = laser.position
        pygame.display.update()

    pygame.quit()
