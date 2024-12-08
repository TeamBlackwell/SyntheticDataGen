import io
import numpy as np
import pygame
import math
import matplotlib.pyplot as plt
from . import env
from . import lidar_funcs as lidar
import utils
from PIL import Image
from matplotlib.gridspec import GridSpec
import os

fig2, ax2 = plt.subplots()
fig2.subplots_adjust(left=0, right=1, top=1, bottom=0)


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


def draw_prediction(surface, prediction, drone_pos, cmap, alpha=0.5):
    prediction = np.linalg.norm(prediction, axis=2)
    ax2.axis("off")
    ax2.imshow(prediction, cmap=cmap, interpolation="bicubic")
    buf = io.BytesIO()
    fig2.savefig(buf, format="png", dpi=50, bbox_inches="tight", pad_inches=0)
    plt.close(fig2)
    buf.seek(0)
    pl = (
        Image.open(buf)
        .resize((prediction.shape[0] * 8, prediction.shape[1] * 8))
        .convert("RGB")
    )
    prediction_img = pygame.image.frombuffer(
        pl.tobytes(), (prediction.shape[0] * 8, prediction.shape[1] * 8), "RGB"
    )

    prediction_img.set_alpha(alpha * 255)

    surface.blit(
        prediction_img,
        (
            drone_pos[0] - (prediction.shape[0] * 4),
            drone_pos[1] - (prediction.shape[0] * 4),
        ),
    )


def run_with_index(data_dir, index, screen_size=800, padding=0, debug=True):
    pygame.init()
    screen_info = pygame.display.Info()
    x = screen_info.current_w - screen_size - 10
    y = 50
    os.environ["SDL_VIDEO_WINDOW_POS"] = "%d,%d" % (x, y)

    previous_position = (0, 0)
    cityimage_path = data_dir / "exportviz" / f"city_{index}.png"
    windflow_path = data_dir / "windflow" / f"city_{index}.npy"

    environment = env.buildEnvironment((screen_size, screen_size), str(cityimage_path))
    environment.originalMap = environment.map.copy()
    laser = lidar.Lidar(200, environment.originalMap, uncertainty=(0.5, 0.01))
    environment.infomap = environment.map.copy()
    running = True

    cmap = utils.make_pastel_colormap("jet", blend_factor=0.5)
    lidar_data = []

    plt.ion()
    fig = plt.figure(figsize=(8, 8))  # Overall figure size
    gs = GridSpec(2, 2, height_ratios=[2, 1])
    fig_ax1 = fig.add_subplot(gs[0, :])
    fig_ax2 = fig.add_subplot(gs[1, 0])
    fig_ax3 = fig.add_subplot(gs[1, 1])

    # Move the plot window to the left
    mng = plt.get_current_fig_manager()
    mng.window.wm_geometry("+0+10")
    while running:
        sensorON = False

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            # if event.type == pygame.MOUSEBUTTONDOWN:
            if pygame.mouse.get_focused():
                sensorON = True
            else:
                sensorON = False

        if sensorON:
            position = pygame.mouse.get_pos()
            laser.position = position
            if position != previous_position:
                sensor_data, lidar_data = laser.sense_obstacles()
                environment.dataStorage(sensor_data)
        if laser.position != previous_position:
            environment.map.blit(environment.infomap, (0, 0))
            pygame.draw.circle(environment.map, (255, 0, 0), laser.position, 5)
            windflow = np.load(str(windflow_path))

            # plotting the lidar data
            fig_ax1.cla()
            angles = np.linspace(-180, 180, 360, False)
            fig_ax1.fill_between(angles, lidar_data, 0, alpha=0.2, color="r")
            fig_ax1.plot(angles, lidar_data, color="r")
            fig_ax1.set_ylim(0, 1.5)
            fig_ax1.set_xlabel("Angle (deg)")
            fig_ax1.set_ylabel("D/D_max")
            fig_ax1.set_title("Model Input")
            # fig_ax2.set_xlabel("Angle (deg)")
            # fig_ax2.set_ylabel("D/D_max")
            fig_ax2.set_title("Velocity Error (m/s)")
            # fig_ax3.set_xlabel("Angle (deg)")
            # fig_ax3.set_ylabel("D/D_max")
            fig_ax3.set_title("Direction Error (deg)")
            prediction = np.random.rand(21, 21, 2)
            prediction = np.linalg.norm(prediction, axis=2)
            fig_ax2.imshow(prediction, cmap=cmap, interpolation="bicubic")
            fig_ax3.imshow(prediction, cmap=cmap, interpolation="bicubic")
            cbar_fig_ax2 = fig.add_axes([0.87, 0.08, 0.02, 0.26])
            cbar_fig_ax3 = fig.add_axes([0.44, 0.08, 0.02, 0.26])
            fig.colorbar(fig_ax2.images[0], cax=cbar_fig_ax2)
            fig.colorbar(fig_ax3.images[0], cax=cbar_fig_ax3)
            fig.canvas.draw()
            fig.canvas.flush_events()
            plt.pause(0.005)

            # TODO: This is hardcoded for now, may need to change this
            data_coords = (
                (laser.position[0] // 8) + 200 - padding,
                (laser.position[1] // 8) + 200 - padding,
            )

            wind_robot = windflow[data_coords[0]][data_coords[1]]
            if debug:
                print(f"Laser: {laser.position} | Data: {data_coords}")

            magnitude = math.sqrt(wind_robot[0] ** 2 + wind_robot[1] ** 2)
            magnitude = math.log1p(magnitude) * 25  # log scaling
            magnitude = math.floor(magnitude)
            angle = math.atan2(wind_robot[1], wind_robot[0])
            print(f"Angle: {math.degrees(angle)}, Magnitude: {magnitude}")
            direction = (math.cos(angle), math.sin(angle))
            direction_sign = (
                1 if direction[0] > 0 else -1,
                1 if direction[1] > 0 else -1,
            )

            wind_robot = (
                laser.position[0]
                + direction_sign[0] * max(min(abs(magnitude * direction[0]), 150), 15),
                laser.position[1]
                + direction_sign[1] * max(min(abs(magnitude * direction[1]), 150), 15),
            )

            # if debug:
            #     print(f"robot_coords and Wind Robot: {laser.position}, {wind_robot}")

            # wind_robot = (wind_robot[0] - 200, wind_robot[1] - 200)
            # wind_robot[1] = wind_robot[1] - 200

            # if debug:
            #     print(f"robot_coords and Wind Robot: {laser.position}, {wind_robot}")

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
            prediction = np.random.rand(11, 11, 2)  # 21x21 grid
            draw_prediction(
                environment.map,
                prediction,
                laser.position,
                cmap=cmap,
                alpha=0.5,
            )

            imgtrans = pygame.image.load(
                f"data/transparent/city_{index}_transparent.png"
            )
            environment.map.blit(imgtrans, (0, 0))

        previous_position = laser.position
        pygame.display.update()

    pygame.quit()
