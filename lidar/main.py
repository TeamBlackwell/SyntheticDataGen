import io
import time
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
import matplotlib
import matplotlib.animation as animation
matplotlib.use('TkAgg')  # for positioning the plot window support on both MacOS and Windows systems

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

    plt.ioff()
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(1, 1))
    prediction = np.linalg.norm(prediction, axis=2)
    ax.axis("off")
    ax.imshow(prediction, cmap=cmap, interpolation="bicubic")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=50, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
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
    

def get_prediction(wind_robot, lidar_data, position):
    # Placeholder for now
    return np.random.rand(11, 11, 2), np.random.rand(11, 11, 1), np.random.rand(11, 11, 1)

def run_with_index(data_dir, index, screen_size=800, padding=0, debug=False):

    # fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(7, 7))
    fig = plt.figure(constrained_layout=True, figsize=(7, 7))
    axs = fig.subplot_mosaic([['Top', 'Top'],['BottomLeft', 'BottomRight']],
                          gridspec_kw={'height_ratios':[1, 2]})
    
    plt.ioff()
    # make equal
    VEL_ERR = axs["BottomLeft"]
    DIR_ERR = axs["BottomRight"]
    LID_SCAN = axs["Top"]

    VEL_ERR.set_aspect('equal', 'box')
    DIR_ERR.set_aspect('equal', 'box')

    VEL_ERR.set_title("Velocity Error (m/s)")
    DIR_ERR.set_title("Direction Error (deg)")
    LID_SCAN.set_title("Lidar Scan")
    LID_SCAN.set_xlabel("Angle (deg)")
    LID_SCAN.set_ylabel("D/D_max")

    dummy_data = np.zeros(shape=(11, 11))
    VEL_ERR.imshow(dummy_data, cmap="jet", interpolation="bicubic")
    DIR_ERR.imshow(dummy_data, cmap="jet", interpolation="bicubic")

    dummy_lidar = np.zeros(360)
    LID_SCAN.fill_between(np.linspace(-180, 180, 360, False), dummy_lidar, 0, alpha=0.2, color="r")
    LID_PLOT, = LID_SCAN.plot(np.linspace(-180, 180, 360, False), dummy_lidar, color="r")
    LID_SCAN.set_ylim(0, 1.5)
    fig.show()
    # Move the plot window to the left
    mng = plt.get_current_fig_manager()
    mng.window.wm_geometry("+0+10")

    pygame.init()
    screen_info = pygame.display.Info()
    x = screen_info.current_w - screen_size - 10
    y = 50
    os.environ["SDL_VIDEO_WINDOW_POS"] = "%d,%d" % (x, y)

    cmap = utils.make_pastel_colormap("jet", blend_factor=0.5)

    cityimage_path = data_dir / "demoviz" / f"city_{index}.png"
    windflow_path = data_dir / "windflow" / f"city_{index}.npy"

    windflow = np.load(str(windflow_path))

    environment = env.buildEnvironment((screen_size, screen_size), str(cityimage_path))
    # environment.originalMap = environment.map.copy()
    laser = lidar.Lidar(200, environment.map, uncertainty=(0.5, 0.01))
    # environment.infomap = environment.map.copy()

    imgtrans = pygame.image.load(
        f"data/transparent/city_{index}_transparent.png"
    )
    # state variables
    running = True
    plot_data = []
    lidar_data = []
    previous_position = (0, 0)

    while running:
        environment.map.blit(environment.map, (0, 0))
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

            pygame.draw.circle(environment.map, (255, 0, 0), laser.position, 5)


            # TODO: This is hardcoded for now, may need to change this
            data_coords = (
                (laser.position[0] // 8) + 25 - padding,
                (laser.position[1] // 8) + 25 - padding,
            )
            wind_robot = windflow[data_coords[0]][data_coords[1]]

            prediction_data, pred_mag_error, pred_deg_error = get_prediction(
                wind_robot, lidar_data, laser.position
            )
            plot_data = [lidar_data, pred_mag_error, pred_deg_error]
            
            if debug:
                print(f"Laser: {laser.position} | Data: {data_coords}")

            magnitude = np.floor(np.linalg.norm(wind_robot))
            magnitude = math.log1p(magnitude) * 25  # log scaling

            angle = math.atan2(wind_robot[1], wind_robot[0])
            if debug: 
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
            # st = time.time()
            draw_prediction(
                environment.map,
                prediction_data,
                laser.position,
                cmap=cmap,
                alpha=0.5,
            )
            # print(f"Time taken: {time.time() - st}")
            # plotting the lidar data
            LID_PLOT.set_ydata(plot_data[0])
            VEL_ERR.imshow(plot_data[1], cmap=cmap, interpolation="bicubic")
            DIR_ERR.imshow(plot_data[2], cmap=cmap, interpolation="bicubic")
            # fig.colorbar(VEL_ERR, ax=VEL_ERR)
            fig.canvas.draw_idle()
            # fig.canvas.flush_events()
            # plt.pause(0.000001)
            environment.map.blit(imgtrans, (0, 0))

        previous_position = laser.position
        pygame.display.update()

    pygame.quit()