from pathlib import Path
from matplotlib import pyplot as plt
import numpy as np
from phi.torch import flow
from tqdm import trange


@flow.math.jit_compile
def _step(v, p, ob_list, bndry_mask, spd_x, spd_y):

    v = flow.advect.semi_lagrangian(v, v, 1.0)
    v = v * (1 - bndry_mask) + bndry_mask * (
        spd_x,
        spd_y,
    )

    v, p = flow.fluid.make_incompressible(v, ob_list, flow.Solve("auto", 1e-5, x0=p))

    return v, p, ob_list, bndry_mask, spd_x, spd_y


def run_flow(
    rect_data: np.ndarray,
    pre_time: int,
    avg_time_window: int,
    map_size: int,
    speed_x: float,
    speed_y: float,
) -> np.ndarray:
    """
    Run a flow simulation with the given parameters. Mainly with the rectangles and the speed of winds.
    Currently supports only one speed_x and speed_y (one wind field).

    :param rect_data: np.ndarray of shape (n, 4) where n is the number of rectangles and 4 is the x1, y1, x2, y2
    :param pre_time: int, the number of time steps to run before average window starts
    :param avg_time_window: int, the number of time steps to average over, counted after pre_time
    :param map_size: int, the size of the map
    :param speed_x: float, the x speed
    :param speed_y: float, the y speed

    :return v_data: np.ndarray of shape (map_size, map_size, 2). the velocity data
    """

    # for each cuboid, make a box
    cuboid_list = []
    for rect in rect_data:
        x1, y1, x2, y2 = rect
        cuboid_list.append(flow.Box(flow.vec(x=x1, y=y1), flow.vec(x=x2, y=y2)))

    # make all of them obstacles
    obstacle_list = []
    for cuboid in cuboid_list:
        obstacle_list.append(flow.Obstacle(cuboid))

    speeds = flow.tensor([speed_x, speed_y])

    # velociry grid, boundary box, boundary mask, pressure
    velocity = flow.StaggeredGrid(
        speeds,
        flow.ZERO_GRADIENT,
        x=map_size,
        y=map_size,
        bounds=flow.Box(x=map_size, y=map_size),
    )
    boundary_box = flow.Box(x=(-1 * flow.INF, 0.5), y=None)
    boundary_mask = flow.StaggeredGrid(
        boundary_box, velocity.extrapolation, velocity.bounds, velocity.resolution
    )
    pressure = None

    v_data, p_data, _, _, _, _ = flow.iterate(
        _step,
        flow.batch(time=(pre_time + avg_time_window)),
        velocity,
        pressure,
        obstacle_list,
        boundary_mask,
        speed_x,
        speed_y,
        range=trange,
    )

    v_numpy = v_data.numpy()

    x_data = v_numpy[0]  # (T, H + 1, W)
    y_data = v_numpy[1]  # (T, H, W + 1)

    # TODO: check if the following remains consitent based on X and Y direction. For now, it is consistent.
    # ignoring LAST in x_data
    # ignoring FIRST in y_data

    x_data = x_data[:, :-1, :]
    y_data = y_data[:, :, 1:]

    x_data = x_data[pre_time:, :, :]
    y_data = y_data[pre_time:, :, :]

    # x_data = np.mean(x_data, axis=0)
    # y_data = np.mean(y_data, axis=0)

    v_stacked = np.stack((x_data, y_data), axis=0)  # (2, H, W)
    v_stacked = np.moveaxis(v_stacked, 0, -1)

    if np.isnan(v_stacked).any():
        raise ValueError("NANs in the velocity data")

    return v_stacked
