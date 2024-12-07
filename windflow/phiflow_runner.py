from pathlib import Path
from matplotlib import pyplot as plt
import numpy as np
from phi import flow
from tqdm import trange


@flow.math.jit_compile
def _step(v, p, ob_list, bndry_mask, spd_x, spd_y):

    v = flow.advect.semi_lagrangian(v, v, 1.0)
    v = v * (1 - bndry_mask) + bndry_mask * (
        spd_x,
        spd_y,
    )  # make sure you dont simulat OOB

    v, p = flow.fluid.make_incompressible(
        v, ob_list, flow.Solve("auto", 1e-5, x0=p)
    )  # make it do the boundary thign

    return v, p, ob_list, bndry_mask, spd_x, spd_y


@flow.math.jit_compile
def _step_3d(v, p, ob_list):

    v = flow.advect.semi_lagrangian(v, v, 1.0)
    # v = v * (1 - bndry_mask) + bndry_mask * (
    #     speeds[0],
    #     speeds[1],
    #     speeds[2],
    # )  # make sure you dont simulat OOB

    # print("\n", v, "\n")
    # print("\n", v, "\n")
    v, p = flow.fluid.make_incompressible(
        v, ob_list, flow.Solve("auto", 1e-5, x0=p)
    )  # make it do the boundary thing
    return v, p, ob_list


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

    # visualization
    anim = flow.plot(
        [v_data.curl(), *cuboid_list[::-1]],
        animate="time",
        size=(6, 6),
        frame_time=10,
        overlay="list",
    )
    plt.show()
    _step.traces.clear()
    _step.recorded_mappings.clear()

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

    x_data = np.mean(x_data, axis=0)
    y_data = np.mean(y_data, axis=0)

    v_stacked = np.stack((x_data, y_data), axis=0)  # (2, H, W)
    # change to (H, W, 2)
    v_stacked = np.moveaxis(v_stacked, 0, -1)

    if np.isnan(v_stacked).any():
        raise ValueError("NANs in the velocity data")

    return v_stacked


def run_flow3d(
    rect_data: np.ndarray,
    pre_time: int,
    avg_time_window: int,
    map_size: int,
    wind_speed: tuple[float],
) -> np.ndarray:
    """
    Run a flow simulation with the given parameters. Mainly with the rectangles and the speed of winds.
    Currently supports only one (speed_x, speed_y, speed_z) (one wind field).

    :param rect_data: np.ndarray of shape (n, 5) where n is the number of rectangles and 4 is the x1, y1, x2, y2, H (H: Height)
    :param pre_time: int, the number of time steps to run before average window starts
    :param avg_time_window: int, the number of time steps to average over, counted after pre_time
    :param map_size: int, the size of the map
    :param wind_speed: float, 3-tuple of the wind speed (speed_x, speed_y, speed_z)

    :return v_data: np.ndarray of shape (map_size, map_size, 3). the velocity data in 3D
    """

    # for each cuboid, make a box
    cuboid_list = []
    for rect in rect_data:
        x1, y1, x2, y2, h = rect

        cuboid_list.append(
            flow.Box(flow.vec(x=x1, y=y1, z=0), flow.vec(x=x2, y=y2, z=h))
        )

    # make all of them obstacles
    obstacle_list = []
    for cuboid in cuboid_list:
        obstacle_list.append(flow.Obstacle(cuboid, angular_velocity=(0, 0, 0)))

    speeds = flow.tensor(wind_speed)  # (3,) tensor for speeds

    # velociry grid, boundary box, boundary mask, pressure
    velocity = flow.StaggeredGrid(
        values=speeds,
        boundary=flow.ZERO_GRADIENT,
        bounds=flow.Box(x=map_size, y=map_size, z=map_size),
        x=map_size,
        y=map_size,
        z=map_size,
    )

    # boundary_box = flow.Box(x=None, y=None, z=None)
    # print("\n", boundary_box, "\n")
    # boundary_mask = flow.StaggeredGrid(
    #     boundary_box, velocity.extrapolation, velocity.bounds, velocity.resolution
    # )
    # print("\n", boundary_mask, "\n")

    pressure = None
    # plt.show()

    # print(velocity)
    v_data, p_data, _ = flow.iterate(
        _step_3d,
        flow.batch(time=(pre_time + avg_time_window)),
        velocity,
        pressure,
        obstacle_list,
        range=trange,
    )

    # visualization
    # anim = flow.plot(
    #     [traj.curl(), *cuboid_list[::-1]],
    #     animate="time",
    #     size=(6, 6),
    #     frame_time=10,
    #     overlay="list",
    # )
    # plt.show()
    # _step.traces.clear()
    # _step.recorded_mappings.clear()

    v_numpy = v_data.numpy()

    x_data = v_numpy[0]  # (T, H + 1, W, D)
    y_data = v_numpy[1]  # (T, H, W + 1, D)
    z_data = v_numpy[2]  # (T, H, W, D + 1)

    # print(x_data.shape, y_data.shape, z_data.shape)

    # TODO: check if the following remains consitent based on X and Y direction. For now, it is consistent.
    # ignoring LAST in x_data
    # ignoring FIRST in y_data
    # ignoring LAST in z_data

    x_data = x_data[:, :-1, :, :]
    y_data = y_data[:, :, 1:, :]
    z_data = z_data[:, :, :, :-1]

    x_data = x_data[pre_time:, :, :]
    y_data = y_data[pre_time:, :, :]
    z_data = z_data[pre_time:, :, :]

    x_data = np.mean(x_data, axis=0)
    y_data = np.mean(y_data, axis=0)
    z_data = np.mean(z_data, axis=0)

    v_stacked = np.stack((x_data, y_data, z_data), axis=0)  # (3, H, W)
    # # change to (H, W, 2)
    v_stacked = np.moveaxis(v_stacked, 0, -1)

    if np.isnan(v_stacked).any():
        raise ValueError("NANs in the velocity data")

    print(v_stacked.shape)
    return v_stacked


def save_flow(flow_data: np.ndarray, path: Path | str) -> None:
    """
    Save the flow data, which is a np.ndarray of shape (map_size, map_size, 2) to the given path.

    Saves any numpy array, of any shape. Saves to the given path. Path should end with .npy.
    """
    if isinstance(path, str):
        path = Path(path)

    if path.suffix == "npy":
        raise ValueError("Path should end with .npy")

    np.save(path, flow_data)
