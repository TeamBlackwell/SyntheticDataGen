import numpy as np
from phi.jax import flow
from tqdm import trange

@flow.math.jit_compile
def _step_2d(v, p, ob_list):
    v = flow.advect.semi_lagrangian(v, v, 1.0)
    v, p = flow.fluid.make_incompressible(v, ob_list, flow.Solve("auto", 1e-5, x0=p))
    return v, p, ob_list


def run_flow2d(
    rect_data: np.ndarray,
    pre_time: int,
    avg_time_window: int,
    map_size: int,
    wind_speed: tuple[float,float],
) -> np.ndarray:
    """
    Run a flow simulation with the given parameters. Mainly with the rectangles and the speed of winds.
    Currently supports only one (speed_x, speed_y) (one wind field).

    :param rect_data: np.ndarray of shape (n, 4) where n is the number of rectangles and 4 is the x1, y1, x2, y2
    :param pre_time: int, the number of time steps to run before average window starts
    :param avg_time_window: int, the number of time steps to average over, counted after pre_time
    :param map_size: int, the size of the map
    :param wind_speed: float, 2-tuple of the wind speed (speed_x, speed_y)
    :return v_data: np.ndarray of shape (map_size, map_size, 2). the velocity data in 2D
    """

    # for each cuboid, make a box
    rect_list = []
    for rect in rect_data:
        x1, y1, x2, y2 = rect
        rect_list.append(
            flow.Box(flow.vec(x=x1, y=y1), flow.vec(x=x2, y=y2))
        )
        break

    # make all of them obstacles
    obstacle_list = []
    for cuboid in rect_list:
        obstacle_list.append(flow.Obstacle(cuboid, 0))

    speeds = flow.tensor(wind_speed)  # (2,) tensor for speeds

    # velocity grid, boundary box, boundary mask, pressure
    velocity = flow.StaggeredGrid(
        values=speeds,
        boundary=flow.ZERO_GRADIENT,
        bounds=flow.Box(x=map_size, y=map_size),
        x=map_size,
        y=map_size,
    )

    pressure = None

    v_data = flow.iterate(
        _step_2d,
        flow.batch(time=(pre_time + avg_time_window)),
        velocity,
        pressure,
        obstacle_list,
        range=trange,
    )
    quit()
    v_numpy = v_data.numpy()

    x_data = v_numpy[0]  # (T, H + 1, W, D)
    y_data = v_numpy[1]  # (T, H, W + 1, D)

    print(x_data.shape, y_data.shape)

    x_data = x_data[:, :-1, :, :]
    y_data = y_data[:, :, 1:, :]

    x_data = x_data[pre_time:, :, :]
    y_data = y_data[pre_time:, :, :]

    x_data = np.mean(x_data, axis=0)
    y_data = np.mean(y_data, axis=0)

    v_stacked = np.stack((x_data, y_data), axis=0)  # (2, H, W)
    v_stacked = np.moveaxis(v_stacked, 0, -1)

    if np.isnan(v_stacked).any():
        raise ValueError("NANs in the velocity data")

    print(v_stacked.shape)

    print(v_stacked[:, :, 0])
    return v_stacked
