from enum import Enum
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from scipy.stats import qmc

from typing import Tuple


class Tag(Enum):
    SKYSCRAPER = 1
    HOUSE = 2


def sample_poisson_disk(
    dimension: int,
    scale: int,
    density: int = 18,
    n_buildings: int = 25,
) -> Tuple[npt.NDArray, npt.NDArray]:
    """
    Samples using the poisson disk algorithm

    :param dimension: the number of dimensions the sample should return
    :param scale: scales the co-ordinates from [0, 1) to [0, X)
    :param density: [0, inf), higher values make denser clusters
    :param n_buildings: int, number of buildings to generate, OFTEN GENERATES LESSER BUILDINGS

    :return: X, Y numpy ndarrays
    """
    radius = 1 / density
    engine = qmc.PoissonDisk(d=dimension, radius=radius)
    sample = engine.random(n_buildings)  # number of buildings
    return sample[:, 0] * scale, sample[:, 1] * scale