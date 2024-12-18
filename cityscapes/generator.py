import matplotlib.pyplot as plt
import numpy.typing as npt
import numpy as np
from typing import Callable, Dict, List, Tuple
from .qtree import QTree, find_children, Point
from .sampling import Tag, sample_poisson_disk
from tqdm import tqdm
import pandas as pd
from pathlib import Path


class CityScapeGenerator(object):
    """
    Main class for Cityscape generation
    """

    def __init__(
        self,
        dimension: int,
        sampling_fncs: List[Tuple[Callable, Tag, Dict[str, int]]],
        *,
        map_size: int = 100,
        world_size: int = 500,
        debug: bool = False,
    ):
        """
        :param dimension: the dimension of the generator
        :param sampling_fnc: the random function used for sampling. 3 tuple in which first one is function Callable.
            Second is complex_buildings: this tells us if the buildings should be rectangles or combined rectangles
            Third is sampling_kwargs: function arguments for the sampling function
        :param map_size: this is the factor by which the map should be expanded
        :param debug: useful to see the output of the generator
        """
        self.debug = debug
        self.dimension = dimension
        self.map_size = map_size
        self.buildings = []

        self.sampling_fncs = sampling_fncs
        for _, _, sampling_kwargs in sampling_fncs:
            if "scale" not in sampling_kwargs:
                sampling_kwargs["scale"] = map_size
            if "dimension" not in sampling_kwargs:
                sampling_kwargs["dimension"] = dimension
        self.sampling_kwargs = sampling_kwargs
        self.qtree = None
        self.mean = np.zeros(2)
        self.world_size = world_size

        if self.debug:
            self.debug_fig, self.debug_ax = plt.subplots(2, 2)

    def generate_sample(self, *, show=False):
        self.qtree = QTree(1, self.map_size)
        for sampling_fnc, tag, kwargs in self.sampling_fncs:
            X, Y = self.get_sample_from_sampling_fnc(sampling_fnc, kwargs)
            self.mean = np.array([np.mean(X), np.mean(Y)])
            self.add_samples_to_qtree(X, Y, tag)
        self.qtree.subdivide()
        if self.debug:
            self.debug_ax[0][1] = self.qtree.plot(self.debug_ax[0][1])
            self.debug_ax[0][0].legend(["Skyscrapers", "Houses"])
        self.populate_with_buildings()

        if show:
            plt.show()

    def get_sample_from_sampling_fnc(self, sampling_fnc, sampling_kwargs):
        X, Y = sampling_fnc(**sampling_kwargs)
        self.center = (np.mean(X), np.mean(Y))
        if self.debug:
            sample_plot = self.debug_ax[0][0]
            sample_plot.scatter(X, Y)
            sample_plot.set_xlim([0, self.map_size])
            sample_plot.set_ylim([0, self.map_size])
            sample_plot.set_title("[DEBUG] Sampling Function")
            sample_plot.axis("equal")
        return X, Y

    def add_samples_to_qtree(self, X, Y, tag):
        for x, y in zip(X, Y):
            self.qtree.add_point(x, y, tag)

    def populate_with_buildings(self) -> npt.NDArray:
        """
        Populates the map with buildings
        """
        children = find_children(self.qtree.root)
        if self.debug:
            self.debug_ax[1][0] = self.qtree.plot(self.debug_ax[1][0])
            building_plot = self.debug_ax[1][0]
            final_plot = self.debug_ax[1][1]
            final_plot.axis("equal")
            final_plot.set_title("Generated City")
            final_plot.scatter([0, 100], [0, 100], alpha=0)
        for child in children:
            if not len(child.points):
                continue
            building_coords = make_buildings(
                child.points[0].tag, child, debug=self.debug
            )
            self.buildings += building_coords
            if self.debug:
                for building_coord in building_coords:
                    plot_building(building_coord, final_plot)
                    plot_building(building_coord, building_plot)
        return self.buildings

    def export(self, path):
        if not len(self.buildings):
            raise Exception("there are no buildings to export")

        # round the coordinates to 0 decimal places
        self.buildings = np.round(self.buildings, 0)
        # do mathematics. This is a very important step.
        # the map_size is 100x100. But world size is 500x500.
        # the mapped buildings should be placed at the center of the world.
        # center of world is 250, 250. So we add 250 to the x and y coordinates.
        # but we place the buildings at the center of the map. So we subtract 50 from the x and y coordinates.
        # this is the same as adding 200 to the x and y coordinates.
        self.buildings[:, :4] += (self.world_size / 2) - (self.map_size / 2)

        df = pd.DataFrame(self.buildings)
        df.columns = ["x1", "y1", "x2", "y2", "height"]
        df.to_csv(f"{path}.csv", index=False)

        # self.robot_coords.to_csv(f"{path}_robot.csv", index=False)


def plot_building(coords, ax):
    ax.add_patch(
        plt.Rectangle(
            (coords[0], coords[1]),
            coords[2] - coords[0],
            coords[3] - coords[1],
        )
    )


def make_buildings(tag, node, *, debug=False) -> List[npt.NDArray]:
    match tag:
        case Tag.SKYSCRAPER:
            return make_square_buildings(node, debug=debug)
        case Tag.HOUSE:
            return make_square_buildings(node, debug=debug)


def make_square_buildings(node, *, debug):
    point = Point(node.x0 + node.width // 2, node.y0 + node.height // 2, 3)
    x1, y1, x2, y2 = get_bounds_of_house(point, node)
    height = 25
    return [np.array([x1, y1, x2, y2, height])]


def make_house_buildings(node, *, debug):
    ans = []
    while np.random.random() > 0.4:
        point = Point(
            node.x0 + node.width * np.random.random(),
            node.y0 + node.height * np.random.random(),
            3,
        )
        x1, y1, x2, y2 = get_bounds_of_house(point, node, alpha=0.3)
        height = 25
        ans.append(np.array([x1, y1, x2, y2, height]))
    return ans


def get_bounds_of_house(point, node, factor=7, alpha=0.3, beta=0.7):
    bounded_random = lambda: np.random.random() * alpha + beta
    height = bounded_random() * factor
    width = bounded_random() * factor
    x1 = max(point.x - width / 2, node.x0)
    y1 = max(point.y - height / 2, node.y0)
    x2 = min(point.x + width / 2, node.width + node.x0)
    y2 = min(point.y + height / 2, node.height + node.y0)

    return x1, y1, x2, y2


def batch_export(
    path: Path,
    cont: bool,
    name_prefix: str,
    map_size: int,
    world_size: int,
    n_cityscapes: int,
    n_buildings: int,
    building_density: int,
):
    """
    path: export the generated cityscape to this directory.
    name_prefix: prepend prefix onto cityscape file name. Eg `{name_prefix}_1.csv`.
    map_size: side length of the city in metres.
    n_cityscapes: number of cityscapes to be generated.
    n_buildings: number of buildings in each cityscape.
    building_density: controls the clustering of the buildings in the cityscape.
    """
    last_index = 0
    if cont:
        # find last index with prefix in path
        for file in path.glob(f"{name_prefix}_*.csv"):
            last_index = max(last_index, int(file.stem.split("_")[-1]))
        pass

    for i in tqdm(range(n_cityscapes)):
        proc_gen = CityScapeGenerator(
            2,
            sampling_fncs=[
                (
                    sample_poisson_disk,
                    Tag.HOUSE,
                    {"density": building_density, "n_buildings": n_buildings},
                ),
            ],
            map_size=map_size,
            world_size=world_size,
        )
        proc_gen.generate_sample()

        proc_gen.export(f"{path}/{name_prefix}_{i+last_index}")
