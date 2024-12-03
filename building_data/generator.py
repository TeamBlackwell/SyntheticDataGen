import matplotlib.pyplot as plt
import numpy.typing as npt
import numpy as np

from typing import Callable, Dict, List, Tuple
from .qtree import QTree, find_children, Point
from .sampling import Tag, sample_poisson_disk
from tqdm import tqdm
import pandas as pd


class CityScapeGenerator(object):
    """
    Main class for Cityscape generation
    """

    def __init__(
        self,
        dimension: int,
        sampling_fncs: List[Tuple[Callable, Tag, Dict[str, any]]],
        *,
        scale: int = 100,
        debug: bool = False,
    ):
        """
        :param dimension: the dimension of the generator
        :param sampling_fnc: the random function used for sampling. 3 tuple in which first one is function Callable.
            Second is complex_buildings: this tells us if the buildings should be rectangles or combined rectangles
            Third is sampling_kwargs: function arguments for the sampling function
        :param scale: this is the factor by which the map should be expanded
        :param debug: useful to see the output of the generator
        """
        self.debug = debug
        self.dimension = dimension
        self.scale = scale
        self.buildings = []

        self.sampling_fncs = sampling_fncs
        for _, _, sampling_kwargs in sampling_fncs:
            if "scale" not in sampling_kwargs:
                sampling_kwargs["scale"] = scale
            if "dimension" not in sampling_kwargs:
                sampling_kwargs["dimension"] = dimension
        self.sampling_kwargs = sampling_kwargs
        self.qtree = None

        if self.debug:
            self.debug_fig, self.debug_ax = plt.subplots(2, 2)

    def generate_sample(self, *, show=False):
        self.qtree = QTree(1, self.scale)
        for sampling_fnc, tag, kwargs in self.sampling_fncs:
            X, Y = self.get_sample_from_sampling_fnc(sampling_fnc, kwargs)
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
        if self.debug:
            sample_plot = self.debug_ax[0][0]
            sample_plot.scatter(X, Y)
            sample_plot.set_xlim([0, self.scale])
            sample_plot.set_ylim([0, self.scale])
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
        df = pd.DataFrame(self.buildings)
        df.columns = ["x1", "y1", "x2", "y2", "height"]
        df.to_csv(path, index=False)


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
            return make_house_buildings(node, debug=debug)


def make_square_buildings(node, *, debug):
    point = Point(node.x0 + node.width // 2, node.y0 + node.height // 2, 3)
    x1, y1, x2, y2 = get_bounds_of_house(point, node)
    height = np.random.random() * 10 + 20
    return [np.array([x1, y1, x2, y2, height])]


def make_house_buildings(node, *, debug):
    ans = []
    while np.random.random() > 0.4:
        point = Point(
            node.x0 + node.width * np.random.random(),
            node.y0 + node.height * np.random.random(),
            3,
        )
        x1, y1, x2, y2 = get_bounds_of_house(
            point, node, width_equal_height=False, factor=6, alpha=0.3
        )
        height = 10
        ans.append(np.array([x1, y1, x2, y2, height]))
    return ans


def get_bounds_of_house(
    point, node, width_equal_height=True, factor=7, alpha=0.5, beta=0.3
):
    bounded_random = lambda: np.random.random() * alpha + beta
    height = bounded_random() * factor
    width = height if width_equal_height else bounded_random() * factor
    x1 = max(point.x - width / 2, node.x0)
    y1 = max(point.y - height / 2, node.y0)
    x2 = min(point.x + width / 2, node.width + node.x0)
    y2 = min(point.y + height / 2, node.height + node.y0)

    return x1, y1, x2, y2


# @TODO: Accept density and n_buildings as arguments
def batch_export(path, n_exports=60, scale=100, name_prefix="sample"):
    """
    path: the path to the directory where you need to export
    """
    for i in tqdm(range(n_exports)):
        proc_gen = CityScapeGenerator(
            2,
            sampling_fncs=[
                (sample_poisson_disk, Tag.SKYSCRAPER, {"density": 28}),
                (sample_poisson_disk, Tag.HOUSE, {"density": 15, "n_buildings": 75}),
            ],
            scale=scale,
        )
        proc_gen.generate_sample()
        proc_gen.export(f"{path}/{name_prefix}-{i}.csv")
