from pathlib import Path

from cityscapes import batch_export
from drone import batch_export_robot
from windflow import generate_windflow
from lidar import gen_iterative_lidar
from visualize import (
    cityscape_visualization,
    windflow_visualization,
    drone_visualization,
)
from pathlib import Path


def generate_cityscapes(args):
    path = Path(args.output_dir)
    if not path.exists():
        path.mkdir(parents=True)

    batch_export(
        path,
        args.prefix,
        args.map_size,
        args.n_cityscapes,
        args.n_buildings,
        args.building_density,
    )


def generate_drone_positions(args):
    cityscapes_dir = Path(args.cityscapes_dir)
    if not cityscapes_dir.exists() and not cityscapes_dir.is_dir():
        raise ValueError(f"{cityscapes_dir} does not exist")

    if not Path(args.output_dir).exists():
        Path(args.output_dir).mkdir(parents=True)

    batch_export_robot(args.output_dir, cityscapes_dir)


def create_windflows(args):
    cityscapes_dir = Path(args.cityscapes_dir)
    output_dir = Path(args.output_dir)
    if not cityscapes_dir.exists() and not cityscapes_dir.is_dir():
        raise ValueError(f"{cityscapes_dir} does not exist")
    if not output_dir.exists():
        output_dir.mkdir(parents=True)
    generate_windflow(cityscapes_dir, output_dir)


def generate_matlab():
    print("To generate the MATLAB meshes please use MATLAB :(")


def generate_lidar_data(args):
    citymaps_dir = Path(args.citymaps_dir)
    positions_dir = Path(args.positions_dir)
    output_dir = Path(args.output_dir)

    if not citymaps_dir.exists() or not citymaps_dir.is_dir():
        raise ValueError(f"{citymaps_dir} does not exist")
    if not positions_dir.exists() or not positions_dir.is_dir():
        raise ValueError(f"{positions_dir} does not exist")

    if not output_dir.exists():
        output_dir.mkdir(parents=True)

    gen_iterative_lidar(citymaps_dir, positions_dir, output_dir)


def visualise_cityscape(args):
    cityscape_path = Path(args.filename)
    if not cityscape_path.exists():
        raise ValueError(f"{cityscape_path} does not exist")

    args.fig_size = tuple(map(int, args.fig_size.strip("()").split(",")))

    cityscape_visualization(cityscape_path, args.map_size, args.fig_size)


def visualize_windflow(args):

    if not args.export_all:
        windflow_path = Path(args.data_dir / "windflow" / f"city_{args.index}.npy")
        cityscape_path = Path(args.data_dir / "cityscapes" / f"city_{args.index}.csv")

        if not windflow_path.exists():
            raise ValueError(f"{windflow_path} does not exist")
        if not cityscape_path.exists():
            raise ValueError(f"{cityscape_path} does not exist")

    args.fig_size = tuple(map(int, args.fig_size.strip("()").split(",")))

    if args.export_all:
        print(
            f"Ignoring value of index and exporting all windflow data in {args.data_dir / 'windflow'} to {args.export_dir}"
        )
        args.export_dir = Path(args.export_dir)
        if not args.export_dir.exists():
            args.export_dir.mkdir(parents=True)

        for i in (args.data_dir / "windflow").glob("*.npy"):
            cityscape_path = args.data_dir / "cityscapes" / f"{i.stem}.csv"
            if not cityscape_path.exists():
                continue
            windflow_visualization(
                cityscape_path,
                i,
                args.map_size,
                args.fig_size,
                args.export_dir / f"{i.stem}.png",
            )
    else:
        windflow_visualization(
            cityscape_path, windflow_path, args.map_size, args.fig_size, args.export
        )


def visuaize_drone(args):
    drone_path = Path(args.data_dir / "drone_positions" / f"city_{args.index}.csv")
    if not drone_path.exists():
        raise ValueError(f"{drone_path} does not exist")
    map_path = Path(args.data_dir / "cityscapes" / f"city_{args.index}.csv")
    if not map_path.exists():
        raise ValueError(f"{map_path} does not exist")

    args.fig_size = tuple(map(int, args.fig_size.strip("()").split(",")))

    drone_visualization(map_path, drone_path, args.map_size, args.fig_size)
