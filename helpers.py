from pathlib import Path

from tqdm import tqdm

from cityscapes import batch_export
from drone import batch_export_robot
from windflow import batch_generate_windflow
from lidar import gen_iterative_lidar, run_with_index
from visualize import (
    cityscape_visualization,
    windflow_visualization,
    drone_visualization,
)
from pathlib import Path
from tqdm import tqdm


def generate_cityscapes(args):
    path = Path(args.output_dir)
    if not path.exists():
        path.mkdir(parents=True)

    batch_export(
        path,
        args.cont,
        "city",
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

    args.radius_choices = list(map(int, args.radius.split(",")))

    batch_export_robot(
        args.output_dir,
        cityscapes_dir,
        args.num_positions,
        args.min_distance,
        args.radius_choices,
    )


def create_windflows(args):
    cityscapes_dir = Path(args.cityscapes_dir)
    output_dir = Path(args.output_dir)
    if not cityscapes_dir.exists() and not cityscapes_dir.is_dir():
        raise ValueError(f"{cityscapes_dir} does not exist")
    if not output_dir.exists():
        output_dir.mkdir(parents=True)

    winds_csv = Path(args.winds_csv)

    args.speed_x = list(map(float, args.speed_x.split(",")))
    args.speed_y = list(map(float, args.speed_y.split(",")))

    if len(args.speed_x) != len(args.speed_y):
        raise ValueError("speed_x and speed_y must be of the same length")

    speed_candidate_list = [(x, y) for x, y in zip(args.speed_x, args.speed_y)]

    batch_generate_windflow(
        cityscapes_dir,
        output_dir,
        winds_csv,
        speed_candidate_list=speed_candidate_list,
        pre_time=args.pre_time,
        post_time=args.post_time,
        map_size=args.map_size,
    )


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

    cityscape_visualization(
        cityscape_path, args.map_size, args.world_size, args.fig_size
    )


def visualize_windflow(args):

    if not args.export_all and not args.export_all_transparent:
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

        for i in tqdm(list((args.data_dir / "windflow").glob("*.npy"))):
            cityscape_path = args.data_dir / "cityscapes" / f"{i.stem}.csv"
            if not cityscape_path.exists():
                continue
            windflow_visualization(
                cityscape_path,
                i,
                args.map_size,
                args.world_size,
                args.fig_size,
                args.export_dir / f"{i.stem}.png",
            )
    elif args.export_all_transparent:
        print(
            f"Ignoring value of index and exporting all transparent windflow data in {args.data_dir / 'windflow'} to {args.export_dir}"
        )
        args.export_dir = Path(args.export_dir)
        if not args.export_dir.exists():
            args.export_dir.mkdir(parents=True)

        if args.export_dir == args.data_dir / "exportviz":
            raise ValueError("Cannot export to exportviz directory")

        for i in tqdm(
            list((args.data_dir / "windflow").glob("*.npy")), desc="Processing"
        ):
            cityscape_path = args.data_dir / "cityscapes" / f"{i.stem}.csv"
            if not cityscape_path.exists():
                continue
            windflow_visualization(
                cityscape_path,
                i,
                args.map_size,
                args.world_size,
                args.fig_size,
                args.export_dir / f"{i.stem}_transparent.png",
                transparent=True,
            )
    else:
        windflow_visualization(
            cityscape_path,
            windflow_path,
            args.map_size,
            args.world_size,
            args.fig_size,
            args.export,
            args.plot_vector,
            args.export_transparent,
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


def run_demo(args):

    data_dir = Path(args.data_dir)
    if not data_dir.exists() or not data_dir.is_dir():
        raise ValueError(f"{data_dir} does not exist")

    run_with_index(Path(args.data_dir), args.index, args.screen_size, args.padding)
