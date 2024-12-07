"""
CLI for generating data
"""

import argparse
from cityscapes import batch_export
from drone import generate_positions
from windflow import generate_windflow
from visualize import cityscape_visualization, windflow_visualization
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
    generate_positions(cityscapes_dir, args.num_positions, Path(args.output_dir))


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


def visualise_cityscape(args):
    cityscape_path = Path(args.filename)
    if not cityscape_path.exists():
        raise ValueError(f"{cityscape_path} does not exist")
    
    args.fig_size = tuple(map(int, args.fig_size.strip("()").split(",")))

    cityscape_visualization(cityscape_path, args.map_size, args.fig_size)

def visualize_windflow(args):
    windflow_path = Path(args.data_dir / "windflow" / f"city_{args.index}.npy")
    cityscape_path = Path(args.data_dir / "cityscapes" / f"city_{args.index}.csv")
    if not windflow_path.exists():
        raise ValueError(f"{windflow_path} does not exist")
    if not cityscape_path.exists():
        raise ValueError(f"{cityscape_path} does not exist")
    
    args.fig_size = tuple(map(int, args.fig_size.strip("()").split(",")))

    windflow_visualization(cityscape_path, windflow_path, args.map_size, args.fig_size)

def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic data for predicting local wind fields"
    )
    subprasers = parser.add_subparsers(dest="command")

    # Cityscapes
    cityscapes_parser = subprasers.add_parser(
        "cityscapes", help="Generate cityscape csv files"
    )
    cityscapes_parser.add_argument(
        "--prefix", type=str, default="city", help="Prefix for the output files"
    )
    cityscapes_parser.add_argument(
        "--n_cityscapes", type=int, default=60, help="Number of cityscapes to generate"
    )
    cityscapes_parser.add_argument(
        "--map_size", type=int, default=100, help="Side length of map in metres"
    )
    cityscapes_parser.add_argument(
        "--n_buildings",
        type=int,
        default=32,
        help="Number of buildings to generate per cityscape",
    )
    cityscapes_parser.add_argument(
        "--building_density",
        type=int,
        default=8,
        help="Controls the clustering of buildings within a cityscape",
    )
    cityscapes_parser.add_argument(
        "--output_dir",
        type=str,
        default="data/cityscapes",
        help="Directory to save buildings",
    )

    # MATLAB
    _ = subprasers.add_parser(
        "matlab", help="Convert cityscape csv files to MATLAB format"
    )

    # Drone
    drone_parser = subprasers.add_parser(
        "drone", help="Generate drone positions for cityscapes"
    )

    drone_parser.add_argument(
        "--cityscapes_dir",
        type=str,
        default="data/cityscapes",
        help="Directory containing cityscape csv files",
    )

    drone_parser.add_argument(
        "--num_positions",
        type=int,
        default=10,
        help="Number of drone positions to generate",
    )

    drone_parser.add_argument(
        "--output_dir",
        type=str,
        default="data/drone_positions",
        help="Directory to save drone positions",
    )

    # Windflow
    windflow_parser = subprasers.add_parser("windflow", help="Generate windflow data")

    windflow_parser.add_argument(
        "--cityscapes_dir",
        type=str,
        default="data/cityscapes",
        help="Directory containing cityscape csv files",
    )

    windflow_parser.add_argument(
        "--output_dir",
        type=str,
        default="data/windflow",
        help="Directory to save windflow data",
    )

    # Visualisation parser
    viz_parser = subprasers.add_parser("visualize", aliases=["viz"], help="Visualize the anything")
    vizsub = viz_parser.add_subparsers(dest="visualize")
    vizsubcity = vizsub.add_parser("cityscape", aliases=["city"], help="Visualize the cityscape only")
    vizsubcity.add_argument(
        "--filename",
        type=str,
        help="File of the cityscape csv",
    )
    vizsubcity.add_argument(
        "--map_size", type=int, default=100, help="Side length of map in metres"
    )
    vizsubcity.add_argument(
        "--fig_size", type=str, default="(5, 5)", help="Size of the figure"
    )

    vizsubwind = vizsub.add_parser("windflow", aliases=["wind"], help="Visualize the windflow data")
    vizsubwind.add_argument("--data_dir", type=Path, help="Directory containing windflow data", default="data")
    vizsubwind.add_argument(
        "--index",
        type=int,
        help="Index of the windflow data to visualize",
        required=True
    )
    vizsubwind.add_argument(
        "--map_size", type=int, default=100, help="Side length of map in metres"
    )
    vizsubwind.add_argument(
        "--fig_size", type=str, default="(5, 5)", help="Size of the figure"
    )

    args = parser.parse_args()

    if args.command == "cityscapes":
        generate_cityscapes(args)
    if args.command == "matlab":
        generate_matlab()

    if args.command == "drone":
        generate_drone_positions(args)

    if args.command == "windflow":
        create_windflows(args)
    if args.command == "visualize" or args.command == "viz":
        if args.visualize == "cityscape" or args.visualize == "city":
            visualise_cityscape(args)
        elif args.visualize == "windflow" or args.visualize == "wind":
            visualize_windflow(args)
    if not args.command:
        parser.print_help()


if __name__ == "__main__":
    main()
