"""
CLI for generating data
"""

import argparse
from cityscapes import batch_export
from drone import generate_positions
from windflow import generate_windflow
from visualize import cityscape_visualization
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
    cityscape_path = Path(args.cityscape)
    if not cityscape_path.exists():
        raise ValueError(f"{cityscape_path} does not exist")
    cityscape_visualization(cityscape_path, args.map_size)


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
    viz_parser = subprasers.add_parser("visualize", help="Visualize the cityscape")
    viz_parser.add_argument(
        "--cityscape",
        type=str,
        help="Directory containing cityscape csv files",
    )
    viz_parser.add_argument(
        "--map_size", type=int, default=100, help="Side length of map in metres"
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
    if args.command == "visualize":
        visualise_cityscape(args)
    if not args.command:
        parser.print_help()


if __name__ == "__main__":
    main()
