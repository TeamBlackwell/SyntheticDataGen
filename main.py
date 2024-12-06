"""
CLI for generating data
"""

import argparse
from cityscapes import batch_export
from drone import generate_positions
from windflow import generate_windflow
from pathlib import Path


def generate_cityscapes(args):
    path = Path(args.output_dir)
    if not path.exists():
        path.mkdir(parents=True)

    batch_export(path, n_exports=args.num_cities, sclae=args.map_size, name_prefix=args.prefix)


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


def generate_matlab(args):
    print("To generate the MATLAB meshes please use MATLAB :(")


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
        "--num_cities", type=int, default=60, help="Number of cityscapes to generate"
    )
    cityscapes_parser.add_argument(
        "--map_size", type=int, default=250, help="Size of the map"
    )
    cityscapes_parser.add_argument(
        "--output_dir",
        type=str,
        default="data/cityscapes",
        help="Directory to save buildings",
    )
    cityscapes_parser.add_argument(
        "--prefix", type=str, default="city", help="Prefix for the output files"
    )

    # MATLAB
    matlab_parser = subprasers.add_parser(
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

    args = parser.parse_args()

    if args.command == "cityscapes":
        generate_cityscapes(args)
    if args.command == "matlab":
        generate_matlab(args)

    if args.command == "drone":
        generate_drone_positions(args)

    if args.command == "windflow":
        create_windflows(args)


if __name__ == "__main__":
    main()
