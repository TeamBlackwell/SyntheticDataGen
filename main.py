"""
CLI for generating data
"""

import argparse
from building_data import batch_export
from pathlib import Path


def generate_cityscapes(args):
    path = Path(args.output_dir)
    if not path.exists():
        path.mkdir(parents=True)

    batch_export(path, args.num_cities, args.map_size, args.prefix)


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
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/cityscapes",
        help="Directory to save buildings",
    )
    cityscapes_parser.add_argument(
        "--prefix", type=str, default="city", help="Prefix for the output files"
    )
    args = parser.parse_args()

    if args.command == "cityscapes":
        generate_cityscapes(args)


if __name__ == "__main__":
    main()
