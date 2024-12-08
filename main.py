"""
CLI for generating data
"""

from os import environ

environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"
import argparse
from pathlib import Path
import helpers as h
import warnings

warnings.filterwarnings("ignore")


def add_generate_commands(genparser):
    gensub = genparser.add_subparsers(dest="generate_what")

    # Cityscapes
    cityscapes_parser = gensub.add_parser(
        "cityscapes", aliases=["city"], help="Generate cityscape csv files"
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
    _ = gensub.add_parser("matlab", help="Convert cityscape csv files to MATLAB format")

    # Drone
    drone_parser = gensub.add_parser(
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
        "--min_distance",
        type=int,
        default=2,
        help="Minimum distance to buildings (buffer)",
    )
    drone_parser.add_argument(
        "--radius",
        type=int,
        default=40,
        help="Radius of the applicable area from the center of the cityscape",
    )
    drone_parser.add_argument(
        "--output_dir",
        type=str,
        default="data/drone_positions",
        help="Directory to save drone positions",
    )

    # Windflow
    windflow_parser = gensub.add_parser(
        "windflow", aliases=["wind"], help="Generate windflow data"
    )

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

    lidar_parser = gensub.add_parser("lidar", help="Generate lidar data")

    lidar_parser.add_argument(
        "--citymaps_dir",
        type=str,
        default="data/exportviz",
        help="Directory containing cityscape csv files",
    )
    lidar_parser.add_argument(
        "--positions_dir",
        type=str,
        default="data/drone_positions",
        help="Directory containing drone positions",
    )
    lidar_parser.add_argument(
        "--output_dir",
        type=str,
        default="data/lidar",
        help="Directory to save lidar data",
    )
    return genparser


def add_vizualiser_commands(viz_parser):
    vizsub = viz_parser.add_subparsers(dest="visualize")
    vizsubcity = vizsub.add_parser(
        "cityscape", aliases=["city"], help="Visualize the cityscape only"
    )
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

    vizsubwind = vizsub.add_parser(
        "windflow", aliases=["wind"], help="Visualize the windflow data"
    )
    vizsubwind.add_argument(
        "--data_dir",
        type=Path,
        help="Directory containing windflow data",
        default="data",
    )
    vizsubwind.add_argument(
        "--index",
        type=int,
        help="Index of the windflow data to visualize",
        required=True,
    )
    vizsubwind.add_argument(
        "--map_size", type=int, default=100, help="Side length of map in metres"
    )
    vizsubwind.add_argument(
        "--fig_size", type=str, default="(5, 5)", help="Size of the figure"
    )
    vizsubwind.add_argument(
        "--export", type=str, default=None, help="Export the figure"
    )
    vizsubwind.add_argument(
        "--export-all", default=False, help="Export all figures", action="store_true"
    )
    vizsubwind.add_argument(
        "--export-dir", type=Path, default="data/exportviz", help="Export directory"
    )

    vizsubdrone = vizsub.add_parser("drone", help="Visualize the drone positions")
    vizsubdrone.add_argument(
        "--data_dir",
        type=Path,
        help="Directory containing windflow data",
        default="data",
    )
    vizsubdrone.add_argument(
        "--index", type=int, help="Index of the drone data to visualize", required=True
    )
    vizsubdrone.add_argument(
        "--map_size", type=int, default=100, help="Side length of map in metres"
    )
    vizsubdrone.add_argument(
        "--fig_size", type=str, default="(5, 5)", help="Size of the figure"
    )

    return viz_parser


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic data for predicting local wind fields"
    )
    subprasers = parser.add_subparsers(dest="command")

    # generation parser
    genparser = subprasers.add_parser(
        "generate", aliases=["gen"], help="Generate synthetic data"
    )
    genparser = add_generate_commands(genparser)
    # visualisation parser
    viz_parser = subprasers.add_parser(
        "visualize", aliases=["viz"], help="Visualize the anything"
    )
    viz_parser = add_vizualiser_commands(viz_parser)

    args = parser.parse_args()

    if args.command == "generate" or args.command == "gen":
        if args.generate_what == "cityscapes" or args.generate_what == "city":
            h.generate_cityscapes(args)
        if args.generate_what == "matlab":
            h.generate_matlab()
        if args.generate_what == "drone":
            h.generate_drone_positions(args)
        if args.generate_what == "windflow" or args.generate_what == "wind":
            h.create_windflows(args)
        if args.generate_what == "lidar":
            h.generate_lidar_data(args)

        if not args.generate_what:
            genparser.print_help()

    elif args.command == "visualize" or args.command == "viz":
        if args.visualize == "cityscape" or args.visualize == "city":
            h.visualise_cityscape(args)
        elif args.visualize == "windflow" or args.visualize == "wind":
            h.visualize_windflow(args)
        elif args.visualize == "drone":
            h.visuaize_drone(args)

        if not args.visualize:
            viz_parser.print_help()

    if not args.command:
        parser.print_help()


if __name__ == "__main__":
    main()
