"""
CLI for generating data
"""

import argparse
from cityscapes import batch_export
from drone import generate_positions
from windflow import generate_windflow
from lidar import gen_iterative_lidar
from visualize import cityscape_visualization, windflow_visualization
from pathlib import Path

import warnings
warnings.filterwarnings("ignore")


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
        print(f"Ignoring value of index and exporting all windflow data in {args.data_dir / 'windflow'} to {args.export_dir}")
        args.export_dir = Path(args.export_dir)
        if not args.export_dir.exists():
            args.export_dir.mkdir(parents=True)
        
        for i in (args.data_dir / "windflow").glob("*.npy"):
            cityscape_path = args.data_dir / "cityscapes" / f"{i.stem}.csv"
            if not cityscape_path.exists():
                continue
            windflow_visualization(cityscape_path, i, args.map_size, args.fig_size, args.export_dir / f"{i.stem}.png")
    else:
        windflow_visualization(cityscape_path, windflow_path, args.map_size, args.fig_size, args.export)

def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic data for predicting local wind fields"
    )
    subprasers = parser.add_subparsers(dest="command")

    genparser = subprasers.add_parser("generate", aliases=["gen"], help="Generate synthetic data")

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
    _ = gensub.add_parser(
        "matlab", help="Convert cityscape csv files to MATLAB format"
    )

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
        "--output_dir",
        type=str,
        default="data/drone_positions",
        help="Directory to save drone positions",
    )

    # Windflow
    windflow_parser = gensub.add_parser("windflow", aliases=["wind"], help="Generate windflow data")

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
    vizsubwind.add_argument(
        "--export", type=str, default=None, help="Export the figure"
    )
    vizsubwind.add_argument(
        "--export-all", default=False, help="Export all figures", action="store_true"
    )
    vizsubwind.add_argument(
        "--export-dir", type=Path, default="data/exportviz", help="Export directory"
    )

    args = parser.parse_args()

    if args.command == "generate" or args.command == "gen":
        if args.generate_what == "cityscapes" or args.generate_what == "city":
            generate_cityscapes(args)
        if args.generate_what == "matlab":
            generate_matlab()
        if args.generate_what == "drone":
            generate_drone_positions(args)
        if args.generate_what == "windflow" or args.generate_what == "wind":
            create_windflows(args)
        if args.generate_what == "lidar":
            generate_lidar_data(args)
        
        if not args.generate_what:
            genparser.print_help()

    elif args.command == "visualize" or args.command == "viz":
        if args.visualize == "cityscape" or args.visualize == "city":
            visualise_cityscape(args)
        elif args.visualize == "windflow" or args.visualize == "wind":
            visualize_windflow(args)
        
        if not args.visualize:
            viz_parser.print_help()
    
    if not args.command:
        parser.print_help()


if __name__ == "__main__":
    main()
