import sys
from pathlib import Path

import pandas as pd
import shutil

def rename(data_dir, out_data_dir, start_count):

    data_dir = Path(data_dir)
    out_data_dir = Path(out_data_dir)
    start_count = int(start_count)

    city_dir = data_dir / "cityscapes"
    demoviz = data_dir / "demoviz"
    drone = data_dir / "drone_positions"
    exportviz = data_dir / "exportviz"
    lidar = data_dir / "lidar"
    trans = data_dir / "transparent"
    wind = data_dir / "windflow"

    out_data_dir.mkdir(parents=True, exist_ok=True)
    (out_data_dir / "cityscapes").mkdir(parents=True, exist_ok=True)
    (out_data_dir / "demoviz").mkdir(parents=True, exist_ok=True)
    (out_data_dir / "drone_positions").mkdir(parents=True, exist_ok=True)
    (out_data_dir / "exportviz").mkdir(parents=True, exist_ok=True)
    (out_data_dir / "lidar").mkdir(parents=True, exist_ok=True)
    (out_data_dir / "transparent").mkdir(parents=True, exist_ok=True)
    (out_data_dir / "windflow").mkdir(parents=True, exist_ok=True)

    for i, file in enumerate(city_dir.iterdir()):
        print(f"{i} -> {start_count + i}")
        shutil.copy(data_dir / "cityscapes" / f"city_{i}.csv", out_data_dir / "cityscapes" / f"city_{start_count + i}.csv")
        shutil.copy(data_dir / "demoviz" / f"city_{i}.png", out_data_dir / "demoviz" / f"city_{start_count + i}.png")
        shutil.copy(data_dir / "drone_positions" / f"city_{i}.csv", out_data_dir / "drone_positions" / f"city_{start_count + i}.csv")
        shutil.copy(data_dir / "exportviz" / f"city_{i}.png", out_data_dir / "exportviz" / f"city_{start_count + i}.png")

        for file in (data_dir / "lidar").glob(f"city_{i}_pos*.npy"):

            shutil.copy(file, out_data_dir / "lidar" / f"city_{start_count + i}_{file.stem.split('_')[-1]}.npy")
    
        shutil.copy(data_dir / "windflow" / f"city_{i}.npy", out_data_dir / "windflow" / f"city_{start_count + i}.npy")
        shutil.copy(data_dir / "transparent" / f"city_{i}_transparent.png", out_data_dir / "transparent" / f"city_{start_count + i}_transparent.png")
    
    # rename in lidar_positions and winds.csv
    df = pd.read_csv(data_dir / "lidar_positions.csv")

    df["city_id"] = df["city_id"].apply(lambda x: x + start_count)
    df.to_csv(out_data_dir / "lidar_positions.csv", index=False)
    
    df = pd.read_csv(data_dir / "winds.csv")
    # cityscape,speed_x,speed_y,pre_time,post_time,out_file
    # data/cityscapes/city_3.csv,2.0,3.0,30,1,data/windflow/city_3.npy
    # change cityscape path and out_file path
    df["cityscape"] = df["cityscape"].apply(lambda x: str(out_data_dir / "cityscapes" / f"city_{int(x.split('_')[-1].split('.')[0]) + start_count}.png"))
    df["out_file"] = df["out_file"].apply(lambda x: str(out_data_dir / "windflow" / f"city_{int(x.split('_')[-1].split('.')[0]) + start_count}.npy"))

    df.to_csv(out_data_dir / "winds.csv", index=False)

    print("Updated csvs.")


def train_test_split(data_dir, out_data_dir, split_ratio):

    data_dir = Path(data_dir)
    out_data_dir = Path(out_data_dir)
    split_ratio = float(split_ratio)

    print("Copying...")
    # copy data_dir to out_data_dir, all its files and subfile
    shutil.copytree(data_dir, out_data_dir / "train")
    shutil.copytree(data_dir, out_data_dir / "val")

    print("Copy Complete...")

    # split_ratio = 0.8
    # read data_dir / "lidar_positions".csv and get unique city_id count
    # get unique city_id count
    df = pd.read_csv(data_dir / "lidar_positions.csv")
    city_ids = df["city_id"].unique()

    train_count = int(len(city_ids) * split_ratio)

    train_city_ids = city_ids[:train_count]
    test_city_ids = city_ids[train_count:]

    # in out_data_dir / "train", update the lidar_positions.csv to only have those corresponding to train-city_ids
    df = pd.read_csv(out_data_dir / "train" / "lidar_positions.csv")
    df = df[df["city_id"].isin(train_city_ids)]
    df.to_csv(out_data_dir / "train" / "lidar_positions.csv", index=False)

    # in out_data_dir / "val", update the lidar_positions.csv to only have those corresponding to test-city_ids
    df = pd.read_csv(out_data_dir / "val" / "lidar_positions.csv")
    df = df[df["city_id"].isin(test_city_ids)]
    df.to_csv(out_data_dir / "val" / "lidar_positions.csv", index=False)

    print("CSVs updated with split")

if __name__ == "__main__":

    if sys.argv[1] == "rename":
        rename(sys.argv[2], sys.argv[3], sys.argv[4])
    elif sys.argv[1] == "split":
        train_test_split(sys.argv[2], sys.argv[3], sys.argv[4])
    else:
        print("Invalid command")
        print("rename <data_dir> <out_data_dir> <start_count>")
        print("split <data_dir> <out_data_dir> <split_ratio>")
        sys.exit(1)