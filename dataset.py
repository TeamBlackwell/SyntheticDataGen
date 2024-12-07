import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


class LocalWindFieldDataset(Dataset):

    def __init__(
        self,
        config,
        point_fol,
        wind_fol,
        root_dir="",
        local=3,
        device=torch.device("cpu"),
    ):

        self.config_data = pd.read_csv(os.path.join(root_dir, config))
        self.lidar_fol = os.path.join(root_dir, point_fol)
        self.local_wind_field_fol = os.path.join(root_dir, wind_fol)
        self.local = local
        self.device = device

    def __len__(self):
        return len(self.config_data)

    def _get_local_winds(self, robo_coords, winds):
        x_min = max(0, robo_coords[0] - self.local)
        x_max = min(winds.shape[0], robo_coords[0] + self.local + 1)
        y_min = max(0, robo_coords[1] - self.local)
        y_max = min(winds.shape[1], robo_coords[1] + self.local + 1)
        z_min = max(0, robo_coords[2] - self.local)
        z_max = min(winds.shape[2], robo_coords[2] + self.local + 1)

        local_winds = np.zeros(
            (2 * self.local + 1, 2 * self.local + 1, 2 * self.local + 1, winds.shape[3])
        )
        local_winds[: x_max - x_min, : y_max - y_min, : z_max - z_min, :] = winds[
            x_min:x_max, y_min:y_max, z_min:z_max, :
        ]
        return torch.tensor(local_winds, dtype=torch.float32, device=self.device)

    def __getitem__(self, idx):

        curr_config = self.config_data.iloc[idx]
        city_id = curr_config["city_id"]
        winds_path = os.path.join(self.local_wind_field_fol, f"city_{city_id}.npy")
        winds = np.load(winds_path)
        robo_coords = torch.tensor(
            [curr_config["x"], curr_config["y"], curr_config["z"]]
        )
        wind_at_robo = torch.tensor(
            winds[curr_config["x"]][curr_config["y"]][curr_config["z"]],
            dtype=torch.float32,
            device=self.device,
        )
        local_winds = self._get_local_winds(robo_coords, winds)
        lidar_path = os.path.join(
            self.lidar_fol, f"city_{city_id}/pointcloud_{(idx%10)+1}.csv"
        )
        lidar = pd.read_csv(lidar_path)
        lidar_tensor = torch.tensor(
            lidar.values, dtype=torch.float32, device=self.device
        )
        lidar_tensor[lidar_tensor != lidar_tensor] = -1  # Replace NaNs with 10000
        return (
            lidar_tensor,
            wind_at_robo,
            local_winds,
        )
