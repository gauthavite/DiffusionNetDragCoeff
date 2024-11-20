import os
import json
import pandas as pd

import torch
from torch.utils.data import Dataset
from utils import get_geometry


class CarMeshDataset(Dataset):
    """Car Mesh Dataset"""

    def __init__(self, data_dir, fold, train, n_eig, device):
        """
        data_dir (string): Directory with the csv label map.
        fold (int): fold number.
        train (bool): If True, use the training set, else use the test set.
        n_eig (int): Number of eigenvectors to use for processing.
        device (str): device (:D)
        """

        self.train = train
        self.n_eig = n_eig
        self.device = device
        self.cache_dir = os.path.join(data_dir, "cache")
        self.all_items = os.listdir(self.cache_dir)
        self.split_path = os.path.join(data_dir, "splits.json")

        self.label_map = pd.read_csv(os.path.join(data_dir, "drag_coeffs.csv"))

        # Keep meshes that belong to train or test of this fold
        with open(self.split_path, "r") as f:
            split = json.load(f)

        this_fold_files = (
            split[f"fold_{fold}"]["train"] if train else split[f"fold_{fold}"]["test"]
        )

        # self.all_items = [item.split("_")[0] for item in self.all_items if item.split("_")[0] in this_fold_files]
        self.all_items = [item for item in self.all_items if item in this_fold_files]

    def __len__(self):
        return len(self.all_items)

    def __getitem__(self, idx):
        cached_filepath = os.path.join(self.cache_dir, self.all_items[idx])
        # cached_filepath = os.path.join(self.cache_dir, f"{self.all_items[idx]}_{self.n_eig}.npz")
        verts, faces, frames, mass, L, evals, evecs, gradX, gradY = get_geometry(
            cached_filepath, self.device
        )
        # possible_labels = [
        #     self.all_items[idx], 
        #     self.all_items[idx] + "_flip",
        #     self.all_items[idx] + "_aug",
        #     self.all_items[idx] + "_flip_aug"
        # ]
        # label = self.label_map[
        #     self.label_map["file"].isin(possible_labels)
        # ]["Cd"].values[0]
        label = self.label_map[self.label_map["file"] == '_'.join(self.all_items[idx].split("_")[:-1])]["Cd"].values[0]

        dict = {
            "vertices": verts,
            "faces": faces,
            "frames": frames,
            "vertex_area": mass,
            "label": torch.tensor(label, dtype=torch.float32),
            "L": L,
            "evals": evals,
            "evecs": evecs,
            "gradX": gradX,
            "gradY": gradY,
        }
        return dict
