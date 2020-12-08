import os
import random

import pandas as pd
import numpy as np
import nibabel as nib
from torch.utils.data import Dataset, DataLoader


def _collect_paths(hgg_dir_path, lgg_dir_path):
    """
    Collects paths to t1 MRIs
    """
    hgg_paths = []
    for root, dirs, files in os.walk(hgg_dir_path):
        for f in files:
            hgg_paths.append(os.path.join(root, f))
    lgg_paths = []
    for root, dirs, files in os.walk(lgg_dir_path):
        for f in files:
            lgg_paths.append(os.path.join(root, f))
    return hgg_paths, lgg_paths


def _create_csv(split, hgg_paths, lgg_paths):
    df_dict = {"path": [], "label": []}

    for hgg_path in hgg_paths:
        df_dict["path"].append(hgg_path)
        df_dict["label"].append(0)

    for lgg_path in lgg_paths:
        df_dict["path"].append(lgg_path)
        df_dict["label"].append(1)

    df = pd.DataFrame.from_dict(df_dict)
    df = df.sample(frac=1)
    df.to_csv(f"{split}.csv", index=False)


def create_data_split(hgg_dir_path, lgg_dir_path, train_ratio=0.7, valid_ratio=0.2):
    """
    Creates train, valid and test csv files to load data easily    

    train_ratio: % of data to be put in train set
    valid_ratio: % of data to be put in valid set
    test_ratio: % of data to be put in test set

    """
    hgg_paths, lgg_paths = _collect_paths(hgg_dir_path, lgg_dir_path)
    train_hgg_paths = hgg_paths[: int(len(hgg_paths) * train_ratio)]
    valid_hgg_paths = hgg_paths[
        int(len(hgg_paths) * train_ratio) : int(len(hgg_paths) * train_ratio)
        + int(len(hgg_paths) * valid_ratio)
    ]
    test_hgg_paths = hgg_paths[
        int(len(hgg_paths) * train_ratio) + int(len(hgg_paths) * valid_ratio) :
    ]

    train_lgg_paths = lgg_paths[: int(len(lgg_paths) * train_ratio)]
    valid_lgg_paths = lgg_paths[
        int(len(lgg_paths) * train_ratio) : int(len(lgg_paths) * train_ratio)
        + int(len(lgg_paths) * valid_ratio)
    ]
    test_lgg_paths = lgg_paths[
        int(len(lgg_paths) * train_ratio) + int(len(lgg_paths) * valid_ratio) :
    ]

    _create_csv("train", train_hgg_paths, train_lgg_paths)
    _create_csv("valid", valid_hgg_paths, valid_lgg_paths)
    _create_csv("test", test_hgg_paths, test_lgg_paths)


class MRIDataset(Dataset):
    """
    Pytorch Custom dataset
    """

    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        path = self.data["path"][idx]
        label = self.data["label"][idx]
        scan = np.float32(nib.load(path).get_fdata())
        scan = scan[np.newaxis, :, :, :]
        scan = (scan - scan.min()) / (scan.max() - scan.min())
        img_name = os.path.basename(path).split('.')[0]
        payload = {"X": scan, "y": label, "img_name": img_name}
        return payload


if __name__ == "__main__":
    train_dataset = MRIDataset(csv_file="train.csv")
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)

    for i, train_batch in enumerate(train_loader):
        print(train_batch["X"])
        print(train_batch["y"])
        break

