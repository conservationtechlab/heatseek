# heatseek/data_utils.py
import os
import yaml
from roboflow import Roboflow


def download_dataset(api_key: str,
                     workspace: str,
                     project: str,
                     version: int,
                     save_format: str = "yolov11") -> str:
    """Downloads from Roboflow and returns the local dataset folder path."""
    rf = Roboflow(api_key=api_key)
    proj = rf.workspace(workspace).project(project)
    ds = proj.version(version).download(save_format)
    return ds.location


def write_data_yaml(dataset_dir: str,
                    nc: int,
                    names: list[str]):
    """Overwrites <dataset_dir>/data.yaml with our train/val/names spec."""
    data = {
        "train": os.path.join(dataset_dir, "train", "images"),
        "val":   os.path.join(dataset_dir, "valid", "images"),
        "nc":    nc,
        "names": names
    }
    path = os.path.join(dataset_dir, "data.yaml")
    with open(path, "w") as f:
        yaml.dump(data, f)
    return path
