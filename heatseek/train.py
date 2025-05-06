# heatseek/train.py
import torch
from ultralytics import YOLO

def train(data_yaml: str,
          weights: str = "yolo11s.pt",
          epochs: int = 400,
          batch: int = 16,
          imgsz: int = 640,
          device_idx: int = 0):
    """
    Train a YOLO model using ultralytics API.

    Args:
        data_yaml (str): Path to data.yaml file.
        weights (str): Pre-trained weights file path.
        epochs (int): Number of training epochs.
        batch (int): Batch size.
        imgsz (int): Image size.
        device_idx (int): GPU device index; uses CPU if CUDA unavailable.

    Returns:
        model: The trained YOLO model instance.
    """
    device = f"cuda:{device_idx}" if torch.cuda.is_available() else "cpu"
    print(f"[train] using device {device}")
    model = YOLO(weights)
    # ultralytics YOLO handles device internally; explicit .to() not required but can be used
    model.train(
        data=data_yaml,
        epochs=epochs,
        batch=batch,
        imgsz=imgsz,
        device=device,
        scale=0.5,
        mosaic=1.0,
        mixup=0.0,
        copy_paste=0.1,
    )
    return model
