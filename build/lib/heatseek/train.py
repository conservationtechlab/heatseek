# heatseek/train.py
import torch
from ultralytics import YOLO


def train(data_yaml: str,
          weights: str = "yolo11s.pt",
          epochs: int = 400,
          batch: int = 16,
          imgsz: int = 640,
          device_idx: int = 0):
    device = f"cuda:{device_idx}" if torch.cuda.is_available() else "cpu"
    print(f"[train] using device {device}")
    model = YOLO(weights)
    model.to(device)
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
    return model  # ultralytics returns a Results object inside model
