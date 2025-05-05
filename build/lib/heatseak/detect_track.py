# heatseek/detect_track.py
import cv2
import torch
from ultralytics import YOLO


def detect_and_track(in_path: str,
                     out_path: str,
                     weights: str,
                     tracker_cfg: str = "botsort.yaml",
                     device_idx: int = 0):
    device = f"cuda:{device_idx}" if torch.cuda.is_available() else "cpu"
    model = YOLO(weights).to(device)

    cap = cv2.VideoCapture(in_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

    seen_ids = set()
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.track(frame, persist=True, tracker=tracker_cfg)
        if hasattr(results[0], "boxes") and results[0].boxes.id is not None:
            for obj_id in results[0].boxes.id.cpu().tolist():
                seen_ids.add(int(obj_id))

        anno = results[0].plot()
        out.write(anno)

    cap.release()
    out.release()
    print(f"[detect_track] done. unique IDs seen: {len(seen_ids)}")
