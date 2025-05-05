# heatseek/preprocess.py
import cv2
import numpy as np
from tqdm import tqdm


def reduce_background(in_path: str, out_path: str, motion_thresh: float = 3.0):
    cap = cv2.VideoCapture(in_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

    ret, frame = cap.read()
    prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    for _ in tqdm(range(total), desc="bg-reduce"):
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None,
                                            0.5, 3, 15, 3, 5, 1.2, 0)
        mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        mask3 = np.uint8((mag > motion_thresh) * 255)
        mask3 = cv2.merge([mask3]*3)
        out.write(cv2.bitwise_and(frame, mask3))
        prev_gray = gray
    cap.release()
    out.release()
    print(f"[video_preproc] saved → {out_path}")
