import cv2
import numpy as np
from tqdm import tqdm
import yaml


def reduce_background(in_path: str, out_path: str, yaml_path: str):
    """
    Optical flow background reduction using parameters from a YAML config.

    Args:
        in_path (str): Path to input video file.
        out_path (str): Path to save processed video.
        yaml_path (str): Path to YAML configuration file containing motion_thresh and flow_params.
    """
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)

    motion_thresh = config.get('motion_thresh', 1.0)
    flow_cfg = config.get('flow_params', {})
    pyr_scale = flow_cfg.get('pyr_scale', 0.5)
    levels = flow_cfg.get('levels', 3)
    winsize = flow_cfg.get('winsize', 7)
    iterations = flow_cfg.get('iterations', 3)
    poly_n = flow_cfg.get('poly_n', 5)
    poly_sigma = flow_cfg.get('poly_sigma', 1.2)
    flags = flow_cfg.get('flags', 0)
    cap = cv2.VideoCapture(in_path)
    if not cap.isOpened():
        raise ValueError(f"[ERROR] Cannot open video file: {in_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    ret, frame = cap.read()
    if not ret:
        cap.release()
        raise ValueError(f"[ERROR] Could not read first frame from {in_path}")

    prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    for _ in tqdm(range(total_frames), desc="bg-reduce"):
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, gray, None,
            pyr_scale, levels, winsize,
            iterations, poly_n, poly_sigma, flags
        )
        mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])

        mask = (mag > motion_thresh).astype(np.uint8) * 255
        #kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3)) 
        #mask = cv2.erode(mask, kernel, iterations=1)  
        dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
        mask = (dist > 1.5).astype(np.uint8) * 255
        mask_color = cv2.merge([mask, mask, mask])
        masked_pixels = np.count_nonzero(mask)      
        total_pixels   = mask.shape[0] * mask.shape[1]
        mask_ratio     = masked_pixels / total_pixels
        #print(f"Mask covers {mask_ratio*100:.1f}% of the frame")

        radius = 3.5                                   # px
        area_per_bat = np.pi * (radius**2)          # ~50.3 px²
        est_bats     = masked_pixels / area_per_bat
        #print(f"≈{est_bats:.1f} ")

        # if you want an integer:
        num_bats = int(round(est_bats))
        #print(f"Estimated bat count: {num_bats}")
        fg = cv2.bitwise_and(frame, mask_color)
        out.write(fg)

        prev_gray = gray

    cap.release()
    out.release()
    
    print(f"[video_preproc] saved → {out_path}")