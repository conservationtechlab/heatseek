import os
import cv2
import logging
import numpy as np
import argparse
import heatseek.counting.bat_functions as bat_functions 
import time
import matplotlib.pyplot as plt
import math
from pathlib import Path
import yaml

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

SHIFT = 4  # 1/16-pixel precision
FACTOR = 1 << SHIFT  # 16

def iter_frames(video_file, process_every_n_frames=1):
    """Yield frames one at a time without storing them all in memory."""
    video_name = os.path.splitext(os.path.basename(video_file))[0]
    cap = cv2.VideoCapture(video_file)
    fps = cap.get(cv2.CAP_PROP_FPS) or 1.0
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % process_every_n_frames == 0:
            yield {
                'frame_idx': frame_count,
                'timestamp': frame_count / fps,
                'image':     frame,
            }
        frame_count += 1

    cap.release()
    logging.info(f'{video_name}: done. {frame_count} total frames.')

# USE ONLY FOR FLIR BOSON
def get_mask_radiometric(image, min_val=28000, max_val=32000, threshold=5.0):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    k100 = (gray.astype(np.float32) / 255.0) * (max_val - min_val) + min_val
    temp_celsius = (k100 / 100.0) - 273.15
    background_temp = np.median(temp_celsius)
    mask = (temp_celsius > (background_temp + threshold)).astype(np.uint8) * 255
    return mask

# USE FOR NON-RADIOMETRIC THERMAL CAMS
def get_mask_nonradiometric(image, threshold=10):
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.astype(np.float32)
    background = np.median(gray)
    mask = (gray < (background - threshold)).astype(np.uint8) * 255
    return mask

def run_inference(video_file, min_val=28000, max_val=32000,
                  threshold=5.0, process_every_n_frames=1, radiometric=False):
    """Process every frame, storing only detection metadata — not images."""
    centers_list = []
    contours_list = []
    sizes_list = []
    rects_list = []
    start_time = time.perf_counter()
    for record in iter_frames(video_file, process_every_n_frames):

        if radiometric:
            mask = get_mask_radiometric(record['image'], min_val, max_val, threshold)
        else:
            mask = get_mask_nonradiometric(record['image'], threshold=threshold)

        if record['frame_idx'] == process_every_n_frames:
            plt.imshow(mask, cmap='gray')
            plt.title(f'Mask preview (threshold={threshold})')
            plt.axis('off')
            plt.show()  

            if input('Mask look okay? [y/n]: ').strip().lower() != 'y':
                raise SystemExit('Aborted: mask unsatisfactory.')

        centers, areas, contours, _, _, rects = bat_functions.get_blob_info(mask)

        centers_list.append(centers)
        sizes_list.append(areas)
        contours_list.append(contours)
        rects_list.append(rects)

        if len(centers_list) % 1000 == 0:
            logging.info(f'Processed {len(centers_list)} frames.')

    end_time = time.perf_counter()
    logging.info(f'Inference completed in {end_time - start_time:.2f} seconds.')
    logging.info(f'Processed {len(centers_list)} frames.')

    return centers_list, contours_list, sizes_list, rects_list

def max_bats(speed, focal_len, pixel_pitch, depth, fps, width, height):
    """
    Find the maximum number of bats that can fit in a frame

    Params:
        - speed [m/s] - bat speed (use upper limit for safest evaluation)
        - focal_len [m] - focal length of camera
        - pixel_pitch [m/px] - physical distance between the centers of each pixel; this is typically measured in micrometers and will need to be converted to m
        - depth [m] - distance between camera and swarm (can be estimated via Google Maps)
        - fps - video's fps
        - width - width of frame
        - height - height of frame
    
    Returns: 
        - n_max - maximum number of bats/frame
    """
    area = width * height
    f = focal_len / pixel_pitch    
    d = (speed * f) / (depth * fps)     
    n_max = area / (16 * d**2)
    return d, n_max

def bats_within_limit(speed, focal_len, pixel_pitch, depth, fps, width, height, centers_list):
    d, n_max_bats = max_bats(speed, focal_len, pixel_pitch, depth, fps, width, height)
    n_bats = np.array([len(c) for c in centers_list])
    n_peak = np.percentile(n_bats, 99)   # or n_bats.max()
    return n_peak <= n_max_bats


def save_detections(centers_list, contours_list, sizes_list, rects_list,
                    output_folder, num_contour_files=15):
    """Save per-frame detections to disk."""
    os.makedirs(output_folder, exist_ok=True)
    file_num = 0
    new_contours = []

    for frame_ind, cs in enumerate(contours_list):
        if frame_ind % int(len(contours_list) / num_contour_files) == 0:
            file_name = f'contours-compressed-{file_num:02d}.npy'
            np.save(os.path.join(output_folder, file_name), np.array(new_contours, dtype=object))
            new_contours = []
            file_num += 1

        new_contours.append([])

        for c in cs:
            cc = np.squeeze(cv2.approxPolyDP(c, 0.1, closed=True))
            new_contours[-1].append(cc)

    file_name = f'contours-compressed-{file_num:02d}.npy'
    np.save(os.path.join(output_folder, file_name), np.array(new_contours, dtype=object))
    np.save(os.path.join(output_folder, 'size.npy'), np.array(sizes_list, dtype=object))
    np.save(os.path.join(output_folder, 'rects.npy'), np.array(rects_list, dtype=object))
    np.save(os.path.join(output_folder, 'centers.npy'), np.array(centers_list, dtype=object))
    logging.info(f'Saved detections for {len(centers_list)} frames to {output_folder}')

def create_overlay_video(input_video_path, output_video_path, centers_list, contours_list):
    cap = cv2.VideoCapture(input_video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    for frame_idx, record in enumerate(iter_frames(input_video_path)):
        frame = record['image']
        centers = centers_list[frame_idx]
        contours = contours_list[frame_idx]

        # Draw contours and centers on the frame
        for contour in contours:
            cv2.drawContours(frame, [contour], -1, (0, 255, 0), 2)
        for center in centers:
            cx = int(round(float(center[0]) * FACTOR))
            cy = int(round(float(center[1]) * FACTOR))
            cv2.circle(frame, (cx, cy), 5 * FACTOR, (0, 0, 255), -1, shift=SHIFT)

        out.write(frame)

    cap.release()
    out.release()
    logging.info(f'Overlay video saved to {output_video_path}')

def main():
    parser = argparse.ArgumentParser(description='Process a thermal video and detect objects.')
    parser.add_argument('--config', help='Path to Video Inference Config File')
    args = parser.parse_args()
    config_path = Path(args.config)

    if not config_path.exists():
        raise FileNotFoundError(f'Config file not found: {config_path}')

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    centers_list, contours_list, sizes_list, rects_list = run_inference(video_file=config['video_path'], 
                                                                        min_val=config['min_val_radiometric'], 
                                                                        max_val=config['max_val_radiometric'], 
                                                                        threshold=config['threshold'], process_every_n_frames=1, 
                                                                        radiometric=config['radiometric'])
    cap = cv2.VideoCapture(config['video_path'])
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    limit_bool = bats_within_limit(speed=config['speed'], focal_len=config['focal_length'], pixel_pitch=config['pixel_pitch'],
                                        depth=config['depth'], fps=fps, width=width, height=height, centers_list=centers_list)
    
    if not limit_bool:
        raise SystemExit('Too many bats in frame to process')
    
    sum = 0

    for i in range(len(centers_list)):
        sum += len(centers_list[i])
    
    logging.info(f'Average detections per frame: {sum / len(centers_list)}')

    save_detections(centers_list, contours_list, sizes_list, rects_list, config['output_folder'])
    create_overlay_video(config['video_path'], os.path.join(config['output_folder'], 'detections_overlay_video.mp4'),
                         centers_list, contours_list)
    min_dist_threshold , _ = max_bats(speed=config['speed'], focal_len=config['focal_length'], pixel_pitch=config['pixel_pitch'],
                                        depth=config['depth'], fps=fps, width=width, height=height)
    logging.info(f'Set Minimum Distance Threshold for known tracks to {math.ceil(min_dist_threshold)} when generating raw tracks')
    logging.info(f'Set Maximum Distance threshold to <= 2 * minimum distance threshold')
    
if __name__ == '__main__':
    main()
