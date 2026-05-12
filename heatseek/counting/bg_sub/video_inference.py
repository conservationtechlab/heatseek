import os
import cv2
import logging
import numpy as np
import argparse
import bat_functions 

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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


def get_mask(image, min_val=28000, max_val=32000, threshold=5.0):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    k100 = (gray.astype(np.float32) / 255.0) * (max_val - min_val) + min_val
    temp_celsius = (k100 / 100.0) - 273.15
    background_temp = np.median(temp_celsius)
    mask = (temp_celsius > (background_temp + threshold)).astype(np.uint8) * 255
    return mask

def run_inference(video_file, min_val=28000, max_val=32000,
                  threshold=5.0, process_every_n_frames=1):
    """Process every frame, storing only detection metadata — not images."""
    centers_list = []
    contours_list = []
    sizes_list = []
    rects_list = []

    for record in iter_frames(video_file, process_every_n_frames):
        mask = get_mask(record['image'], min_val, max_val, threshold)
        centers, areas, contours, _, _, rects = bat_functions.get_blob_info(mask)

        centers_list.append(centers)
        sizes_list.append(areas)
        contours_list.append(contours)
        rects_list.append(rects)
        # record['image'] goes out of scope here and gets GC'd

        if len(centers_list) % 1000 == 0:
            logging.info(f'Processed {len(centers_list)} frames.')

    logging.info(f'Processed {len(centers_list)} frames.')
    return centers_list, contours_list, sizes_list, rects_list

def save_detections(centers_list, contours_list, sizes_list, rects_list,
                    output_folder, num_contour_files=15):
    """Save per-frame detections to disk."""
    os.makedirs(output_folder, exist_ok=True)
    file_num = 0
    new_contours = []
    for frame_ind, cs in enumerate(contours_list):
        if frame_ind % int(len(contours_list) / num_contour_files) == 0:
            file_name = f'contours-compressed-{file_num:02d}.npy'
            np.save(os.path.join(output_folder, file_name),
                    np.array(new_contours, dtype=object))
            new_contours = []
            file_num += 1
        new_contours.append([])
        for c in cs:
            cc = np.squeeze(cv2.approxPolyDP(c, 0.1, closed=True))
            new_contours[-1].append(cc)
    file_name = f'contours-compressed-{file_num:02d}.npy'
    np.save(os.path.join(output_folder, file_name),
            np.array(new_contours, dtype=object))
    np.save(os.path.join(output_folder, 'size.npy'), np.array(sizes_list, dtype=object))
    np.save(os.path.join(output_folder, 'rects.npy'), np.array(rects_list, dtype=object))
    np.save(os.path.join(output_folder, 'centers.npy'), np.array(centers_list, dtype=object))
    print(f'Saved detections for {len(centers_list)} frames to {output_folder}')

def main():
    parser = argparse.ArgumentParser(description='Process a thermal video and detect objects.')
    parser.add_argument('--vid_path', help='Path to the input video file.')
    parser.add_argument('--output_folder', help='Path to the output folder.')
    args = parser.parse_args()

    centers_list, contours_list, sizes_list, rects_list = run_inference(args.vid_path)
    save_detections(centers_list, contours_list, sizes_list, rects_list, args.output_folder)

if __name__ == '__main__':
    main()
