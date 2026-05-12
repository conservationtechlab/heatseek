import glob
import os
import cv2
import logging
import argparse

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def pull_out_frames(video_file, process_every_n_frames=1):
    """Extract frames from a video file and optionally save them to disk.
    Parameters:
    - video_file: path to the input video file
    - save_frames: whether to save the extracted frames to disk
    - process_every_n_frames: only process every nth frame to speed up extraction
    """
    video_dir = os.path.dirname(video_file)
    video_name = os.path.splitext(os.path.basename(video_file))[0]
    output_folder = os.path.join(video_dir, video_name, 'frames')
    os.makedirs(output_folder, exist_ok=True)
    logging.info(f'Saving frames to {output_folder}')
    cap = cv2.VideoCapture(video_file)
    frame_count = 0

    while True:         #read every frame until the end of the video
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % process_every_n_frames == 0:
            frame_name = f'{video_name}_{frame_count:05d}.jpg'
            frame_file = os.path.join(output_folder, frame_name)
            cv2.imwrite(frame_file, frame)

        frame_count += 1

    cap.release()
    logging.info(f'{video_name}: done. {frame_count} total frames.')



def main():
    parser = argparse.ArgumentParser(description='Extract frames from videos')
    parser.add_argument('--clips_dir', help='Parent folder containing video files')
    parser.add_argument('--process_every_n_frames', type=int, default=1, help='Process every nth frame')
    args = parser.parse_args()

    video_files = []
    video_files.extend(glob.glob(os.path.join(args.clips_dir, '**', '*.mp4'), recursive=True))
    video_files.sort()

    for video_file in video_files:
        pull_out_frames(video_file, process_every_n_frames=args.process_every_n_frames)

if __name__ == '__main__':
    main()
