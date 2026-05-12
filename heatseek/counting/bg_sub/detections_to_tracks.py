import numpy as np
import os
import glob
import bat_functions as kbf
from multiprocessing import Pool
import argparse

def track(camera_dict):
    """
    Run multi-object tracking on precomputed detection data for a single camera.

    Loads saved contour, center, and size data from a camera's output folder,
    then runs the tracking algorithm (kbf.find_tracks) over a specified frame
    range. The first contours file is skipped to avoid an off-by-one alignment
    issue with the centers/sizes arrays. Results are saved to a .npy file in
    the camera folder.

    If no contours files are found, a warning is printed and tracking is skipped.

    Parameters:
        camera_dict (dict): Configuration dict with the following keys:
            - 'camera_folder' (str): Path to the directory containing
              precomputed detection files (contours, centers, sizes).
            - 'first_frame' (int): Index of the first frame to track.
            - 'max_frame' (int): Index of the last frame to track (inclusive).

    Output:
        Writes a raw tracks file to:
            <camera_folder>/first_frame_<first_frame>_max_val_<max_frame>_raw_tracks.npy

    Returns:
        None
    """
    camera_folder = camera_dict['camera_folder']
    first_frame = camera_dict['first_frame']
    max_frame = camera_dict['max_frame']
    print(f"begun: frames {first_frame} to {max_frame}")

    contours_files = sorted(
        glob.glob(os.path.join(camera_folder, 'contours-compressed-*.npy'))
    )
    if contours_files:
        contours_files = contours_files[1:]
        centers = np.load(os.path.join(camera_folder, 'centers.npy'), allow_pickle=True)
        sizes = np.load(os.path.join(camera_folder, 'size.npy'), allow_pickle=True)
        tracks_file = os.path.join(camera_folder, f'first_frame_{first_frame}_max_val_{max_frame}_raw_tracks.npy')
        raw_tracks = kbf.find_tracks(first_frame, centers, contours_files=contours_files,
                                     sizes_list=sizes, tracks_file=tracks_file,
                                     max_frame=max_frame)
    else:
        print("Missing contour files.")

def build_camera_dicts(output_folder, num_groups=10, fps=60, overlap_seconds=15):
    """
    Divide a video's detection data into overlapping frame groups and return
    a list of tracking job descriptors for any groups not yet processed.

    Loads the precomputed centers array to determine total frame count, then
    splits the full range into num_groups evenly spaced segments. Adjacent
    segments overlap by overlap_seconds * fps frames so that tracks crossing
    a segment boundary can still be recovered. The final segment always runs
    to the end of the video (max_frame=None).

    Segments whose raw tracks output file already exists are skipped, allowing
    interrupted runs to resume without reprocessing completed chunks.

    Parameters:
        output_folder (str): Path to the camera output directory containing
            'centers.npy' and where raw tracks files will be written.
        num_groups (int): Number of frame segments to divide the video into.
            Defaults to 10.
        fps (int): Frames per second of the source video, used to convert
            overlap_seconds to a frame count. Defaults to 60.
        overlap_seconds (int or float): Seconds of overlap between adjacent
            segments. Defaults to 15.

    Returns:
        list[dict]: One dict per unprocessed segment, each containing:
            - 'camera_folder' (str): Same as output_folder.
            - 'first_frame' (int): First frame index for this segment
              (overlap-adjusted for all segments after the first).
            - 'max_frame' (int or None): Exclusive end frame index,
              or None for the final segment.
    """
    centers_file = os.path.join(output_folder, 'centers.npy')
    centers = np.load(centers_file, allow_pickle=True)
    
    overlap_frames = int(fps * overlap_seconds)
    camera_dicts = []

    max_vals = np.linspace(0, len(centers), num_groups, dtype=int)[1:].tolist()
    max_vals[-1] = None
    min_vals = np.linspace(0, len(centers), num_groups, dtype=int)[:-1]
    min_vals[1:] = min_vals[1:] - overlap_frames

    for min_val, max_val in zip(min_vals, max_vals):
        min_val = int(np.max([min_val, 0]))
        camera_dict = {'camera_folder': output_folder,
                       'first_frame': min_val,
                       'max_frame': max_val}
        tracks_basename = f'first_frame_{min_val}_max_val_{max_val}_raw_tracks.npy'
        tracks_file = os.path.join(output_folder, tracks_basename)
        if not os.path.exists(tracks_file):
            camera_dicts.append(camera_dict)
            print(tracks_file)

    return camera_dicts

def combine_overlapping_tracks(output_folder, first_group=0, last_group=None, save=False):
    """
    Merge per-segment raw track files into a single deduplicated track list.

    Loads all 'first_frame_*.npy' track files from output_folder, sorts them
    by their start frame, and stitches them together by retaining only tracks
    that began before the overlap region of each segment. This prevents tracks
    detected in the overlapping frames of adjacent segments from being counted
    twice. For the final segment, all remaining tracks are included regardless
    of start frame.

    Any tracks whose 'track', 'pos_index', or 'size' fields are plain lists
    are converted to numpy arrays before being appended.

    Parameters:
        output_folder (str): Path to the directory containing per-segment
            'first_frame_*.npy' track files.
        first_group (int): Index of the first segment file to include.
            Defaults to 0 (all segments).
        last_group (int or None): Exclusive index of the last segment file to
            include. Defaults to None (all segments through the end).
        save (bool): If True, writes the merged track list to
            <output_folder>/raw_tracks.npy and prints a confirmation message.
            Defaults to False.

    Returns:
        None. Results are printed to stdout and optionally saved to disk.
    """
    track_files = glob.glob(os.path.join(output_folder, 'first_frame*.npy'))
    track_files = sorted(track_files, key=lambda f: int(os.path.basename(f).split('_')[2]))

    track_groups = []
    for file in track_files:
        track_groups.append(np.load(file, allow_pickle=True))

    for track_file in track_files:
        print(os.path.basename(track_file))

    first_overlap_frames = [int(os.path.basename(f).split('_')[2]) for f in track_files[1:]]
    first_overlap_frames.append(None)
    print(first_overlap_frames)

    all_tracks = []

    for group_ind, track_group in enumerate(track_groups[first_group:last_group]):
        if group_ind >= len(track_groups) - 1:
            for track in track_group:
                if type(track['track']) == list:
                    track['track'] = np.stack(track['track'])
                    track['pos_index'] = np.stack(track['pos_index'])
                    if 'size' in track:
                        track['size'] = np.stack(track['size'])
                all_tracks.append(track)
            break

        for track_ind, track in enumerate(track_group):
            if track['first_frame'] < first_overlap_frames[first_group + group_ind]:
                all_tracks.append(track)

    all_tracks_file = os.path.join(output_folder, 'raw_tracks.npy')
    if save:
        np.save(all_tracks_file, all_tracks)
        print(f'Saved {len(all_tracks)} tracks to {all_tracks_file}')

def combine_tracks(output_folder):
    """
    Merge overlapping track segments into a single file, skipping if already done.

    Calls combine_overlapping_tracks() only if raw_tracks.npy does not already
    exist in output_folder, making this safe to call multiple times without
    overwriting a completed merge.

    Parameters:
        output_folder (str): Path to the directory containing per-segment
            'first_frame_*.npy' track files and where 'raw_tracks.npy'
            will be written.

    Returns:
        None
    """
    if not os.path.exists(os.path.join(output_folder, 'raw_tracks.npy')):
        combine_overlapping_tracks(output_folder, save=True)

def run_tracking(output_folder, processes=5):
    """
    Run multi-object tracking across all unprocessed frame segments in parallel.

    Builds the list of pending tracking jobs via build_camera_dicts(), then
    distributes them across a multiprocessing pool. Each worker calls track()
    on one segment dict. Segments whose output file already exists are
    automatically skipped by build_camera_dicts().

    Parameters:
        output_folder (str): Path to the camera output directory containing
            precomputed detection files and where raw track files will be saved.
        processes (int): Number of parallel worker processes. Defaults to 5.

    Returns:
        None
    """
    camera_dicts = build_camera_dicts(output_folder)
    print(f'Tracking {len(camera_dicts)} groups')
    with Pool(processes=processes) as pool:
        pool.map(track, camera_dicts)

def main():
    parser = argparse.ArgumentParser(description='Get centers and contours from detections from model inference')
    parser.add_argument('--output_folder', type=str, help='Path of output folder containing the centers, contours, etc from video inference')
    args = parser.parse_args()
    run_tracking(args.output_folder)
    combine_tracks(args.output_folder)

if __name__ == '__main__':
    main()
