from heatseek.counting.bat_functions import threshold_short_tracks, measure_crossing_bats
import numpy as np
import matplotlib.pyplot as plt
import argparse
import cv2
from heatseek.counting.detections_to_tracks import make_palette, build_frame_lookup, _fade
from collections import Counter
from pathlib import Path
import os
import yaml

def save_crossing_tracks_from_raw_tracks(file, out_file, count_across, count_out, frame_height=176, frame_width=176, verbose=True):
    """ Save all crossing tracks after preproccesing all tracks in observation.
    
    Parameters:
        file: full path to a numpy file that is a list of track objects
        out_file: full path where list of crossing tracks should be saved
        count_across: count horizontal tracks
        count_out: count vertical tracks
    
    Nothing is returned
    """
    
    raw_track_list = np.load(file, allow_pickle=True)

    if verbose:
        print(f"{len(raw_track_list)} raw tracks in observation.")

    # Get rid of tracks less than two points long
    tracks_list = threshold_short_tracks(raw_track_list, min_length_threshold=2)
    # Get list of tracks that cross the mid line
    crossing_tracks_list = measure_crossing_bats(tracks_list, frame_height=frame_height, frame_width=frame_width, 
                                                 count_across=count_across, count_out=count_out)
    
    if verbose:
        print(f"{len(crossing_tracks_list)} tracks crossing counting line",
              "in observation.")

    np.save(out_file, np.array(crossing_tracks_list, dtype=object))

def visualize_crossing_tracks(crossing_tracks_file, count_out, count_across, frame_height=176, frame_width=176, num_tracks=100):
    if count_out == count_across:
        raise ValueError("Provide exactly one of count_out=True or count_across=True.")
    if count_across and frame_width is None:
        raise ValueError("frame_width required when count_across=True.")

    midline_y = frame_height // 2
    midline_x = frame_width // 2

    crossing_tracks = np.load(crossing_tracks_file, allow_pickle=True)

    print(f'Total crossing tracks: {len(crossing_tracks)}')
    if len(crossing_tracks) == 0:
        print('No crossing tracks found.')
        return

    print(f'Track keys: {crossing_tracks[0].keys()}')

    if count_out:
        line_dim, line_value = 1, midline_y
        direction_labels = 'blue=upward, red=downward'
        pos_label, neg_label = 'down-to-up', 'up-to-down'
    else:
        line_dim, line_value = 0, midline_x
        direction_labels = 'blue=leftward, red=rightward'
        pos_label, neg_label = 'right-to-left', 'left-to-right'

    events = []          # (frame, direction, track_ind); direction +1 = down/right
    for ti, tr in enumerate(crossing_tracks):
        for f, d in find_crossing_frames(tr, line_value, line_dim):
            events.append((f, d, ti))

    net_by_track = Counter()
    for f, d, ti in events:
        net_by_track[ti] += d

    num_tracks = min(num_tracks, len(crossing_tracks))
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    ax = axes[0]
    for track_ind in np.linspace(0, len(crossing_tracks), num_tracks, endpoint=False, dtype=int):
        track = crossing_tracks[track_ind]
        # net > 0 means net downward motion across the line
        color = 'red' if net_by_track[track_ind] > 0 else 'blue'
        ax.plot(track['track'][:, 0], track['track'][:, 1], color=color, alpha=0.3, linewidth=1.5)

    if count_out:
        ax.axhline(y=midline_y, color='green', linewidth=2, linestyle='--', label='midline')
        ax.invert_yaxis()
    else:
        ax.axvline(x=midline_x, color='green', linewidth=2, linestyle='--', label='midline')

    ax.set_title(f'Sample of {num_tracks} crossing tracks\n{direction_labels}')
    ax.legend()

    ax = axes[1]
    crossing_frames = [f for f, _, _ in events]
    ax.hist(crossing_frames, bins=100)
    ax.set_xlabel('Frame number')
    ax.set_ylabel('Number of crossings')
    ax.set_title('Bat crossings over time')

    pos = sum(1 for _, d, _ in events if d > 0)   # down / right
    neg = sum(1 for _, d, _ in events if d < 0)   # up / left

    print('Crossing frames:')
    print(f'{pos_label}: {neg}')
    print(f'{neg_label}: {pos}')
    print(f'total events: {len(events)}   net: {neg - pos}')

    for f, d, ti in sorted(events):
        print(f'  frame {f}: {pos_label if d < 0 else neg_label}  (track {ti})')

    plt.tight_layout()
    plt.show()

def find_crossing_frames(track, line_value, line_dim):
    """Yield (abs_frame, direction). direction = +1 increasing along line_dim
    (down / right), -1 decreasing (up / left)."""
    coord = np.asarray(track['track'], dtype=float)[:, line_dim] - line_value
    first = int(track['first_frame'])
    events = []

    for i in range(1, len(coord)):
        if coord[i - 1] * coord[i] < 0:
            events.append((first + i, 1 if coord[i] > 0 else -1))

    return events


def create_crossing_overlay_video(input_video_path, output_video_path, crossing_tracks,
                                  count_out=True, count_across=False,
                                  frame_height=176, frame_width=176,
                                  lookback=4, min_arrow_len=1.5, arrow_scale=2.5,
                                  dot_radius=4):
    if count_out == count_across:
        raise ValueError("Set exactly one of count_out / count_across.")

    if count_out:
        line_dim, line_value = 1, int(frame_height // 2)   # horizontal line, y
    else:
        line_dim, line_value = 0, int(frame_width // 2)    # vertical line, x

    palette = make_palette(len(crossing_tracks))
    per_frame = build_frame_lookup(crossing_tracks, lookback=lookback)

    crossing_events = set()     # (frame, track_id) — for the flash ring
    pos_per_frame = Counter()   # +1: increasing along line_dim (down / right)
    neg_per_frame = Counter()   # -1: decreasing (up / left)

    for track_id, tr in enumerate(crossing_tracks):
        for f, d in find_crossing_frames(tr, line_value, line_dim):
            crossing_events.add((f, track_id))
            if d > 0:
                pos_per_frame[f] += 1
            else:
                neg_per_frame[f] += 1

    cum_pos = 0
    cum_neg = 0

    cap = cv2.VideoCapture(input_video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 60.0
    out = None
    frame_idx = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        if frame.dtype != np.uint8:
            frame = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        if frame.ndim == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

        frame = np.ascontiguousarray(frame)

        if out is None:
            h, w = frame.shape[:2]

            if (count_out and h != frame_height) or (count_across and w != frame_width):
                print(f"WARNING: video is {w}x{h} but counting used "
                      f"frame_width={frame_width}, frame_height={frame_height}. "
                      f"Line and track coords will be misaligned unless these match.")
                
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_video_path, fourcc, fps, (w, h))

            if not out.isOpened():
                raise RuntimeError(f"VideoWriter failed to open for {output_video_path}")

        if count_out:
            cv2.line(frame, (0, line_value), (w, line_value), (255, 255, 255), 1)
        else:
            cv2.line(frame, (line_value, 0), (line_value, h), (255, 255, 255), 1)

        for (x, y, track_id, is_coasting, dx, dy) in per_frame.get(frame_idx, []):
            color = palette[track_id]
            cx, cy = int(round(x)), int(round(y))

            if is_coasting:
                draw_color = _fade(color)
                cv2.circle(frame, (cx, cy), dot_radius, draw_color, 1)
                arrow_thick = 1
            else:
                draw_color = color
                cv2.circle(frame, (cx, cy), dot_radius, draw_color, -1)
                arrow_thick = 2

            mag = (dx * dx + dy * dy) ** 0.5

            if mag >= min_arrow_len:
                ex = int(round(x + dx * arrow_scale))
                ey = int(round(y + dy * arrow_scale))
                cv2.arrowedLine(frame, (cx, cy), (ex, ey), draw_color, arrow_thick, tipLength=0.35)

            if (frame_idx, track_id) in crossing_events:
                cv2.circle(frame, (cx, cy), dot_radius + 5, (255, 255, 255), 2)  # crossing flash

        cum_pos += pos_per_frame.get(frame_idx, 0)
        cum_neg += neg_per_frame.get(frame_idx, 0)

        lines = [f"OUT: {cum_neg}", f"IN:  {cum_pos}", f"NET: {cum_neg - cum_pos}"]

        for i, txt in enumerate(lines):
            y = 15 + i * 14
            cv2.putText(frame, txt, (5, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,0), 3, cv2.LINE_AA)
            cv2.putText(frame, txt, (5, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1, cv2.LINE_AA)
        
        out.write(frame)
        frame_idx += 1

    cap.release()

    if out is not None:
        out.release()
        
    print(f'Crossing overlay saved to {output_video_path} '
          f'({len(crossing_tracks)} crossing tracks)')

def main():
    parser = argparse.ArgumentParser(description='Get bat crossing tracks across the video')
    parser.add_argument('--config', help='Path to Video Inference Config File')
    parser.add_argument('--count_across', action='store_true', help='Count horizontal crossings across vertical midline')
    parser.add_argument('--count_out', action='store_true', help='Count vertical crossings across horizontal midline')
    args = parser.parse_args()
    config_path = Path(args.config)

    if not config_path.exists():
        raise FileNotFoundError(f'Config file not found: {config_path}')

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    cap = cv2.VideoCapture(config['input_video_path'])
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    frame_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    output_video_file = os.path.join(os.path.dirname(config['input_video_path']), 'ct_overlay.mp4')
    crossing_tracks_file = os.path.join(os.path.dirname(config['raw_tracks_file']), 'crossing_tracks.npy')
    save_crossing_tracks_from_raw_tracks(file=config['raw_tracks_file'], out_file=crossing_tracks_file, 
                                         count_across=args.count_across, count_out=args.count_out, frame_height=frame_height,
                                         frame_width=frame_width)
    visualize_crossing_tracks(crossing_tracks_file=crossing_tracks_file, count_out=args.count_out, count_across=args.count_across,
                              frame_height=frame_height, frame_width=frame_width)
    crossing_tracks = np.load(crossing_tracks_file, allow_pickle=True)
    create_crossing_overlay_video(input_video_path=config['input_video_path'], output_video_path=output_video_file, 
                                  crossing_tracks=crossing_tracks, count_out=args.count_out, count_across=args.count_across,
                                  frame_height=frame_height, frame_width=frame_width)
if __name__ == '__main__':
    main()
