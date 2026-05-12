from bat_functions import threshold_short_tracks, measure_crossing_bats
import numpy as np
import matplotlib.pyplot as plt
import argparse

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
    crossing_tracks_list = measure_crossing_bats(tracks_list, 
                                                 frame_height=frame_height, frame_width=frame_width, count_across=count_across, count_out=count_out)
    if verbose:
        print(f"{len(crossing_tracks_list)} tracks crossing counting line",
              "in observation.")

    np.save(out_file, np.array(crossing_tracks_list, dtype=object))

def visualize_crossing_tracks(crossing_tracks_file, count_out, count_across,
                               frame_height=176, frame_width=176, num_tracks=100):
    if count_out == count_across:
        raise ValueError("Provide exactly one of count_out=True or count_across=True.")
    if count_across and frame_width is None:
        raise ValueError("frame_width required when count_across=True.")

    midline_y = frame_height // 2
    midline_x = frame_width // 2

    crossed_key = 'crossed' if count_out else 'across_crossed'

    crossing_tracks = np.load(crossing_tracks_file, allow_pickle=True)

    print(f'Total crossing tracks: {len(crossing_tracks)}')
    if len(crossing_tracks) == 0:
        print('No crossing tracks found.')
        return

    print(f'Track keys: {crossing_tracks[0].keys()}')
    print(f'Forward crossings: {sum(1 for t in crossing_tracks if t[crossed_key] > 0)}')
    print(f'Backward crossings: {sum(1 for t in crossing_tracks if t[crossed_key] < 0)}')

    num_tracks = min(num_tracks, len(crossing_tracks))
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    ax = axes[0]
    for track_ind in np.linspace(0, len(crossing_tracks), num_tracks, endpoint=False, dtype=int):
        track = crossing_tracks[track_ind]
        color = 'blue' if track[crossed_key] > 0 else 'red'
        ax.plot(track['track'][:, 0], track['track'][:, 1], color=color, alpha=0.3, linewidth=1.5)

    if count_out:
        ax.axhline(y=midline_y, color='green', linewidth=2, linestyle='--', label='midline')
        ax.invert_yaxis()
        direction_labels = 'blue=downward, red=upward'
    else:
        ax.axvline(x=midline_x, color='green', linewidth=2, linestyle='--', label='midline')
        direction_labels = 'blue=rightward, red=leftward'

    ax.set_title(f'Sample of {num_tracks} crossing tracks\n{direction_labels}')
    ax.legend()

    ax = axes[1]
    crossing_frames = [abs(t[crossed_key]) for t in crossing_tracks]
    ax.hist(crossing_frames, bins=100)
    ax.set_xlabel('Frame number')
    ax.set_ylabel('Number of crossings')
    ax.set_title('Bat crossings over time')

    print('Crossing frames:')
    for t in crossing_tracks:
        direction = 'forward' if t[crossed_key] > 0 else 'backward'
        print(f'  frame {abs(t[crossed_key])}: {direction}')

    plt.tight_layout()
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Get bat crossing tracks across the video')
    parser.add_argument('--raw_tracks_file', type=str, help='Path to raw tracks file')
    parser.add_argument('--crossing_tracks_file', type=str, help='Intended path for the crossing tracks file')
    parser.add_argument('--count_across', action='store_true', help='Count horizontal crossings across vertical midline')
    parser.add_argument('--count_out', action='store_true', help='Count vertical crossings across horizontal midline')
    args = parser.parse_args()

    save_crossing_tracks_from_raw_tracks(args.raw_tracks_file, args.crossing_tracks_file, args.count_across, args.count_out)
    visualize_crossing_tracks(args.crossing_tracks_file, count_out=args.count_out, count_across=args.count_across)

if __name__ == '__main__':
    main()
