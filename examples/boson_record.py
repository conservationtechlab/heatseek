'''Example script to record thermal imagery using FLIR Boson
Radiometric camera.

This script demonstrates how to use the 'BosonCapture' class from
the 'heatseek' package to record both radiometric data and a normalized
viewable video file from a connected FLIR Boson camera.

Usage:
    python boson_record.py \
    --raw_output_path path/to/output.npy \
    --video_output_path path/to/output.mp4 \
    --bosonsdk_path path/to/BosonSDK/SDK_USER_PERMISSIONS \
    --recording_time 60

If no arguments are provided, default filenames are used and script records
for 10s.
'''

import argparse
from time import sleep
from heatseek.boson_capture import BosonCapture


def main():
    '''Run exmaple thermal recording using BosonCapture

    Parses arguments, initializes camera, starts recording for specified
    duration, then stops and releases camera.

    Command-line args:
        --raw_output_path (str, opt): Filepath to save
            radiometric output (.npy)
        --video_output_path (str, opt): Filepath to save
            normalized video (.mp4)
        --bosonsdk_path (str, opt) Filepath to boson sdk folder
        --recording_time (int, opt): Duration of recording in seconds

    Returns:
        None
    '''

    parser = argparse.ArgumentParser()
    parser.add_argument('--raw_output_path',
                        type=str,
                        help='filepath to save raw radiometric output')
    parser.add_argument('--video_output_path',
                        type=str,
                        help='filepath to save normalized video output')
    parser.add_argument('--bosonsdk_path',
                        type=str,
                        help='filepath to boson sdk')
    parser.add_argument('--recording_time',
                        type=int,
                        help='time in s to record')
    args = parser.parse_args()

    camera = BosonCapture()
    camera.start_recording(raw=args.raw_output_path,
                           norm=args.video_output_path)
    sleep(args.recording_time or 10)
    camera.stop_recording()
    camera.release_camera()


if __name__ == '__main__':
    main()
