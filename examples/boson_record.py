"""Example script to record thermal imagery using FLIR Boson
Radiometric camera.

This script demonstrates how to use the 'BosonCapture' class from
the 'heatseek' package to record both radiometric data and a normalized
viewable video file from a connected FLIR Boson camera.

Usage:
    python boson_record.py \
    --serial_port path/to/serial/port \
    --video_port path/to/video/port \
    --autonorm_path path/to/output.mp4 \
    --globalnorm_path path/to/output.mp4 \
    --metadata_path path/to/metadata.yaml \
    --bosonsdk_path path/to/BosonSDK/SDK_USER_PERMISSIONS \
    --recording_time 60

If no arguments are provided, default filenames are used and script records
for 10s.
"""

import argparse
from time import time, sleep
import keyboard
from heatseek.boson_capture import BosonCapture


def main():
    """Run exmaple thermal recording using BosonCapture

    Parses arguments, initializes camera, starts recording for specified
    duration, then stops and releases camera.

    Command-line args:
        --serial_port (str, opt): Path to serial port.
            Default is \dev\ttyACM0.
        --video_port (str, opt): Path to video port.
            Default is \dev\video0.
        --autonorm_path (str, opt): Filepath to save
            radiometric output (.mp4)
        --globalnorm_path (str, opt): Filepath to save
            normalized video (.mp4)
        --metadata_path (str, opt): Filepath to save 
            radiometric metadata
        --bosonsdk_path (str, opt) Filepath to boson sdk folder
        --recording_time (int, opt): Duration of recording in seconds.
            Default is 10s.

    Returns:
        None
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--serial_port',
                        type=str,
                        help='path to serial port for radiometric data. Default is /dev/ttyACM0.')
    parser.add_argument('--video_port',
                        type=str,
                        help='path to video port. Default is /dev/video0')
    parser.add_argument('--autonorm_path',
                        type=str,
                        help='filepath to save raw radiometric output')
    parser.add_argument('--globalnorm_path',
                        type=str,
                        help='filepath to save normalized video output')
    parser.add_argument('--metadata_path',
                        type=str,
                        help='filepath to save radiometric metadata')
    parser.add_argument('--bosonsdk_path',
                        type=str,
                        help='filepath to boson sdk')
    parser.add_argument('--recording_time',
                        type=int,
                        help='time in s to record')
    args = parser.parse_args()


    
    camera = BosonCapture(serial_port=args.serial_port,
                          video_port=args.video_port,
                          sdkpath=args.bosonsdk_path)
    camera.start_recording(autonorm=args.autonorm_path,
                           globalnorm=args.globalnorm_path,
                           meta=args.metadata_path)


    max_recording_time = args.recording_time or 10800 # 3hrs default
    start_time = time()
    
    try:
        print('Recording ... Press CTRL+C to stop manually')
        while True:
            elapsed = time() - start_time
            if elapsed >= max_recording_time:
                print('Reached max recording time. Stopping recording')
                break
            sleep(.01)

    except KeyboardInterrupt:
        print('Stopping Recording')

    finally:
    # sleep(args.recording_time or 10)
        camera.stop_recording()
        camera.release_camera()


if __name__ == '__main__':
    main()
