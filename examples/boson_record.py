'''example file to record using flir boson'''   

from heatseek.boson_capture import BosonCapture
import argparse
from time import sleep

if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--raw_output_path',
                        help='filepath to save raw radiometric output')
    parser.add_argument('--video_output_path',
                        help='filepath to save normalized video output')
    parser.add_argument('--bosonsdk_path',
                        help='filepath to boson sdk')
    parser.add_argument('--recording_time',
                        help='time in s to record')
    args = parser.parse_args()
    
    camera = BosonCapture()
    camera.start_recording(raw=args.raw_output_path,
                           norm=args.video_output_path)
    sleep(10)
    camera.stop_recording()
    camera.release_camera()
