import cv2
import numpy as np
from SDK_USER_PERMISSIONS import *
from time import sleep
import argparse

# setup camera parameters
def setup(func, *args, delay=1, success_code=0, description=''):
    '''run camera setup fucntion until it succeeds'''

    result = func(*args)
    while result != success_code:
        sleep(delay)
        result = func(*args)
    print(f'{description} Set')

def get_center_temp(frame):
    '''get frame center temp in C and F'''

    center_raw = frame[int(frame.shape[0]/2), int(frame.shape[1]/2)]
    center_c = round((center_raw/100) - 273, 1)
    center_f = round(center_c * 9/5 + 32, 1)

    return center_c, center_f

if __name__ == "__main__":

    # Connect to Camera
    myCam = CamAPI.pyClient(manualport="/dev/ttyACM0") #Boson COM port on windows, check device manager

    # User set parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw_output_path',
                        default='output.npy',
                        help='filepath to save raw radiometric output')
    parser.add_argument('--video_output_path',
                        default='output.mp4',
                        help='filepath to save normalized video output')
    args = parser.parse_args()

    # Set Radiometric Parameters
    setup(myCam.bosonSetGainMode,
          FLR_BOSON_GAINMODE_E.FLR_BOSON_HIGH_GAIN,
          description='High Gain Mode')
    setup(myCam.TLinearSetControl,
          FLR_ENABLE_E.FLR_ENABLE,
          description='TLinear Mode')
    setup(myCam.sysctrlSetUsbVideoIR16Mode,
          FLR_SYSCTRL_USBIR16_MODE_E.FLR_SYSCTRL_USBIR16_MODE_TLINEAR,
          description='IR16 Mode')
    setup(myCam.radiometrySetTransmissionWindow,
          100,
          description='Window Transimission')
    setup(myCam.TLinearRefreshLUT,
          FLR_BOSON_GAINMODE_E.FLR_BOSON_HIGH_GAIN,
          description='LUT Refresh')
    setup(myCam.bosonRunFFC,
          description='Flat Field Correction')

    #open camera
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 256)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)     # don't auto-convert to RGB
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('Y','1', '6', ' '))


    # output video settings
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(args.video_output_path, fourcc, 30, (320, 256))

    raw_frames = []
    print('recording...')

    while True:
        ret, frame = cap.read()
        if not ret:
            print('frame grab failed, stopping')
            break

        raw_frames.append(frame)

        # Viewable Window
        min_val, max_val = np.min(frame), np.max(frame)
        frame_8bit = ((frame-min_val)/(max_val-min_val) *255).astype(np.uint8)
        frame_color = cv2.applyColorMap(frame_8bit, cv2.COLORMAP_INFERNO)
        writer.write(frame_color)

        # Center Temp
        temp_c, temp_f = get_center_temp(frame)
        print(f'Center temp: Temp C - {temp_c} | Temp F - {temp_f}')

        cv2.imshow('color', frame_color)
        if cv2.waitKey(1)==ord('q'):
            print(f'recording saved at {args.video_output_path}')
            break

    raw_frames = np.stack(raw_frames, axis=0)
    np.save(args.raw_output_path, raw_frames)

    cap.release()
    writer.release()

    myCam.Close()

