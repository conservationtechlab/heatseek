from capture import Capture
from time import sleep
from importlib import import_module
import sys
import os
import numpy as np
import cv2


class Boson_Capture(Capture):
    def __init__(self, camera_id=0, sdkpath='~/BosonSDK/SDK_USER_PERMISSIONS'):
        super().__init__(camera_id)
        self.isRecording = False
        self.height = 256
        self.width = 320

        # import Boson SDK
        path = os.path.expanduser(sdkpath)
        assert os.path.exists(path), 'SDK Path does not exist'
        sys.path.append(path)
        self.cam_api = import_module('SDK_USER_PERMISSIONS.ClientFiles_Python.Client_API')
        self.enums = import_module('SDK_USER_PERMISSIONS.ClientFiles_Python.EnumTypes')

        # create camera object & configure radiometric parameters
        port = f'/dev/ttyACM{self.camera_id}'
        self.camera = self.cam_api.pyClient(manualport=port)
        self.setup()

    def setup(self):
        print('Setting up Radiometry...')

        # Set Radiometric Parameters
        self.configure(self.camera.bosonSetGainMode,
                  self.enums.FLR_BOSON_GAINMODE_E.FLR_BOSON_HIGH_GAIN,
                  description='High Gain Mode')
        self.configure(self.camera.TLinearSetControl,
                  self.enums.FLR_ENABLE_E.FLR_ENABLE,
                  description='TLinear Mode')
        self.configure(self.camera.sysctrlSetUsbVideoIR16Mode,
                  self.enums.FLR_SYSCTRL_USBIR16_MODE_E.FLR_SYSCTRL_USBIR16_MODE_TLINEAR,
                  description='IR16 Mode')
        self.configure(self.camera.radiometrySetTransmissionWindow,
                  100,
                  description='Window Transimission')
        self.configure(self.camera.TLinearRefreshLUT,
                  self.enums.FLR_BOSON_GAINMODE_E.FLR_BOSON_HIGH_GAIN,
                  description='LUT Refresh')
        self.configure(self.camera.bosonRunFFC,
                  description='Flat Field Correction')

    def configure(self, func, *args, delay=1, success_code=0, description=''):
        '''run camera paramter configuration function until it succeeds

        Args:
            func: function to update camera parameter
            *args: args for func
            delay (int): delay before trying again
            success_code (int): success code returned from func
            description (str): name of setting

        Returns:
            None
        '''

        result = func(*args)
        while result != success_code:
            sleep(delay)
            result = func(*args)
        print(f'{description} Set')

    def take_image(self):
        print('Taking Image')

    def start_recording(self, raw='output.npy', norm='output.mp4'):
        print('Starting Recording...')

        # video settings
        self.cap = cv2.VideoCapture(self.camera_id, cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('Y', '1', '6', ' '))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(norm, fourcc, 30, (self.width, self.width))

        # raw video setup
        self.raw_mm = np.memmap(raw, dtype=np.uint16, mode='w+', shape=(50000, self.height, self.width))
        self.frame_index = 0

        self.isRecording = True

        while self.isRecording:
            ret, frame = cap.read()
            if not ret:
                print('frame grab failed, stopping')
                break

            # Viewable frame
            min_val, max_val = np.min(frame), np.max(frame)
            frame_8bit = ((frame - min_val)/max_val - min_val) * 255).astype(np.uint8)
            frame_color = cv2.applyColorMap(frame_8bit, cv2.COLORMAP_INFERNO)
            writer.write(frame_color)

            # Raw Video
            if self.frame_index > self.raw_mm.shape[0]:
                print('reached preallocated size, stopping')
                break
            self.raw_mm[self.frame_index] = frame
            self.frame_index += 1
            self.raw_mm.flush()

    def stop_recording(self):
        print('Stopping Recording...')

    def release_camera(self):
        self.camera.Close()
        print('Release Camera')


# TEST CODE
boson_capture = Boson_Capture()
boson_capture.release_camera()

