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


    def start_recording(self):
        print('Starting Recording...')

    def stop_recording(self):
        print('Stopping Recording...')

    def release_camera(self):
        self.camera.Close()
        print('Release Camera')

boson_capture = Boson_Capture()
boson_capture.release_camera()

