from capture import Capture
from time import sleep
from importlib import import_module
import threading
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
        if self.isRecording:
            print('Recording in Progress!')
            return

        self.raw_data_fpath = raw
        self.viewable_video_fpath = norm

        self.isRecording = True
        self.recording_thread = threading.Thread(target=self._record_loop, daemon=True)
        self.recording_thread.start()
        print('Staring Recording...')

    def _record_loop(self):

        # video settings
        self.cap = cv2.VideoCapture(self.camera_id, cv2.CAP_V4L2)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('Y', '1', '6', ' '))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(self.viewable_video_fpath, fourcc, 30, (self.width, self.height))


        # raw video setup
        self.raw_mm = np.memmap(self.raw_data_fpath, dtype=np.uint16, mode='w+', shape=(50000, self.height, self.width))
        self.frame_index = 0

        try:
            while self.isRecording:
                ret, frame = self.cap.read()
                if not ret:
                    print('frame grab failed, stopping')
                    break

                # Viewable frame
                min_val, max_val = np.min(frame), np.max(frame)
                #frame_8bit = ((frame - min_val)/(max_val - min_val) * 255).astype(np.uint8)
                frame_8bit = cv2.normalize(frame, None, 0, 255, norm_type=cv2.NORM_MINMAX).astype(np.uint8)
                frame_color = cv2.applyColorMap(frame_8bit, cv2.COLORMAP_INFERNO).astype(np.uint8)
                writer.write(frame_color)

                # Raw Video
                if self.frame_index > self.raw_mm.shape[0]:
                    print('reached preallocated size, stopping')
                    break
                self.raw_mm[self.frame_index] = frame
                self.frame_index += 1
                self.raw_mm.flush()

        finally:
            self._finalize_recording()

    def stop_recording(self):
        if not self.isRecording:
            print('No recording in progress')
            return

        print('Stopping Recording...')
        self.isRecording = False

        if self.recording_thread is not None:
            self.recording_thread.join()
            self.record_thread = None

    def _finalize_recording(self):
        self.isRecording = False

        if self.raw_mm is not None:
            self.raw_mm.flush()
            del self.raw_mm
            self.raw_mm = None

        print('Recording Successfully Completed')
        print('Radiometric data saved: {self.raw_data_fpath}')
        print('Viewable Video: {self.viewable_video_fpath}')

    def release_camera(self):
        self.camera.Close()
        print('Release Camera')


# TEST CODE
boson_capture = Boson_Capture()
boson_capture.start_recording()
sleep(10)
boson_capture.stop_recording()
boson_capture.release_camera()

