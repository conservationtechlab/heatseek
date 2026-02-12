"""Implementation of Capture abstract class for FLIR Boson
Radiometric Thermal camera

This module provides the 'Boson_Capture' class, which implements the standard
camera interface defined by the 'Capture' abstract base class. It handles
initializing the BosonSDK, capturing single frames, recording radiometric
frames, and creating a viewable mp4 video with a color map.
"""

import threading
import sys
import os
from time import sleep
import subprocess
from importlib import import_module
from datetime import datetime
import yaml
import numpy as np
import cv2
from heatseek.capture import Capture


class BosonCapture(Capture):
    """Camera interface for FLIR Boson Radiometric Thermal camera

    Implements 'Capture' abstract base class for Boson Radiometric thermal
    camera.

    Attributes:
        serial_port (str): path to serial port for radiometric data
        video_port (str): path to video port for noramlized video data
        cam_api (module): imported FLIR Boson SDK API modulde
        enums (module): imported FLIR Boson Enum definitions
        camera (pyClient): camera object created using boson SDK
        isRecording (bool): True when recording session in progress
        raw_data_fpath (str): file path to save raw radiometric data
        viewable_video_fpath (str): file path to save normalized thermal video
        raw_mm (np.memmap): memory mapped array for storing raw
            radiometric frames
        recording_thread (threading.Thread): thread running recording loop
        height (int): frame pixel height
        width (int): frame pixel width
    """

    def __init__(self, serial_port=None, video_port=None, sdkpath=None):
        """Initializes boson camera interface

        Loads Boson SDK, connects to camera and configures radiometric
        parameters

        Args:
            serial_port (str, opt): path to serial port.
                Defaults to \dev\ttyACM0.
            video_port (str, opt): path to video port.
                Defaults to \dev\video0.
            skdpath (str, opt): path to FLIR Boson SDK folder.
                Defaults to ~/BosonSDK/SDK_USER_PERMISSIONS
        """

        self.serial_port = serial_port or '/dev/ttyACM0'
        self.video_port = video_port or '/dev/video0'
        self.sdkpath = sdkpath or '~/BosonSDK/SDK_USER_PERMISSIONS'
        self.recording = False
        self.height = 256
        self.width = 320
        self.GLOBAL_MIN = 28000
        self.GLOBAL_MAX = 32000
        self.n_frames = 0
        self.autonorm_fpath = None
        self.globalnorm_fpath = None
        self.metadata_fpath = None
        self.recording_thread = None
        self.raw_mm = None

        # import Boson SDK
        path = os.path.expanduser(self.sdkpath)
        assert os.path.exists(path), 'SDK Path does not exist'
        sys.path.append(path)
        self.cam_api = import_module(
            'SDK_USER_PERMISSIONS.ClientFiles_Python.Client_API'
            )
        self.enums = import_module(
            'SDK_USER_PERMISSIONS.ClientFiles_Python.EnumTypes'
            )

        # create camera object & configure radiometric parameters
        self.camera = self.cam_api.pyClient(manualport=self.serial_port)
        self.setup()

    def setup(self):
        """Configure radiometric parameters for capture

        Sets High Gain Mode, enables TLinear mode, sets IR16 mode,
        configures window transmissivity, refreshes LUT, & performs FFC
        """

        print('Setting up Radiometry...')

        # Set Radiometric Parameters
        self._configure(self.camera.bosonSetGainMode,
                        self.enums.FLR_BOSON_GAINMODE_E.FLR_BOSON_HIGH_GAIN,
                        description='High Gain Mode')
        self._configure(self.camera.TLinearSetControl,
                        self.enums.FLR_ENABLE_E.FLR_ENABLE,
                        description='TLinear Mode')
        self._configure(self.camera.sysctrlSetUsbVideoIR16Mode,
                        self.enums.FLR_SYSCTRL_USBIR16_MODE_E.FLR_SYSCTRL_USBIR16_MODE_TLINEAR,
                        description='IR16 Mode')
        self._configure(self.camera.radiometrySetTransmissionWindow,
                        100,
                        description='Window Transimission')
        self._configure(self.camera.TLinearRefreshLUT,
                        self.enums.FLR_BOSON_GAINMODE_E.FLR_BOSON_HIGH_GAIN,
                        description='LUT Refresh')
        self._configure(self.camera.bosonRunFFC,
                        description='Flat Field Correction')

    def _configure(self, func, *args, delay=1, success_code=0, description=''):
        """run camera paramter configuration function until it succeeds

        Args:
            func: function to update camera parameter
            *args: args for func
            delay (int): delay before trying again
            success_code (int): success code returned from func
            description (str): name of setting
        """

        print(f'\nSetting {description}')
        sleep(0.5)
        result = func(*args)
        while result != success_code:
            sleep(delay)
            result = func(*args)
        print('Success!')

    def take_image(self):
        """Capture single image frame from camera.

        Currently placeholder method - will be implemented to return ndarray
        """

        print('Taking Image- PLACEHOLDER')

    def start_recording(self, autonorm=None, globalnorm=None, meta=None):
        """Begin thread for continuous recording

        Args:
            autonorm (str, opt): filepath to save frame by frame
                noramlized video. Defaults to autonorm_{timestamp}.mp4
            globalnorm (str, opt): filepath to save globally
                normalized mp4 video. Defaults to globalnorm_{timestamp}.mp4
        """

        if self.recording:
            print('Recording in Progress!')
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        self.autonorm_fpath = autonorm or f'autonorm_{timestamp}.mp4'
        self.globalnorm_fpath = globalnorm or f'globalnorm_{timestamp}.mkv'
        self.metadata_fpath = meta or f'metadata_{timestamp}.yaml'

        self.recording = True
        self.recording_thread = threading.Thread(
            target=self._record_loop, daemon=True
            )
        self.recording_thread.start()
        print('Staring Recording...')

    def _get_center_temp(self, frame):
        """Internal method: returns pixel value of frame center
        """

        center = frame[int(self.height/2), int(self.width/2)]
        return center

    def _record_loop(self):
        """Internal method: recording loop to continuously capture frames

        Saves raw radiometric frames and writes a viewable mp4 video.
        Stops recording when self.recording is False or preallocated
        memory is full
        """

        # MP4 Setup
        cap = cv2.VideoCapture(self.video_port, cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)
        cap.set(cv2.CAP_PROP_FOURCC,
                cv2.VideoWriter_fourcc('Y', '1', '6', ' '))
        mpv4_fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        autonorm_writer = cv2.VideoWriter(self.autonorm_fpath,
                                          mpv4_fourcc,
                                          60,
                                          (self.width, self.height))

        # MKV Setup
        proc = subprocess.Popen([
            "ffmpeg", "-y",
            "-f", "rawvideo",
            "-pixel_format", "gray",
            "-video_size", f"{self.width}x{self.height}",
            "-framerate", "60",
            "-i", "-",
            "-c:v", "libx264",
            self.globalnorm_fpath
        ], stdin=subprocess.PIPE)

        # raw video setup
        # self.raw_mm = np.memmap('raw_test.raw',
                                # dtype=np.uint16,
                                # mode='w+',
                                # shape=(600, self.height, self.width))

        self.n_frames = 0

        try:
            while self.recording:
                ret, frame = cap.read()
                if not ret:
                    print('Frame grab failed. Stopping Recording')
                    break

                # Autonorm Frame (MP4)
                frame_8bit_autonorm = cv2.normalize(frame, None, 0, 255,
                    norm_type=cv2.NORM_MINMAX).astype(np.uint8)
                frame_color = cv2.applyColorMap(frame_8bit_autonorm,
                    cv2.COLORMAP_INFERNO).astype(np.uint8)
                autonorm_writer.write(frame_color)

                # Globalnorm Frame (MKV)
                frame_32bit = frame.astype(np.float32)
                frame_8bit_globalnorm = (255.0 * (frame_32bit-self.GLOBAL_MIN)/(self.GLOBAL_MAX-self.GLOBAL_MIN)).astype(np.uint8)
                proc.stdin.write(frame_8bit_globalnorm.tobytes())

                # Raw Video
                # if frame_index < self.raw_mm.shape[0]:
                    # if self.n_frames > self.raw_mm.shape[0]:
                        # print('Reached Preallocated Size, Stopping Recording')
                        # break
                    # self.raw_mm[self.n_frames] = frame
                    # frame_index += 1
                    # self.n_frames += 1
                    # self.raw_mm.flush()

                self.n_frames += 1

        finally:
            cap.release()
            autonorm_writer.release()

            self._finalize_recording()

            proc.stdin.close()
            proc.wait()

    def stop_recording(self):
        """Stop ongoing recording session.

        Sets isRecording flag to False, waits for recording thread to finish,
        and cleans up resources.
        """

        if not self.recording:
            print('No recording in progress')
            return

        print('Stopping Recording...')
        self.recording = False

        if self.recording_thread is not None:
            self.recording_thread.join()
            self.recording_thread = None

    def _finalize_recording(self):
        """Finalize recording.

        Flushes and deletes memory-mapped raw frame array, closes video writer,
        and prints file locations.
        """

        self.recording = False

        # if self.raw_mm is not None:
            # self.raw_mm.flush()
            # del self.raw_mm
            # self.raw_mm = None

        self._write_radiometric_metadata()

        print('Recording Successfully Completed')
        print(f'Viewable Video: {self.autonorm_fpath}')
        print(f'Globally normalized Video: {self.globalnorm_fpath}')
        print(f'Radiometric Metadata: {self.metadata_fpath}')

    def _write_radiometric_metadata(self):
        """Saves out all radiometric video metadata
        """

        meta = {
            'width': self.width,
            'height': self.height,
            'n_frames': self.n_frames,
            'min': self.GLOBAL_MIN,
            'max': self.GLOBAL_MAX,
            'dtype': 'uint16',
        }

        with open(self.metadata_fpath, "w", encoding="utf-8") as f:
            yaml.safe_dump(meta, f, sort_keys=False)

    def release_camera(self):
        """Release camera resources.

        Closes camera connection via boson SDK
        """

        self.camera.Close()
        print('Release Camera')
