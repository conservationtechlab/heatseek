'''
Deifnes an abstract base class for thermal camera interfaces.

This modele provides a generic 'Capture' abstract base class that
defines a consistent API for different camera hardware implementations.
Concrete subclasses ('Boson_Capture') can inherit from 'Capture' and
implement all abstract methods to support their specific SDKs.
'''

from abc import ABC, abstractmethod


class Capture(ABC):
    '''Abstract base class for thermal camera interfaces

    Provides standard API for intializing, taking images, and
    managing recordings across different camera types.
    '''

    def __init__(self, camera_id=0):
        '''Initialize capture interface

        This method should handle hardware initalization and
        SDK loading

        Args:
            camera_id (int, opt): Device identifier for camera.
                Defaults to 0.
        '''
        self.camera_id = camera_id

    @abstractmethod
    def setup(self):
        '''Prepare camera for capture

        This method shoudl handle any configuration of camera settings
        that needs to happen before data collection
        '''

    @abstractmethod
    def take_image(self):
        '''Capture and return a single frame

        This method should capture and return a single frame as a np.array

        Returns:
            Any (np.array): captured image
        '''

    @abstractmethod
    def start_recording(self):
        '''Begin video recording

        This method should capture frames continuously and save to disk
        '''

    @abstractmethod
    def stop_recording(self):
        '''Stop ongoing recording

        This method should stop any current video recordings and ensure
        all collected data is saved properly to disk
        '''

    @abstractmethod
    def release_camera(self):
        '''Release camera hardware and close

        This method should handle any releasing and/or shutting down of
        camera hardware
        '''
