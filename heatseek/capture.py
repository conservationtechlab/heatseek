from abc import ABC, abstractmethod

class Capture(ABC):
    def __init__(self, camera_id=0):
        self.camera_id = camera_id

    @abstractmethod
    def setup(self):
        pass

    @abstractmethod
    def take_image(self):
        pass

    @abstractmethod
    def start_recording(self):
        pass

    @abstractmethod
    def stop_recording(self):
        pass

    @abstractmethod
    def release_camera(self):
        pass
