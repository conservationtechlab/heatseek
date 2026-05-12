from torch.utils.data import IterableDataset
import cv2

class BatIterableDataset(IterableDataset):
    """ An iterable PyTorch dataset that streams grayscale frames from one or
        more video files for bat detection inference.

        Frames are read sequentially using OpenCV, converted to grayscale, and
        optionally passed through an augmentation pipeline. Videos are processed
        one at a time in the order provided; when a video is exhausted, the next
        one starts automatically.

        Consecutive read failures are tolerated up to max_bad_reads before the
        current video is closed and iteration ends.

        Attributes:
            video_files (list[str]): Ordered list of video file paths.
            augmentor (MaskCompose or None): Optional transform applied to each frame.
            max_bad_reads (int): Max consecutive failed reads before closing a video.
            total_frames_read (int): Running count of successfully read frames.
            total_bad_reads (int): Running count of failed frame reads.
            video_number (int): Index of the currently open video in video_files.
            more_frames (bool): Whether the current video still has frames.
    """

    def __init__(self, video_files, augmentor=None, max_bad_reads=300):
        """
        Initialize the dataset and open the first video file.

        Parameters:
            video_files (list[str]): Ordered list of paths to video files.
                The first file is opened immediately on construction.
            augmentor (MaskCompose, optional): Transform pipeline applied to
                each frame dict before yielding. Defaults to None.
            max_bad_reads (int): Maximum consecutive failed cv2 reads before
                the current video is considered exhausted. Defaults to 300.

        Raises:
            AssertionError: If the first video file cannot be opened by OpenCV.
        """
        self.vid_cap = cv2.VideoCapture(video_files[0])
        self.video_files = video_files
        assert self.vid_cap.isOpened()
        self.more_frames = True
        self.max_bad_reads = max_bad_reads
        self.total_frames_read = 0
        self.total_bad_reads = 0
        self.augmentor = augmentor
        self.video_number = 0

    def more_videos(self):
        """
        Check whether there are additional unprocessed videos in the queue.

        Returns:
            bool: True if video_number is within bounds of video_files.
        """
        return self.video_number < len(self.video_files)

    def start_next_video(self):
        """
        Release the current video capture and open the next video in the queue.

        Increments video_number, releases any open capture, and opens a new
        cv2.VideoCapture for the next file. Does nothing if all videos have
        already been processed. Prints a status message and frame read info
        when a new video starts.
        """
        if self.vid_cap.isOpened():
            self.vid_cap.release()
        self.video_number += 1
        if self.video_number < len(self.video_files):
            print('starting new video')
            print(self.get_read_frame_info())
            self.vid_cap = cv2.VideoCapture(self.video_files[self.video_number])

    def video_generator(self):
        """
        Generator that yields preprocessed frames across all video files.

        For each video, reads frames sequentially. On a successful read the
        frame is converted from BGR to grayscale, cropped by 2 pixels on each
        edge, wrapped in a dict as {'image': frame}, and optionally transformed
        by the augmentor before being yielded.

        Consecutive failed reads are counted; if max_bad_reads is reached the
        video is released and iteration moves on. Transitions between videos
        are handled automatically via start_next_video().

        Yields:
            dict or augmentor output: {'image': np.ndarray} if no augmentor,
                otherwise the transformed output of augmentor({'image': frame}).
        """

        while(self.vid_cap.isOpened() or self.more_videos()):
            if not self.vid_cap.isOpened():
                self.start_next_video()
            good_read = False
            num_bad_reads = 0
            while (not good_read and (num_bad_reads < self.max_bad_reads)):
                grabbed, frame = self.vid_cap.read()
                if grabbed:
                    good_read = True
                    self.total_frames_read += 1
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    frame = {'image': frame[2:-2, 2:-2]}
                    if self.augmentor:
                        frame = self.augmentor(frame)
                    yield frame
                else:
                    num_bad_reads += 1
                    self.total_bad_reads += 1
            if not good_read:
                self.vid_cap.release()
                print("video capture closed")

    def __iter__(self):
        """
        Return the frame generator, making this dataset compatible with
        PyTorch DataLoader.

        Returns:
            generator: The video_generator() iterator.
        """
        return self.video_generator()

    def __del__(self):
        """
        Release the OpenCV video capture on object destruction to free
        file handles and decoder resources.
        """
        if self.vid_cap.isOpened():
            self.vid_cap.release()

    def is_more_frames(self):
        """
        Check whether the current video capture is still open and readable.

        Returns:
            bool: True if vid_cap is currently open.
        """
        return self.vid_cap.isOpened()

    def get_read_frame_info(self):
        """
        Print a summary of total frames read and total bad reads so far.

        Example output:
            142 frames have been read with 3 bad reads
        """
        print('{} frames have been read with {} bad reads'.format(
            self.total_frames_read, self.total_bad_reads))
        