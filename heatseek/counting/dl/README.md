# Bat Track Counting Pipeline (Deep Learning)
## Overview:
This file contains information related to generating crossing tracks using deep learning. This is the method that was originally implemented by Koger et al, but modified to fit our platform and data. Deep learning is useful when the data contains noisy objects (e.g. small bats in an RGB image).

## Pipeline:
1) In order to train a model, we must create a training set consisting of images and segmentation masks as the model that is used in this pipeline is a UNet model. The steps for this stage are:

    1) `crop_vids.py` - Cut the long videos into short clips of motion using the command:
        ```bash
        cd path/to/working/dir
        python -m crop_vids --input path/to/input/file --output_dir path/to/desired/output/folder 
        ```
        There are additional parameters that may be set. Please refer to file for fine-tuning.

        The clips must go through human validation to ensure there are no false positives for motion. Once this is done, they must be arranged in a directory structure that looks like this:

        ```
        clips 
        |---- vid1_clips
        |---------------- clip1.mp4
        |---------------- clip2.mp4
                                .
                                .
                                .
        |----- vid2_clips
        |---------------- clip1.mp4
        |---------------- clip2.mp4
                                .
                                .
                                .
                    .
                    .
                    .
    
    2) `extract_frames.py` - Extract frames from the clips
        ```bash
        cd path/to/working/dir
        python -m extract_frames --clips_dir path/to/directory/containing/clips
        ```
        There are additional parameters that may be set. Please refer to file for fine-tuning.
        
    3) `segment_frames.py` - Get masks for the frames using temperature-based background segmentation (assumes availability of globally normalized video with minimum and maximum pixel values corresponding to temperature from a radiometric thermal camera)
        ```bash
        cd path/to/working/dir
        python -m segment_frames --clips_dir path/to/directory/containing/clips 
        ```
        There are additional parameters that may be set. Please refer to file for fine-tuning.

2) Train the model on the training img, mask pairs using `trainer.py`
    ```bash
    cd path/to/working/dir
    python -m trainer --clips_dir path/to/directory/containing/clips
    ```
    There are additional parameters that may be set. Please refer to file for fine-tuning.

3) Perform inference on the frames of the video to generate detection centroids, contours, and boundaries for blobs using `video_inference.py`
    ```bash
    cd path/to/working/dir
    python -m video_inference --vid_path path/to/inference/video --output_folder desired/path/of/output/folder --model_filepath path/to/model.tar --clips_dir path/to/directory/containing/clips
    ```

4) Convert the detection data into raw tracks using `detections_to_tracks.py`
    ```bash
    cd path/to/working/dir
    python -m detections_to_tracks --output_folder path/to/folder/with/detections   
    #should ideally be the same as output_folder in video_inference
    ```

5) Generate a file and visualization of tracks that cross the midline using `crossing_tracks.py`
    ```bash
    cd path/to/working/dir
    python -m crossing_tracks --raw_tracks_file path/to/raw/tracks/file --crossing_tracks_file desired/path/of/crossing_tracks/file
    ```

    An additional parameter must be added at the end of the second command. If the user wishes to generate a file containing tracks vertically crossing a horizontal midline and its corresponding visualization, they must add the additional parameter `--count_out`. If they instead wish to generate a file containing tracks horizontally crossing a vertical midline, they should use the parameter `--count_across` instead.



    