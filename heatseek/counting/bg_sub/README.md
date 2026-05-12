# Bat Track Counting Pipeline (Temperature-based Background Subtraction)
## Overview:
This file contains information related to generating crossing tracks using temperature-based background subtraction. This method skips the model training and relies on the pixel values from a radiometric camera image and their converted temperature value.The camera is assumed to be a FLIR Boson, and therefore, the raw temperature values are assumed to be encoded in centikelvins. Given this data, inference can directly be performed on the video to generate detection centroids, contours, and boundaries of bat blobs. From there, it can generate all the information related to the bat tracks.

## Pipeline
1) Perform inference on the frames of the video to generate detection centroids, contours, and boundaries for blobs using `video_inference.py`
    ```bash
    cd path/to/working/dir
    python -m video_inference --vid_path path/to/inference/video --output_folder desired/path/of/output/folder
    ```

2) Convert the detection data into raw tracks using `detections_to_tracks.py`
    ```bash
    cd path/to/working/dir
    python -m detections_to_tracks --output_folder path/to/folder/with/detections   
    #should ideally be the same as output_folder in video_inference
    ```

3) Generate a file and visualization of tracks that cross the midline using `crossing_tracks.py`
    ```bash
    cd path/to/working/dir
    python -m crossing_tracks --raw_tracks_file path/to/raw/tracks/file --crossing_tracks_file desired/path/of/crossing_tracks/file
    ```

    An additional parameter must be added at the end of the second command. If the user wishes to generate a file containing tracks vertically crossing a horizontal midline and its corresponding visualization, they must add the additional parameter `--count_out`. If they instead wish to generate a file containing tracks horizontally crossing a vertical midline, they should use the parameter `--count_across` instead.





