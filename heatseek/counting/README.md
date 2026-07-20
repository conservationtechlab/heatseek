# Counting Pipeline
This pipeline consists of 3 stages: 

1. Segmentation
2. Raw Tracking
3. Counting-Line Tracking

## Segmentation

This script is the **first stage** of the counting pipeline. It turns a raw thermal video into a set of per-frame blob detections that every downstream stage (tracking, line-crossing counting) consumes.

### What it does

For each frame of the input video it:

1. **Builds a foreground mask.** Two modes are supported, selected by the `radiometric` flag:
   - **Radiometric** (FLIR Boson): the 8-bit pixel is mapped back to a temperature in °C, and any pixel warmer than `median(background) + threshold` is flagged. Bats read *hotter* (brighter) than the scene.
   - **Non-radiometric** (generic thermal): any pixel *darker* than `median(background) - threshold` is flagged. Bats are assumed to read *darker* than the scene. This is the active path for the config below (`radiometric: False`).
2. **Extracts blobs** from the mask via `bat_functions.get_blob_info` — center, area, contour, and bounding rect per detection.
3. **Previews the mask** on the first processed frame and prompts for confirmation before committing to the full run, so a bad threshold is caught early.

After the pass it:

- **Runs a density sanity check.** `max_bats` computes the per-frame pixel displacement of a bat, `d = (speed × f) / (depth × fps)`, where `f = focal_length / pixel_pitch` is the focal length in pixels. From this it derives a maximum trackable count, `n_max = (width × height) / (16 · d²)`. If the 99th-percentile per-frame detection count exceeds `n_max`, the swarm is too dense to associate reliably and the run aborts. (Further information about this math provided [here](https://drive.google.com/uc?export=download&id=1JrdbtFeGjzSSB6oRCtV8dOb5-wmQ3uFg))
- **Saves detections** to the output folder as `centers.npy`, `size.npy`, `rects.npy`, and chunked `contours-compressed-*.npy`.
- **Writes an overlay video** with contours and centers drawn on the original frames for visual QA.
- **Emits tracker thresholds.** It logs a recommended minimum distance threshold of `ceil(d)` and a maximum of `≤ 2 × min`, which parameterize the downstream tracking stage.

### Why it matters

Detection quality is the ceiling on everything that follows. A missed or split blob here cannot be recovered by the tracker. The density check also acts as a gate. It prevents feeding a swarm into the tracker when the geometry makes correct frame-to-frame association impossible, and it hands the tracker the distance thresholds derived from the same physics.

### Config parameters

| Parameter | Units / Type | Description |
|---|---|---|
| `video_path` | path | Input thermal video to process. |
| `output_folder` | path | Directory where detections and the overlay video are written. |
| `threshold` | intensity or °C | Mask sensitivity. In **radiometric** mode, degrees Celsius above background. In **non-radiometric** mode, intensity units below background. |
| `min_val_radiometric` | cK | Lower bound of the radiometric scale used to map 8-bit pixels back to temperature. **Radiometric mode only.** |
| `max_val_radiometric` | cK | Upper bound of the radiometric scale. **Radiometric mode only.** |
| `radiometric` | bool | Selects the mask function: `True` → FLIR Boson temperature path, `False` → generic intensity path. |
| `speed` | m/s | Bat flight speed; use an upper-bound estimate for the safest density check. |
| `focal_length` | m | Camera focal length. |
| `pixel_pitch` | m/px | Physical spacing between adjacent pixel centers (typically given in µm; convert to meters if necessary). |
| `depth` | m | Distance from the camera to the swarm. |

## Raw Tracking

This script is the **second stage** of the counting pipeline. It consumes the per-frame detections produced by the previous stage (`centers.npy`, `size.npy`, `contours-compressed-*.npy`) and links them into persistent, identity-bearing tracks (one trajectory per bat) that the downstream line-crossing counter can use to count each individual exactly once.

### What it does

1. **Splits the video into overlapping segments.** `build_camera_dicts` divides the full frame range into `num_video_segments` chunks. Adjacent chunks overlap by `segment_overlap_seconds × fps` frames so a bat crossing a segment boundary can still be recovered on both sides. The final segment always runs to the end of the video. Segments whose output already exists are skipped, so interrupted runs resume without reprocessing.
2. **Tracks each segment in parallel.** `run_tracking` distributes the segments across a multiprocessing pool; each worker calls `kbf.find_tracks`, which associates detections frame-to-frame under the distance thresholds below. A track may coast (be interpolated forward without a detection) for up to `max_unseen_frames` frames before it is terminated, so a brief missed detection doesn't fragment one bat into two tracks. The first contours file is dropped to correct an off-by-one alignment against the centers/sizes arrays.
3. **Merges segments into one track list.** `combine_tracks` stitches the per-segment files into a single `raw_tracks.npy`, keeping only tracks that started before each segment's overlap region so a bat seen in two overlapping segments isn't counted twice. This step is idempotent. It won't overwrite an existing merge.
4. **Writes an overlay video** (`rt_overlay_video.mp4`) with a distinctly colored dot and heading arrow per track. Coasting frames (where the tracker guessed the position, i.e. `pos_index` is `nan`) are drawn faded and hollow so real detections and interpolated ones are visually distinguishable.

### Why it matters

Detections on their own have no identity. The same bat in two consecutive frames is just two unrelated blobs. This stage resolves those blobs into single trajectories, which is exactly what counting requires (count the track, not the blob). The coasting mechanism prevents count inflation from momentary detection dropouts, and the overlapping-segment scheme lets long videos be tracked in parallel without losing tracks at chunk boundaries.

### Config parameters

| Parameter | Units / Type | Description |
|---|---|---|
| `video_path` | path | Source thermal video. Used to read `fps` and to render the overlay; detections are read from `output_folder`. |
| `output_folder` | path | Directory holding the detection files from stage 1 and where segment tracks, `raw_tracks.npy`, and the overlay are written. |
| `max_distance_threshold` | px | Maximum association distance. A detection farther than this from a track is not linked to it — effectively a ceiling on per-frame displacement. |
| `min_distance_threshold` | px | Minimum association distance used by the tracking logic; detections closer than this may be merged or filtered. |
| `max_distance_threshold_noise` | px | Distance threshold applied when handling noise / spurious detections during association. |
| `max_unseen_frames` | frames | How many consecutive frames a track may coast without a matching detection before it is terminated. |
| `num_video_segments` | int | Number of overlapping segments the video is split into for parallel tracking. |
| `segment_overlap_seconds` | seconds | Overlap between adjacent segments, converted to frames via `fps`, so boundary-crossing tracks are recoverable. |

## Crossing & Counting

This script is the **final stage** of the pipeline. It takes the trajectories from raw tracking (`raw_tracks.npy`) and turns them into an actual directional count by measuring how tracks cross a counting line through the middle of the frame.

### What it does

1. **Filters and selects crossing tracks.** `save_crossing_tracks_from_raw_tracks` loads the raw tracks, drops anything shorter than two points (`threshold_short_tracks`) to remove spurious fragments, then keeps only the tracks that actually cross the midline (`measure_crossing_bats`). The result is written to `crossing_tracks.npy`.
2. **Counts crossings and their direction.** `find_crossing_frames` walks each track and emits an event at every frame where the track's position changes sign relative to the counting line, i.e. every geometric crossing. Direction is signed: `+1` = increasing along the counting axis (down / right), `-1` = decreasing (up / left).
3. **Visualizes the result.** `visualize_crossing_tracks` plots a sample of crossing tracks colored by net direction alongside a histogram of crossings over time, and prints per-direction totals and the net count.
4. **Renders a counting overlay** (`ct_overlay.mp4`) showing the counting line, per-track dots and heading arrows, a white flash ring on the exact crossing frame, and a live cumulative **OUT / IN / NET** tally burned into each frame.

Counting axis is chosen at runtime by exactly one flag:

- `--count_out` — horizontal counting line at `height / 2`; counts vertical (up/down) crossings. Upward crossings tally as **OUT**, downward as **IN**.
- `--count_across` — vertical counting line at `width / 2`; counts horizontal (left/right) crossings.

### Why it matters

This is where the pipeline produces its deliverable: a number. Tracks by themselves describe motion. This stage reduces them to counted, direction-resolved line crossings. Filtering short tracks first keeps detector noise out of that number, and the signed counting gives net flow (e.g. how many bats left the roost) rather than just a raw total.

### Config parameters

| Parameter | Units / Type | Description |
|---|---|---|
| `input_video_path` | path | Source thermal video. Used to read `fps`, `frame_width`, and `frame_height`, and to render the counting overlay. |
| `raw_tracks_file` | path | The `raw_tracks.npy` produced by the tracking stage. Its parent directory is where `crossing_tracks.npy` is written. |

**Runtime flags** (not in the config; pass exactly one on the command line):

| Flag | Description |
|---|---|
| `--count_out` | Count vertical crossings over a horizontal midline (OUT = upward, IN = downward). |
| `--count_across` | Count horizontal crossings over a vertical midline. |

## Running the Pipeline

1) Perform inference on the frames of the video to generate detection centroids, contours, and boundaries for blobs using `video_inference.py`
    ```bash
    cd path/to/working/dir
    python -m video_inference --config path/to/video_inference_config.yaml
    ```

2) Convert the detection data into raw tracks using `detections_to_tracks.py`
    ```bash
    cd path/to/working/dir
    python -m detections_to_tracks --config path/to/detections_to_tracks_config.yaml
    ```

3) Generate a file and visualization of tracks that cross the midline using `crossing_tracks.py`
    ```bash
    cd path/to/working/dir
    python -m crossing_tracks --config path/to/crossing_tracks_config.yaml
    ```