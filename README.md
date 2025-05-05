# HeatSeek

**HeatSeek** is a command-line tool for downloading thermal datasets, preprocessing video, training YOLO models, and tracking objects in thermal imagery. The current focus of the package is on thermal imagery of bats.

<p align="center">
  <img src="images/OpticalFlow_Example.png" alt="Optical Flow" width="400"/>
  <img src="images/PB_Example.png" alt="PB Example" width="400"/>
</p>

## Installation

First, clone the repository:

```bash
git clone ...
cd heatseek
```

Then install the package:

```bash
pip install . --no-build-isolation
```

Alternatively, for development:

```bash
pip install -e . --no-build-isolation
```

---

## Usage

After installation, use the `heatseek` CLI:

### Download Dataset

```bash
heatseek download \
  --api-key YOUR_API_KEY \
  --workspace your_workspace \
  --project your_project \
  --version 1 \
  --nc 3 \
  --names hot_bat cold_bat other
```

This downloads data from Weights & Biases and writes a `data.yaml`.

---

### Train YOLO Model

```bash
heatseek train \
  --data-yaml path/to/data.yaml \
  --weights yolov8s.pt \
  --epochs 400 \
  --batch 16 \
  --imgsz 640
```

---

### Preprocess Thermal Video

```bash
heatseek preprocess \
  --input raw_video.mp4 \
  --output bg_reduced.mp4 \
  --thresh 3.0
```

Removes background using thermal contrast thresholding.

---

### Track Objects

```bash
heatseek track \
  --input video.mp4 \
  --output tracks.json \
  --weights yolov8.pt
```

Runs YOLO-based detection and object tracking.

---

### Preprocess + Track in One Step

```bash
heatseek pretrack \
  --input raw.mp4 \
  --preprocessed bg_reduced.mp4 \
  --output tracks.json \
  --weights yolov8.pt \
  --thresh 3.0
```

Runs background reduction then tracking in one go.

---

## Dependencies

Ensure you have the following installed (or install via `requirements.txt` if provided):

- `torch`
- `opencv-python`
- `argparse`
- `wandb`
- `ultralytics` (if using YOLOv8)

---

## Project Structure

```
heatseek/
├── cli.py           # Entry point
├── data_utils.py    # Dataset download and YAML writing
├── train.py         # Training logic
├── preprocess.py    # Background reduction
├── detect_track.py  # Detection and tracking
```

---

## 📃 License

MIT or specify your license here.
