#!/usr/bin/env python3
# heatseek/cli.py

import argparse
import requests
from .data_utils import download_dataset, write_data_yaml
from .train import train
from .preprocess import reduce_background
from .detect_track import detect_and_track

def main():
    parser = argparse.ArgumentParser(prog="heatseek")
    subs = parser.add_subparsers(dest="cmd", required=True)

    # getweights
    getw = subs.add_parser("getweights", help="Download pretrained YOLO weights")
    getw.add_argument(
        "--url",
        default="https://sandiegozoo.box.com/shared/static/0edcehha4y8yli9h0bzettitftbd1jdl.pt",
        help="URL to download weights from",
    )
    getw.add_argument(
        "--output", default="yolo_model.pt", help="Where to save the downloaded weights"
    )

    # download (Roboflow)
    dl = subs.add_parser("download", help="Fetch Roboflow dataset and write data.yaml")
    dl.add_argument("--api-key", required=True, help="Roboflow API key")
    dl.add_argument("--workspace", required=True, help="Roboflow workspace name")
    dl.add_argument("--project", required=True, help="Roboflow project name")
    dl.add_argument("--version", type=int, required=True, help="Roboflow version number")
    dl.add_argument("--nc", type=int, default=3, help="Number of classes")
    dl.add_argument(
        "--names",
        nargs="+",
        default=["hot_bat", "cold_bat", "other"],
        help="List of class names",
    )

    # train
    tr = subs.add_parser("train", help="Train YOLO on your dataset")
    tr.add_argument("--data-yaml", required=True, help="Path to your data.yaml")
    tr.add_argument("--weights", default="yolo11s.pt", help="Pretrained weights file")
    tr.add_argument("--epochs", type=int, default=400, help="Number of epochs")
    tr.add_argument("--batch", type=int, default=16, help="Batch size")
    tr.add_argument("--imgsz", type=int, default=640, help="Image size (pixels)")
    tr.add_argument(
        "--device", type=int, default=0, help="CUDA device index (uses CPU if none)"
    )

    # preprocess
    pre = subs.add_parser("preprocess", help="Reduce background in a video")
    pre.add_argument("--input", required=True, help="Input video path")
    pre.add_argument("--output", required=True, help="Output video path")
    pre.add_argument(
        "--config",
        default="heatseek/config/preproc_config.yaml",
        help="YAML config for optical‐flow thresholds",
    )

    # track
    track = subs.add_parser("track", help="Detect + track objects in a video")
    track.add_argument("--input", required=True, help="Input video path")
    track.add_argument("--output", required=True, help="Output path for results")
    track.add_argument("--weights", required=True, help="Weights for detection")

    # pretrack
    pretrack = subs.add_parser(
        "pretrack", help="Preprocess (bg-reduce) then detect+track"
    )
    pretrack.add_argument("--input", required=True, help="Input video path")
    pretrack.add_argument(
        "--preprocessed", required=True, help="Where to dump preprocessed video"
    )
    pretrack.add_argument("--output", required=True, help="Output path for tracking")
    pretrack.add_argument("--weights", required=True, help="Weights for detection")
    pretrack.add_argument(
        "--thresh", type=float, default=3.0, help="Motion threshold for pretracking"
    )

    args = parser.parse_args()

    if args.cmd == "getweights":
        print(f"Downloading weights from {args.url}…")
        resp = requests.get(args.url)
        resp.raise_for_status()
        with open(args.output, "wb") as f:
            f.write(resp.content)
        print(f"Weights saved to {args.output}")

    elif args.cmd == "download":
        ds_dir = download_dataset(
            args.api_key, args.workspace, args.project, args.version
        )
        yaml_path = write_data_yaml(ds_dir, args.nc, args.names)
        print("→ data.yaml written at", yaml_path)

    elif args.cmd == "train":
        train(
            data_yaml=args.data_yaml,
            weights=args.weights,
            epochs=args.epochs,
            batch=args.batch,
            imgsz=args.imgsz,
            device_idx=args.device,
        )

    elif args.cmd == "preprocess":
        reduce_background(args.input, args.output, args.config)

    elif args.cmd == "track":
        detect_and_track(args.input, args.output, args.weights)

    elif args.cmd == "pretrack":
        reduce_background(args.input, args.preprocessed, args.config)
        detect_and_track(args.preprocessed, args.output, args.weights)

if __name__ == "__main__":
    main()
