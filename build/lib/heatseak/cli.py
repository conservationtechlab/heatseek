# heatseek/cli.py
import argparse
from .data_utils import download_dataset, write_data_yaml
from .train import train
from .preprocess import reduce_background
from .detect_track import detect_and_track


def main():
    p = argparse.ArgumentParser(prog="heatseek")
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("download")
    d.add_argument("--api-key", required=True)
    d.add_argument("--workspace", required=True)
    d.add_argument("--project", required=True)
    d.add_argument("--version", type=int, required=True)
    d.add_argument("--nc", type=int, default=3)
    d.add_argument("--names", nargs="+",
                   default=["hot_bat", "cold_bat", "other"])

    t = sub.add_parser("train")
    t.add_argument("--data-yaml", required=True)
    t.add_argument("--weights", default="yolo11s.pt")
    t.add_argument("--epochs", type=int, default=400)
    t.add_argument("--batch", type=int, default=16)
    t.add_argument("--imgsz", type=int, default=640)

    v = sub.add_parser("preprocess")
    v.add_argument("--input", required=True)
    v.add_argument("--output", required=True)
    v.add_argument("--thresh", type=float, default=3.0)

    g = sub.add_parser("track")
    g.add_argument("--input", required=True)
    g.add_argument("--output", required=True)
    g.add_argument("--weights", required=True)

    args = p.parse_args()

    if args.cmd == "download":
        ds_dir = download_dataset(args.api_key,
                                  args.workspace,
                                  args.project,
                                  args.version)
        yaml_p = write_data_yaml(ds_dir, args.nc, args.names)
        print("→ data.yaml written at", yaml_p)

    elif args.cmd == "train":
        train(args.data_yaml, args.weights, args.epochs,
              args.batch, args.imgsz)

    elif args.cmd == "preprocess":
        reduce_background(args.input, args.output, args.thresh)

    elif args.cmd == "track":
        detect_and_track(args.input, args.output, args.weights)


if __name__ == "__main__":
    main()
