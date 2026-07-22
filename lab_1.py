"""Run YOLOv8 detection or tracking on an image, video or webcam."""

import argparse
from pathlib import Path

from ultralytics import YOLO


def parse_source(value):
    """Convert a numeric webcam value while leaving file paths unchanged."""
    return int(value) if value.isdigit() else value


def build_parser():
    parser = argparse.ArgumentParser(
        description="Detect or track objects with a pretrained YOLOv8 model."
    )
    parser.add_argument(
        "--source",
        default="Asian market.jpg",
        help="Image/video path, stream URL, or webcam index such as 0.",
    )
    parser.add_argument(
        "--mode",
        choices=("auto", "detect", "track"),
        default="auto",
        help="Use detection, tracking, or infer the mode from the source.",
    )
    parser.add_argument("--model", default="yolov8n.pt", help="YOLO model path or name.")
    parser.add_argument("--confidence", type=float, default=0.35)
    parser.add_argument("--show", action="store_true", help="Open a live preview window.")
    return parser


def choose_mode(source, requested_mode):
    if requested_mode != "auto":
        return requested_mode
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    if isinstance(source, str) and Path(source).suffix.lower() in image_extensions:
        return "detect"
    return "track"


def run(source, mode="auto", model_name="yolov8n.pt", confidence=0.35, show=False):
    if not 0 < confidence <= 1:
        raise ValueError("confidence must be greater than 0 and at most 1")

    resolved_source = parse_source(str(source))
    resolved_mode = choose_mode(resolved_source, mode)
    model = YOLO(model_name)

    options = {
        "source": resolved_source,
        "conf": confidence,
        "show": show,
        "save": True,
    }
    if resolved_mode == "detect":
        return model.predict(**options)
    return model.track(tracker="bytetrack.yaml", **options)


def main():
    args = build_parser().parse_args()
    run(
        source=args.source,
        mode=args.mode,
        model_name=args.model,
        confidence=args.confidence,
        show=args.show,
    )
    print("Complete. Annotated output is available in the runs/ directory.")


if __name__ == "__main__":
    main()
