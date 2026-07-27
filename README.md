# AI Object Detection and Tracking Lab

A configurable computer-vision prototype using YOLOv8 for image detection, video tracking and real-time webcam experiments.

This project began as a CMPG 313 practical. I am presenting it here as a foundation for responsible applied AI: computer vision can support traffic analysis, accessibility and safety research, but only when accuracy, privacy and context are treated seriously.

## Capabilities

- Detect objects in still images
- Track objects across video frames with ByteTrack
- Accept a webcam index for real-time experiments
- Adjust confidence threshold and model from the command line
- Save annotated results in the generated `runs/` directory
- Run without opening a preview window, which is useful on remote systems

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Ultralytics downloads `yolov8n.pt` automatically when it is not already available.

## Usage

Detect objects in an image:

```bash
python lab_1.py --source "Asian market.jpg" --mode detect
```

Track objects in a video:

```bash
python lab_1.py --source "Cars Moving On Road Footage.mp4" --mode track
```

Use a webcam and display the live result:

```bash
python lab_1.py --source 0 --mode track --show
```

## Responsible use

The model is pretrained on a general-purpose dataset. Its output can be incomplete, inaccurate or less reliable in settings that differ from its training data.

Before using a similar system for a real community or public-space project, I would:

- measure precision and recall on relevant, consented data;
- avoid identifying individuals;
- define a clear retention and privacy policy;
- document failure cases across lighting, weather and camera angles;
- keep a human responsible for interpreting results.

The sample media is provided for learning and demonstration. Large future datasets and model files should be stored as release assets, Git LFS objects or documented external downloads rather than normal Git history.

## Portfolio direction

A meaningful next step is anonymous vehicle counting for road-safety or congestion research—not facial recognition or individual surveillance. The goal is to learn how AI can provide useful signals while respecting the people represented by the data.

## Author

Built by Ofentse Seko while developing practical AI skills with a focus on responsible social impact.
