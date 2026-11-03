# Camera Tamper Detection

Real-time detection of camera tampering — movement/reframing, defocus, blocked/covered
lens, and blackout/whiteout — from a live video stream or file, using classical CV
signals (no ML model, no training data required).

## Why

Downstream analytics (detection, tracking, recognition) implicitly assume the camera
feed reflects the intended scene. A moved, covered, or defocused camera silently
breaks that assumption while continuing to produce output. This detector runs
upstream of those pipelines and raises an explicit tamper flag when the assumption
no longer holds.

## Detection approach

A single reference (baseline) frame is captured once, and every subsequent frame is
compared against it — the baseline is never replaced by a "tampered" frame, so a
sustained tamper event doesn't get normalized into the new "normal" over time.

# Camera Tampering Detector

Real-time webcam tamper detection using classical computer vision. The detector
compares each frame with a reference view and identifies camera movement, a changed
view, a covered lens, defocus, blackout, and whiteout. It does not use a machine
learning model, training data, or a GPU.

## Requirements

- Python 3
- OpenCV
- NumPy

Install the dependencies with:

```bash
pip install opencv-python numpy
```

## Run

From the repository directory, start the webcam detector with:

```bash
python CameraTamperingDetector.py
```

The program opens webcam device `0` at 640x480 and displays the live feed in a
window named `Camera Tamper Detection`. It first waits 45 frames for automatic
exposure, white balance, and focus to settle, then captures the current frame as the
baseline. Confirmed tamper frames are written to `tamper_frames/` as JPEG files,
with at most one saved image every three seconds.

Controls:

- Press `B` to replace the baseline with the current frame.
- Press `Q` to quit.

The program prints the total number of processed frames and confirmed tamper events
when it exits. On Windows, it tries the DirectShow capture backend first and falls
back to the default OpenCV backend if necessary.

## Detection

The baseline is retained until it is explicitly reset with `B` or through
`set_baseline()`. Each subsequent frame is evaluated using independent signals:

| Signal | Method | Detects |
|---|---|---|
| Scene similarity | HSV histogram cosine similarity | Large changes in the scene or lighting |
| Camera movement | ORB feature matching and median keypoint shift | Physical movement or reframing |
| View change | Blurred-frame difference and insufficient ORB matches | A camera pointed at a substantially different view |
| Blocked view | Grayscale mean and standard deviation | A covered or nearly uniform lens |
| Defocus | Laplacian variance relative to the baseline | Loss of focus |
| Blackout/whiteout | Grayscale mean plus histogram deviation | Very dark or very bright frames |

Suspicious frames must remain consecutive before an alert is confirmed. This avoids
triggering on a single blurred frame or a person briefly crossing the view. A
confirmed frame receives a red `TAMPER DETECTED` banner in the display and saved
image.

## Python API

Frames passed to `process_frame()` must be OpenCV BGR NumPy arrays with shape
`HxWx3`.

```python
import cv2
from CameraTamperingDetector import CameraTamperDetector

detector = CameraTamperDetector(
    tamper_confirm_frames=60,
    output_dir="tamper_frames",
)

result = detector.process_frame(bgr_frame)
if result["tampered"]:
    print(result["note"])
    annotated_frame = result["frame"]
```

`process_frame()` returns a dictionary containing `tampered`, `note`, and the
processed `frame`. For non-empty frames after baseline setup it also includes
`similarity`, `changed_ratio`, and `lap_ratio`. Empty frames return
`{"tampered": False, "note": "empty_frame"}`.

Call `set_baseline(frame)` when the intended scene changes. The output directory is
created automatically.

## Configuration

| Parameter | Default | Purpose |
|---|---:|---|
| `similarity_threshold` | `0.70` | Histogram similarity below this can indicate a blocked view |
| `black_mean_threshold` | `35` | Mean grayscale value below which a frame can be black |
| `white_mean_threshold` | `220` | Mean grayscale value above which a frame can be white |
| `covered_std_threshold` | `15.0` | Grayscale standard deviation below which a view can be covered |
| `defocus_ratio_threshold` | `0.20` | Current-to-baseline Laplacian variance ratio for defocus |
| `defocus_similarity_max` | `0.65` | Histogram similarity ceiling for defocus detection |
| `tamper_confirm_frames` | `60` | Consecutive frames needed for movement, defocus, or blocked-view confirmation |
| `black_white_confirm_frames` | `45` | Consecutive frames needed for blackout/whiteout confirmation |
| `warmup_frames` | `45` | Frames to skip before setting the initial baseline |
| `shift_threshold_px` | `45` | Median ORB keypoint shift needed to classify camera movement |
| `view_changed_ratio` | `0.60` | Blurred-pixel change ratio used for a substantially changed view |
| `output_dir` | `tamper_frames` | Directory for saved confirmed tamper images |

## Limitations

- The current entry point uses webcam device `0`; video-file processing is not
  implemented in `main()`.
- A legitimate permanent scene change should be followed by a manual baseline reset.
- Detection depends on having enough textured features for ORB matching; low-texture
  scenes rely more heavily on the histogram and frame-difference signals.
- The confirmation defaults are tuned for roughly 30 FPS, so alerts can take about
  two seconds for general tamper and 1.5 seconds for blackout/whiteout.
