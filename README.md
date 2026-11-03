# Camera Tampering Detector

Real-time webcam tamper detection using classical computer vision. The detector
compares each frame with a reference view and identifies camera movement, a changed
view, a covered lens, defocus, blackout, and whiteout. It does not use a machine
learning model, training data, or a GPU.

## Why

Downstream analytics (detection, tracking, recognition) implicitly assume the camera
feed reflects the intended scene. A moved, covered, or defocused camera silently
breaks that assumption while continuing to produce output. This detector runs
upstream of those pipelines and raises an explicit tamper flag when the assumption
no longer holds.

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
`set_baseline()`. It is never replaced by a tampered frame, so a sustained tamper
event does not become the new "normal". Each subsequent frame is evaluated with the
signals below. Histogram similarity is a supporting signal used inside several of the
checks rather than a stand-alone trigger.

| Signal | Method | Detects |
|---|---|---|
| Scene similarity | HSV histogram cosine similarity | Supports the checks below; large changes in scene or lighting |
| Camera movement | ORB feature matching and median keypoint shift | Physical movement or reframing |
| View change | Blurred-frame difference, too few ORB matches, and low histogram similarity | A camera pointed at a substantially different view |
| Blocked view | Grayscale standard deviation and histogram similarity | A covered or nearly uniform lens |
| Defocus | Laplacian variance relative to the baseline, and histogram similarity | Loss of focus |
| Blackout/whiteout | Grayscale mean and histogram similarity | Very dark or very bright frames |

Suspicious frames must remain consecutive before an alert is confirmed. This avoids
triggering on a single blurred frame or a person briefly crossing the view. A
confirmed frame receives a red `TAMPER DETECTED` banner in the display and saved
image.

## Python API

Frames passed to `process_frame()` must be OpenCV BGR NumPy arrays with shape
`HxWx3`.

```python
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

`process_frame()` returns a dictionary containing `tampered`, `note`, and `frame`.
For non-empty frames after baseline setup it also includes `similarity`,
`changed_ratio`, and `lap_ratio`. During warm-up the note is `warming_up`, and empty
frames return the note `empty_frame`.

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
| `tamper_confirm_frames` | `60` | Consecutive suspicious frames needed to confirm any tamper alert |
| `black_white_confirm_frames` | `45` | Consecutive dark or bright frames before a black/white frame counts as suspicious |
| `warmup_frames` | `45` | Frames to skip before setting the initial baseline |
| `shift_threshold_px` | `45` | Median ORB keypoint shift needed to classify camera movement |
| `view_changed_ratio` | `0.60` | Fraction of changed pixels used for a substantially changed view |
| `output_dir` | `tamper_frames` | Directory for saved confirmed tamper images |

The following values are fixed in the code: histogram similarity below `0.60` for
black/white frames and below `0.40` for a changed view, a per-pixel change cutoff of
`40`, a minimum of `10` good ORB matches, and a ratio-test value of `0.75`.

## Limitations

- The current entry point uses webcam device `0`; video-file processing is not
  implemented in `main()`.
- Alerts are shown on screen, printed to the console, and saved as images. There is
  no external notification yet.
- A legitimate permanent scene change should be followed by a manual baseline reset.
- Detection depends on having enough textured features for ORB matching; low-texture
  scenes rely more heavily on the histogram and frame-difference signals.
- The confirmation defaults are tuned for roughly 30 FPS. A general tamper alert
  takes about two seconds (60 frames). Blackout and whiteout take about 3.5 seconds,
  because a frame must first be dark or bright for 45 frames and then the general
  60-frame confirmation applies (about 105 frames).
