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

Two independent signals are computed per frame:

| Signal | Method | Catches |
|---|---|---|
| Structural correlation | Grid of patch-level `matchTemplate` (TM_CCOEFF_NORMED) against baseline | Reframing, pan/tilt, physical repositioning |
| Color/tone similarity | HSV histogram cosine similarity | Blackout, whiteout, gross lighting change |

These are treated as **independent triggers**, not an AND condition — in low
color-diversity scenes (e.g. grayscale/metal environments), histogram similarity can
stay high through a full reframe since the color *mix* doesn't change even though the
spatial layout does. Gating on both simultaneously misses real movement events.

Once a frame looks suspicious, it's sub-classified:

- **`camera_moved`** — structural mismatch, texture/focus otherwise normal
- **`defocus`** — Laplacian focus-measure has collapsed relative to baseline
- **`blocked_view`** — grayscale std-dev has collapsed (near-uniform frame, e.g. covered lens)
- **`black_frame` / `white_frame`** — luminance threshold + histogram deviation

Day/night (IR) transitions are detected separately (B/G channel divergence) and
handled as a legitimate re-baseline trigger, not a tamper event.

Every tamper class requires **N consecutive suspicious frames** (configurable) before
firing, to suppress single-frame false positives from motion blur, a person briefly
crossing the FOV, or transient noise.

## Tech stack

- **OpenCV** — does all the heavy lifting, no other CV library involved:
  - `cv2.matchTemplate` (`TM_CCOEFF_NORMED`) — grid patch correlation
  - `cv2.calcHist` / `cv2.normalize` — HSV histogram + cosine similarity
  - `cv2.Laplacian` — focus-measure (defocus detection)
  - `cv2.cvtColor`, `cv2.VideoCapture`, `cv2.VideoWriter` — color conversion and I/O
- **NumPy** — array ops, thresholds, cosine similarity
- **Python 3** — no other runtime dependencies

No ML model, no training data, no GPU requirement — runs on CPU in real time per stream.

## Installation

```bash
pip install opencv-python numpy
```

## Usage

```python
from camera_tamper_detector import CameraTamperDetector

detector = CameraTamperDetector(output_dir="tamper_frames", tamper_confirm_frames=5)

result = detector.process_frame(bgr_frame)  # np.ndarray, BGR, HxWx3
if result["tampered"]:
    print(result["note"])          # e.g. "camera_moved_confirmed"
    print(result["frame"])         # annotated frame (only drawn when tampered)
```

Run against a video file directly:

```bash
python CameraTamperingDetector_fresh.py
```

writes an annotated `output_inference.mp4` (tampered frames get a red banner,
normal frames pass through untouched) and saves confirmed tamper frames to
`tamper_frames/tampered/`.

## Configuration

| Parameter | Default | Purpose |
|---|---|---|
| `grid_size` | 100 |   Number of patch points for structural correlation (10×10 grid; patch/cell size auto-scales to the input frame's resolution, not a fixed pixel size) |
| `match_threshold` | 0.65 | Per-patch correlation cutoff to count as "matched" |
| `index_threshold` | 20 | Max matched-patch count still considered suspicious |
| `similarity_threshold` | 0.7 | Max histogram cosine similarity still considered suspicious |
| `covered_std_threshold` | 15.0 | Grayscale std-dev below which a frame is "blocked/covered" |
| `defocus_ratio_threshold` | 0.5 | Focus-measure ratio (current/baseline) below which a frame is "defocused" |
| `tamper_confirm_frames` | 5 | Consecutive suspicious frames required to confirm movement/defocus/blocked-view |
| `black_white_confirm_frames` | 10 | Consecutive suspicious frames required to confirm blackout/whiteout |
| `ir_change_confirm_frames` | 5 | Consecutive frames required to confirm a day/night switch and re-baseline |

## Known limitations

- Small-translation shake/vibration degrades gracefully rather than triggering
  cleanly — structural patch correlation is tuned for clear reframes, not
  sub-pixel jitter.
- HSV histogram similarity is a weak signal in low color-diversity environments;
  the structural signal carries most of the detection weight there by design.
- No re-baseline mechanism for legitimate scene changes other than day/night
  switching (e.g. permanent furniture rearrangement) — call `set_baseline()`
  manually if the "true" scene changes on purpose.

## License

[choose a license]
