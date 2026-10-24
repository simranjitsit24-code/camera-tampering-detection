import os
import time
import cv2
import numpy as np


class CameraTamperDetector:
    def __init__(
        self,
        similarity_threshold=0.70,
        black_mean_threshold=35,
        white_mean_threshold=220,
        covered_std_threshold=15.0,
        defocus_ratio_threshold=0.20,    # lower = less sensitive to blur
        defocus_similarity_max=0.65,     # lower = less sensitive to blur
        tamper_confirm_frames=60,        # higher = less sensitive (~2 s at 30 fps)
        black_white_confirm_frames=45,
        warmup_frames=45,
        shift_threshold_px=45,           # higher = less sensitive to camera movement
        view_changed_ratio=0.60,         # higher = less sensitive to scene change
        output_dir="tamper_frames",
    ):
        self.similarity_threshold = similarity_threshold
        self.black_mean_threshold = black_mean_threshold
        self.white_mean_threshold = white_mean_threshold
        self.covered_std_threshold = covered_std_threshold
        self.defocus_ratio_threshold = defocus_ratio_threshold
        self.defocus_similarity_max = defocus_similarity_max
        self.tamper_confirm_frames = tamper_confirm_frames
        self.black_white_confirm_frames = black_white_confirm_frames
        self.warmup_frames = warmup_frames
        self.shift_threshold_px = shift_threshold_px
        self.view_changed_ratio = view_changed_ratio
        self.output_dir = output_dir

        os.makedirs(output_dir, exist_ok=True)

        # ORB for camera movement
        self.orb = cv2.ORB_create(nfeatures=1000)
        self.baseline_keypoints = None
        self.baseline_descriptors = None

        self.baseline_gray = None
        self.baseline_histogram = None
        self.baseline_laplacian_var = None

        self.frame_count = 0
        self.tamper_count = 0
        self.black_count = 0
        self.white_count = 0
        self.tamper_event_count = 0
        self.currently_tampered = False

    @staticmethod
    def get_histogram(frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
        cv2.normalize(hist, hist)
        return hist.flatten()

    @staticmethod
    def similarity(hist1, hist2):
        h1 = hist1.astype(np.float32)
        h2 = hist2.astype(np.float32)
        denominator = np.linalg.norm(h1) * np.linalg.norm(h2)
        if denominator == 0:
            return 0.0
        return float(np.dot(h1, h2) / denominator)

    @staticmethod
    def laplacian_variance(gray):
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())

    def set_baseline(self, frame):
        self.baseline_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.baseline_histogram = self.get_histogram(frame)
        self.baseline_laplacian_var = self.laplacian_variance(self.baseline_gray)

        self.baseline_keypoints, self.baseline_descriptors = self.orb.detectAndCompute(
            self.baseline_gray, None
        )

        self.tamper_count = 0
        self.black_count = 0
        self.white_count = 0
        self.currently_tampered = False

    def process_frame(self, frame):
        if frame is None or frame.size == 0:
            return {"tampered": False, "note": "empty_frame", "frame": frame}

        # Warm-up: let auto exposure / white balance / focus settle first
        if self.baseline_gray is None:
            self.frame_count += 1
            if self.frame_count < self.warmup_frames:
                return {"tampered": False, "note": "warming_up", "frame": frame.copy()}
            self.set_baseline(frame)
            return {"tampered": False, "note": "baseline_set", "frame": frame.copy()}

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)
        base_blur = cv2.GaussianBlur(self.baseline_gray, (5, 5), 0)

        mean_value = float(np.mean(gray))
        std_value = float(np.std(gray))
        similarity = self.similarity(self.baseline_histogram, self.get_histogram(frame))

        lap_var = self.laplacian_variance(gray)
        baseline_lap = self.baseline_laplacian_var
        lap_ratio = lap_var / baseline_lap if baseline_lap > 0 else 1.0

        changed_ratio = float(np.mean(cv2.absdiff(base_blur, gray_blur) > 40))

        # --- ORB: measure how far background keypoints moved ---
        orb_good_matches = 0
        median_shift = 0.0

        if self.baseline_descriptors is not None and len(self.baseline_descriptors) > 10:
            kp, des = self.orb.detectAndCompute(gray, None)

            if des is not None and len(des) > 10:
                bf = cv2.BFMatcher(cv2.NORM_HAMMING)
                pairs = [
                    p for p in bf.knnMatch(self.baseline_descriptors, des, k=2)
                    if len(p) == 2
                ]
                good = [m for m, n in pairs if m.distance < 0.75 * n.distance]
                orb_good_matches = len(good)

                if orb_good_matches >= 10:
                    p0 = np.float32([self.baseline_keypoints[m.queryIdx].pt for m in good])
                    p1 = np.float32([kp[m.trainIdx].pt for m in good])
                    # Median is robust to a person moving in front of the scene
                    median_shift = float(np.median(np.linalg.norm(p1 - p0, axis=1)))

        enough_matches = orb_good_matches >= 10

        # Black / white detection
        black_candidate = mean_value < self.black_mean_threshold and similarity < 0.60
        white_candidate = mean_value > self.white_mean_threshold and similarity < 0.60

        self.black_count = self.black_count + 1 if black_candidate else 0
        self.white_count = self.white_count + 1 if white_candidate else 0

        note = "normal"
        candidate = False

        if self.black_count >= self.black_white_confirm_frames:
            candidate = True
            note = "black_frame_confirmed"
        elif self.white_count >= self.black_white_confirm_frames:
            candidate = True
            note = "white_frame_confirmed"
        else:
            blocked_view = (
                std_value < self.covered_std_threshold
                and similarity < self.similarity_threshold
            )
            defocused = (
                lap_ratio < self.defocus_ratio_threshold
                and similarity < self.defocus_similarity_max
                and not blocked_view
            )
            # Camera physically moved: background keypoints consistently shifted
            camera_moved = enough_matches and median_shift > self.shift_threshold_px
            # Pointed somewhere else entirely: almost no matches + big change
            view_changed = (
                not enough_matches
                and changed_ratio > self.view_changed_ratio
                and similarity < 0.40
                and not blocked_view
                and not defocused
            )

            if blocked_view:
                candidate = True
                note = "possible_blocked_view"
            elif defocused:
                candidate = True
                note = "possible_defocus"
            elif camera_moved or view_changed:
                candidate = True
                note = "possible_camera_moved"

        if candidate:
            self.tamper_count += 1
        else:
            self.tamper_count = 0
            self.currently_tampered = False

        tampered = candidate and self.tamper_count >= self.tamper_confirm_frames

        if tampered:
            if not self.currently_tampered:
                self.tamper_event_count += 1
            self.currently_tampered = True
            if note.startswith("possible_"):
                note = note.replace("possible_", "") + "_confirmed"

        display_frame = frame.copy()
        if tampered:
            h, w = display_frame.shape[:2]
            overlay = display_frame.copy()
            cv2.rectangle(overlay, (0, 0), (w, 55), (0, 0, 255), -1)
            cv2.addWeighted(overlay, 0.60, display_frame, 0.40, 0, display_frame)
            cv2.putText(
                display_frame, "TAMPER DETECTED", (10, 38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2
            )

        self.frame_count += 1

        print(
            f"\rFrame: {self.frame_count} | "
            f"Sim: {similarity:.3f} | "
            f"Chg: {changed_ratio:.3f} | "
            f"ORB: {orb_good_matches} | "
            f"Shift: {median_shift:.1f}px | "
            f"Lap: {lap_ratio:.2f} | "
            f"Tamper: {self.tamper_count} | "
            f"{note}       ",
            end="", flush=True
        )

        return {
            "tampered": tampered,
            "note": note,
            "similarity": similarity,
            "changed_ratio": changed_ratio,
            "lap_ratio": lap_ratio,
            "frame": display_frame,
        }


def main():
    detector = CameraTamperDetector(output_dir="tamper_frames")

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("ERROR: Webcam could not be opened.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("Webcam started. Press B to reset baseline, Q to quit.")

    alert_visible = False
    last_saved_time = 0.0

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("\nUnable to read webcam frame.")
                break

            result = detector.process_frame(frame)
            cv2.imshow("Camera Tamper Detection", result["frame"])

            if result["tampered"]:
                if not alert_visible:
                    print(f"\nTAMPER ALERT: {result['note']}")
                    alert_visible = True

                now = time.time()
                if now - last_saved_time >= 3:
                    filename = os.path.join(
                        detector.output_dir, f"tamper_{int(now)}.jpg"
                    )
                    cv2.imwrite(filename, result["frame"])
                    last_saved_time = now
            else:
                alert_visible = False

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("b"):
                detector.set_baseline(frame)
                print("\nBaseline reset.")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("\nFrames processed:", detector.frame_count)
        print("Tamper events:", detector.tamper_event_count)


if __name__ == "__main__":
    main()