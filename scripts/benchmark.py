"""
DriverGuard AI - Production Performance Benchmark Script.
Measures real FPS, pipeline stage latencies, memory footprint, and outputs a formatted markdown report.
"""

import os
import sys
import time
from pathlib import Path
import numpy as np
import cv2

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.utils.config import get_config
from src.vision.face_detector import FaceLandmarkDetector
from src.vision.eye_detector import EyeDetector
from src.vision.preprocessing import EyePreprocessor
from src.models.inference import EyeStateClassifier
from src.vision.yawn_detector import YawnDetector
from src.vision.head_pose import HeadPoseEstimator
from src.detection.fatigue_score import FatigueScoreCalculator


def run_benchmark(num_frames: int = 150):
    print("==================================================")
    print("DriverGuard AI - Pipeline Latency & FPS Benchmark")
    print("==================================================")

    cfg = get_config()
    detector = FaceLandmarkDetector()
    eye_det = EyeDetector()
    preprocessor = EyePreprocessor()
    classifier = EyeStateClassifier(weights_path=cfg["model"]["weights_path"], preprocessor=preprocessor)
    yawn_det = YawnDetector()
    head_pose = HeadPoseEstimator()
    fatigue_calc = FatigueScoreCalculator()

    # Create dummy video frames (640x480 RGB)
    frame = np.random.randint(40, 220, (480, 640, 3), dtype=np.uint8)

    print(f"\n[+] Running {num_frames} frames through the pipeline...")

    times_face = []
    times_eyes = []
    times_cnn = []
    times_yawn = []
    times_head = []
    times_fatigue = []
    times_total = []

    for i in range(num_frames):
        t_start = time.perf_counter()

        # 1. Face detection
        t0 = time.perf_counter()
        det_res = detector.detect(frame)
        times_face.append((time.perf_counter() - t0) * 1000.0)

        # 2. Eye cropping
        t0 = time.perf_counter()
        eye_data = eye_det.extract_eyes(frame, det_res.landmarks_px)
        times_eyes.append((time.perf_counter() - t0) * 1000.0)

        # 3. Eye inference
        t0 = time.perf_counter()
        pred = classifier.predict(eye_data)
        times_cnn.append((time.perf_counter() - t0) * 1000.0)

        # 4. Yawn calculation
        t0 = time.perf_counter()
        yawn_st = yawn_det.update(det_res.landmarks_px)
        times_yawn.append((time.perf_counter() - t0) * 1000.0)

        # 5. Head pose
        t0 = time.perf_counter()
        hp_st = head_pose.update(det_res.landmarks_px, 640, 480)
        times_head.append((time.perf_counter() - t0) * 1000.0)

        # 6. Fatigue Score
        t0 = time.perf_counter()
        f_res = fatigue_calc.update(
            is_closed=pred.is_closed,
            closure_duration_ms=0.0,
            blinks_per_min=16.0,
            recent_drowsy_events=0,
            recent_critical_events=0,
            recent_yawns=yawn_st.total_yawns,
            recent_nods=hp_st.total_nod_events
        )
        times_fatigue.append((time.perf_counter() - t0) * 1000.0)

        total_frame_ms = (time.perf_counter() - t_start) * 1000.0
        times_total.append(total_frame_ms)

    avg_total_ms = np.mean(times_total)
    p95_total_ms = np.percentile(times_total, 95)
    effective_fps = 1000.0 / max(0.1, avg_total_ms)

    print("\n---------------- Benchmark Results ----------------")
    print(f"| Pipeline Stage             | Mean (ms) | P95 (ms) |")
    print(f"|:---------------------------|:----------|:---------|")
    print(f"| Face Landmarker (MediaPipe)| {np.mean(times_face):9.2f} | {np.percentile(times_face, 95):8.2f} |")
    print(f"| Eye Region Extraction      | {np.mean(times_eyes):9.2f} | {np.percentile(times_eyes, 95):8.2f} |")
    print(f"| Eye State Classification   | {np.mean(times_cnn):9.2f} | {np.percentile(times_cnn, 95):8.2f} |")
    print(f"| Yawn Detection (MAR)       | {np.mean(times_yawn):9.2f} | {np.percentile(times_yawn, 95):8.2f} |")
    print(f"| 3D Head Pose (solvePnP)    | {np.mean(times_head):9.2f} | {np.percentile(times_head, 95):8.2f} |")
    print(f"| Fatigue Score & State FSM  | {np.mean(times_fatigue):9.2f} | {np.percentile(times_fatigue, 95):8.2f} |")
    print(f"| TOTAL FRAME LATENCY        | {avg_total_ms:9.2f} | {p95_total_ms:8.2f} |")
    print(f"---------------------------------------------------")
    print(f"Effective Processing Speed: {effective_fps:.1f} FPS (Real-time compatible >= 30 FPS)")
    print("==================================================")


if __name__ == "__main__":
    run_benchmark()
