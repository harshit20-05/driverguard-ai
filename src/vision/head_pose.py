"""
Head Pose Estimator & Driver Gaze/Nodding Detector.
Calculates 3D Head Pose (Pitch, Yaw, Roll) via cv2.solvePnP with anthropometric 3D face model.
Detects:
1. Head Nodding: pitch dropping forward (characteristic of microsleep episodes)
2. Road Inattention: yaw or pitch turned away from the driving direction for > threshold seconds.
"""

from dataclasses import dataclass
from typing import Optional, Tuple
import time
import cv2
import numpy as np


@dataclass
class HeadPoseStatus:
    """Encapsulates estimated 3D head pose and attention status."""
    pitch: float = 0.0          # Positive = looking up, Negative = looking down
    yaw: float = 0.0            # Positive = looking right, Negative = looking left
    roll: float = 0.0           # Tilt toward shoulder
    is_nodding: bool = False    # Head dropped forward
    is_looking_away: bool = False # Off-road gaze distraction
    nodding_duration_s: float = 0.0
    looking_away_duration_s: float = 0.0
    total_nod_events: int = 0
    total_distraction_events: int = 0


class HeadPoseEstimator:
    """
    Solves Perspective-n-Point (solvePnP) problem using 6 key 2D facial landmarks
    matched to canonical 3D facial feature coordinates.
    """

    # Canonical 3D anthropometric face model coordinates (in millimeters)
    # Origin centered at nose tip
    _MODEL_POINTS = np.array([
        (0.0, 0.0, 0.0),             # Nose tip (index 1)
        (0.0, -330.0, -65.0),        # Chin (index 152)
        (-225.0, 170.0, -135.0),     # Left eye outer corner (index 33)
        (225.0, 170.0, -135.0),      # Right eye outer corner (index 263)
        (-150.0, -150.0, -125.0),    # Left mouth corner (index 61)
        (150.0, -150.0, -125.0),     # Right mouth corner (index 291)
    ], dtype=np.float64)

    # Corresponding MediaPipe landmark indices
    _LANDMARK_INDICES = [1, 152, 33, 263, 61, 291]

    def __init__(
        self,
        nod_pitch_threshold: float = -15.0,     # Pitch below -15 deg indicates forward head drop
        distract_yaw_threshold: float = 24.0,   # Yaw magnitude > 24 deg indicates looking away
        nod_alert_duration_s: float = 0.8,
        distract_alert_duration_s: float = 2.0
    ):
        self.nod_pitch_threshold = nod_pitch_threshold
        self.distract_yaw_threshold = distract_yaw_threshold
        self.nod_alert_duration_s = nod_alert_duration_s
        self.distract_alert_duration_s = distract_alert_duration_s

        self.baseline_pitch: float = 0.0
        self.baseline_yaw: float = 0.0

        self.total_nod_events: int = 0
        self.total_distraction_events: int = 0

        self._nod_start_time: Optional[float] = None
        self._distract_start_time: Optional[float] = None
        self._nod_event_recorded: bool = False
        self._distract_event_recorded: bool = False

    def estimate_pose(
        self,
        landmarks_px: np.ndarray,
        frame_width: int,
        frame_height: int
    ) -> Tuple[float, float, float]:
        """
        Calculates Euler angles (pitch, yaw, roll) in degrees.
        """
        if landmarks_px is None or len(landmarks_px) <= max(self._LANDMARK_INDICES):
            return 0.0, 0.0, 0.0

        # Extract 2D points
        image_points = np.array([
            landmarks_px[idx][:2] for idx in self._LANDMARK_INDICES
        ], dtype=np.float64)

        # Approximate camera intrinsic matrix
        focal_length = frame_width
        center = (frame_width / 2.0, frame_height / 2.0)
        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype=np.float64)

        dist_coeffs = np.zeros((4, 1), dtype=np.float64)  # Assume minimal camera distortion

        # Solve PnP
        success, rvec, tvec = cv2.solvePnP(
            self._MODEL_POINTS,
            image_points,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )

        if not success:
            return 0.0, 0.0, 0.0

        # Convert rotation vector to rotation matrix
        rmat, _ = cv2.Rodrigues(rvec)

        # Extract Euler angles from rotation matrix
        # Pitch (X-rotation), Yaw (Y-rotation), Roll (Z-rotation)
        sy = np.sqrt(rmat[0, 0] ** 2 + rmat[1, 0] ** 2)
        singular = sy < 1e-6

        if not singular:
            pitch = np.arctan2(rmat[2, 1], rmat[2, 2])
            yaw = np.arctan2(-rmat[2, 0], sy)
            roll = np.arctan2(rmat[1, 0], rmat[0, 0])
        else:
            pitch = np.arctan2(-rmat[1, 2], rmat[1, 1])
            yaw = np.arctan2(-rmat[2, 0], sy)
            roll = 0.0

        # Convert radians to degrees
        pitch_deg = float(np.degrees(pitch))
        yaw_deg = float(np.degrees(yaw))
        roll_deg = float(np.degrees(roll))

        return pitch_deg, yaw_deg, roll_deg

    def update(
        self,
        landmarks_px: Optional[np.ndarray],
        frame_width: int,
        frame_height: int,
        current_time: Optional[float] = None
    ) -> HeadPoseStatus:
        """
        Evaluates current head pose angles and inattention/nodding state machine.
        """
        now = current_time if current_time is not None else time.time()

        if landmarks_px is None:
            self._nod_start_time = None
            self._distract_start_time = None
            return HeadPoseStatus(
                total_nod_events=self.total_nod_events,
                total_distraction_events=self.total_distraction_events
            )

        pitch, yaw, roll = self.estimate_pose(landmarks_px, frame_width, frame_height)

        # Adjust relative to baseline
        rel_pitch = pitch - self.baseline_pitch
        rel_yaw = yaw - self.baseline_yaw

        # 1. Nodding evaluation (pitch dropped forward)
        nod_active = rel_pitch <= self.nod_pitch_threshold
        nod_duration = 0.0
        if nod_active:
            if self._nod_start_time is None:
                self._nod_start_time = now
            nod_duration = now - self._nod_start_time
            if nod_duration >= self.nod_alert_duration_s and not self._nod_event_recorded:
                self.total_nod_events += 1
                self._nod_event_recorded = True
        else:
            self._nod_start_time = None
            self._nod_event_recorded = False

        # 2. Distraction / Looking away evaluation
        distract_active = abs(rel_yaw) >= self.distract_yaw_threshold or rel_pitch >= 22.0
        distract_duration = 0.0
        if distract_active:
            if self._distract_start_time is None:
                self._distract_start_time = now
            distract_duration = now - self._distract_start_time
            if distract_duration >= self.distract_alert_duration_s and not self._distract_event_recorded:
                self.total_distraction_events += 1
                self._distract_event_recorded = True
        else:
            self._distract_start_time = None
            self._distract_event_recorded = False

        return HeadPoseStatus(
            pitch=pitch,
            yaw=yaw,
            roll=roll,
            is_nodding=nod_duration >= self.nod_alert_duration_s,
            is_looking_away=distract_duration >= self.distract_alert_duration_s,
            nodding_duration_s=nod_duration,
            looking_away_duration_s=distract_duration,
            total_nod_events=self.total_nod_events,
            total_distraction_events=self.total_distraction_events
        )

    def reset(self):
        self.total_nod_events = 0
        self.total_distraction_events = 0
        self._nod_start_time = None
        self._distract_start_time = None
        self._nod_event_recorded = False
        self._distract_event_recorded = False
