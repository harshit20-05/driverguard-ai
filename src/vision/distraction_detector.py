"""
Distraction Detector — MediaPipe Tasks HandLandmarker-based Phone Usage Detection.

Detects when a driver's hand is raised to face/ear level, the primary
indicator of phone usage visible from a forward-facing cabin camera.
Smoothed via a consecutive-frame counter to suppress false positives.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple
import numpy as np

try:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision
    _MP_AVAILABLE = True
except ImportError:
    _MP_AVAILABLE = False


@dataclass
class DistractionStatus:
    """Encapsulates per-frame distraction detection output."""
    phone_detected: bool = False
    hand_near_face: bool = False
    confidence: float = 0.0
    consecutive_frames: int = 0
    alert_message: str = ""


class PhoneDistractionDetector:
    """
    Detects phone usage by tracking hand proximity to the driver's face.

    Strategy:
      - MediaPipe HandLandmarker locates hand landmarks each frame.
      - If a hand wrist/index-tip is within `face_proximity_ratio` of the
        face bounding box centre, it is flagged as 'near face'.
      - After `alert_frames` consecutive flagged frames an alert fires.
      - The counter decays by 2 per open frame to reduce sensitivity to
        momentary hand movements (e.g. scratching nose).
    """

    # Hand landmark indices in MediaPipe HandLandmarker:
    # 0: Wrist, 4: Thumb tip, 8: Index tip, 12: Middle tip
    _WRIST = 0
    _INDEX_TIP = 8
    _MIDDLE_TIP = 12
    _THUMB_TIP = 4

    def __init__(
        self,
        min_detection_confidence: float = 0.50,
        min_tracking_confidence: float = 0.50,
        alert_frames: int = 18,        # ~0.6s at 30 fps before beep
        face_proximity_ratio: float = 0.30,
        model_path: Optional[str] = None,
    ):
        self.face_proximity_ratio = face_proximity_ratio
        self.alert_frames = alert_frames
        self._consecutive = 0
        self._landmarker = None

        if model_path is None:
            model_path = str(Path(__file__).resolve().parent.parent.parent / "models" / "mediapipe" / "hand_landmarker.task")
        self.model_path = model_path

        if _MP_AVAILABLE and Path(self.model_path).exists():
            try:
                base_options = mp_python.BaseOptions(model_asset_path=self.model_path)
                options = mp_vision.HandLandmarkerOptions(
                    base_options=base_options,
                    running_mode=mp_vision.RunningMode.IMAGE,
                    num_hands=2,
                    min_hand_detection_confidence=min_detection_confidence,
                    min_hand_presence_confidence=min_tracking_confidence,
                    min_tracking_confidence=min_tracking_confidence,
                )
                self._landmarker = mp_vision.HandLandmarker.create_from_options(options)
            except Exception:
                self._landmarker = None

    def detect(
        self,
        frame_rgb: np.ndarray,
        face_bbox: Optional[Tuple[int, int, int, int]] = None,
    ) -> DistractionStatus:
        """
        Run hand detection and evaluate proximity to the driver's face.

        Args:
            frame_rgb: RGB uint8 frame (H, W, 3)
            face_bbox: Optional (x, y, w, h) face bounding box in pixels.

        Returns:
            DistractionStatus dataclass.
        """
        if self._landmarker is None or frame_rgb is None or frame_rgb.size == 0:
            return DistractionStatus()

        h, w = frame_rgb.shape[:2]
        try:
            if not frame_rgb.flags.c_contiguous:
                frame_rgb = np.ascontiguousarray(frame_rgb)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            results = self._landmarker.detect(mp_image)
        except Exception:
            return DistractionStatus()

        hand_near = False
        max_conf = 0.0

        if results and results.hand_landmarks:
            for hand_lm in results.hand_landmarks:
                if len(hand_lm) > max(self._WRIST, self._INDEX_TIP, self._MIDDLE_TIP):
                    wrist   = hand_lm[self._WRIST]
                    idx_tip = hand_lm[self._INDEX_TIP]
                    mid_tip = hand_lm[self._MIDDLE_TIP]

                    # Use midpoint of key landmarks for stability
                    hand_y = (wrist.y + idx_tip.y + mid_tip.y) / 3.0
                    hand_x = (wrist.x + idx_tip.x + mid_tip.x) / 3.0

                    if face_bbox:
                        fx, fy, fw, fh = face_bbox
                        face_cy = (fy + fh * 0.5) / h
                        face_cx = (fx + fw * 0.5) / w
                        face_r  = max(fw, fh) * 0.5 / max(w, h)

                        dist = ((hand_y - face_cy) ** 2 + (hand_x - face_cx) ** 2) ** 0.5
                        if dist < (face_r + self.face_proximity_ratio):
                            hand_near = True
                            conf = float(np.clip(1.0 - dist / (face_r + self.face_proximity_ratio), 0, 1))
                            max_conf = max(max_conf, conf)
                    else:
                        # Fallback: flag if hand is in upper 55 % of frame
                        if hand_y < 0.55:
                            hand_near = True
                            conf = float(np.clip(1.0 - hand_y / 0.55, 0, 1))
                            max_conf = max(max_conf, conf)

        # Smooth with counter
        if hand_near:
            self._consecutive = min(self._consecutive + 1, self.alert_frames * 2)
        else:
            self._consecutive = max(0, self._consecutive - 2)

        phone_alert = self._consecutive >= self.alert_frames

        return DistractionStatus(
            phone_detected=phone_alert,
            hand_near_face=hand_near,
            confidence=max_conf,
            consecutive_frames=self._consecutive,
            alert_message="📵 DISTRACTION ALERT — Keep both hands on the wheel!" if phone_alert else "",
        )

    def reset(self):
        self._consecutive = 0

    def close(self):
        if self._landmarker is not None:
            try:
                self._landmarker.close()
            except Exception:
                pass
            self._landmarker = None
