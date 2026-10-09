"""
MediaPipe Face Mesh Detector wrapper.
Robustly extracts facial landmarks with multi-face handling and missing face recovery.
Uses MediaPipe 1.x Tasks API (FaceLandmarker).
"""

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional, List, Tuple, Any
import cv2
import numpy as np

try:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision
    _MEDIAPIPE_AVAILABLE = True
except ImportError:
    _MEDIAPIPE_AVAILABLE = False


class FaceStatus(str, Enum):
    FACE_DETECTED = "FACE_DETECTED"
    NO_FACE_DETECTED = "NO_FACE_DETECTED"
    MULTIPLE_FACES = "MULTIPLE_FACES"
    INIT_ERROR = "INIT_ERROR"


@dataclass
class FaceDetectionResult:
    """Encapsulates facial landmark detection results for a single frame."""
    status: FaceStatus = FaceStatus.NO_FACE_DETECTED
    landmarks_px: Optional[np.ndarray] = None  # Shape (478, 3) in pixel coordinates
    landmarks_normalized: Optional[np.ndarray] = None
    face_bbox: Optional[Tuple[int, int, int, int]] = None  # (x, y, w, h)
    face_count: int = 0
    confidence: float = 0.0


class FaceLandmarkDetector:
    """
    Production-grade Face Mesh detector using MediaPipe Tasks API.
    Manages resource initialization and multi-face prioritization.
    """
    def __init__(
        self,
        max_num_faces: int = 1,
        refine_landmarks: bool = True,
        min_detection_confidence: float = 0.6,
        min_tracking_confidence: float = 0.6,
        model_path: Optional[str] = None,
    ):
        self.max_num_faces = max_num_faces
        self.refine_landmarks = refine_landmarks
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence

        if model_path is None:
            model_path = str(Path(__file__).resolve().parent.parent.parent / "models" / "mediapipe" / "face_landmarker.task")
        self.model_path = model_path

        self._landmarker = None
        self._initialized = False
        self._init_mediapipe()

    def _init_mediapipe(self) -> bool:
        """Initializes the MediaPipe FaceLandmarker model."""
        if not _MEDIAPIPE_AVAILABLE or not Path(self.model_path).exists():
            self._initialized = False
            return False

        try:
            base_options = mp_python.BaseOptions(model_asset_path=self.model_path)
            options = mp_vision.FaceLandmarkerOptions(
                base_options=base_options,
                running_mode=mp_vision.RunningMode.IMAGE,
                num_faces=self.max_num_faces,
                min_face_detection_confidence=self.min_detection_confidence,
                min_face_presence_confidence=self.min_tracking_confidence,
                min_tracking_confidence=self.min_tracking_confidence,
            )
            self._landmarker = mp_vision.FaceLandmarker.create_from_options(options)
            self._initialized = True
            return True
        except Exception:
            self._initialized = False
            return False

    @property
    def is_available(self) -> bool:
        return self._initialized

    def detect(self, frame_bgr: np.ndarray) -> FaceDetectionResult:
        """
        Processes a BGR video frame and returns extracted facial landmarks.
        
        Args:
            frame_bgr: BGR video frame from camera
            
        Returns:
            FaceDetectionResult with pixel landmarks and face bounding box
        """
        if frame_bgr is None or frame_bgr.size == 0:
            return FaceDetectionResult(status=FaceStatus.NO_FACE_DETECTED)

        if not self._initialized:
            # Attempt re-init if not ready
            if not self._init_mediapipe():
                return FaceDetectionResult(status=FaceStatus.INIT_ERROR)

        h, w = frame_bgr.shape[:2]
        # MediaPipe expects RGB
        rgb_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        if not rgb_frame.flags.c_contiguous:
            rgb_frame = np.ascontiguousarray(rgb_frame)

        try:
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            results = self._landmarker.detect(mp_image)
        except Exception:
            return FaceDetectionResult(status=FaceStatus.INIT_ERROR)

        if not results or not results.face_landmarks:
            return FaceDetectionResult(status=FaceStatus.NO_FACE_DETECTED, face_count=0)

        num_faces = len(results.face_landmarks)
        status = FaceStatus.FACE_DETECTED if num_faces == 1 else FaceStatus.MULTIPLE_FACES

        # Select primary face (largest face or first face)
        selected_face = self._select_primary_face(results.face_landmarks, w, h)

        # Convert normalized coordinates to pixel space
        landmarks_px = np.zeros((len(selected_face), 3), dtype=np.float32)
        landmarks_norm = np.zeros((len(selected_face), 3), dtype=np.float32)

        for i, lm in enumerate(selected_face):
            landmarks_norm[i] = [lm.x, lm.y, lm.z]
            landmarks_px[i] = [
                np.clip(lm.x * w, 0, w - 1),
                np.clip(lm.y * h, 0, h - 1),
                lm.z * w
            ]

        # Calculate face bounding box
        x_min = int(np.min(landmarks_px[:, 0]))
        y_min = int(np.min(landmarks_px[:, 1]))
        x_max = int(np.max(landmarks_px[:, 0]))
        y_max = int(np.max(landmarks_px[:, 1]))
        bbox = (x_min, y_min, max(1, x_max - x_min), max(1, y_max - y_min))

        return FaceDetectionResult(
            status=status,
            landmarks_px=landmarks_px,
            landmarks_normalized=landmarks_norm,
            face_bbox=bbox,
            face_count=num_faces,
            confidence=1.0  # MediaPipe returns faces matching threshold
        )

    def _select_primary_face(self, faces: List[Any], width: int, height: int):
        """Picks the largest face bounding area (most prominent driver)."""
        if len(faces) == 1:
            return faces[0]

        best_face = faces[0]
        max_area = 0.0

        for face in faces:
            xs = [lm.x * width for lm in face]
            ys = [lm.y * height for lm in face]
            area = (max(xs) - min(xs)) * (max(ys) - min(ys))
            if area > max_area:
                max_area = area
                best_face = face

        return best_face

    def close(self):
        """Release MediaPipe resources."""
        if self._landmarker is not None:
            try:
                self._landmarker.close()
            except Exception:
                pass
            self._landmarker = None
            self._initialized = False
