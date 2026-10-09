"""
Eye Region Detector and Extractor.
Locates left and right eye regions from facial landmarks and provides cropped image patches.
"""

from dataclasses import dataclass
from typing import Optional, Tuple, Dict
import cv2
import numpy as np

from .landmarks import (
    LEFT_EYE_LANDMARKS,
    RIGHT_EYE_LANDMARKS,
    LEFT_EYE_EAR_INDICES,
    RIGHT_EYE_EAR_INDICES,
    calculate_ear,
    get_eye_bbox
)


@dataclass
class EyeData:
    """Encapsulates eye extraction results for a single frame."""
    left_crop: Optional[np.ndarray] = None
    right_crop: Optional[np.ndarray] = None
    left_bbox: Optional[Tuple[int, int, int, int]] = None   # (x, y, w, h)
    right_bbox: Optional[Tuple[int, int, int, int]] = None  # (x, y, w, h)
    left_ear: float = 0.0
    right_ear: float = 0.0
    avg_ear: float = 0.0
    eyes_detected: bool = False


class EyeDetector:
    """
    Extracts eye region patches and geometric metrics from detected facial landmarks.
    """
    def __init__(self, crop_padding_ratio: float = 0.28):
        self.crop_padding_ratio = crop_padding_ratio

    def extract_eyes(self, frame_bgr: np.ndarray, landmarks_px: Optional[np.ndarray]) -> EyeData:
        """
        Extracts both eye crops, bounding boxes, and geometric EAR from landmark coordinates.
        
        Args:
            frame_bgr: Source video frame (H, W, 3)
            landmarks_px: Array of (N, 2) or (N, 3) pixel coordinates for face mesh
            
        Returns:
            EyeData object containing crops and geometric metrics
        """
        if frame_bgr is None or landmarks_px is None or len(landmarks_px) < 468:
            return EyeData()

        h, w = frame_bgr.shape[:2]

        # 1. Calculate bounding boxes
        left_bbox = get_eye_bbox(
            landmarks_px,
            LEFT_EYE_LANDMARKS,
            frame_width=w,
            frame_height=h,
            padding_ratio=self.crop_padding_ratio
        )
        right_bbox = get_eye_bbox(
            landmarks_px,
            RIGHT_EYE_LANDMARKS,
            frame_width=w,
            frame_height=h,
            padding_ratio=self.crop_padding_ratio
        )

        # 2. Extract image patches
        lx, ly, lw, lh = left_bbox
        rx, ry, rw, rh = right_bbox

        left_crop = frame_bgr[ly:ly + lh, lx:lx + lw].copy() if lw > 0 and lh > 0 else None
        right_crop = frame_bgr[ry:ry + rh, rx:rx + rw].copy() if rw > 0 and rh > 0 else None

        # 3. Compute geometric EAR
        left_ear = calculate_ear(landmarks_px, LEFT_EYE_EAR_INDICES)
        right_ear = calculate_ear(landmarks_px, RIGHT_EYE_EAR_INDICES)
        avg_ear = (left_ear + right_ear) / 2.0

        eyes_detected = (left_crop is not None and right_crop is not None and
                         left_crop.size > 0 and right_crop.size > 0)

        return EyeData(
            left_crop=left_crop,
            right_crop=right_crop,
            left_bbox=left_bbox,
            right_bbox=right_bbox,
            left_ear=left_ear,
            right_ear=right_ear,
            avg_ear=avg_ear,
            eyes_detected=eyes_detected
        )

    def draw_eye_overlays(
        self,
        frame_bgr: np.ndarray,
        eye_data: EyeData,
        landmarks_px: Optional[np.ndarray] = None,
        color: Tuple[int, int, int] = (0, 255, 170)
    ) -> np.ndarray:
        """
        Draws eye bounding boxes and landmark points onto a copy of the frame.
        """
        annotated = frame_bgr.copy()
        if not eye_data.eyes_detected:
            return annotated

        # Draw left eye bounding box
        if eye_data.left_bbox:
            x, y, w, h = eye_data.left_bbox
            cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)
            cv2.putText(annotated, f"L EAR: {eye_data.left_ear:.2f}", (x, max(15, y - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

        # Draw right eye bounding box
        if eye_data.right_bbox:
            x, y, w, h = eye_data.right_bbox
            cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)
            cv2.putText(annotated, f"R EAR: {eye_data.right_ear:.2f}", (x, max(15, y - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

        # Draw key eye mesh points if landmarks provided
        if landmarks_px is not None:
            for idx in LEFT_EYE_LANDMARKS + RIGHT_EYE_LANDMARKS:
                px, py = int(landmarks_px[idx][0]), int(landmarks_px[idx][1])
                cv2.circle(annotated, (px, py), 1, (255, 255, 0), -1)

        return annotated
