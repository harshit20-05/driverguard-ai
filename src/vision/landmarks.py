"""
MediaPipe Face Mesh landmark indices and geometric calculations for eye regions.
Uses standard 468 / 478 MediaPipe facial landmark layout.
"""

from typing import List, Tuple, Dict
import numpy as np

# Left Eye Landmark Indices (MediaPipe Face Mesh)
LEFT_EYE_LANDMARKS: List[int] = [
    33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246
]

# Right Eye Landmark Indices (MediaPipe Face Mesh)
RIGHT_EYE_LANDMARKS: List[int] = [
    362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398
]

# Key points for Eye Aspect Ratio (EAR) as an auxiliary geometric signal:
# EAR = (|p2 - p6| + |p3 - p5|) / (2 * |p1 - p4|)
LEFT_EYE_EAR_INDICES = {
    "corner_left": 33,
    "corner_right": 133,
    "top_1": 160,
    "bottom_1": 144,
    "top_2": 158,
    "bottom_2": 153
}

RIGHT_EYE_EAR_INDICES = {
    "corner_left": 362,
    "corner_right": 263,
    "top_1": 385,
    "bottom_1": 380,
    "top_2": 387,
    "bottom_2": 373
}


def calculate_ear(landmarks_np: np.ndarray, eye_indices: Dict[str, int]) -> float:
    """
    Calculates Eye Aspect Ratio (EAR) from 2D facial landmark coordinates.
    Used as an auxiliary metric alongside the deep CNN classifier.
    
    Args:
        landmarks_np: Array of shape (N, 2) or (N, 3) in pixel coordinates
        eye_indices: Dictionary containing corner and vertical landmark indices
        
    Returns:
        EAR float value (typically 0.15 - 0.35)
    """
    p1 = landmarks_np[eye_indices["corner_left"]][:2]
    p4 = landmarks_np[eye_indices["corner_right"]][:2]
    p2 = landmarks_np[eye_indices["top_1"]][:2]
    p6 = landmarks_np[eye_indices["bottom_1"]][:2]
    p3 = landmarks_np[eye_indices["top_2"]][:2]
    p5 = landmarks_np[eye_indices["bottom_2"]][:2]

    vertical_1 = np.linalg.norm(p2 - p6)
    vertical_2 = np.linalg.norm(p3 - p5)
    horizontal = np.linalg.norm(p1 - p4)

    if horizontal <= 1e-6:
        return 0.0

    ear = (vertical_1 + vertical_2) / (2.0 * horizontal)
    return float(ear)


def get_eye_bbox(
    landmarks_np: np.ndarray,
    indices: List[int],
    frame_width: int,
    frame_height: int,
    padding_ratio: float = 0.28
) -> Tuple[int, int, int, int]:
    """
    Calculates padded bounding box (x, y, w, h) around eye landmarks.
    
    Args:
        landmarks_np: Array of coordinates in pixel space
        indices: Landmark indices corresponding to the eye
        frame_width: Width of image frame
        frame_height: Height of image frame
        padding_ratio: Relative expansion applied to width and height
        
    Returns:
        (x_min, y_min, box_width, box_height) clamped within image boundaries
    """
    pts = landmarks_np[indices][:, :2]
    x_min, y_min = np.min(pts, axis=0)
    x_max, y_max = np.max(pts, axis=0)

    w = x_max - x_min
    h = y_max - y_min

    # Add margin for eyelid context and CNN receptive field
    pad_w = w * padding_ratio
    pad_h = max(h * padding_ratio, pad_w * 0.7)  # Ensure sufficient vertical context

    x1 = int(max(0, x_min - pad_w))
    y1 = int(max(0, y_min - pad_h))
    x2 = int(min(frame_width, x_max + pad_w))
    y2 = int(min(frame_height, y_max + pad_h))

    return x1, y1, max(1, x2 - x1), max(1, y2 - y1)
