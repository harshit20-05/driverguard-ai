"""
Image preprocessing and enhancement pipeline for eye region patches.
Includes CLAHE adaptive histogram equalization, robust resizing, and normalization.
"""

from typing import Tuple, Optional
import cv2
import numpy as np


class EyePreprocessor:
    """
    Handles cropping, lighting normalization, and format conversions for eye CNN inputs.
    """
    def __init__(
        self,
        target_size: Tuple[int, int] = (64, 64),
        apply_clahe: bool = True,
        clip_limit: float = 2.0,
        grid_size: Tuple[int, int] = (8, 8)
    ):
        self.target_size = target_size
        self.apply_clahe = apply_clahe
        self.clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)

    def enhance_lighting(self, img_bgr: np.ndarray) -> np.ndarray:
        """
        Enhances contrast in low-light or uneven vehicle cockpit lighting
        using CLAHE on the luminance channel (LAB color space).
        """
        if not self.apply_clahe or img_bgr is None or img_bgr.size == 0:
            return img_bgr
            
        try:
            lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
            l_channel, a_channel, b_channel = cv2.split(lab)
            l_enhanced = self.clahe.apply(l_channel)
            merged_lab = cv2.merge((l_enhanced, a_channel, b_channel))
            return cv2.cvtColor(merged_lab, cv2.COLOR_LAB2BGR)
        except Exception:
            return img_bgr

    def check_lighting(self, frame_bgr: np.ndarray, low_light_thresh: float = 50.0) -> Tuple[float, bool, np.ndarray]:
        """
        Analyzes vehicle cabin illumination level.
        
        Args:
            frame_bgr: Current camera frame
            low_light_thresh: Luminance threshold below which lighting is flagged as poor
            
        Returns:
            (mean_luminance, is_poor_lighting, enhanced_frame)
        """
        if frame_bgr is None or frame_bgr.size == 0:
            return 0.0, True, frame_bgr

        # Compute average luminance on L channel of LAB
        lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
        l_channel = lab[:, :, 0]
        mean_lum = float(np.mean(l_channel))
        is_poor = mean_lum < low_light_thresh

        enhanced = self.enhance_lighting(frame_bgr) if is_poor else frame_bgr
        return mean_lum, is_poor, enhanced

    def preprocess_crop(self, crop_bgr: np.ndarray) -> np.ndarray:
        """
        Processes a raw cropped eye patch into a normalized tensor ready for CNN inference.
        
        Args:
            crop_bgr: BGR image crop of the eye
            
        Returns:
            Normalized RGB float32 array with shape (H, W, 3), range [0.0, 1.0]
        """
        if crop_bgr is None or crop_bgr.size == 0:
            # Return blank zero tensor if crop is invalid
            return np.zeros((self.target_size[0], self.target_size[1], 3), dtype=np.float32)

        # 1. Lighting enhancement
        enhanced = self.enhance_lighting(crop_bgr)

        # 2. Convert BGR to RGB
        rgb = cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)

        # 3. Resize with area interpolation
        resized = cv2.resize(rgb, self.target_size, interpolation=cv2.INTER_AREA)

        # 4. Normalize to [0.0, 1.0] float32
        normalized = resized.astype(np.float32) / 255.0

        return normalized

    def prepare_batch(self, left_eye_crop: Optional[np.ndarray], right_eye_crop: Optional[np.ndarray]) -> np.ndarray:
        """
        Prepares both eyes as a single batch tensor of shape (2, H, W, 3).
        """
        left_tensor = self.preprocess_crop(left_eye_crop)
        right_tensor = self.preprocess_crop(right_eye_crop)
        return np.stack([left_tensor, right_tensor], axis=0)
