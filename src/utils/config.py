"""
Configuration manager for DriverGuard AI.
Loads YAML configurations with environment variable resolution.
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional
import yaml
from dotenv import load_dotenv

# Load local .env if present
load_dotenv()

_DEFAULT_CONFIG_PATH = Path("config/config.yaml")
_CACHED_CONFIG: Optional[Dict[str, Any]] = None


def load_config(config_path: Optional[Path | str] = None) -> Dict[str, Any]:
    """
    Loads YAML configuration file and parses environment overrides.
    
    Args:
        config_path: Path to config YAML file. Defaults to config/config.yaml
        
    Returns:
        Dictionary representation of config
    """
    global _CACHED_CONFIG
    
    target_path = Path(config_path) if config_path else _DEFAULT_CONFIG_PATH
    
    if not target_path.exists():
        # Fallback default configuration dictionary
        return _get_fallback_config()
        
    try:
        with open(target_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
    except Exception as e:
        print(f"Error loading config at {target_path}: {e}. Falling back to default.")
        cfg = _get_fallback_config()
        
    # Inject environment variables
    cfg["ai"] = cfg.get("ai", {})
    cfg["ai"]["api_key"] = os.getenv("GEMINI_API_KEY", "")
    
    _CACHED_CONFIG = cfg
    return cfg


def get_config(reload: bool = False) -> Dict[str, Any]:
    """
    Returns the cached configuration or loads it if not already loaded.
    """
    global _CACHED_CONFIG
    if _CACHED_CONFIG is None or reload:
        return load_config()
    return _CACHED_CONFIG


def _get_fallback_config() -> Dict[str, Any]:
    """Provides safe baseline configuration."""
    return {
        "app": {
            "name": "DRIVERGUARD AI",
            "tagline": "Real-Time AI for Safer Driving",
            "version": "1.0.0",
            "debug": False,
            "theme": "dark"
        },
        "vision": {
            "camera_index": 0,
            "frame_width": 640,
            "frame_height": 480,
            "target_fps": 30,
            "face_mesh": {
                "max_num_faces": 1,
                "refine_landmarks": True,
                "min_detection_confidence": 0.6,
                "min_tracking_confidence": 0.6
            },
            "eye_processing": {
                "input_size": [64, 64],
                "crop_padding_ratio": 0.28,
                "clahe_clip_limit": 2.0,
                "clahe_grid_size": [8, 8],
                "normalize_range": [0.0, 1.0]
            }
        },
        "model": {
            "weights_path": "models/eye_classifier.keras",
            "metadata_path": "models/model_metadata.json",
            "input_shape": [64, 64, 3],
            "open_threshold": 0.50,
            "confidence_smoothing_window": 5
        },
        "detection": {
            "blink_min_duration_ms": 80,
            "blink_max_duration_ms": 380,
            "warning_threshold_ms": 1200,
            "critical_threshold_ms": 2200,
            "recovery_open_frames": 6
        },
        "fatigue": {
            "window_duration_seconds": 60,
            "weights": {
                "perclos": 0.40,
                "prolonged_closures": 0.35,
                "blink_frequency": 0.15,
                "event_severity": 0.10
            },
            "levels": {
                "low": [0, 30],
                "moderate": [31, 60],
                "high": [61, 80],
                "critical": [81, 100]
            }
        },
        "alerts": {
            "enable_sound": False,
            "sound_frequency": 1000,
            "sound_duration_ms": 300,
            "warning_color": "#EAB308",
            "critical_color": "#EF4444",
            "alert_color": "#10B981"
        },
        "analytics": {
            "db_path": "data/sessions.db",
            "export_dir": "data/exports",
            "log_interval_seconds": 1
        },
        "ai": {
            "model_name": "gemini-2.0-flash",
            "temperature": 0.2,
            "max_output_tokens": 1500,
            "timeout_seconds": 15,
            "api_key": os.getenv("GEMINI_API_KEY", "")
        },
        "privacy": {
            "store_raw_video": False,
            "store_eye_crops": False,
            "anonymize_session_data": False
        }
    }
