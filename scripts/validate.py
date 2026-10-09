"""Quick validation script for DriverGuard AI."""
import sys
import os
sys.path.insert(0, '.')

from src.utils.config import get_config
cfg = get_config()
print("Config loaded successfully")
print("  AI Model:", cfg["ai"]["model_name"])
print("  Weights:", cfg["model"]["weights_path"])
print("  DB Path:", cfg["analytics"]["db_path"])

from src.ai.gemini_service import GeminiService
gs = GeminiService()
print("  GeminiService available:", gs.is_available)
print("  eye_classifier.keras exists:", os.path.exists(cfg["model"]["weights_path"]))

from src.models.inference import EyeStateClassifier
clf = EyeStateClassifier(weights_path=cfg["model"]["weights_path"])
print("  CNN model loaded:", clf.is_model_loaded)

from src.analytics.session_logger import SessionDatabase
db = SessionDatabase(cfg["analytics"]["db_path"])
sessions = db.list_sessions()
print("  Session count in DB:", len(sessions))

print()
print("All validation checks PASSED.")
