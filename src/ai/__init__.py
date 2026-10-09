"""AI package for DriverGuard AI."""
from .gemini_service import GeminiService
from .prompts import SAFETY_REPORT_SYSTEM_PROMPT, SAFETY_REPORT_USER_PROMPT, ASSISTANT_SYSTEM_PROMPT
from .report_generator import SessionReportChain, SafetyAssistant

__all__ = [
    "GeminiService",
    "SAFETY_REPORT_SYSTEM_PROMPT",
    "SAFETY_REPORT_USER_PROMPT",
    "ASSISTANT_SYSTEM_PROMPT",
    "SessionReportChain",
    "SafetyAssistant"
]
