"""
Structured AI Safety Report Generator and Conversational Assistant.
Uses LangChain and Google Gemini to transform driving telemetry into structured insights.
"""

from typing import Dict, Any, Optional
import json
import re
from ..analytics.session_logger import SessionSummary
from ..utils.helpers import format_duration
from .prompts import SAFETY_REPORT_SYSTEM_PROMPT, SAFETY_REPORT_USER_PROMPT, ASSISTANT_SYSTEM_PROMPT
from .gemini_service import GeminiService


class SessionReportChain:
    """
    Produces structured safety analysis from SessionSummary metrics.
    Employs LangChain / Gemini API with deterministic fallback if the API is offline.
    """
    def __init__(self, gemini_service: Optional[GeminiService] = None):
        self.gemini = gemini_service or GeminiService()

    def generate_report(self, summary: SessionSummary) -> Dict[str, Any]:
        """
        Generates a validated structured safety report.
        """
        # Duration formatted
        duration_fmt = format_duration(summary.duration_seconds)
        blink_rate = (summary.blink_count / (summary.duration_seconds / 60.0)) if summary.duration_seconds > 0 else 0.0

        if self.gemini.is_available:
            prompt = SAFETY_REPORT_USER_PROMPT.format(
                session_id=summary.session_id,
                duration_formatted=duration_fmt,
                duration_seconds=summary.duration_seconds,
                blink_count=summary.blink_count,
                blink_rate=blink_rate,
                drowsiness_events=summary.drowsiness_events,
                critical_events=summary.critical_events,
                avg_fatigue_score=summary.avg_fatigue_score,
                max_fatigue_score=summary.max_fatigue_score,
                avg_confidence=summary.avg_confidence,
                avg_latency_ms=summary.avg_latency_ms
            )

            full_prompt = f"{SAFETY_REPORT_SYSTEM_PROMPT}\n\n{prompt}"
            raw_response = self.gemini.generate_content(full_prompt)

            if raw_response:
                parsed = self._extract_json(raw_response)
                if parsed:
                    parsed["source"] = "Google Gemini 1.5 (LangChain Structured Pipeline)"
                    return parsed

        # Robust Rule-Based Deterministic Fallback if API key is not supplied
        return self._generate_rule_based_report(summary, duration_fmt, blink_rate)

    def _extract_json(self, text: str) -> Optional[Dict[str, Any]]:
        """Extracts JSON object from LLM markdown code blocks or raw text."""
        try:
            # Try direct JSON parse
            return json.loads(text)
        except Exception:
            pass

        # Try extracting code block
        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except Exception:
                pass
        return None

    def _generate_rule_based_report(self, summary: SessionSummary, duration_fmt: str, blink_rate: float) -> Dict[str, Any]:
        """Deterministic safety report generator ensuring full functionality without Gemini."""
        if summary.max_fatigue_score >= 80 or summary.critical_events > 0:
            risk = "CRITICAL"
            summary_txt = f"Critical drowsiness detected during driving session ({summary.critical_events} severe events). Immediate rest is urgently advised."
        elif summary.max_fatigue_score >= 50 or summary.drowsiness_events > 0:
            risk = "HIGH"
            summary_txt = f"Multiple drowsiness warnings occurred ({summary.drowsiness_events} events). Driver alertness degraded significantly over the session."
        elif summary.avg_fatigue_score > 30:
            risk = "MODERATE"
            summary_txt = "Moderate fatigue detected. Driver exhibited signs of reduced alertness and fluctuating blink intervals."
        else:
            risk = "LOW"
            summary_txt = "Driver remained predominantly alert throughout the monitored driving session with consistent blink rates."

        observations = [
            f"Monitored driving duration: {duration_fmt} ({summary.duration_seconds:.1f}s) across {summary.blink_count} total blinks.",
            f"Calculated average blink frequency: {blink_rate:.1f} blinks/minute.",
            f"Mean fatigue score was {summary.avg_fatigue_score:.1f}/100 with a peak of {summary.max_fatigue_score:.1f}/100.",
            f"Recorded {summary.drowsiness_events} warning events and {summary.critical_events} critical prolonged closure events."
        ]

        recommendations = []
        if risk in ("HIGH", "CRITICAL"):
            recommendations.append("Pull over safely at the nearest rest stop or service area immediately.")
            recommendations.append("Take a minimum 20-minute nap or switch with an alert co-driver.")
            recommendations.append("Avoid relying solely on caffeine or loud music to combat physiological sleep pressure.")
        else:
            recommendations.append("Maintain good cabin ventilation and keep air temperature cool.")
            recommendations.append("Schedule regular rest intervals every 2 hours during long-distance highway trips.")
            recommendations.append("Stay well-hydrated to reduce eye strain and fatigue.")

        critical_breakdown = [
            f"Logged {summary.critical_events} prolonged closures (> 2.2 seconds) posing severe collision hazard.",
            f"Logged {summary.drowsiness_events} moderate warning events (> 1.2 seconds)."
        ] if (summary.critical_events > 0 or summary.drowsiness_events > 0) else [
            "No prolonged eye closures or critical drowsiness events were detected during this session."
        ]

        return {
            "summary": summary_txt,
            "risk_level": risk,
            "key_observations": observations,
            "recommendations": recommendations,
            "critical_events_breakdown": critical_breakdown,
            "source": "Rule-Based Deterministic Engine (Gemini API Key not set or offline)"
        }


class SafetyAssistant:
    """Conversational assistant for in-cabin driver queries."""
    def __init__(self, gemini_service: Optional[GeminiService] = None):
        self.gemini = gemini_service or GeminiService()

    def ask(self, question: str, session_summary: SessionSummary) -> str:
        """Answers a user question based on current session statistics."""
        duration_fmt = format_duration(session_summary.duration_seconds)
        level = "LOW"
        if session_summary.max_fatigue_score >= 80:
            level = "CRITICAL"
        elif session_summary.max_fatigue_score >= 60:
            level = "HIGH"
        elif session_summary.max_fatigue_score >= 30:
            level = "MODERATE"

        system_prompt = ASSISTANT_SYSTEM_PROMPT.format(
            duration=duration_fmt,
            fatigue_score=session_summary.max_fatigue_score,
            fatigue_level=level,
            drowsy_events=session_summary.drowsiness_events,
            critical_events=session_summary.critical_events,
            blinks=session_summary.blink_count
        )

        if self.gemini.is_available:
            full_prompt = f"{system_prompt}\n\nDriver Question: {question}\nAssistant Response:"
            res = self.gemini.generate_content(full_prompt)
            if res:
                return res.strip()

        # Deterministic conversational response if Gemini is offline
        q_lower = question.lower()
        if "why" in q_lower or "drowsy" in q_lower:
            return (f"Your session recorded {session_summary.drowsiness_events} warning events and "
                    f"{session_summary.critical_events} critical closures where your eyes stayed closed longer than "
                    f"1.2 seconds. This exceeds the normal blink threshold.")
        elif "score" in q_lower or "fatigue" in q_lower:
            return (f"Your peak fatigue score was {session_summary.max_fatigue_score:.1f}/100 ({level}). "
                    f"It is derived from PERCLOS (percentage of eye closure time), blink frequency, and prolonged closures.")
        elif "sleep" in q_lower or "tired" in q_lower or "do" in q_lower:
            return ("If you feel sleepy, the safest action is to pull over at the nearest safe rest area. "
                    "Studies show a 15-20 minute nap is the most effective countermeasure for driver drowsiness.")
        else:
            return (f"During your {duration_fmt} session, we tracked {session_summary.blink_count} blinks and "
                    f"{session_summary.drowsiness_events} drowsiness events. Please drive responsibly and take breaks.")
