"""
Google Gemini API Service Wrapper.
Provides robust connection management, key validation, and non-blocking fallbacks.
Uses the latest google-genai SDK (google.genai).
"""

import os
from typing import Optional

try:
    from google import genai
    from google.genai import types as genai_types
    _GEMINI_LIB_AVAILABLE = True
except ImportError:
    _GEMINI_LIB_AVAILABLE = False


class GeminiService:
    """
    Client for interacting with Google's Gemini models using the modern google-genai SDK.
    Gracefully degrades to offline mode if the API key is absent or invalid.
    """
    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "gemini-2.0-flash",
        temperature: float = 0.2
    ):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY", "")
        self.model_name = model_name
        self.temperature = temperature
        self._client = None
        self._initialized = False

        if self.api_key and _GEMINI_LIB_AVAILABLE:
            self._init_client()

    def _init_client(self):
        try:
            self._client = genai.Client(api_key=self.api_key)
            self._initialized = True
        except Exception:
            self._initialized = False

    @property
    def is_available(self) -> bool:
        """Returns True if the API key is present and client is initialized."""
        return self._initialized and self._client is not None

    def generate_content(self, prompt: str) -> Optional[str]:
        """
        Generates content from the Gemini model.
        Returns response string, or None if failed or unavailable.
        """
        if not self.is_available:
            return None

        try:
            response = self._client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=genai_types.GenerateContentConfig(
                    temperature=self.temperature,
                    max_output_tokens=2048,
                )
            )
            if response and response.text:
                return response.text
        except Exception:
            return None
        return None
