"""
Synthetic Audio Generator and Alert Sound Engine.
Generates WAV audio tones entirely in Python via NumPy without external binary dependencies.
Supports sound library presets (beep, soft chime, double beep, siren, rising tone),
volume scaling, sound testing, escalation management, and HTML5 Web Audio embedding.
"""

import io
import wave
import base64
import time
import threading
import sys
from pathlib import Path
from typing import Dict, Optional
import numpy as np


class SoundLibrary:
    """Generates synthetic 16-bit PCM WAV audio for safety alerts."""

    @staticmethod
    def generate_tone(
        sound_type: str = "beep",
        volume: float = 0.8,
        sample_rate: int = 44100
    ) -> bytes:
        """
        Synthesizes a 16-bit mono PCM WAV in memory.
        
        Args:
            sound_type: 'beep', 'soft_chime', 'double_beep', 'siren', 'rising'
            volume: 0.0 to 1.0 amplitude scaler
            sample_rate: sampling frequency in Hz
            
        Returns:
            bytes of a standard .wav file
        """
        volume = float(np.clip(volume, 0.0, 1.0))

        if sound_type == "soft_chime":
            # Gentle harmonic chime (523.25 Hz C5 + 659.25 Hz E5) with exponential fade
            duration = 0.9
            t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
            decay = np.exp(-3.5 * t)
            wave_c = 0.6 * np.sin(2 * np.pi * 523.25 * t) + 0.4 * np.sin(2 * np.pi * 659.25 * t)
            samples = wave_c * decay

        elif sound_type == "double_beep":
            # Two crisp 1100 Hz pulses with 80ms silence between
            pulse_dur = 0.14
            gap_dur = 0.08
            t_pulse = np.linspace(0, pulse_dur, int(sample_rate * pulse_dur), endpoint=False)
            pulse = np.sin(2 * np.pi * 1100.0 * t_pulse)
            
            # Envelope on pulse to prevent clicking
            fade_samples = int(sample_rate * 0.01)
            env = np.ones_like(pulse)
            env[:fade_samples] = np.linspace(0, 1, fade_samples)
            env[-fade_samples:] = np.linspace(1, 0, fade_samples)
            pulse = pulse * env
            
            silence = np.zeros(int(sample_rate * gap_dur))
            samples = np.concatenate([pulse, silence, pulse])

        elif sound_type == "siren":
            # Urgent high-low frequency modulated emergency siren
            duration = 1.0
            t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
            # Sweeping between 850 Hz and 1450 Hz
            mod_freq = 3.5  # 3.5 cycles per second
            instant_freq = 1150.0 + 300.0 * np.sin(2 * np.pi * mod_freq * t)
            phase = 2 * np.pi * np.cumsum(instant_freq) / sample_rate
            samples = np.sin(phase)

        elif sound_type == "rising":
            # Rising alert chirp: 650 Hz -> 1500 Hz
            duration = 0.55
            t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
            freqs = np.linspace(650.0, 1500.0, len(t))
            phase = 2 * np.pi * np.cumsum(freqs) / sample_rate
            samples = np.sin(phase)

        else:  # default 'beep'
            duration = 0.28
            t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
            samples = np.sin(2 * np.pi * 950.0 * t)

        # Apply global attack/release envelope to avoid clicks
        fade_len = min(int(sample_rate * 0.015), len(samples) // 4)
        if fade_len > 0:
            samples[:fade_len] *= np.linspace(0, 1, fade_len)
            samples[-fade_len:] *= np.linspace(1, 0, fade_len)

        # Normalize and scale to 16-bit PCM
        peak = np.max(np.abs(samples))
        if peak > 1e-6:
            samples = (samples / peak) * volume
        else:
            samples = samples * volume

        int_samples = (samples * 32767).astype(np.int16)

        buf = io.BytesIO()
        with wave.open(buf, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(int_samples.tobytes())

        return buf.getvalue()

    @classmethod
    def get_base64_data_uri(cls, sound_type: str = "beep", volume: float = 0.8) -> str:
        """Returns data:audio/wav;base64 string for embedding in HTML5 audio tags."""
        wav_bytes = cls.generate_tone(sound_type, volume)
        b64 = base64.b64encode(wav_bytes).decode("ascii")
        return f"data:audio/wav;base64,{b64}"


class AudioAlertManager:
    """
    Manages alert playback, escalation, repeat intervals, and volume controls.
    Supports escalation:
      - Warning: Soft chime / rising tone
      - Critical: Loud siren repeating until eyes open
    """
    def __init__(
        self,
        warning_sound: str = "soft_chime",
        critical_sound: str = "siren",
        phone_sound: str = "rising",
        volume: float = 0.85,
        repeat_interval_sec: float = 1.6,
        enabled: bool = True
    ):
        self.warning_sound = warning_sound
        self.critical_sound = critical_sound
        self.phone_sound = phone_sound
        self.volume = volume
        self.repeat_interval_sec = repeat_interval_sec
        self.enabled = enabled

        self._last_alert_time: float = 0.0
        self._current_active_sound: Optional[str] = None
        self._play_lock = threading.Lock()
        self._cache: Dict[str, bytes] = {}
        self._pregenerate_sounds()

    def _pregenerate_sounds(self):
        """Pre-generates tone bytes for all sound types to minimize latency."""
        types = ["beep", "soft_chime", "double_beep", "siren", "rising"]
        for st in types:
            self._cache[st] = SoundLibrary.generate_tone(st, volume=self.volume)

    def set_volume(self, volume: float):
        self.volume = float(np.clip(volume, 0.0, 1.0))
        self._pregenerate_sounds()

    def trigger(self, alert_level: str, force: bool = False) -> Optional[str]:
        """
        Triggers an alert sound if enabled and repeat cooldown has expired.
        
        Args:
            alert_level: 'WARNING', 'CRITICAL', 'DISTRACTION', or 'TEST'
            force: if True, bypasses cooldown
            
        Returns:
            HTML audio embed snippet for browser rendering, or None if suppressed.
        """
        if not self.enabled and alert_level != "TEST":
            return None

        now = time.time()
        # Critical alert repeats faster
        cooldown = self.repeat_interval_sec * 0.7 if alert_level == "CRITICAL" else self.repeat_interval_sec

        if not force and (now - self._last_alert_time < cooldown):
            return None

        self._last_alert_time = now

        sound_name = self.warning_sound
        if alert_level == "CRITICAL":
            sound_name = self.critical_sound
        elif alert_level == "DISTRACTION":
            sound_name = self.phone_sound
        elif alert_level == "TEST":
            sound_name = self.warning_sound

        self._current_active_sound = sound_name

        # 1. Asynchronous native audio playback fallback (winsound or terminal bell)
        self._play_native_async(sound_name)

        # 2. Return HTML5 base64 audio string for browser frontend
        data_uri = SoundLibrary.get_base64_data_uri(sound_name, self.volume)
        # Hidden audio tag with unique key so browser plays immediately
        audio_html = f"""<audio autoplay style="display:none">
            <source src="{data_uri}" type="audio/wav">
        </audio>"""
        return audio_html

    def _play_native_async(self, sound_name: str):
        """Plays tone via OS audio driver in a background thread."""
        def _run():
            with self._play_lock:
                try:
                    if sys.platform == "win32":
                        import winsound
                        freq_map = {
                            "beep": (1000, 250),
                            "soft_chime": (650, 400),
                            "double_beep": (1100, 150),
                            "siren": (1400, 450),
                            "rising": (900, 300),
                        }
                        freq, dur = freq_map.get(sound_name, (1000, 250))
                        winsound.Beep(freq, dur)
                    else:
                        sys.stdout.write("\a")
                        sys.stdout.flush()
                except Exception:
                    pass

        t = threading.Thread(target=_run, daemon=True)
        t.start()
