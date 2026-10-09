"""
Session Logger and SQLite Database Manager.
Stores structured driving sessions, time-series telemetry, and event logs.
Provides automated CSV and JSON export capabilities using native sqlite3 and csv.
"""

from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Optional
import sqlite3
import os
import json
import time
import csv


@dataclass
class SessionSummary:
    """Consolidated summary for a finished driving session."""
    session_id: str
    start_time: str
    end_time: str
    duration_seconds: float
    blink_count: int
    drowsiness_events: int
    critical_events: int
    avg_fatigue_score: float
    max_fatigue_score: float
    avg_confidence: float
    avg_latency_ms: float
    avg_fps: float


class SessionDatabase:
    """
    Manages persistent SQLite storage of driver sessions and telemetry samples.
    """
    def __init__(self, db_path: str = "data/sessions.db"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            # Sessions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    start_time TEXT NOT NULL,
                    end_time TEXT NOT NULL,
                    duration_seconds REAL NOT NULL,
                    blink_count INTEGER NOT NULL,
                    drowsiness_events INTEGER NOT NULL,
                    critical_events INTEGER NOT NULL,
                    avg_fatigue_score REAL NOT NULL,
                    max_fatigue_score REAL NOT NULL,
                    avg_confidence REAL NOT NULL,
                    avg_latency_ms REAL NOT NULL,
                    avg_fps REAL NOT NULL
                )
            """)

            # Time-series telemetry samples table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS telemetry (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    timestamp REAL NOT NULL,
                    elapsed_seconds REAL NOT NULL,
                    fatigue_score REAL NOT NULL,
                    eye_prob REAL NOT NULL,
                    is_closed INTEGER NOT NULL,
                    drowsiness_state TEXT NOT NULL,
                    blinks_per_min REAL NOT NULL,
                    latency_ms REAL NOT NULL,
                    fps REAL NOT NULL,
                    FOREIGN KEY(session_id) REFERENCES sessions(session_id)
                )
            """)

            # Event timeline table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    timestamp REAL NOT NULL,
                    elapsed_seconds REAL NOT NULL,
                    event_type TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    duration_ms REAL NOT NULL,
                    details TEXT NOT NULL,
                    FOREIGN KEY(session_id) REFERENCES sessions(session_id)
                )
            """)
            conn.commit()

    def save_session(
        self,
        summary: SessionSummary,
        samples: List[Dict[str, Any]],
        events: Optional[List[Dict[str, Any]]] = None
    ):
        """Saves session summary, telemetry samples, and timeline events transactionally."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO sessions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                summary.session_id,
                summary.start_time,
                summary.end_time,
                summary.duration_seconds,
                summary.blink_count,
                summary.drowsiness_events,
                summary.critical_events,
                summary.avg_fatigue_score,
                summary.max_fatigue_score,
                summary.avg_confidence,
                summary.avg_latency_ms,
                summary.avg_fps
            ))

            for s in samples:
                cursor.execute("""
                    INSERT INTO telemetry (
                        session_id, timestamp, elapsed_seconds, fatigue_score,
                        eye_prob, is_closed, drowsiness_state, blinks_per_min, latency_ms, fps
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    summary.session_id,
                    s.get("timestamp", time.time()),
                    s.get("elapsed_seconds", 0.0),
                    s.get("fatigue_score", 0.0),
                    s.get("eye_prob", 0.5),
                    int(s.get("is_closed", False)),
                    s.get("drowsiness_state", "AWAKE"),
                    s.get("blinks_per_min", 0.0),
                    s.get("latency_ms", 0.0),
                    s.get("fps", 0.0)
                ))

            if events:
                for ev in events:
                    cursor.execute("""
                        INSERT INTO events (
                            session_id, timestamp, elapsed_seconds, event_type, severity, duration_ms, details
                        ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """, (
                        summary.session_id,
                        ev.get("timestamp", time.time()),
                        ev.get("elapsed_seconds", 0.0),
                        ev.get("event_type", "WARNING"),
                        ev.get("severity", "WARNING"),
                        ev.get("duration_ms", 0.0),
                        ev.get("details", "")
                    ))

            conn.commit()

    def delete_session(self, session_id: str) -> bool:
        """Deletes a session and its associated telemetry and event records."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM events WHERE session_id = ?", (session_id,))
            cursor.execute("DELETE FROM telemetry WHERE session_id = ?", (session_id,))
            cursor.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
            conn.commit()
            return cursor.rowcount > 0

    def list_sessions(self) -> List[Dict[str, Any]]:
        """Returns all completed sessions ordered by most recent."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM sessions ORDER BY start_time DESC")
            rows = cursor.fetchall()
            return [dict(row) for row in rows]

    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Returns single session details."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM sessions WHERE session_id = ?", (session_id,))
            row = cursor.fetchone()
            return dict(row) if row else None

    def get_telemetry_records(self, session_id: str) -> List[Dict[str, Any]]:
        """Returns session telemetry samples as a list of dictionaries."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM telemetry WHERE session_id = ? ORDER BY elapsed_seconds ASC", (session_id,))
            rows = cursor.fetchall()
            return [dict(r) for r in rows]

    def get_session_events(self, session_id: str) -> List[Dict[str, Any]]:
        """Returns timeline event records for the specified session."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM events WHERE session_id = ? ORDER BY elapsed_seconds ASC", (session_id,))
            rows = cursor.fetchall()
            return [dict(r) for r in rows]

    def export_session_csv_text(self, session_id: str) -> str:
        """Exports session telemetry to CSV formatted string."""
        import io
        records = self.get_telemetry_records(session_id)
        if not records:
            return ""

        output = io.StringIO()
        fieldnames = list(records[0].keys())
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
        return output.getvalue()

    def export_session_csv(self, session_id: str, output_path: str) -> bool:
        """Exports session telemetry to CSV file on disk."""
        csv_text = self.export_session_csv_text(session_id)
        if not csv_text:
            return False
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            f.write(csv_text)
        return True
