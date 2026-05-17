"""
Shared utility helpers for AirSim tools.
"""

from datetime import datetime, timezone


def _timestamp_utc() -> str:
    """
    Return the current UTC timestamp in ISO-8601 format.
    """

    return datetime.now(timezone.utc).isoformat()
