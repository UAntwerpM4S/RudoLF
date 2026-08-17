
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


class AsyncioWin995Filter(logging.Filter):
    """
    Suppress the noisy Windows asyncio self-pipe shutdown traceback.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        """
        Return False for the known benign WinError 995 shutdown record.
        """

        exc = record.exc_info[1] if record.exc_info else None
        message = record.getMessage().lower()
        if (
            "event loop self pipe" in message
            and (
                (
                    isinstance(exc, ConnectionResetError)
                    and getattr(exc, "winerror", None) == 995
                )
                or "winerror 995" in message
                or "connectionreseterror" in message
            )
        ):
            return False
        return True
