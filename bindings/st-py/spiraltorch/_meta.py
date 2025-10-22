"""Build metadata helpers for SpiralTorch Python bindings."""

from __future__ import annotations

import datetime as _datetime
import platform as _platform
import uuid as _uuid

_BUILD_PREFIX = "RyoST"

BUILD_ID = (
    f"{_BUILD_PREFIX}-{_platform.node()}-"
    f"{_datetime.datetime.utcnow().isoformat()}-"
    f"{_uuid.uuid4().hex[:8]}"
)

__all__ = ["BUILD_ID"]
