"""Investigation session identifiers for interactive MOSAIC clients."""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4


def new_investigation_id(*, prefix: str = "INV") -> str:
    """Create a readable, collision-resistant investigation identifier."""

    prefix = prefix.strip()
    if not prefix:
        raise ValueError("prefix must be non-empty")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return f"{prefix}-{stamp}-{uuid4().hex[:8]}"
