from __future__ import annotations

import json
from typing import Any


def log_event(event: str, **fields: Any) -> None:
    payload = {"event": event, **fields}
    print(json.dumps(payload, default=str, sort_keys=True), flush=True)
