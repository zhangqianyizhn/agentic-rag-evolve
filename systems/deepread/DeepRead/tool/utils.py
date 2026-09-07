from __future__ import annotations

import re
from typing import List, Optional, Tuple

try:
    import tiktoken  # type: ignore
except Exception:  # pragma: no cover
    tiktoken = None


# ------------------------------
# Neighbor window helpers
# ------------------------------
def _normalize_neighbor_window(window: Optional[Tuple[int, int]]) -> Optional[Tuple[int, int]]:
    if window is None:
        return None
    if not isinstance(window, tuple) or len(window) != 2:
        raise ValueError("neighbor_window must be a tuple of 2 integers, e.g. (1, -1) or (0, 0)")
    up, down = window
    if not isinstance(up, int) or not isinstance(down, int):
        raise ValueError("neighbor_window elements must be integers")
    if up < 0:
        raise ValueError(f"neighbor_window[0] (up) must be >= 0; got {up!r}")
    if down > 0:
        raise ValueError(f"neighbor_window[1] (down) must be <= 0; got {down!r}")
    if up == 0 and down == 0:
        return None
    return (up, down)


def _round_score(val: float) -> float:
    try:
        return float(f"{float(val):.2f}")
    except Exception:
        return float(val)

# ------------------------------
# Tokenization + counting
# ------------------------------
_ws_re = re.compile(r"\s+")


def simple_tokenize(text: str) -> List[str]:
    text = text.lower()
    return [t for t in _ws_re.split(text.strip()) if t]


def count_model_tokens(text: str) -> int:
    if not text:
        return 0
    try:
        if tiktoken is None:
            raise ImportError
        try:
            enc = tiktoken.get_encoding("o200k_base")
        except Exception:
            enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:
        return len(simple_tokenize(text))


normalize_neighbor_window = _normalize_neighbor_window
round_score = _round_score
