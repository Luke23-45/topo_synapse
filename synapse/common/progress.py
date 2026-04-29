from __future__ import annotations

from typing import Iterable


def iter_progress(iterable: Iterable, desc: str | None = None):
    try:
        from tqdm import tqdm

        return tqdm(iterable, desc=desc)
    except Exception:
        return iterable
