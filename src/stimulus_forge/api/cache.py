"""In-memory caching layer for related concept generation."""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from typing import Any


class ConceptCache:
    """Thread-safe LRU cache with time-to-live expiration."""

    def __init__(self, maxsize: int = 256, ttl_seconds: float = 3600.0):
        self._maxsize = maxsize
        self._ttl = ttl_seconds
        self._lock = threading.Lock()
        self._cache: OrderedDict[str, tuple[float, list[dict[str, str]]]] = OrderedDict()

    def _normalize_key(self, concept: str, count: int, context: str = "") -> str:
        clean = " ".join(concept.casefold().strip().split())
        clean_ctx = " ".join(context.casefold().strip().split())
        return f"{clean}::{clean_ctx}::{count}"

    def get(self, concept: str, count: int, context: str = "") -> list[dict[str, str]] | None:
        key = self._normalize_key(concept, count, context)
        with self._lock:
            if key not in self._cache:
                return None
            inserted_at, value = self._cache[key]
            if time.time() - inserted_at > self._ttl:
                del self._cache[key]
                return None
            # Move to end (most recently used)
            self._cache.move_to_end(key)
            return [dict(item) for item in value]

    def set(self, concept: str, count: int, concepts: list[dict[str, str]], context: str = "") -> None:
        key = self._normalize_key(concept, count, context)
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
            self._cache[key] = (time.time(), [dict(item) for item in concepts])
            if len(self._cache) > self._maxsize:
                self._cache.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()

    def size(self) -> int:
        with self._lock:
            return len(self._cache)


# Global singleton cache instance
concept_cache = ConceptCache()
