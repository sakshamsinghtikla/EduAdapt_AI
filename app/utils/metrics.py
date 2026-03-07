from __future__ import annotations

from collections import defaultdict
from time import perf_counter
from typing import Callable


class MetricsStore:
    def __init__(self) -> None:
        self.endpoint_stats: dict[str, dict[str, float]] = defaultdict(
            lambda: {"count": 0.0, "total_ms": 0.0, "max_ms": 0.0}
        )
        self.events_processed = 0

    def record(self, endpoint: str, duration_ms: float) -> None:
        stats = self.endpoint_stats[endpoint]
        stats["count"] += 1
        stats["total_ms"] += duration_ms
        stats["max_ms"] = max(stats["max_ms"], duration_ms)

    def snapshot(self) -> dict:
        result = {}
        for name, stats in self.endpoint_stats.items():
            count = max(stats["count"], 1.0)
            result[name] = {
                "count": int(stats["count"]),
                "avg_ms": round(stats["total_ms"] / count, 3),
                "max_ms": round(stats["max_ms"], 3),
            }
        result["events_processed"] = self.events_processed
        return result


metrics_store = MetricsStore()


def timed(endpoint: str) -> Callable:
    def decorator(func: Callable) -> Callable:
        async def wrapper(*args, **kwargs):
            start = perf_counter()
            response = await func(*args, **kwargs)
            duration_ms = (perf_counter() - start) * 1000
            metrics_store.record(endpoint, duration_ms)
            return response

        return wrapper

    return decorator
