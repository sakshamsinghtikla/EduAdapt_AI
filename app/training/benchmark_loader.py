from __future__ import annotations

import json
from pathlib import Path
from typing import Any


RUNTIME_DIR = Path(__file__).resolve().parents[1] / "data" / "runtime"
BENCHMARK_SUMMARY_PATH = RUNTIME_DIR / "benchmark_summary.json"


class BenchmarkLoader:
    def load_summary(self) -> dict[str, Any]:
        if not BENCHMARK_SUMMARY_PATH.exists():
            return {}
        return json.loads(BENCHMARK_SUMMARY_PATH.read_text(encoding="utf-8"))


benchmark_loader = BenchmarkLoader()
