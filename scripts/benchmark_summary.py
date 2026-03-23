from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNTIME_DIR = PROJECT_ROOT / "app" / "data" / "runtime"
BENCHMARK_SUMMARY_PATH = RUNTIME_DIR / "benchmark_summary.json"


def _safe_float(value: str | None, default: float = 0.0) -> float:
    try:
        return float(value) if value not in (None, "") else default
    except Exception:
        return default


def _safe_int(value: str | None, default: int = 0) -> int:
    try:
        return int(float(value)) if value not in (None, "") else default
    except Exception:
        return default


def build_summary(stats_csv_path: Path) -> dict[str, Any]:
    if not stats_csv_path.exists():
        raise FileNotFoundError(f"Stats CSV not found: {stats_csv_path}")

    rows = []
    with stats_csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    endpoint_rows = [row for row in rows if row.get("Type") in ("GET", "POST")]

    total_requests = sum(_safe_int(row.get("Request Count")) for row in endpoint_rows)
    total_failures = sum(_safe_int(row.get("Failure Count")) for row in endpoint_rows)

    weighted_avg_latency_numerator = sum(
        _safe_float(row.get("Average Response Time")) * _safe_int(row.get("Request Count"))
        for row in endpoint_rows
    )
    weighted_avg_latency = (
        weighted_avg_latency_numerator / total_requests if total_requests > 0 else 0.0
    )

    max_p95 = max(
        (_safe_float(row.get("95%")) for row in endpoint_rows),
        default=0.0,
    )
    max_rps = max(
        (_safe_float(row.get("Requests/s")) for row in endpoint_rows),
        default=0.0,
    )

    per_endpoint = []
    for row in endpoint_rows:
        per_endpoint.append(
            {
                "method": row.get("Type"),
                "name": row.get("Name"),
                "request_count": _safe_int(row.get("Request Count")),
                "failure_count": _safe_int(row.get("Failure Count")),
                "avg_response_time_ms": round(_safe_float(row.get("Average Response Time")), 3),
                "min_response_time_ms": round(_safe_float(row.get("Min Response Time")), 3),
                "max_response_time_ms": round(_safe_float(row.get("Max Response Time")), 3),
                "p95_response_time_ms": round(_safe_float(row.get("95%")), 3),
                "requests_per_sec": round(_safe_float(row.get("Requests/s")), 3),
            }
        )

    summary = {
        "total_requests": total_requests,
        "total_failures": total_failures,
        "error_rate": round((total_failures / total_requests), 4) if total_requests > 0 else 0.0,
        "weighted_avg_latency_ms": round(weighted_avg_latency, 3),
        "max_p95_latency_ms": round(max_p95, 3),
        "max_requests_per_sec": round(max_rps, 3),
        "per_endpoint": per_endpoint,
        "interpretation": _interpret(
            weighted_avg_latency=weighted_avg_latency,
            max_p95=max_p95,
            total_failures=total_failures,
        ),
    }

    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    BENCHMARK_SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _interpret(weighted_avg_latency: float, max_p95: float, total_failures: int) -> str:
    parts = []

    if weighted_avg_latency <= 150:
        parts.append("Average latency is strong for an interactive adaptive system.")
    elif weighted_avg_latency <= 300:
        parts.append("Average latency is acceptable under load.")
    else:
        parts.append("Average latency is high and may need optimization.")

    if max_p95 <= 400:
        parts.append("Tail latency remains under good control.")
    elif max_p95 <= 800:
        parts.append("Tail latency is moderate under concurrency.")
    else:
        parts.append("Tail latency is high and indicates scaling pressure.")

    if total_failures == 0:
        parts.append("No request failures were observed during the benchmark.")
    else:
        parts.append("Some request failures were observed and should be investigated.")

    return " ".join(parts)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        raise SystemExit("Usage: python scripts/benchmark_summary.py <locust_stats_csv_path>")

    csv_path = Path(sys.argv[1])
    result = build_summary(csv_path)
    print(json.dumps(result, indent=2))
