"""FinOps and observability telemetry engine for Stimulus Forge."""

from __future__ import annotations

import threading
import time
from datetime import datetime, timezone
from typing import Any

from stimulus_forge.api.schemas import TelemetryMetric, TelemetrySummary


# Standard pricing benchmarks (per 1M tokens in USD)
MODEL_PRICING: dict[str, dict[str, float]] = {
    # OpenAI
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-5.6-luna": {"input": 2.50, "output": 10.00},
    # Anthropic
    "claude-3-5-sonnet": {"input": 3.00, "output": 15.00},
    "claude-3-5-haiku": {"input": 0.80, "output": 4.00},
    # Google
    "gemini-1.5-flash": {"input": 0.075, "output": 0.30},
    "gemini-2.0-flash": {"input": 0.10, "output": 0.40},
    "gemini-3.8-flash": {"input": 0.10, "output": 0.40},
    "gemini-1.5-pro": {"input": 1.25, "output": 5.00},
    # Local & self-hosted
    "local": {"input": 0.0, "output": 0.0},
    "compatible": {"input": 0.20, "output": 0.20},  # generic hosted Llama
}

IMAGE_PRICING: dict[str, float] = {
    "dall-e-3": 0.040,
    "dall-e-2": 0.020,
    "local": 0.0,
    "bing": 0.0,
}

# Baseline cloud model used to calculate cost avoidance (savings) when using local or cache
BASELINE_CLOUD_MODEL = "gpt-4o"


class TelemetryTracker:
    """Thread-safe telemetry and FinOps metrics collector."""

    def __init__(self, max_history: int = 500):
        self._lock = threading.Lock()
        self._max_history = max_history
        self._metrics: list[TelemetryMetric] = []
        self._cache_hits = 0
        self._cache_misses = 0

    def estimate_cost(
        self,
        provider: str,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
    ) -> float:
        """Estimate the USD cost of a token-based generation call."""
        if provider == "local":
            return 0.0

        key = model.lower()
        pricing = None
        for model_prefix, rate in MODEL_PRICING.items():
            if model_prefix in key:
                pricing = rate
                break

        if pricing is None:
            pricing = MODEL_PRICING.get(provider, {"input": 1.0, "output": 2.0})

        cost_in = (prompt_tokens / 1_000_000.0) * pricing["input"]
        cost_out = (completion_tokens / 1_000_000.0) * pricing["output"]
        return round(cost_in + cost_out, 6)

    def record_concept_generation(
        self,
        provider: str,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        latency_ms: float,
        cached: bool = False,
    ) -> TelemetryMetric:
        """Record an inference event and update running counters."""
        estimated_cost = (
            0.0
            if cached
            else self.estimate_cost(provider, model, prompt_tokens, completion_tokens)
        )

        metric = TelemetryMetric(
            timestamp=datetime.now(timezone.utc),
            provider=provider,
            model=model,
            task_type="concept_generation",
            latency_ms=round(latency_ms, 2),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            estimated_cost_usd=estimated_cost,
            cached=cached,
        )

        with self._lock:
            if cached:
                self._cache_hits += 1
            else:
                self._cache_misses += 1

            self._metrics.append(metric)
            if len(self._metrics) > self._max_history:
                self._metrics.pop(0)

        return metric

    def get_summary(self) -> TelemetrySummary:
        """Compute aggregated operational metrics and cost avoidance."""
        with self._lock:
            metrics = list(self._metrics)
            cache_hits = self._cache_hits
            cache_misses = self._cache_misses

        total_requests = len(metrics)
        total_cache_events = cache_hits + cache_misses
        hit_ratio = (
            round(cache_hits / total_cache_events, 3) if total_cache_events > 0 else 0.0
        )

        if not metrics:
            return TelemetrySummary(
                total_requests=0,
                cache_hits=cache_hits,
                cache_misses=cache_misses,
                cache_hit_ratio=hit_ratio,
                total_prompt_tokens=0,
                total_completion_tokens=0,
                total_cost_usd=0.0,
                total_cost_avoided_usd=0.0,
                avg_latency_ms=0.0,
                p95_latency_ms=0.0,
                requests_by_provider={},
                cost_by_provider={},
            )

        total_prompt_tokens = sum(m.prompt_tokens for m in metrics)
        total_completion_tokens = sum(m.completion_tokens for m in metrics)
        total_cost = round(sum(m.estimated_cost_usd for m in metrics), 4)

        # Calculate cost avoided (what would have been spent if all calls ran on baseline cloud model)
        baseline = MODEL_PRICING[BASELINE_CLOUD_MODEL]
        hypothetical_cloud_cost = sum(
            (m.prompt_tokens / 1_000_000.0) * baseline["input"]
            + (m.completion_tokens / 1_000_000.0) * baseline["output"]
            for m in metrics
        )
        cost_avoided = max(0.0, round(hypothetical_cloud_cost - total_cost, 4))

        latencies = sorted(m.latency_ms for m in metrics)
        avg_latency = round(sum(latencies) / len(latencies), 2)
        p95_idx = int(len(latencies) * 0.95)
        p95_latency = round(latencies[min(p95_idx, len(latencies) - 1)], 2)

        req_by_provider: dict[str, int] = {}
        cost_by_provider: dict[str, float] = {}
        for m in metrics:
            req_by_provider[m.provider] = req_by_provider.get(m.provider, 0) + 1
            cost_by_provider[m.provider] = round(
                cost_by_provider.get(m.provider, 0.0) + m.estimated_cost_usd, 4
            )

        return TelemetrySummary(
            total_requests=total_requests,
            cache_hits=cache_hits,
            cache_misses=cache_misses,
            cache_hit_ratio=hit_ratio,
            total_prompt_tokens=total_prompt_tokens,
            total_completion_tokens=total_completion_tokens,
            total_cost_usd=total_cost,
            total_cost_avoided_usd=cost_avoided,
            avg_latency_ms=avg_latency,
            p95_latency_ms=p95_latency,
            requests_by_provider=req_by_provider,
            cost_by_provider=cost_by_provider,
        )

    def reset_cache_stats(self) -> None:
        """Reset cache hit/miss counters when cache is cleared."""
        with self._lock:
            self._cache_hits = 0
            self._cache_misses = 0

    def reset(self) -> None:
        """Reset telemetry counters."""
        with self._lock:
            self._metrics.clear()
            self._cache_hits = 0
            self._cache_misses = 0


# Global singleton instance for the process
telemetry = TelemetryTracker()
