import unittest
from stimulus_forge.api.telemetry import TelemetryTracker, MODEL_PRICING


class TelemetryTrackerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tracker = TelemetryTracker(max_history=50)

    def test_local_model_has_zero_cost(self) -> None:
        cost = self.tracker.estimate_cost("local", "qwen-3-4b-gguf", 300, 100)
        self.assertEqual(cost, 0.0)

    def test_cloud_model_cost_estimation(self) -> None:
        # GPT-4o: $2.50 input / 1M, $10.00 output / 1M
        # 10,000 prompt tokens = $0.025, 1,000 completion tokens = $0.01 -> total $0.035
        cost = self.tracker.estimate_cost("openai", "gpt-4o", 10_000, 1_000)
        self.assertAlmostEqual(cost, 0.035, places=4)

    def test_record_and_summary_aggregation(self) -> None:
        self.tracker.record_concept_generation(
            provider="local",
            model="qwen-gguf",
            prompt_tokens=280,
            completion_tokens=50,
            latency_ms=120.0,
            cached=False,
        )
        self.tracker.record_concept_generation(
            provider="local",
            model="qwen-gguf",
            prompt_tokens=0,
            completion_tokens=0,
            latency_ms=1.5,
            cached=True,
        )

        summary = self.tracker.get_summary()
        self.assertEqual(summary.total_requests, 2)
        self.assertEqual(summary.cache_hits, 1)
        self.assertEqual(summary.cache_misses, 1)
        self.assertEqual(summary.cache_hit_ratio, 0.5)
        self.assertEqual(summary.total_cost_usd, 0.0)
        # Cost avoided should be greater than zero because local inference saved cloud API budget
        self.assertGreater(summary.total_cost_avoided_usd, 0.0)

    def test_reset_clears_metrics(self) -> None:
        self.tracker.record_concept_generation(
            provider="openai",
            model="gpt-4o",
            prompt_tokens=500,
            completion_tokens=100,
            latency_ms=800.0,
            cached=False,
        )
        self.tracker.reset()
        summary = self.tracker.get_summary()
        self.assertEqual(summary.total_requests, 0)
        self.assertEqual(summary.cache_hits, 0)


if __name__ == "__main__":
    unittest.main()
