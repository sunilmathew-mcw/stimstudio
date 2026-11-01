import unittest
from datetime import datetime, timezone
from pydantic import ValidationError

from stimulus_forge.api.schemas import (
    ConceptGenerateRequest,
    ConceptGenerateResponse,
    ConceptItem,
    ImageSearchRequest,
    ImageGenerateRequest,
    PipelineBatchRequest,
    PipelineJobStatus,
    ProviderInfo,
    TelemetrySummary,
)


class ApiSchemasTest(unittest.TestCase):
    def test_concept_item_validation(self) -> None:
        item = ConceptItem(
            name="Michael Jordan",
            char="Bulls Legend",
            description="Legendary basketball player",
        )
        self.assertEqual(item.name, "Michael Jordan")
        self.assertEqual(item.char, "Bulls Legend")
        self.assertEqual(item.description, "Legendary basketball player")

        # Test extra fields forbidden
        with self.assertRaises(ValidationError):
            ConceptItem(name="Test", extra_key="invalid")

    def test_concept_generate_request_validation(self) -> None:
        req = ConceptGenerateRequest(
            concept="basketball",
            count=10,
            provider="openai",
            use_cache=True,
            context="Sports: Basketball",
        )
        self.assertEqual(req.concept, "basketball")
        self.assertEqual(req.count, 10)
        self.assertEqual(req.provider, "openai")
        self.assertEqual(req.context, "Sports: Basketball")

        # Count bounds
        with self.assertRaises(ValidationError):
            ConceptGenerateRequest(concept="basketball", count=0)
        with self.assertRaises(ValidationError):
            ConceptGenerateRequest(concept="basketball", count=100)

        # Invalid provider
        with self.assertRaises(ValidationError):
            ConceptGenerateRequest(concept="basketball", provider="unsupported_provider")

    def test_concept_generate_response_serialization(self) -> None:
        resp = ConceptGenerateResponse(
            concept="tennis",
            provider="local",
            model="qwen-gguf",
            concepts=[
                ConceptItem(name="Racket", char="", description="Tennis equipment"),
                ConceptItem(name="Wimbledon", char="", description="Grand Slam tournament"),
            ],
            cached=False,
            latency_ms=145.2,
            prompt_tokens=280,
            completion_tokens=65,
            estimated_cost_usd=0.0,
        )
        data = resp.model_dump()
        self.assertEqual(len(data["concepts"]), 2)
        self.assertEqual(data["latency_ms"], 145.2)
        self.assertEqual(data["estimated_cost_usd"], 0.0)

    def test_image_search_request_validation(self) -> None:
        req = ImageSearchRequest(
            concept="apple",
            limit=5,
            safe_search="strict",
            output_dir="/tmp/test_images",
            context="Fruit: Apple",
        )
        self.assertEqual(req.concept, "apple")
        self.assertEqual(req.limit, 5)
        self.assertEqual(req.output_dir, "/tmp/test_images")
        self.assertEqual(req.max_faces, 5)
        self.assertEqual(req.context, "Fruit: Apple")

        with self.assertRaises(ValidationError):
            ImageSearchRequest(concept="apple", limit=50)

        with self.assertRaises(ValidationError):
            ImageSearchRequest(concept="apple", max_faces=15)

    def test_image_generate_request_validation(self) -> None:
        req = ImageGenerateRequest(prompt="basketball court", count=2, provider="openai", output_dir="/tmp/custom_dir")
        self.assertEqual(req.prompt, "basketball court")
        self.assertEqual(req.count, 2)
        self.assertEqual(req.output_dir, "/tmp/custom_dir")
        self.assertEqual(req.max_faces, 5)

        # Text stimulus provider
        req_text = ImageGenerateRequest(
            prompt="basketball",
            concept="basketball",
            provider="text",
            text_bg_color="#000000",
            text_color="#FFFFFF",
            count=1,
        )
        self.assertEqual(req_text.provider, "text")
        self.assertEqual(req_text.text_bg_color, "#000000")
        self.assertEqual(req_text.text_color, "#FFFFFF")

    def test_pipeline_batch_request_validation(self) -> None:
        req = PipelineBatchRequest(
            concepts=["basketball", "tennis"],
            concepts_per_seed=5,
            images_per_concept=3,
            text_provider="local",
            image_source="text",
            output_dir="/tmp/pipeline_out",
        )
        self.assertEqual(len(req.concepts), 2)
        self.assertEqual(req.concepts_per_seed, 5)
        self.assertEqual(req.image_source, "text")
        self.assertEqual(req.output_dir, "/tmp/pipeline_out")
        self.assertEqual(req.max_faces, 5)

    def test_telemetry_summary_defaults(self) -> None:
        summary = TelemetrySummary(
            total_requests=10,
            cache_hits=4,
            cache_misses=6,
            cache_hit_ratio=0.4,
            total_prompt_tokens=2000,
            total_completion_tokens=500,
            total_cost_usd=0.005,
            total_cost_avoided_usd=0.025,
            avg_latency_ms=150.0,
            p95_latency_ms=250.0,
            requests_by_provider={"local": 6, "openai": 4},
            cost_by_provider={"local": 0.0, "openai": 0.005},
        )
        self.assertEqual(summary.total_requests, 10)
        self.assertEqual(summary.cache_hit_ratio, 0.4)
        self.assertEqual(summary.total_cost_avoided_usd, 0.025)


if __name__ == "__main__":
    unittest.main()
