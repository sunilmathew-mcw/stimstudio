import unittest
from unittest.mock import MagicMock, patch

from stimulus_forge.api.cache import ConceptCache
from stimulus_forge.api.schemas import (
    ConceptGenerateRequest,
    ConceptItem,
)
from stimulus_forge.api.server import create_app


class ApiServerAndRoutesTest(unittest.TestCase):
    def setUp(self) -> None:
        self.app = create_app()

    def test_routes_mounted_correctly(self) -> None:
        route_paths = []
        for r in self.app.routes:
            if hasattr(r, "path") and r.path:
                route_paths.append(r.path)
            elif hasattr(r, "original_router"):
                prefix = getattr(getattr(r, "include_context", None), "prefix", "")
                for sub in getattr(r.original_router, "routes", []):
                    sub_path = getattr(sub, "path", "")
                    full = f"{prefix}{sub_path}" if not sub_path.startswith(prefix) else sub_path
                    route_paths.append(full)
        expected_paths = [
            "/api/v1/health",
            "/api/v1/health/diagnostics",
            "/api/v1/providers",
            "/api/v1/concepts/generate",
            "/api/v1/concepts/cache/clear",
            "/api/v1/concepts/batch",
            "/api/v1/images/search",
            "/api/v1/images/generate",
            "/api/v1/images/file",
            "/api/v1/images/crop-face",
            "/api/v1/images/detect-face",
            "/api/v1/pipeline/batch",
            "/api/v1/pipeline/jobs",
            "/api/v1/telemetry/stats",
            "/api/v1/telemetry/reset",
            "/api/v1/system/browse-folder",
            "/api/v1/system/validate-folder",
            "/api/v1/system/config-paths",
        ]
        for path in expected_paths:
            self.assertIn(path, route_paths, f"Expected route {path} to be registered")

    def test_concept_cache_lru_and_expiration(self) -> None:
        cache = ConceptCache(maxsize=2, ttl_seconds=60.0)
        cache.set("apple", 5, [{"name": "iPhone", "char": "", "description": "Smartphone"}])
        cache.set("banana", 5, [{"name": "Fruit", "char": "", "description": "Yellow fruit"}])

        self.assertEqual(cache.size(), 2)
        cached_apple = cache.get("apple", 5)
        self.assertIsNotNone(cached_apple)
        self.assertEqual(cached_apple[0]["name"], "iPhone")

        # Exceed maxsize
        cache.set("cherry", 5, [{"name": "Pie", "char": "", "description": "Dessert"}])
        self.assertEqual(cache.size(), 2)
        # Banana should have been evicted (apple was accessed recently)
        self.assertIsNone(cache.get("banana", 5))
        self.assertIsNotNone(cache.get("cherry", 5))

    def test_clear_concept_cache_endpoint(self) -> None:
        from stimulus_forge.api.cache import concept_cache
        from stimulus_forge.api.routes.concepts import clear_concept_cache

        concept_cache.set("basketball", 5, [{"name": "Michael Jordan", "char": "", "description": "NBA legend"}])
        self.assertGreaterEqual(concept_cache.size(), 1)
        result = clear_concept_cache()
        self.assertEqual(result["status"], "success")
        self.assertGreaterEqual(result["cleared_entries"], 1)
        self.assertEqual(concept_cache.size(), 0)

    def test_app_metadata(self) -> None:
        self.assertEqual(self.app.title, "Stimulus Forge API")
        self.assertEqual(self.app.version, "0.1.0")

    def test_folder_validation_and_config_paths(self) -> None:
        import tempfile
        from stimulus_forge.api.routes.system import get_config_paths, validate_folder_path

        # 1. Config paths
        config = get_config_paths()
        self.assertIn("default_image_dir", config)
        self.assertIn("current_image_dir", config)

        # 2. Existing folder
        with tempfile.TemporaryDirectory() as tmp_dir:
            res_valid = validate_folder_path({"path": tmp_dir})
            self.assertTrue(res_valid["valid"])
            self.assertTrue(res_valid["exists"])

        # 3. Empty path
        res_empty = validate_folder_path({"path": "  "})
        self.assertFalse(res_empty["valid"])
        self.assertFalse(res_empty["exists"])

    def test_list_providers_includes_google_available_models(self) -> None:
        from stimulus_forge.api.routes.providers import list_providers

        providers = list_providers()
        google_p = next((p for p in providers if p.id == "google"), None)
        self.assertIsNotNone(google_p)
        self.assertIn("gemini-3.8-flash", google_p.available_models)
        self.assertIn("gemini-3.8-pro", google_p.available_models)
        self.assertIn("gemini-2.5-flash", google_p.available_models)


if __name__ == "__main__":
    unittest.main()
