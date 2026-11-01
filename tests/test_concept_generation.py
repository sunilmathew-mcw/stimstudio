import unittest
from types import SimpleNamespace

from stimulus_forge.services.concept_generation import (
    AnthropicRelatedConceptBackend,
    CompatibleRelatedConceptBackend,
    GoogleRelatedConceptBackend,
    OpenAIRelatedConceptBackend,
    RelatedConcept,
    RelatedConceptCollection,
    RelatedConceptGenerator,
    normalize_related_concepts,
)


class _FakeResponses:
    def __init__(self, parsed):
        self.parsed = parsed
        self.request = None

    def parse(self, **kwargs):
        self.request = kwargs
        return SimpleNamespace(output_parsed=self.parsed)


class ConceptGenerationTests(unittest.TestCase):
    def test_local_is_the_default_provider(self) -> None:
        generator = RelatedConceptGenerator()
        self.assertEqual(generator.selected_provider, "local")
        self.assertEqual(generator.local_gpu_layers, -1)

    def test_auto_provider_is_no_longer_accepted(self) -> None:
        with self.assertRaises(ValueError):
            RelatedConceptGenerator(provider="auto")

    def test_normalizes_deduplicates_and_excludes_parent(self) -> None:
        concepts = normalize_related_concepts(
            [
                {"name": " Basketball ", "char": "", "description": "same"},
                {"name": "Michael  Jordan", "description": "NBA player"},
                {"name": "michael jordan", "char": "", "description": "duplicate"},
                {"name": "Wilson", "char": "Basketball", "description": "duplicate parent"},
                {"name": "NBA", "char": "", "description": "  Pro league  "},
            ],
            "basketball",
        )

        self.assertEqual(
            concepts,
            [
                {"name": "Michael Jordan", "char": "", "description": "NBA player"},
                {"name": "NBA", "char": "", "description": "Pro league"},
            ],
        )

    def test_openai_backend_uses_structured_outputs(self) -> None:
        parsed = RelatedConceptCollection(
            concepts=[
                RelatedConcept(
                    name="Michael Jordan",
                    char="",
                    description="Chicago Bulls legend",
                )
            ]
        )
        responses = _FakeResponses(parsed)
        client = SimpleNamespace(responses=responses)

        values = OpenAIRelatedConceptBackend(client, "test-model").generate(
            "basketball", count=1
        )

        self.assertEqual(values[0]["name"], "Michael Jordan")
        self.assertEqual(responses.request["text_format"], RelatedConceptCollection)
        self.assertIn("exactly 1", responses.request["input"][1]["content"])

    def test_openai_backend_passes_category_context_in_prompt(self) -> None:
        from stimulus_forge.services.concept_generation import concept_user_prompt
        prompt_with_ctx = concept_user_prompt("entourage", 5, context="TV Shows (Other): Entourage")
        self.assertIn('Concept: "entourage"', prompt_with_ctx)
        self.assertIn('Context / Category: "TV Shows (Other): Entourage"', prompt_with_ctx)

        parsed = RelatedConceptCollection(
            concepts=[
                RelatedConcept(
                    name="Jeremy Piven",
                    char="Ari Gold",
                    description="Super-agent in Entourage",
                )
            ]
        )
        responses = _FakeResponses(parsed)
        client = SimpleNamespace(responses=responses)
        values = OpenAIRelatedConceptBackend(client, "test-model").generate(
            "entourage", count=1, context="TV Shows (Other): Entourage"
        )
        self.assertEqual(values[0]["char"], "Ari Gold")
        self.assertIn('Context / Category: "TV Shows (Other): Entourage"', responses.request["input"][1]["content"])

    def test_anthropic_backend_parses_schema_validated_json(self) -> None:
        request = {}
        def create(**kwargs):
            request.update(kwargs)
            return SimpleNamespace(content=[SimpleNamespace(
                type='text',
                text='{"concepts":[{"name":"NBA","char":"","description":"Pro league"}]}',
            )])
        client = SimpleNamespace(messages=SimpleNamespace(create=create))
        values = AnthropicRelatedConceptBackend(client, 'claude-test').generate(
            'basketball', 1
        )
        self.assertEqual(values[0]['name'], 'NBA')
        self.assertEqual(request['model'], 'claude-test')

    def test_google_backend_requests_json_schema(self) -> None:
        request = {}
        def create(**kwargs):
            request.update(kwargs)
            return SimpleNamespace(output_text=(
                '{"concepts":[{"name":"NBA","char":"","description":"Pro league"}]}'
            ))
        client = SimpleNamespace(interactions=SimpleNamespace(create=create))
        values = GoogleRelatedConceptBackend(client, 'gemini-test').generate(
            'basketball', 1
        )
        self.assertEqual(values[0]['name'], 'NBA')
        self.assertEqual(request['response_format']['mime_type'], 'application/json')

    def test_google_backend_auto_upgrades_deprecated_model(self) -> None:
        request = {}
        def create(**kwargs):
            request.update(kwargs)
            return SimpleNamespace(output_text=(
                '{"concepts":[{"name":"NBA","char":"","description":"Pro league"}]}'
            ))
        client = SimpleNamespace(interactions=SimpleNamespace(create=create))
        backend = GoogleRelatedConceptBackend(client, 'gemini-2.0-flash')
        self.assertEqual(backend.model, 'gemini-3.8-flash')
        values = backend.generate('basketball', 1)
        self.assertEqual(request['model'], 'gemini-3.8-flash')
        self.assertEqual(values[0]['name'], 'NBA')

    def test_google_backend_falls_back_on_404_deprecation_error(self) -> None:
        calls = []
        def create(**kwargs):
            calls.append(kwargs)
            if len(calls) == 1:
                raise RuntimeError(
                    "Error code: 404 - {'error': {'message': 'This model models/gemini-2.5-preview is no longer available. Please update your code to use models/gemini-3.8-flash for the latest features and improvements.', 'code': 'not_found'}}"
                )
            return SimpleNamespace(output_text=(
                '{"concepts":[{"name":"NBA","char":"","description":"Pro league"}]}'
            ))
        client = SimpleNamespace(interactions=SimpleNamespace(create=create))
        backend = GoogleRelatedConceptBackend(client, 'gemini-2.5-preview')
        values = backend.generate('basketball', 1)
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0]['model'], 'gemini-2.5-preview')
        self.assertEqual(calls[1]['model'], 'gemini-3.8-flash')
        self.assertEqual(values[0]['name'], 'NBA')

    def test_openai_compatible_backend_supports_hosted_llama(self) -> None:
        request = {}
        def create(**kwargs):
            request.update(kwargs)
            message = SimpleNamespace(content=(
                '```json\n{"concepts":[{"name":"NBA","char":"","description":"Pro league"}]}\n```'
            ))
            return SimpleNamespace(choices=[SimpleNamespace(message=message)])
        client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create))
        )
        values = CompatibleRelatedConceptBackend(client, 'llama-test').generate(
            'basketball', 1
        )
        self.assertEqual(values[0]['name'], 'NBA')
        self.assertEqual(request['model'], 'llama-test')

    def test_concept_cache_differentiates_context(self) -> None:
        from stimulus_forge.api.cache import ConceptCache
        cache = ConceptCache()
        cache.set("entourage", 5, [{"name": "bodyguard", "char": "", "description": "definition"}])
        cache.set("entourage", 5, [{"name": "Jeremy Piven", "char": "Ari Gold", "description": "agent"}], context="TV Shows (Other): Entourage")

        generic = cache.get("entourage", 5)
        show_specific = cache.get("entourage", 5, context="TV Shows (Other): Entourage")
        self.assertIsNotNone(generic)
        self.assertIsNotNone(show_specific)
        self.assertEqual(generic[0]["name"], "bodyguard")
        self.assertEqual(show_specific[0]["name"], "Jeremy Piven")


if __name__ == "__main__":
    unittest.main()
