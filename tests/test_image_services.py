import base64
from io import BytesIO
import json
import os
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch, MagicMock
import urllib.parse

from PIL import Image

from stimulus_forge.services.bing_image_search import BingDownloader, build_bing_query
from stimulus_forge.services.google_images import GoogleImageService
from stimulus_forge.services.huggingface_images import HuggingFaceImageService
from stimulus_forge.services.local_images import LocalImageService
from stimulus_forge.services.openai_images import (
    OpenAIImageService,
    build_stimulus_image_prompt,
    image_name_stem,
)
from stimulus_forge.services.text_images import (
    TextImageService,
    render_centered_text_image,
)


def _png_bytes() -> bytes:
    output = BytesIO()
    Image.new("RGB", (200, 200), "orange").save(output, format="PNG")
    return output.getvalue()


class _FakeImageClient:
    def __init__(self):
        self.images = SimpleNamespace(generate=self.generate)
        self.moderations = SimpleNamespace(create=self.moderate)
        self.image_request = None
        self.moderation_result = False

    def generate(self, **kwargs):
        self.image_request = kwargs
        data = [
            SimpleNamespace(b64_json=base64.b64encode(_png_bytes()).decode("ascii"))
            for _ in range(kwargs["n"])
        ]
        return SimpleNamespace(data=data)

    def moderate(self, **kwargs):
        del kwargs
        categories = SimpleNamespace(
            model_dump=lambda **_: {"sexual": self.moderation_result}
        )
        return SimpleNamespace(results=[SimpleNamespace(categories=categories)])


class _FakeGoogleImageClient:
    def __init__(self):
        self.models = SimpleNamespace(generate_images=self.generate_images)
        self.last_request = None

    def generate_images(self, **kwargs):
        self.last_request = kwargs
        n = kwargs.get("config", {}).get("number_of_images", 1)
        generated_images = [
            SimpleNamespace(image=SimpleNamespace(image_bytes=_png_bytes()))
            for _ in range(n)
        ]
        return SimpleNamespace(generated_images=generated_images)


class _FakeHuggingFaceClient:
    def __init__(self):
        self.last_request = None

    def text_to_image(self, prompt, **kwargs):
        self.last_request = {"prompt": prompt, **kwargs}
        return Image.new("RGB", (256, 256), "green")


class _FakeSearchResponse:
    def __init__(self, data=b'murl&quot;:&quot;https://example.com/basketball.jpg&quot;'):
        self._data = data

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self):
        return self._data


class _FakeLocalPipeline:
    def __init__(self):
        self.requests = []

    def __call__(self, **kwargs):
        self.requests.append(kwargs)
        return SimpleNamespace(images=[Image.new("RGB", (128, 128), "blue")])


class ImageServiceTests(unittest.TestCase):
    def test_bing_query_targets_character_and_performer_without_parent_concept(self) -> None:
        self.assertEqual(
            build_bing_query("the office", "Michael Scott", "Steve Carell"),
            "Michael Scott Steve Carell",
        )

    def test_bing_query_targets_association_without_parent_concept(self) -> None:
        self.assertEqual(
            build_bing_query("thanksgiving", "", "Lincoln Financial Field"),
            "Lincoln Financial Field",
        )

    def test_bing_query_deduplicates_repeated_context(self) -> None:
        self.assertEqual(
            build_bing_query("basketball", "NBA", "nba"),
            "NBA",
        )

    def test_bing_query_uses_concept_when_no_association(self) -> None:
        self.assertEqual(
            build_bing_query("basketball"),
            "basketball",
        )

    def test_bing_query_disambiguates_concept_with_context(self) -> None:
        # Resolves show ambiguity (e.g. Entourage show vs literal entourage)
        self.assertEqual(
            build_bing_query("entourage", context="TV Shows (Other): Entourage"),
            "entourage tv show",
        )
        self.assertEqual(
            build_bing_query("packers", context="Sports (Other): packers"),
            "packers sports",
        )
        self.assertEqual(
            build_bing_query("nirvana", context="Music (Other): nirvana"),
            "nirvana band",
        )
        # Does not append redundant hints when specific association performer/role exists
        self.assertEqual(
            build_bing_query("entourage", "ari gold", "jeremy piven", context="TV Shows (Other): Entourage"),
            "ari gold jeremy piven",
        )

    def test_bing_query_includes_parent_concept_for_generic_association(self) -> None:
        self.assertEqual(
            build_bing_query("basketball", "dunk", ""),
            "basketball dunk",
        )
        self.assertEqual(
            build_bing_query("basketball", "hoops", ""),
            "basketball hoops",
        )
        self.assertEqual(
            build_bing_query("electric guitar", "amplifier", ""),
            "electric guitar amplifier",
        )

    def test_bing_query_contextualizes_topic_associations(self) -> None:
        self.assertEqual(
            build_bing_query("Albert Einstein", "theory of relativity", ""),
            "Albert Einstein theory of relativity",
        )
        self.assertEqual(
            build_bing_query("Albert Einstein", "einsteins brain", ""),
            "Albert Einstein einsteins brain",
        )

    def test_bing_query_keeps_person_context(self) -> None:
        self.assertEqual(build_bing_query("Michael Jordan"), "Michael Jordan")
        self.assertEqual(
            build_bing_query("Michael Jordan", "slam dunk", ""),
            "Michael Jordan slam dunk",
        )
        self.assertEqual(
            build_bing_query("Michael Jordan", "Chicago Bulls", ""),
            "Michael Jordan Chicago Bulls",
        )


    def test_downloader_passes_association_query_into_bing_query(self) -> None:
        with patch.dict(sys.modules, {'cv2': SimpleNamespace()}):
            from stimulus_forge.services.image_workspace import TaskImagesDownloader

            with tempfile.TemporaryDirectory() as directory:
                with patch(
                    'stimulus_forge.services.image_workspace.BingDownloader'
                ) as downloader:
                    downloader.return_value.run.return_value = []
                    TaskImagesDownloader().download(
                        concept='electric guitar',
                        related_concept='Eddie Van Halen',
                        name='Edward Van Halen',
                        limit=1,
                        output_dir=directory,
                        verbose=False,
                    )
            self.assertEqual(
                downloader.call_args.kwargs['query'],
                'Eddie Van Halen Edward Van Halen',
            )

    def test_image_prompt_and_filename_are_stimulus_specific(self) -> None:
        prompt = build_stimulus_image_prompt(
            "the office",
            "Michael Scott",
            "Steve Carell",
            "regional manager",
        )
        self.assertIn("single centered subject", prompt)
        self.assertIn("No collage", prompt)
        self.assertIn("regional manager", prompt)
        self.assertEqual(
            image_name_stem("The Office", "Michael Scott", "Steve Carell"),
            "theoffice~michaelscott~stevecarell",
        )

    def test_openai_image_generation_saves_valid_png_files(self) -> None:
        client = _FakeImageClient()
        service = OpenAIImageService(client=client)
        with tempfile.TemporaryDirectory() as directory:
            reported = []
            paths = service.generate(
                "basketball", directory, limit=2, on_image=reported.append
            )

            self.assertEqual(
                [path.name for path in paths],
                ["basketball_1.png", "basketball_2.png"],
            )
            self.assertTrue(all(path.is_file() for path in paths))
            self.assertEqual(reported, paths)
            self.assertEqual(client.image_request["moderation"], "auto")
            self.assertEqual(client.image_request["size"], "1024x1024")

    def test_openai_image_service_dynamic_api_key(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            # No env var, no key supplied
            service = OpenAIImageService()
            self.assertFalse(service.is_configured)

            # Pass key dynamically via property
            service.api_key = "sk-testkey123"
            self.assertTrue(service.is_configured)
            self.assertEqual(service.api_key, "sk-testkey123")

            # Init with key
            service2 = OpenAIImageService(api_key="sk-initkey456")
            self.assertTrue(service2.is_configured)
            self.assertEqual(service2.api_key, "sk-initkey456")

    def test_google_image_generation_saves_valid_png_files(self) -> None:
        client = _FakeGoogleImageClient()
        service = GoogleImageService(client=client)
        with tempfile.TemporaryDirectory() as directory:
            reported = []
            paths = service.generate(
                "basketball", directory, limit=2, on_image=reported.append
            )

            self.assertEqual(
                [path.name for path in paths],
                ["basketball_1.png", "basketball_2.png"],
            )
            self.assertTrue(all(path.is_file() for path in paths))
            self.assertEqual(reported, paths)
            self.assertEqual(client.last_request["config"]["number_of_images"], 2)
            self.assertEqual(client.last_request["config"]["aspect_ratio"], "1:1")

    def test_google_image_service_dynamic_api_key(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            # No env var, no key supplied
            service = GoogleImageService()
            self.assertFalse(service.is_configured)

            # Pass key dynamically via property
            service.api_key = "AIzaSyTest123"
            self.assertTrue(service.is_configured)
            self.assertEqual(service.api_key, "AIzaSyTest123")

            # Init with key
            service2 = GoogleImageService(api_key="AIzaSyInit456")
            self.assertTrue(service2.is_configured)
            self.assertEqual(service2.api_key, "AIzaSyInit456")

    def test_huggingface_image_generation_generates_and_saves_png(self) -> None:
        client = _FakeHuggingFaceClient()
        service = HuggingFaceImageService(client=client, api_key="hf_test_token_123")
        self.assertTrue(service.is_configured)
        self.assertEqual(service.api_key, "hf_test_token_123")
        with tempfile.TemporaryDirectory() as directory:
            reported = []
            paths = service.generate(
                "basketball", directory, limit=2, on_image=reported.append
            )
            self.assertEqual(
                [path.name for path in paths],
                ["basketball_1.png", "basketball_2.png"],
            )
            self.assertTrue(all(path.is_file() for path in paths))
            self.assertEqual(reported, paths)
            self.assertEqual(client.last_request["model"], "black-forest-labs/FLUX.1-schnell")
            self.assertIn("basketball", client.last_request["prompt"])

    def test_local_image_generation_uses_safe_prompt_and_saves_png(self) -> None:
        pipeline = _FakeLocalPipeline()
        service = LocalImageService(pipeline=pipeline)
        with tempfile.TemporaryDirectory() as directory:
            reported = []
            paths = service.generate(
                "basketball", directory, limit=2, on_image=reported.append
            )

            self.assertEqual(
                [path.name for path in paths],
                ["basketball_1.png", "basketball_2.png"],
            )
            self.assertTrue(all(path.is_file() for path in paths))
            self.assertEqual(reported, paths)
            self.assertIn("nudity", pipeline.requests[0]["negative_prompt"])
            self.assertIn("safe-for-work", pipeline.requests[0]["prompt"])

    def test_multimodal_moderation_rejects_sexual_image(self) -> None:
        client = _FakeImageClient()
        service = OpenAIImageService(client=client)
        self.assertTrue(service.is_sexually_safe(_png_bytes(), "image/png"))
        client.moderation_result = True
        self.assertFalse(service.is_sexually_safe(_png_bytes(), "image/png"))

    def test_bing_search_cannot_disable_safe_search(self) -> None:
        with self.assertRaisesRegex(ValueError, "must be strict"):
            BingDownloader(
                query="basketball",
                concept="basketball",
                related_concept="",
                name="",
                limit=1,
                output_dir="unused",
                timeout=5,
                safe_search="off",
            )

    def test_bing_request_is_strict_and_excludes_adult_terms(self) -> None:
        downloader = BingDownloader(
            query="basketball",
            concept="basketball",
            related_concept="",
            name="",
            limit=1,
            output_dir="unused",
            timeout=5,
        )
        captured = {}

        def fake_urlopen(request, timeout):
            captured["url"] = request.full_url
            captured["timeout"] = timeout
            return _FakeSearchResponse()

        with patch("urllib.request.urlopen", fake_urlopen):
            links = downloader._search_page()

        query = urllib.parse.parse_qs(urllib.parse.urlsplit(captured["url"]).query)
        self.assertEqual(query["adlt"], ["strict"])
        self.assertIn("/images/search", captured["url"])
        # Negative terms degrade relevance; safety uses adlt + blocklist instead.
        self.assertNotIn("-porn", query["q"][0])
        self.assertEqual(links, ["https://example.com/basketball.jpg"])
        self.assertTrue(downloader._is_unsafe("https://example.com/a.jpg", "free porn pics"))

    def test_candidates_ranked_by_concept_relevance(self) -> None:
        downloader = BingDownloader(
            query="Michael Jordan slam dunk",
            concept="Michael Jordan",
            related_concept="slam dunk",
            name="",
            limit=1,
            output_dir="unused",
            timeout=5,
        )
        downloader.titles = {
            "https://x.com/hoop.jpg": "Basketball hoop at sunset",
            "https://x.com/mj.jpg": "Michael Jordan wins 1988 slam dunk contest",
        }
        ranked = sorted(downloader.titles, key=downloader._relevance, reverse=True)
        self.assertEqual(ranked[0], "https://x.com/mj.jpg")

    def test_bing_validator_rejects_before_writing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            downloader = BingDownloader(
                query="basketball",
                concept="basketball",
                related_concept="",
                name="",
                limit=1,
                output_dir=directory,
                timeout=5,
                verbose=False,
                image_validator=lambda *_: False,
            )
            with patch.object(downloader, "_fetch_and_normalize", return_value=_png_bytes()):
                self.assertFalse(downloader.download_image("https://example.com/image.png"))
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_bing_reports_each_image_after_saving(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            reported = []
            downloader = BingDownloader(
                query="basketball",
                concept="basketball",
                related_concept="",
                name="",
                limit=1,
                output_dir=directory,
                timeout=5,
                verbose=False,
                on_image=reported.append,
            )
            with patch.object(
                downloader, "_fetch_and_normalize", return_value=_png_bytes()
            ):
                self.assertTrue(
                    downloader.download_image("https://example.com/image.png")
                )
            self.assertEqual(len(reported), 1)
            self.assertTrue(Path(reported[0]).is_file())

    def test_bing_deduplicates_images_across_related_concept_downloaders(self) -> None:
        shared_hashes = set()
        shared_lock = threading.Lock()
        with tempfile.TemporaryDirectory() as directory:
            first_dir = Path(directory) / 'elaine'
            second_dir = Path(directory) / 'george'
            first = BingDownloader(
                query='"Elaine Benes" "Seinfeld"',
                concept='Seinfeld',
                related_concept='Elaine Benes',
                name='',
                limit=1,
                output_dir=str(first_dir),
                timeout=5,
                verbose=False,
                seen_images=shared_hashes,
                seen_images_lock=shared_lock,
            )
            second = BingDownloader(
                query='"George Costanza" "Seinfeld"',
                concept='Seinfeld',
                related_concept='George Costanza',
                name='',
                limit=1,
                output_dir=str(second_dir),
                timeout=5,
                verbose=False,
                seen_images=shared_hashes,
                seen_images_lock=shared_lock,
            )
            first_dir.mkdir()
            second_dir.mkdir()
            with patch.object(first, '_fetch_and_normalize', return_value=_png_bytes()):
                self.assertTrue(first.download_image('https://example.com/elaine.png'))
            with patch.object(second, '_fetch_and_normalize', return_value=_png_bytes()):
                self.assertFalse(second.download_image('https://example.com/george.png'))
            self.assertEqual(len(list(first_dir.iterdir())), 1)
            self.assertEqual(len(list(second_dir.iterdir())), 0)

    def test_bing_rejects_explicit_adult_urls_before_fetching(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            downloader = BingDownloader(
                query="basketball",
                concept="basketball",
                related_concept="",
                name="",
                limit=1,
                output_dir=directory,
                timeout=5,
                verbose=False,
            )
            with patch.object(downloader, "_fetch_and_normalize") as fetch:
                accepted = downloader.download_image(
                    "https://example.com/nsfw/basketball.png"
                )
            self.assertFalse(accepted)
            fetch.assert_not_called()

    def test_wikimedia_search_collects_valid_candidates(self) -> None:
        downloader = BingDownloader(
            query="Michael Jordan",
            concept="Michael Jordan",
            related_concept="",
            name="",
            limit=2,
            output_dir="unused",
            timeout=5,
            provider="wikimedia",
        )
        fake_api_response = {
            "query": {
                "pages": {
                    "101": {
                        "title": "File:Michael Jordan 1984.jpg",
                        "imageinfo": [
                            {
                                "thumburl": "https://upload.wikimedia.org/thumb/mj.jpg",
                                "url": "https://upload.wikimedia.org/orig/mj.jpg",
                                "mime": "image/jpeg",
                            }
                        ],
                    }
                }
            }
        }
        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_urlopen.return_value = _FakeSearchResponse(
                json.dumps(fake_api_response).encode("utf-8")
            )
            links = downloader._search_wikimedia()

        self.assertEqual(links, ["https://upload.wikimedia.org/thumb/mj.jpg"])
        self.assertEqual(downloader.titles["https://upload.wikimedia.org/thumb/mj.jpg"], "Michael Jordan 1984.jpg")

    def test_relevance_filter_rejects_single_first_name_false_positives(self) -> None:
        downloader = BingDownloader(
            query="Michael Scott Steve Carell",
            concept="The Office",
            related_concept="Michael Scott",
            name="Steve Carell",
            limit=3,
            output_dir="unused",
            timeout=5,
        )
        downloader.titles = {
            "https://x.com/mj.jpg": "Michael Jackson performs in concert 1984",
            "https://x.com/office_bldg.jpg": "Modern office building interior",
            "https://x.com/scott.jpg": "Steve Carell as Michael Scott on The Office",
        }
        self.assertFalse(downloader._is_relevant("https://x.com/mj.jpg"))
        self.assertFalse(downloader._is_relevant("https://x.com/office_bldg.jpg"))
        self.assertTrue(downloader._is_relevant("https://x.com/scott.jpg"))

    def test_openverse_search_caps_page_size_to_anonymous_limit(self) -> None:
        downloader = BingDownloader(
            query="Golden Retriever",
            concept="Golden Retriever",
            related_concept="",
            name="",
            limit=5,
            output_dir="unused",
            timeout=5,
            provider="openverse",
        )
        captured_urls: list[str] = []
        fake_openverse = {
            "results": [
                {
                    "url": "https://example.com/golden.jpg",
                    "thumbnail": "https://api.openverse.org/v1/images/123/thumb/",
                    "title": "Golden Retriever puppy",
                }
            ]
        }

        def fake_urlopen(request, timeout):
            captured_urls.append(request.full_url)
            return _FakeSearchResponse(json.dumps(fake_openverse).encode("utf-8"))

        with patch("urllib.request.urlopen", fake_urlopen):
            links = downloader._search_openverse()

        self.assertEqual(links, ["https://example.com/golden.jpg"])
        self.assertEqual(
            downloader.thumbnails["https://example.com/golden.jpg"],
            "https://api.openverse.org/v1/images/123/thumb/",
        )
        qs = urllib.parse.parse_qs(urllib.parse.urlsplit(captured_urls[0]).query)
        self.assertLessEqual(int(qs["page_size"][0]), 20)

    def test_try_fetch_falls_back_to_thumbnail_when_primary_url_fails(self) -> None:
        downloader = BingDownloader(
            query="Golden Retriever",
            concept="Golden Retriever",
            related_concept="",
            name="",
            limit=1,
            output_dir="unused",
            timeout=5,
            verbose=False,
        )
        primary = "https://blocked-host.example/dog.jpg"
        thumb = "https://cdn-thumb.example/dog_thumb.jpg"
        downloader.thumbnails[primary] = thumb

        def fake_fetch(url: str) -> bytes:
            if url == primary:
                raise RuntimeError("HTTP 403 Forbidden")
            return _png_bytes()

        with patch.object(downloader, "_fetch_and_normalize", side_effect=fake_fetch) as mock_fetch:
            result = downloader._try_fetch(primary)

        self.assertIsNotNone(result)
        self.assertEqual(mock_fetch.call_count, 2)

    def test_web_downloader_enforces_custom_min_width_and_height(self) -> None:
        downloader = BingDownloader(
            query="basketball",
            concept="basketball",
            related_concept="",
            name="",
            limit=1,
            output_dir="unused",
            timeout=5,
            verbose=False,
            min_width=300,
            min_height=250,
        )
        # _png_bytes() is 200x200, which is smaller than 300x250
        class _FakeImgHttpResponse:
            headers = {"Content-Type": "image/png"}
            def __enter__(self):
                return self
            def __exit__(self, *args):
                return False
            def read(self, *_):
                return _png_bytes()

        with patch("urllib.request.urlopen", return_value=_FakeImgHttpResponse()):
            with self.assertRaisesRegex(ValueError, "smaller than 300x250px"):
                downloader._fetch_and_normalize("https://example.com/small.png")

    def test_face_detection_caps_at_10_and_filters_background_crowd(self) -> None:
        from stimulus_forge.services.image_workspace import _filter_and_rank_faces

        # Simulate a 1200x800 basketball photo with 2 foreground players (180x180, 160x160)
        # and 25 smaller crowd spectators in the background (30x30 to 75x75)
        candidates = [
            {"x": 400, "y": 200, "w": 180, "h": 180, "confidence": 0.99},
            {"x": 700, "y": 220, "w": 160, "h": 160, "confidence": 0.97},
        ]
        for i in range(25):
            candidates.append({
                "x": (i * 45) % 1100,
                "y": 50 + (i // 10) * 60,
                "w": 35 + (i % 5) * 10,
                "h": 35 + (i % 5) * 10,
                "confidence": 0.92,
            })

        filtered = _filter_and_rank_faces(candidates, img_width=1200, img_height=800, max_faces=10)
        # Only the 2 prominent foreground player faces should pass the >=80x80 & >=1/25th head-area filter
        self.assertEqual(len(filtered), 2)
        self.assertEqual(filtered[0]["w"], 180)
        self.assertEqual(filtered[1]["w"], 160)

        # Now simulate 15 equally prominent faces in a group portrait: must cap at 10
        group_faces = [
            {"x": i * 85, "y": 100, "w": 120, "h": 120, "confidence": 0.95}
            for i in range(15)
        ]
        capped = _filter_and_rank_faces(group_faces, img_width=1400, img_height=900, max_faces=10)
        self.assertEqual(len(capped), 10)

    def test_crop_face_supports_custom_wxh_dimensions(self) -> None:
        from stimulus_forge.services.image_workspace import TaskImagesDownloader

        with tempfile.TemporaryDirectory() as directory:
            img_path = Path(directory) / "player.png"
            Image.new("RGB", (600, 600), "green").save(img_path, format="PNG")
            downloader = TaskImagesDownloader()
            fake_detection = {
                "width": 600,
                "height": 600,
                "faces": [{"x": 150, "y": 150, "w": 200, "h": 200, "confidence": 0.98}],
                "count": 1,
                "face_detected": True,
                "detector": "mtcnn",
                "path": str(img_path),
            }
            with patch.object(downloader, "detect_faces", return_value=fake_detection):
                ok = downloader.crop_face(str(img_path), target_size=(256, 192))
            self.assertTrue(ok)
            with Image.open(img_path) as cropped_img:
                self.assertEqual(cropped_img.size, (256, 192))

    def test_compute_face_crop_box_square_mirrors_desktop_version(self) -> None:
        from stimulus_forge.services.image_workspace import compute_face_crop_box

        # Rectangular face (tall and narrow, typical of MTCNN output: 120w x 160h)
        face = {"x": 300, "y": 200, "w": 120, "h": 160}
        # In a 1000x800 image, target size is 160x160 (square stimulus)
        x1, y1, x2, y2 = compute_face_crop_box(1000, 800, face, target_size=(160, 160))

        # 1. Must be perfectly square (no horizontal stretching or vertical squishing)
        crop_w = x2 - x1
        crop_h = y2 - y1
        self.assertEqual(crop_w, crop_h)

        # 2. Must apply desktop's 60% expansion around max(w, h): max(120, 160) * 1.6 = 256
        self.assertEqual(crop_w, 256)
        self.assertEqual(crop_h, 256)

        # 3. Must be centered around face center (cx = 360, cy = 280)
        self.assertEqual(x1, 360 - 128)
        self.assertEqual(y1, 280 - 128)
        self.assertEqual(x2, 360 + 128)
        self.assertEqual(y2, 280 + 128)

    def test_compute_face_crop_box_boundary_shifting(self) -> None:
        from stimulus_forge.services.image_workspace import compute_face_crop_box

        # Face near top-left edge: must shift away from 0 while keeping exact square shape
        face_tl = {"x": 10, "y": 10, "w": 100, "h": 120}
        x1, y1, x2, y2 = compute_face_crop_box(500, 500, face_tl, target_size=(160, 160))
        self.assertEqual(x1, 0)
        self.assertEqual(y1, 0)
        self.assertEqual(x2 - x1, 192)
        self.assertEqual(y2 - y1, 192)

        # Face near bottom-right edge: must shift away from borders while keeping exact square shape
        face_br = {"x": 380, "y": 370, "w": 100, "h": 120}
        bx1, by1, bx2, by2 = compute_face_crop_box(500, 500, face_br, target_size=(160, 160))
        self.assertEqual(bx2, 500)
        self.assertEqual(by2, 500)
        self.assertEqual(bx2 - bx1, 192)
        self.assertEqual(by2 - by1, 192)

    def test_crop_face_avoids_stretching_rectangular_faces(self) -> None:
        from stimulus_forge.services.image_workspace import TaskImagesDownloader

        with tempfile.TemporaryDirectory() as directory:
            img_path = Path(directory) / "portrait.png"
            # Create a 800x800 image
            Image.new("RGB", (800, 800), "blue").save(img_path, format="PNG")
            downloader = TaskImagesDownloader()

            # Face is 100x150 (ratio 0.67, tall rectangle)
            fake_detection = {
                "width": 800,
                "height": 800,
                "faces": [{"x": 350, "y": 300, "w": 100, "h": 150, "confidence": 0.99}],
                "count": 1,
                "face_detected": True,
                "detector": "mtcnn",
                "path": str(img_path),
            }
            with patch.object(downloader, "detect_faces", return_value=fake_detection):
                ok = downloader.crop_face(str(img_path), target_size=(160, 160))
            self.assertTrue(ok)
            with Image.open(img_path) as cropped_img:
                self.assertEqual(cropped_img.size, (160, 160))

    def test_detect_face_rois_includes_crop_box(self) -> None:
        from stimulus_forge.services.image_workspace import detect_face_rois

        with tempfile.TemporaryDirectory() as directory:
            img_path = Path(directory) / "test_face.png"
            Image.new("RGB", (600, 600), "white").save(img_path, format="PNG")

            with patch("stimulus_forge.services.image_workspace._get_face_detector") as mock_det:
                mock_detector = MagicMock()
                mock_detector.detect_faces.return_value = [
                    {
                        "box": [200, 150, 100, 140],
                        "confidence": 0.98,
                        "keypoints": {"left_eye": (220, 180), "right_eye": (260, 180)},
                    }
                ]
                mock_det.return_value = mock_detector
                res = detect_face_rois(str(img_path), target_size=(160, 160))

            self.assertTrue(res["face_detected"])
            self.assertEqual(len(res["faces"]), 1)
            face = res["faces"][0]
            self.assertIn("crop_box", face)
            crop_box = face["crop_box"]
            # Square crop box
            self.assertEqual(crop_box["w"], crop_box["h"])
            # Expanded by 60%: 140 * 1.6 = 224
            self.assertEqual(crop_box["w"], 224)

    def test_sanitize_subject_id(self) -> None:
        from stimulus_forge.api.routes.images import _sanitize_subject_id

        self.assertEqual(_sanitize_subject_id("SUBJ-07"), "SUBJ-07")
        self.assertEqual(_sanitize_subject_id("pilot_01"), "pilot_01")
        self.assertEqual(_sanitize_subject_id("../../etc/passwd"), "etc_passwd")
        self.assertEqual(_sanitize_subject_id("   "), "")
        self.assertEqual(_sanitize_subject_id(None), "")
        self.assertEqual(_sanitize_subject_id("..."), "")

    def test_search_images_routes_to_subject_folder(self) -> None:
        from stimulus_forge.api.routes.images import search_images
        from stimulus_forge.api.schemas import ImageSearchRequest

        with tempfile.TemporaryDirectory() as tmpdir:
            captured_output_dir = []

            def fake_download(*args, **kwargs):
                out = kwargs.get("output_dir", "")
                captured_output_dir.append(out)
                f = Path(out) / "img1.png"
                f.parent.mkdir(parents=True, exist_ok=True)
                f.write_bytes(b"dummy")
                return [str(f)]

            with patch("stimulus_forge.api.routes.images.TaskImagesDownloader.download", side_effect=fake_download):
                # 1. With subject_id
                req = ImageSearchRequest(
                    concept="seinfeld",
                    related_concept="jerry",
                    subject_id="SUBJ-07",
                    output_dir=tmpdir,
                )
                search_images(req)
                self.assertTrue(len(captured_output_dir) > 0)
                self.assertIn("SUBJ-07", captured_output_dir[0])
                self.assertIn("seinfeld", captured_output_dir[0])
                self.assertIn("jerry", captured_output_dir[0])

                # 2. Without subject_id (backward compatibility fallback)
                captured_output_dir.clear()
                req_no_subj = ImageSearchRequest(
                    concept="basketball",
                    output_dir=tmpdir,
                )
                search_images(req_no_subj)
                self.assertIn("downloads", captured_output_dir[0])
                self.assertIn("basketball", captured_output_dir[0])

    def test_generate_images_routes_to_subject_folder(self) -> None:
        from stimulus_forge.api.routes.images import generate_images
        from stimulus_forge.api.schemas import ImageGenerateRequest

        with tempfile.TemporaryDirectory() as tmpdir:
            captured_dir = []

            def fake_gen(*args, **kwargs):
                out = kwargs.get("output_dir")
                captured_dir.append(str(out))
                f = Path(out) / "gen.png"
                f.parent.mkdir(parents=True, exist_ok=True)
                f.write_bytes(b"dummy")
                return [f]

            with patch("stimulus_forge.services.openai_images.OpenAIImageService.generate", side_effect=fake_gen):
                req = ImageGenerateRequest(
                    prompt="a cartoon dog",
                    concept="dog",
                    subject_id="SUBJ-07",
                    output_dir=tmpdir,
                )
                generate_images(req)
                self.assertTrue(len(captured_dir) > 0)
                self.assertIn("SUBJ-07", captured_dir[0])
                self.assertIn("generated", captured_dir[0])

            with patch("stimulus_forge.services.google_images.GoogleImageService.generate", side_effect=fake_gen):
                captured_dir.clear()
                req_google = ImageGenerateRequest(
                    prompt="a cartoon cat",
                    concept="cat",
                    provider="google",
                    subject_id="SUBJ-08",
                    output_dir=tmpdir,
                )
                generate_images(req_google)
                self.assertTrue(len(captured_dir) > 0)
                self.assertIn("SUBJ-08", captured_dir[0])
                self.assertIn("generated", captured_dir[0])

            # Text stimulus generation endpoint
            req_text = ImageGenerateRequest(
                prompt="basketball",
                concept="basketball",
                related_concept="Michael Jordan",
                provider="text",
                count=2,
                subject_id="SUBJ-09",
                output_dir=tmpdir,
            )
            resp_text = generate_images(req_text)
            self.assertEqual(len(resp_text.images), 2)
            self.assertEqual(resp_text.images[0].source, "text")
            self.assertTrue(Path(resp_text.images[0].local_path).exists())
            self.assertIn("SUBJ-09", resp_text.images[0].local_path)


class TextImageServiceUnitTests(unittest.TestCase):
    def test_render_centered_text_image(self) -> None:
        img = render_centered_text_image("Basketball", size=(400, 400), bg_color="#FFFFFF", text_color="#000000")
        self.assertEqual(img.size, (400, 400))
        self.assertEqual(img.mode, "RGB")

    def test_text_image_service_generation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            service = TextImageService(target_size=(300, 300))
            self.assertTrue(service.is_configured)
            self.assertTrue(service.is_available)

            notified = []
            paths = service.generate(
                concept="Tennis",
                related_concept="Racket",
                output_dir=tmp_dir,
                limit=3,
                on_image=lambda p: notified.append(p),
            )
            self.assertEqual(len(paths), 3)
            self.assertEqual(len(notified), 3)
            for p in paths:
                self.assertTrue(p.exists())
                self.assertTrue(p.name.startswith("tennis~racket_"))
                with Image.open(p) as img:
                    self.assertEqual(img.size, (300, 300))

    def test_text_image_service_should_continue(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            service = TextImageService()
            called = [0]

            def should_continue():
                called[0] += 1
                return called[0] <= 1

            paths = service.generate(
                concept="Soccer",
                output_dir=tmp_dir,
                limit=5,
                should_continue=should_continue,
            )
            self.assertEqual(len(paths), 1)


if __name__ == "__main__":
    unittest.main()

