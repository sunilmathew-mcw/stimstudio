import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from stimulus_forge.cli import (
    build_parser,
    check_gpu_environment,
    collect_concepts,
    run_cli,
)


class StimStudioCLITests(unittest.TestCase):
    def test_parser_defaults(self):
        parser = build_parser()
        args = parser.parse_args(["apple", "banana"])
        self.assertEqual(args.concepts, ["apple", "banana"])
        self.assertEqual(args.provider, "local")
        self.assertEqual(args.gpu_layers, -1)
        self.assertEqual(args.ctx_size, 4096)
        self.assertEqual(args.count, 10)
        self.assertFalse(args.with_images)
        self.assertFalse(args.no_cache)

    def test_collect_concepts_from_args_and_comma(self):
        parser = build_parser()
        args = parser.parse_args(["apple, banana", "cherry", "apple"])
        concepts = collect_concepts(args)
        self.assertEqual(concepts, ["apple", "banana", "cherry"])

    def test_collect_concepts_from_file(self):
        with tempfile.NamedTemporaryFile("w+", suffix=".txt", delete=False) as f:
            f.write("Einstein\nCurie\nNewton\n")
            f_path = f.name

        try:
            parser = build_parser()
            args = parser.parse_args(["-f", f_path])
            concepts = collect_concepts(args)
            self.assertEqual(concepts, ["einstein", "curie", "newton"])
        finally:
            Path(f_path).unlink(missing_ok=True)

    def test_collect_concepts_from_stdin(self):
        parser = build_parser()
        args = parser.parse_args(["-"])
        with patch("sys.stdin", io.StringIO("Jupiter\nSaturn\nMars\n")):
            concepts = collect_concepts(args)
        self.assertEqual(concepts, ["Jupiter", "Saturn", "Mars"])

    def test_check_gpu_environment_structure(self):
        diag = check_gpu_environment()
        self.assertIn("python_version", diag)
        self.assertIn("torch_available", diag)
        self.assertIn("cuda_available", diag)
        self.assertIn("llama_cpp_available", diag)
        self.assertIn("llama_gpu_offload", diag)

    def test_check_gpu_flag_exits_zero(self):
        code = run_cli(["--check-gpu"])
        self.assertEqual(code, 0)

    @patch("stimulus_forge.cli.RelatedConceptGenerator")
    def test_run_cli_json_output(self, mock_gen_cls):
        mock_instance = MagicMock()
        mock_instance.generate.return_value = [
            {"name": "General Relativity", "char": "", "description": "Theory of gravitation"},
            {"name": "Photoelectric Effect", "char": "", "description": "Nobel Prize winning discovery"},
        ]
        mock_gen_cls.return_value = mock_instance

        with tempfile.NamedTemporaryFile("w+", suffix=".json", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            code = run_cli([
                "Einstein",
                "--provider", "compatible",
                "--model", "google/gemma-2-27b-it",
                "--base-url", "http://localhost:8000/v1",
                "-o", tmp_path,
                "-q",
            ])
            self.assertEqual(code, 0)

            # Check that RelatedConceptGenerator was initialized with compatible settings
            mock_gen_cls.assert_called_once_with(
                provider="compatible",
                openai_model="google/gemma-2-27b-it",
                local_model_path="",
                local_gpu_layers=-1,
                api_key="",
                base_url="http://localhost:8000/v1",
                n_ctx=4096,
            )

            # Check output json file
            content = json.loads(Path(tmp_path).read_text(encoding="utf-8"))
            self.assertEqual(content["total_seeds"], 1)
            self.assertEqual(content["provider"], "compatible")
            self.assertEqual(len(content["results"][0]["related_concepts"]), 2)
            self.assertEqual(
                content["results"][0]["related_concepts"][0]["name"],
                "General Relativity",
            )
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    @patch("stimulus_forge.cli.RelatedConceptGenerator")
    def test_run_cli_csv_output(self, mock_gen_cls):
        mock_instance = MagicMock()
        mock_instance.generate.return_value = [
            {"name": "Mount Vernon", "char": "", "description": "Historic estate"},
        ]
        mock_gen_cls.return_value = mock_instance

        with tempfile.NamedTemporaryFile("w+", suffix=".csv", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            code = run_cli([
                "George Washington",
                "-p", "local",
                "-m", "/dummy/path/model.gguf",
                "-o", tmp_path,
                "-q",
            ])
            self.assertEqual(code, 0)

            lines = Path(tmp_path).read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(lines[0], "seed,name,character,description")
            self.assertIn("George Washington,Mount Vernon,,Historic estate", lines[1])
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_run_cli_no_concepts_fails(self):
        with patch("sys.stdin.isatty", return_value=True):
            code = run_cli([])
            self.assertEqual(code, 1)


if __name__ == "__main__":
    unittest.main()
