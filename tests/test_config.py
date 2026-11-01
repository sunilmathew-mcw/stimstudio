import unittest
from pathlib import Path

from stimulus_forge.config import config


class ConfigTests(unittest.TestCase):
    def test_default_paths_are_absolute(self) -> None:
        path_keys = (
            "concepts_file",
            "experiment_folder",
            "pics_main_folder",
            "task_images_folder",
        )
        for key in path_keys:
            with self.subTest(key=key):
                self.assertTrue(Path(config[key]).is_absolute())

    def test_default_concepts_file_exists(self) -> None:
        self.assertTrue(Path(config["concepts_file"]).is_file())

    def test_load_env_file_loads_key_value_pairs(self) -> None:
        import os
        import tempfile
        from stimulus_forge import config as cfg_module

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_env = Path(tmpdir) / ".env"
            tmp_env.write_text("TEST_KEY_STIMULUS=custom_value\n# Comment\nINVALID_LINE\n")
            orig_root = cfg_module.PROJECT_ROOT
            try:
                cfg_module.PROJECT_ROOT = Path(tmpdir)
                cfg_module._load_env_file()
                self.assertEqual(os.environ.get("TEST_KEY_STIMULUS"), "custom_value")
            finally:
                cfg_module.PROJECT_ROOT = orig_root
                os.environ.pop("TEST_KEY_STIMULUS", None)


if __name__ == "__main__":
    unittest.main()
