import tempfile
import unittest
from pathlib import Path

import pandas as pd

from stimulus_forge.services.concept_loader import load_concepts, workspace_name_for


class ConceptLoaderTests(unittest.TestCase):
    def test_loads_and_deduplicates_text_concepts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "My Concept List.txt"
            path.write_text("Dog\nChicago; dog\nTaylor Swift", encoding="utf-8")

            self.assertEqual(load_concepts(path), ["dog", "chicago", "taylor swift"])
            self.assertEqual(workspace_name_for(path), "My_Concept_List")

    def test_loads_named_concept_column_from_csv(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "concepts.csv"
            path.write_text("notes,Concept\nignore,Dog\nignore,Chicago", encoding="utf-8")

            self.assertEqual(load_concepts(path), ["dog", "chicago"])

    def test_loads_plain_excel_grid(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "concepts.xlsx"
            pd.DataFrame([["Dog"], ["Chicago"]]).to_excel(
                path, index=False, header=False
            )

            self.assertEqual(load_concepts(path), ["dog", "chicago"])

    def test_rejects_an_empty_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "concepts.txt"
            path.write_text("\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "does not contain any concepts"):
                load_concepts(path)


if __name__ == "__main__":
    unittest.main()
