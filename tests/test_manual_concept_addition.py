import unittest
from unittest.mock import MagicMock

from stimulus_forge.ui.forge import StimulusForge


class ManualConceptAdditionTests(unittest.TestCase):
    def setUp(self):
        # Create a mock instance with required attributes
        self.forge = object.__new__(StimulusForge)
        self.forge.related_concepts_dict = {}
        self.forge.patient_concepts = ["basketball"]
        self.forge.subject_id = "SUBJ-01"
        self.forge.update_related_concepts = MagicMock()
        self.forge._sync_workflow_tracker = MagicMock()
        self.forge.create_images_button = MagicMock()
        self.forge.c = MagicMock()

    def test_add_single_related_concept(self):
        res = self.forge.add_related_concepts_to("basketball", "Steph Curry")
        self.assertTrue(res)
        self.assertEqual(len(self.forge.related_concepts_dict["basketball"]), 1)
        item = self.forge.related_concepts_dict["basketball"][0]
        self.assertEqual(item["name"], "Steph Curry")
        self.assertEqual(item["char"], "")
        self.assertEqual(item["description"], "Manually added association")
        self.forge.update_related_concepts.assert_called_once_with(b_llm=True)
        self.forge.create_images_button.setEnabled.assert_called_with(True)

    def test_add_multiple_comma_separated_concepts(self):
        res = self.forge.add_related_concepts_to(
            "basketball", "Steph Curry, Golden State Warriors, LeBron James"
        )
        self.assertTrue(res)
        self.assertEqual(len(self.forge.related_concepts_dict["basketball"]), 3)
        names = [item["name"] for item in self.forge.related_concepts_dict["basketball"]]
        self.assertEqual(names, ["Steph Curry", "Golden State Warriors", "LeBron James"])

    def test_deduplicates_against_existing_and_parent(self):
        self.forge.related_concepts_dict["basketball"] = [
            {"name": "Steph Curry", "char": "", "description": "Existing"}
        ]
        # Attempt to add duplicate "steph curry" and parent "basketball" and new "Klay Thompson"
        res = self.forge.add_related_concepts_to(
            "basketball", "steph curry, basketball, Klay Thompson"
        )
        self.assertTrue(res)
        self.assertEqual(len(self.forge.related_concepts_dict["basketball"]), 2)
        names = [item["name"] for item in self.forge.related_concepts_dict["basketball"]]
        self.assertEqual(names, ["Steph Curry", "Klay Thompson"])

    def test_rejects_empty_or_all_duplicates(self):
        self.forge.related_concepts_dict["basketball"] = [
            {"name": "Steph Curry", "char": "", "description": "Existing"}
        ]
        res = self.forge.add_related_concepts_to("basketball", "   ")
        self.assertFalse(res)

        res2 = self.forge.add_related_concepts_to("basketball", "steph curry, basketball")
        self.assertFalse(res2)


if __name__ == "__main__":
    unittest.main()
