import unittest
from pathlib import Path
from unittest.mock import MagicMock

from stimulus_forge.api.routes.concepts import upload_concept_file
from stimulus_forge.services.agentic_rag import (
    extract_concepts_with_rag,
    extract_raw_text,
    is_straightforward_concept_list,
)
from stimulus_forge.services.concept_loader import load_concepts, workspace_name_for
from stimulus_forge.services.questionnaire_intelligence import (
    parse_questionnaire,
    is_questionnaire_dataframe,
)


class QuestionnaireRagTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.csv_path = Path("assets/sample-responses.csv")
        self.assertTrue(self.csv_path.exists(), "assets/sample-responses.csv should exist for test")

    def test_parse_questionnaire_detects_subjects_and_extracts_subj02(self) -> None:
        result = parse_questionnaire(self.csv_path)
        self.assertTrue(result.is_questionnaire)
        self.assertIn("SUBJ-02", result.subjects)
        self.assertIn("SUBJ-01", result.subjects)
        self.assertEqual(result.selected_subject, "SUBJ-02")

        # SUBJ-02 has Seinfeld, Frasier, Entourage, bucks, packers
        expected = ["seinfeld", "frasier", "entourage", "bucks", "packers"]
        self.assertEqual(result.selected_concepts, expected)

        # Ensure excerpts match
        self.assertIn("seinfeld", result.selected_excerpts)
        self.assertIn("bucks", result.selected_excerpts)

    def test_parse_questionnaire_subject_filtering(self) -> None:
        # Request SUBJ-01
        res_01 = parse_questionnaire(self.csv_path, subject_id="SUBJ-01")
        self.assertEqual(res_01.selected_subject, "SUBJ-01")
        self.assertGreater(len(res_01.selected_concepts), 50)
        self.assertIn("abbott elementary", res_01.selected_concepts)
        self.assertIn("stranger things", res_01.selected_concepts)
        self.assertIn("will smith", res_01.selected_concepts)

        # Ensure demographic columns are not included
        forbidden = {"patientid", "first name", "last name", "email", "address", "city/town", "surgery date:"}
        for c in res_01.selected_concepts:
            self.assertNotIn(c, forbidden)

        # Request 'all'
        res_all = parse_questionnaire(self.csv_path, subject_id="all")
        self.assertEqual(res_all.selected_subject, "all")
        self.assertGreater(len(res_all.selected_concepts), len(res_01.selected_concepts))
        self.assertIn("seinfeld", res_all.selected_concepts)
        self.assertIn("bucks", res_all.selected_concepts)

    def test_concept_loader_with_questionnaire(self) -> None:
        # Default loads SUBJ-02
        concepts = load_concepts(self.csv_path)
        self.assertEqual(concepts, ["seinfeld", "frasier", "entourage", "bucks", "packers"])

        # Explicit subject_id
        concepts_01 = load_concepts(self.csv_path, subject_id="SUBJ-01")
        self.assertIn("the office", concepts_01)
        self.assertIn("harry potter", concepts_01)

        # Workspace name
        name = workspace_name_for(self.csv_path, subject_id="SUBJ-02")
        self.assertEqual(name, "SUBJ-02")

    def test_extract_raw_text_generates_semantic_summary_for_questionnaire(self) -> None:
        content = self.csv_path.read_bytes()
        text = extract_raw_text(self.csv_path.name, content)
        self.assertIn("CLINICAL / RESEARCH SUBJECT QUESTIONNAIRE RESPONSES", text)
        self.assertIn("SUBJ-02", text)
        self.assertIn("seinfeld", text)

    def test_extract_concepts_with_rag_fallback_on_questionnaire(self) -> None:
        # Generator without LLM backend triggers fallback
        dummy_generator = MagicMock()
        dummy_generator.generate_raw.side_effect = Exception("No LLM connection")

        text = self.csv_path.read_text(encoding="utf-8", errors="replace")
        concepts, excerpts = extract_concepts_with_rag(text, dummy_generator, max_concepts=10)
        self.assertEqual(concepts, ["seinfeld", "frasier", "entourage", "bucks", "packers"])
        self.assertIn("seinfeld", excerpts)
        self.assertIn("bucks", excerpts)

    async def test_upload_concept_file_endpoint_with_questionnaire(self) -> None:
        from fastapi import UploadFile
        import io

        content = self.csv_path.read_bytes()
        upload_file = UploadFile(
            filename="sample-responses.csv",
            file=io.BytesIO(content),
        )

        res = await upload_concept_file(file=upload_file, provider="local")
        self.assertEqual(res["method"], "questionnaire_rag")
        self.assertEqual(res["selected_subject"], "SUBJ-02")
        self.assertEqual(res["concepts"], ["seinfeld", "frasier", "entourage", "bucks", "packers"])
        self.assertEqual(res["subjects"], ["SUBJ-01", "SUBJ-02"])
        self.assertIn("SUBJ-01", res["concepts_by_subject"])
        self.assertIn("SUBJ-01", res["excerpts_by_subject"])
        self.assertIn("all_concepts", res)
        self.assertIn("all_excerpts", res)
        self.assertGreater(len(res["all_concepts"]), len(res["concepts"]))

    def test_phi_scrubbing_and_rejection(self) -> None:
        from stimulus_forge.services.questionnaire_intelligence import (
            contains_phi,
            is_familial_disclosure,
            sanitize_phi_from_text,
        )

        self.assertTrue(contains_phi("patient email: user@domain.org"))
        self.assertTrue(contains_phi("call 414-555-0199"))
        self.assertTrue(contains_phi("living at 8701 W Watertown Plank Rd, Milwaukee"))
        self.assertFalse(contains_phi("The Simpsons"))

        self.assertTrue(is_familial_disclosure("George Peeke (my paternal grandfather)"))
        self.assertTrue(is_familial_disclosure("my mother Mary"))
        self.assertFalse(is_familial_disclosure("Will Smith"))

        sanitized = sanitize_phi_from_text("meet me at 8701 W Watertown Plank Rd or email test@mcw.edu")
        self.assertNotIn("test@mcw.edu", sanitized)
        self.assertNotIn("8701 W Watertown Plank Rd", sanitized)

    def test_semantic_summary_zero_phi_leakage(self) -> None:
        res = parse_questionnaire(self.csv_path)
        summary = res.semantic_summary

        # Check that none of the participants' emails, street addresses, zip codes, or occupations appear in prompt summary
        forbidden_phi = [
            "alex.taylor@example.org",
            "jordan.smith@example.org",
            "123 Science Park Rd",
            "456 Research Blvd",
            "53027",
            "53202",
            "Rutgers University",
            "Unemployed/Disabled",
        ]
        for phi in forbidden_phi:
            self.assertNotIn(phi.lower(), summary.lower())

    def test_cloud_guards_block_phi_leakage(self) -> None:
        from fastapi import HTTPException
        from stimulus_forge.api.routes.concepts import generate_concepts
        from stimulus_forge.api.routes.images import generate_images, search_images
        from stimulus_forge.api.schemas import (
            ConceptGenerateRequest,
            ImageGenerateRequest,
            ImageSearchRequest,
        )

        # 1. Concept generation with PHI is blocked
        with self.assertRaises(HTTPException) as ctx:
            generate_concepts(ConceptGenerateRequest(concept="John Doe DOB 1980-05-12 SSN 123-45-6789"))
        self.assertEqual(ctx.exception.status_code, 400)

        # 2. Image search with PHI is blocked
        with self.assertRaises(HTTPException) as ctx:
            search_images(ImageSearchRequest(concept="patient at 8701 W Watertown Plank Rd"))
        self.assertEqual(ctx.exception.status_code, 400)

        # 3. Image generation prompt with PHI is blocked
        with self.assertRaises(HTTPException) as ctx:
            generate_images(ImageGenerateRequest(prompt="Photo of patient email jdoe@example.com"))
        self.assertEqual(ctx.exception.status_code, 400)



if __name__ == "__main__":
    unittest.main()
