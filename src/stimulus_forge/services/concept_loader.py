"""Load a normalized concept list from common text and spreadsheet formats."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from stimulus_forge.services.questionnaire_intelligence import parse_questionnaire


SUPPORTED_CONCEPT_FILE_TYPES = {".txt", ".csv", ".xlsx"}


def _concept_cells(file_path: Path) -> list[object]:
    suffix = file_path.suffix.lower()
    if suffix == ".txt":
        return [file_path.read_text(encoding="utf-8-sig")]
    if suffix == ".csv":
        table = pd.read_csv(file_path, header=None, dtype=str, keep_default_na=False)
    elif suffix == ".xlsx":
        table = pd.read_excel(file_path, header=None, dtype=str, keep_default_na=False)
    else:
        supported = ", ".join(sorted(SUPPORTED_CONCEPT_FILE_TYPES))
        raise ValueError(f"Unsupported concept file type. Choose one of: {supported}.")

    if table.empty:
        return []

    first_row = [str(value).strip().casefold() for value in table.iloc[0].tolist()]
    concept_columns = [
        index
        for index, heading in enumerate(first_row)
        if heading in {"name", "stimulus", "stimuli"} or "concept" in heading
    ]
    if concept_columns:
        return table.iloc[1:, concept_columns].to_numpy().ravel().tolist()
    return table.to_numpy().ravel().tolist()


def load_concepts(file_path: str | Path, subject_id: str | None = None) -> list[str]:
    """Return unique, lowercase concepts while preserving their file order."""
    path = Path(file_path).expanduser().resolve()
    if not path.is_file():
        raise ValueError(f"Concept file does not exist: {path}")

    # Check if the file is a clinical or neuroscience survey response / questionnaire
    suffix = path.suffix.lower()
    if suffix in {".csv", ".tsv", ".xlsx", ".xls"}:
        q_res = parse_questionnaire(path, subject_id=subject_id)
        if q_res.is_questionnaire and q_res.selected_concepts:
            return q_res.selected_concepts

    concepts: list[str] = []
    seen: set[str] = set()
    for cell in _concept_cells(path):
        if cell is None:
            continue
        for value in re.split(r"[;,\n]+", str(cell)):
            concept = value.strip().casefold()
            if concept and concept not in seen:
                concepts.append(concept)
                seen.add(concept)

    if not concepts:
        raise ValueError("The selected file does not contain any concepts.")
    return concepts


def workspace_name_for(file_path: str | Path, subject_id: str | None = None) -> str:
    """Create a filesystem-safe workspace name from a concept-list filename or subject ID."""
    if subject_id and subject_id.strip():
        safe_sub = re.sub(r"[^\w.-]+", "_", subject_id.strip(), flags=re.UNICODE).strip("._")
        if safe_sub:
            return safe_sub

    stem = Path(file_path).stem.strip()
    safe_name = re.sub(r"[^\w.-]+", "_", stem, flags=re.UNICODE).strip("._")
    return safe_name or "concepts"
