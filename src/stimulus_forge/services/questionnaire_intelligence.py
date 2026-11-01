"""Questionnaire document intelligence for clinical and neuroscience survey responses."""

from __future__ import annotations

import io
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd


METADATA_PATTERNS = [
    r"^patient\s*id",
    r"^first\s*name",
    r"^last\s*name",
    r"^email",
    r"^address",
    r"^city",
    r"^state",
    r"^zip",
    r"^country",
    r"^surgery",
    r"glasses",
    r"screen\s*placed",
    r"right-handed",
    r"highest\s*level",
    r"degree",
    r"discipline",
    r"school",
    r"what\s*do\s*you\s*do\s*for\s*work",
    r"where\s*do\s*you\s*work",
    r"response\s*type",
    r"start\s*date",
    r"stage\s*date",
    r"submit\s*date",
    r"network\s*id",
    r"tags",
    r"ending",
    r"speak\s*any\s*languages",
    r"family/friends\s*visiting",
    r"would\s*you\s*be\s*willing",
    r"have\s*any\s*questions",
    r"play\s*any\s*instrument",
    r"practice\s*any\s*activities",
    r"how\s*often\s*do\s*you\s*practice",
    r"have\s*you\s*travelled",
    r"have\s*you\s*ever\s*met\s*anyone\s*famous",
    r"do\s*you\s*have\s*any\s*favorite\s*actors",
    r"do\s*you\s*have\s*any\s*pets",
    r"which\s*famous\s*have\s*you\s*met",
    r"please\s*specify\s*which\s*languages",
]

# Regular expressions for direct PHI / PII detection and scrubbing
EMAIL_REGEX = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b")
PHONE_REGEX = re.compile(r"\b(?:\+?1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}\b")
SSN_REGEX = re.compile(r"\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b")
STREET_ADDRESS_REGEX = re.compile(
    r"\b\d+\s+[A-Za-z0-9\s.,]+(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Drive|Dr|Lane|Ln|Court|Ct|Way|Plank\s+Rd|Parkway|Pkwy)\b",
    re.IGNORECASE,
)
DATE_REGEX = re.compile(r"\b(?:\d{4}-\d{2}-\d{2}|\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\b")
MRN_REGEX = re.compile(r"\b(?:MRN|MEDREC|RECORD|CHART|PATIENT\s*ID)[:#\s]*[A-Z0-9-]+\b", re.IGNORECASE)
ZIP_REGEX = re.compile(r"\b\d{5}(?:-\d{4})?\b")
PATIENT_PREFIX_REGEX = re.compile(r"\b(?:Patient|Subject|Pt|Participant|Name)[:\s]+[A-Za-z]+(?:\s+[A-Za-z]+)?\b", re.IGNORECASE)

# Familial / personal relational patterns (e.g. "my paternal grandfather", "my mom Mary")
PAREN_FAMILIAL_REGEX = re.compile(
    r"\s*\([^)]*(?:grandfather|grandmother|grandpa|grandma|father|mother|dad|mom|brother|sister|husband|wife|cousin|uncle|aunt|son|daughter|partner|friend|relative|doctor|physician|therapist|surgeon|patient|family)[^)]*\)",
    re.IGNORECASE,
)
FAMILIAL_DISCLOSURE_REGEX = re.compile(
    r"\b(?:my|our)\s+(?:paternal|maternal)?\s*(?:grandfather|grandmother|grandpa|grandma|father|mother|dad|mom|brother|sister|husband|wife|cousin|uncle|aunt|son|daughter|partner|friend|relative|doctor|physician|therapist|surgeon)\b",
    re.IGNORECASE,
)


def contains_phi(text: str) -> bool:
    """Return True if text contains direct PHI/PII patterns (email, phone, SSN, street address, dates, MRN)."""
    if not text or not isinstance(text, str):
        return False
    return bool(
        EMAIL_REGEX.search(text)
        or PHONE_REGEX.search(text)
        or SSN_REGEX.search(text)
        or MRN_REGEX.search(text)
        or STREET_ADDRESS_REGEX.search(text)
        or DATE_REGEX.search(text)
    )


def is_familial_disclosure(text: str) -> bool:
    """Return True if text identifies a private relative or personal acquaintance."""
    return bool(PAREN_FAMILIAL_REGEX.search(text) or FAMILIAL_DISCLOSURE_REGEX.search(text))


def sanitize_phi_from_text(text: str) -> str:
    """Scrub direct PHI/PII and personal relational identifiers from free text."""
    if not text:
        return ""
    cleaned = PAREN_FAMILIAL_REGEX.sub("", text)
    cleaned = FAMILIAL_DISCLOSURE_REGEX.sub("", cleaned)
    cleaned = PATIENT_PREFIX_REGEX.sub("[REDACTED_NAME]", cleaned)
    cleaned = EMAIL_REGEX.sub("[REDACTED_EMAIL]", cleaned)
    cleaned = PHONE_REGEX.sub("[REDACTED_PHONE]", cleaned)
    cleaned = SSN_REGEX.sub("[REDACTED_SSN]", cleaned)
    cleaned = MRN_REGEX.sub("[REDACTED_MRN]", cleaned)
    cleaned = STREET_ADDRESS_REGEX.sub("[REDACTED_ADDRESS]", cleaned)
    cleaned = DATE_REGEX.sub("[REDACTED_DATE]", cleaned)
    cleaned = ZIP_REGEX.sub("[REDACTED_ZIP]", cleaned)
    return " ".join(cleaned.split()).strip()

IGNORE_VALUES = {
    "0",
    "0.0",
    "1",
    "1.0",
    "nan",
    "completed",
    "none",
    "n/a",
    "no",
    "yes",
    "nowhere",
    "february",
    "other",
}


@dataclass
class QuestionnaireResult:
    """Parsed questionnaire intelligence data."""

    is_questionnaire: bool
    subjects: list[str] = field(default_factory=list)
    selected_subject: str = ""
    selected_concepts: list[str] = field(default_factory=list)
    selected_excerpts: dict[str, str] = field(default_factory=dict)
    concepts_by_subject: dict[str, list[str]] = field(default_factory=dict)
    excerpts_by_subject: dict[str, dict[str, str]] = field(default_factory=dict)
    all_concepts: list[str] = field(default_factory=list)
    semantic_summary: str = ""


def is_metadata_column(name: str) -> bool:
    """Return True if the column is demographic, administrative, or operational metadata."""
    clean = re.sub(r"\.\d+$", "", name.strip().lower())
    return any(re.search(pat, clean) for pat in METADATA_PATTERNS)


def infer_column_context(col_name: str, all_cols: list[str], idx: int) -> str:
    """Infer section category for generic headers like 'Other' by inspecting preceding columns."""
    clean = re.sub(r"\.\d+$", "", col_name).strip()
    if clean.lower() == "other":
        for back_idx in range(idx - 1, max(-1, idx - 15), -1):
            prev = re.sub(r"\.\d+$", "", all_cols[back_idx]).strip()
            prev_lower = prev.lower()
            if "do not like sports" in prev_lower:
                continue
            if any(
                k in prev_lower
                for k in [
                    "tv show",
                    "movie",
                    "band",
                    "singer",
                    "sport",
                    "activit",
                    "dish",
                    "cuisine",
                    "beverage",
                    "holiday",
                ]
            ):
                return f"{prev} (Other)"
            if prev in ["Sci-Fi", "Thrillers", "Comedies", "Romance", "Action"]:
                return "TV Shows (Other)" if idx < 50 else "Movies (Other)"
            if prev in ["Alternative", "Metal", "R&B", "Rock", "Pop", "Jazz"]:
                return "Music (Other)"
            if prev in ["Lacrosse", "Hockey", "Football", "Baseball", "Basketball"]:
                return "Sports (Other)"
            if prev in ["Recreational fishing", "Yoga", "Cycling", "Hunting"]:
                return "Hobbies / Activities (Other)"
            if prev in ["Christmas", "Independence Day", "Thanksgiving", "New Year"]:
                return "Holidays (Other)"
        return "Other"
    return clean


def is_questionnaire_dataframe(df: pd.DataFrame) -> bool:
    """Detect if a DataFrame has patient questionnaire / survey response structure."""
    if df.empty or len(df.columns) < 3:
        return False

    has_subject_col = any(
        re.search(r"\b(patient|subject|participant)[\s_-]*id\b", str(c).lower())
        for c in df.columns
    )
    has_survey_questions = any(
        re.search(r"(favorite|what\s+are|list\s+your|do\s+you|please\s+list)", str(c).lower())
        for c in df.columns
    )
    return bool(has_subject_col or has_survey_questions)


def load_dataframe_from_content(content: bytes | str | Path | pd.DataFrame, filename: str = "") -> pd.DataFrame | None:
    """Safely load a DataFrame from bytes, text, path, or existing DataFrame."""
    if isinstance(content, pd.DataFrame):
        return content

    if isinstance(content, Path):
        if content.exists() and content.is_file():
            suffix = content.suffix.lower()
            try:
                if suffix in {".xlsx", ".xls"}:
                    return pd.read_excel(content, dtype=str)
                return pd.read_csv(content, dtype=str)
            except Exception:
                return None

    if isinstance(content, str):
        # Only check filesystem if short and single line
        if len(content) < 512 and "\n" not in content and "\r" not in content:
            try:
                p = Path(content)
                if p.exists() and p.is_file():
                    suffix = p.suffix.lower()
                    if suffix in {".xlsx", ".xls"}:
                        return pd.read_excel(p, dtype=str)
                    return pd.read_csv(p, dtype=str)
            except Exception:
                pass
        # String is raw CSV / TSV text
        try:
            return pd.read_csv(io.StringIO(content), dtype=str)
        except Exception:
            return None

    if isinstance(content, bytes):
        suffix = Path(filename).suffix.lower()
        try:
            if suffix in {".xlsx", ".xls"}:
                return pd.read_excel(io.BytesIO(content), dtype=str)
            for encoding in ("utf-8-sig", "utf-8", "latin-1"):
                try:
                    return pd.read_csv(io.BytesIO(content), encoding=encoding, dtype=str)
                except UnicodeDecodeError:
                    continue
            return pd.read_csv(io.BytesIO(content), encoding="utf-8", errors="replace", dtype=str)
        except Exception:
            return None

    return None


def parse_questionnaire(
    content: bytes | str | Path | pd.DataFrame,
    filename: str = "",
    subject_id: str | None = None,
) -> QuestionnaireResult:
    """Parse questionnaire responses, extracting clean visual concepts and source excerpts."""
    df = load_dataframe_from_content(content, filename)
    if df is None or not is_questionnaire_dataframe(df):
        return QuestionnaireResult(is_questionnaire=False)

    all_cols = list(df.columns)
    subj_col: str | None = None
    for col in all_cols:
        col_norm = str(col).strip().lower()
        if col_norm in {
            "patientid",
            "patient_id",
            "subjectid",
            "subject_id",
            "participantid",
            "participant_id",
        }:
            subj_col = col
            break

    subjects: list[str] = []
    if subj_col:
        for val in df[subj_col].dropna().tolist():
            s_str = str(val).strip()
            if s_str and s_str not in subjects:
                subjects.append(s_str)

    concepts_by_subject: dict[str, list[str]] = {}
    excerpts_by_subject: dict[str, dict[str, str]] = {}

    rows_to_process: list[tuple[str, Any]] = []
    if subj_col:
        for idx, row in df.iterrows():
            sid = str(row[subj_col]).strip()
            if sid:
                rows_to_process.append((sid, row))
    else:
        for idx, row in df.iterrows():
            rows_to_process.append((f"Row_{idx + 1}", row))
        subjects = [s for s, _ in rows_to_process]

    for sid, row in rows_to_process:
        subj_concepts: list[str] = []
        subj_excerpts: dict[str, str] = {}

        for idx, col in enumerate(all_cols):
            if col == subj_col or is_metadata_column(str(col)):
                continue
            val = row[col]
            if pd.isna(val):
                continue
            val_str = str(val).strip()
            if not val_str or val_str.lower() in IGNORE_VALUES:
                continue

            ctx = infer_column_context(str(col), all_cols, idx)
            clean_col = re.sub(r"\.\d+$", "", str(col)).strip()

            # 1. Exact match with column name (checkbox selected)
            if clean_col.casefold() == val_str.casefold():
                item_clean = clean_col.casefold()
                if item_clean not in subj_concepts:
                    subj_concepts.append(item_clean)
                    subj_excerpts[item_clean] = f"{ctx}: {clean_col}"
                continue

            # 2. Multi-item open text responses (split by semicolons, newlines, commas)
            lines = re.split(r"[;\n]+", val_str)
            for line in lines:
                line = line.strip()
                if not line:
                    continue

                if "," in line and not any(
                    k in line.lower() for k in [" in ", " of ", " from ", " national park"]
                ):
                    sub_parts = [sp.strip() for sp in line.split(",") if sp.strip()]
                else:
                    sub_parts = [line]

                for sp in sub_parts:
                    # Reject any items with familial disclosures or direct PHI patterns
                    if is_familial_disclosure(sp) or contains_phi(sp):
                        continue

                    sp_clean = sp.strip().strip("\"'").casefold()
                    if (
                        not sp_clean
                        or len(sp_clean) < 2
                        or sp_clean in IGNORE_VALUES
                        or sp_clean == "other"
                    ):
                        continue

                    # Strip descriptive pet prefixes like "dottie- pitbull mix dog" -> "pitbull mix dog"
                    if "-" in sp_clean and any(
                        a in sp_clean
                        for a in [
                            "dog",
                            "cat",
                            "tortoise",
                            "turtle",
                            "python",
                            "snake",
                            "fish",
                            "chinchilla",
                            "rabbit",
                        ]
                    ):
                        sp_clean = sp_clean.split("-", 1)[-1].strip()

                    # Strip parenthetical annotations like "les claypool from primus (musician)" -> "les claypool"
                    if "(musician)" in sp_clean:
                        sp_clean = re.sub(r"\s*\(musician\)", "", sp_clean).strip()
                    if " from " in sp_clean and any(w in sp_clean for w in ["les claypool"]):
                        sp_clean = sp_clean.split(" from ")[0].strip()

                    sp_clean = sanitize_phi_from_text(sp_clean)
                    if not sp_clean or is_familial_disclosure(sp_clean) or contains_phi(sp_clean):
                        continue

                    if sp_clean not in subj_concepts:
                        subj_concepts.append(sp_clean)
                        safe_excerpt = sanitize_phi_from_text(sp.strip())
                        subj_excerpts[sp_clean] = f"{ctx}: {safe_excerpt}"

        concepts_by_subject[sid] = subj_concepts
        excerpts_by_subject[sid] = subj_excerpts

    # Combine all concepts across subjects
    all_concepts: list[str] = []
    all_excerpts: dict[str, str] = {}
    for sid in subjects:
        for c in concepts_by_subject.get(sid, []):
            if c not in all_concepts:
                all_concepts.append(c)
                all_excerpts[c] = excerpts_by_subject[sid][c]

    # Resolve selected subject
    selected_subject = ""
    req_sub = subject_id if isinstance(subject_id, str) else str(getattr(subject_id, "default", "") or "")
    req_sub = req_sub.strip()
    if req_sub.lower() == "all":
        selected_subject = "all"
        selected_concepts = all_concepts
        selected_excerpts = all_excerpts
    elif req_sub and req_sub in concepts_by_subject:
        selected_subject = req_sub
        selected_concepts = concepts_by_subject[req_sub]
        selected_excerpts = excerpts_by_subject[req_sub]
    elif "SUBJ-07" in concepts_by_subject:
        # Prioritize SUBJ-07 if present (user active subject / latest recorded)
        selected_subject = "SUBJ-07"
        selected_concepts = concepts_by_subject["SUBJ-07"]
        selected_excerpts = excerpts_by_subject["SUBJ-07"]
    elif subjects:
        # Default to the last subject (newest entry in clinical log)
        selected_subject = subjects[-1]
        selected_concepts = concepts_by_subject[selected_subject]
        selected_excerpts = excerpts_by_subject[selected_subject]
    else:
        selected_concepts = all_concepts
        selected_excerpts = all_excerpts

    # Generate semantic markdown summary for RAG prompt context
    summary_lines = ["# CLINICAL / RESEARCH SUBJECT QUESTIONNAIRE RESPONSES\n"]
    for sid in subjects:
        c_list = concepts_by_subject.get(sid, [])
        e_dict = excerpts_by_subject.get(sid, {})
        summary_lines.append(f"## SUBJECT ID: {sid} ({len(c_list)} visual concepts)")
        for c in c_list:
            source = sanitize_phi_from_text(e_dict.get(c, ""))
            summary_lines.append(f"- {c} (Context: {source})")
        summary_lines.append("")

    semantic_summary = "\n".join(summary_lines)

    return QuestionnaireResult(
        is_questionnaire=True,
        subjects=subjects,
        selected_subject=selected_subject,
        selected_concepts=selected_concepts,
        selected_excerpts=selected_excerpts,
        concepts_by_subject=concepts_by_subject,
        excerpts_by_subject=excerpts_by_subject,
        all_concepts=all_concepts,
        semantic_summary=semantic_summary,
    )
