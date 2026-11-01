"""Prompt injection defense, input sanitization, and adversarial protection services."""

from __future__ import annotations

import re
from typing import Tuple


# Common direct and indirect prompt injection, jailbreak, and system override patterns
INJECTION_PATTERNS = [
    r"\bignore\s+(?:all\s+)?(?:previous|prior|above|former)\s+(?:instructions|rules|prompts|directives)\b",
    r"\bdisregard\s+(?:all\s+)?(?:previous|prior|above)\s+(?:instructions|rules|prompts|directives)\b",
    r"\bforget\s+(?:all\s+)?(?:previous|prior|above)\s+(?:instructions|rules|prompts)\b",
    r"\bsystem\s*override\b",
    r"\byou\s+are\s+now\s+(?:in\s+)?(?:dan|developer|jailbreak|unrestricted|god)\s+mode\b",
    r"\byou\s+are\s+no\s+longer\s+(?:an?\s+)?(?:ai|assistant)\b",
    r"\bfrom\s+now\s+on\s+you\s+(?:will|must|can)\s+do\s+anything\b",
    r"\brepeat\s+(?:the\s+)?(?:system\s+prompt|instructions\s+above|everything\s+above)\b",
    r"\boutput\s+(?:the\s+)?(?:system\s+prompt|secret|api\s*key|initial\s+prompt)\b",
    r"\bwhat\s+(?:is|are)\s+your\s+(?:system\s+prompt|hidden\s+instructions)\b",
    r"<\s*\|\s*(?:im_start|im_end|endoftext|system|user|assistant)\s*\|\s*>",
    r"\[\s*/?\s*INST\s*\]",
    r"<<\s*/?\s*SYS\s*>>",
    r"<\s*/?\s*system\s*>",
]

INJECTION_REGEX = re.compile("|".join(INJECTION_PATTERNS), re.IGNORECASE)


def detect_prompt_injection(text: str) -> Tuple[bool, str]:
    """Check text for known prompt injection, jailbreak, or system override patterns.

    Returns (is_injection, matched_pattern).
    """
    if not text or not isinstance(text, str):
        return False, ""
    match = INJECTION_REGEX.search(text)
    if match:
        return True, match.group(0)
    return False, ""


def sanitize_prompt_text(text: str, max_length: int = 100) -> str:
    """Sanitize a concept or user input string:
    - Strips special LLM control and delimiter tokens
    - Collapses multiple newlines/whitespace
    - Truncates to max_length
    """
    if not text:
        return ""
    # Strip dangerous LLM special control tokens
    cleaned = re.sub(r"<\s*\|\s*[^|>]+\s*\|\s*>", "", text)
    cleaned = re.sub(r"\[\s*/?\s*INST\s*\]", "", cleaned)
    cleaned = re.sub(r"<<\s*/?\s*SYS\s*>>", "", cleaned)
    # Remove quotes that might attempt prompt breakout
    cleaned = cleaned.replace('"', "'").replace("`", "")
    # Single-line normalized
    cleaned = " ".join(cleaned.split())
    return cleaned[:max_length].strip()


def sanitize_document_content_for_rag(text: str) -> str:
    """Neutralize potential prompt injection vectors inside untrusted document text:
    - Escapes boundary tags
    - Strips special control tokens
    - Neutralizes prompt override instructions
    """
    if not text:
        return ""
    # Strip LLM control tokens
    cleaned = re.sub(r"<\s*\|\s*[^|>]+\s*\|\s*>", "", text)
    cleaned = re.sub(r"\[\s*/?\s*INST\s*\]", "", cleaned)
    cleaned = re.sub(r"<<\s*/?\s*SYS\s*>>", "", cleaned)
    # Prevent XML boundary tag spoofing
    cleaned = cleaned.replace("<untrusted_document_content>", "[DOCUMENT_START]")
    cleaned = cleaned.replace("</untrusted_document_content>", "[DOCUMENT_END]")
    cleaned = cleaned.replace('"""', "'''")
    # Redact injection phrases inside document text
    cleaned = INJECTION_REGEX.sub("[BLOCKED_INSTRUCTION]", cleaned)
    return cleaned
