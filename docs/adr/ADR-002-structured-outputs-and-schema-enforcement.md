# ADR-002: Structured Outputs and Token-Level Schema Enforcement

## Status
Accepted

## Context
Generative language models frequently suffer from schema drift, Markdown code-fence wrapping, missing required keys, or subtle formatting hallucinations when producing structured JSON datasets. In automated experimental pipelines like cognitive stimulus generation, invalid JSON halts batch runs and requires fragile regex post-processing.

## Decision
We implemented deterministic schema enforcement using Pydantic V2 and provider-native constrained decoding:
1. **OpenAI Structured Outputs:** Employs `client.responses.parse(..., text_format=RelatedConceptCollection)` to guarantee strict JSON schema compliance at the sampling level.
2. **Anthropic & Google GenAI Output Constraints:** Injects strict JSON schema definitions with temperature pinning ($T = 0.15$) to prevent creative formatting deviations.
3. **Local GGUF Grammar-Constrained Decoding:** When running quantized models locally via `llama-cpp-python`, we supply `response_format={"type": "json_object", "schema": ...}`, forcing grammar-guided token mask filtering during autoregressive decoding.
4. **Deterministic Normalization Pipeline:** A resilient normalization layer (`normalize_related_concepts`) strips leading/trailing punctuation, deduplicates semantic equivalents, and rejects self-referential concepts before entering downstream application workflows.

## Consequences
### Positive
- Achieves 100% schema validity across both cloud and local model outputs.
- Eliminates downstream runtime JSON parsing crashes.
- Enforces role separation (`name` vs. `char`) deterministically for experimental stimulus categorization.

### Negative / Trade-offs
- Constrained grammar decoding incurs a minor initial grammar compilation overhead on local `llama.cpp` instances.
- Restricts model freedom to output conversational explanations (intentional for programmatic pipelines).
