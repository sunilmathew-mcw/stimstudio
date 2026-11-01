# ADR-003: FinOps Telemetry, Semantic Caching, and Hybrid Local/Cloud Routing

## Status
Accepted

## Context
Deploying Generative AI pipelines in research labs and enterprise healthcare requires strict financial governance, latency accountability, and compliance with data privacy mandates (HIPAA, IRB). Unmonitored API calls lead to cost explosions, redundant queries for common concepts, and potential vendor lock-in.

## Decision
We implemented a comprehensive FinOps and hybrid execution strategy:
1. **Telemetry & Cost Tracking Engine (`TelemetryTracker`):**
   - Captures per-transaction metrics: execution latency ($ms$), prompt tokens, completion tokens, and real-time dollar cost based on model-specific pricing tables.
   - Calculates **Cost Avoidance** metrics by comparing actual expenditures against baseline cloud models (e.g. GPT-4o).
   - Computes statistical percentiles ($p50$, $p95$, average latency) across providers.
2. **Two-Tier Model Routing (Local vs. Cloud):**
   - **Local Tier (Zero Cloud Leakage):** Quantized GGUF models (`llama.cpp`) and local Diffusers execute with $0.00 cost and zero network transmission, satisfying air-gapped lab privacy.
   - **Cloud Tier (High-Throughput Elasticity):** OpenAI, Anthropic, and Google Gemini endpoints handle complex associative reasoning with cost attribution.
3. **Semantic In-Memory LRU Cache:**
   - Caches generated concepts using normalized semantic keys with configurable TTL and LRU eviction.
   - Cache hits return in $<2$ ms with $0$ tokens and $0$ cost, drastically reducing redundant upstream calls.

## Consequences
### Positive
- Full visibility into token consumption, latency budgets, and cost breakdown per provider.
- Transparent quantification of cost savings achieved by utilizing local open-source models and caching.
- Seamless developer inspection via `/api/v1/telemetry/stats` and the web dashboard telemetry bar.

### Negative / Trade-offs
- In-memory cache is currently process-bound (can be backed by Redis in multi-instance horizontal deployments).
