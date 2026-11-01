# ADR-001: Decoupled Client-Server & Headless API Architecture

## Status
Accepted

## Context
Stimulus Forge was originally implemented as a monolithic desktop application using Qt/PySide. While this provided direct local filesystem access and native GPU acceleration on research lab workstations, it suffered from three critical limitations:
1. **Deployment Lock-In:** Researchers and collaborators could not access or run stimulus generation workflows without installing local Python environments, C++ Qt bindings, and platform-specific dependencies.
2. **Integration Inflexibility:** External lab automation scripts, web platforms, and data pipelines had no programmable interface to invoke stimulus generation as a microservice.
3. **Showcase Friction:** Potential reviewers, enterprise architects, and users faced high friction trying to clone, build, and run the desktop app locally.

## Decision
We refactored Stimulus Forge into a decoupled client-server architecture:
1. **Headless Engine (FastAPI Backend):** Encapsulates all domain logic, model selection, schema validation, image acquisition, and asynchronous batch processing behind REST and Server-Sent Events (SSE) streaming APIs.
2. **Pluggable Client Layer:**
   - **Modern Web Client:** A responsive Single-Page Application (SPA) served directly from the backend, providing live latency/cost counters, interactive concept curation, and real-time SSE progress indicators.
   - **Native Qt Desktop Client:** Retained as a rich client for air-gapped lab workstations requiring zero network dependencies.
3. **Containerization:** Packaged with multi-stage Docker and Docker Compose definitions for one-command deployment across cloud or on-premises servers.

## Consequences
### Positive
- Enables zero-install, browser-based usage and live cloud portfolio demonstrations.
- Provides interactive OpenAPI/Swagger (`/docs`) and Redoc documentation for third-party system integration.
- Supports asynchronous long-running batch stimulus generation with real-time SSE progress streaming.
- Preserves offline/local execution capabilities via local model providers (`llama.cpp`, local Diffusers).

### Negative / Trade-offs
- Introduces HTTP serialization and network overhead compared to in-memory Qt signals/slots.
- Requires network port binding and basic CORS management when deployed in distributed environments.
