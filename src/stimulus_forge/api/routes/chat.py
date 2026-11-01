"""Conversational workflow assistant API endpoint matching desktop studio capabilities."""

from __future__ import annotations

from typing import Any
from fastapi import APIRouter

from stimulus_forge.api.schemas import ChatChoice, ChatMessageRequest, ChatMessageResponse
from stimulus_forge.services.workflow_agent import (
    WorkflowCommand,
    command_from_message,
    concept_from_message,
)


router = APIRouter(prefix="/chat", tags=["Workflow Assistant"])


@router.post("/message", response_model=ChatMessageResponse, summary="Send Message to Workflow Assistant")
def chat_message(request: ChatMessageRequest) -> ChatMessageResponse:
    """Process a conversational query or app command and return assistant guidance and UI actions."""
    text = request.message.strip()
    lowered = text.casefold()

    # 1. Help Command
    if lowered in {"help", "?", "commands"}:
        return ChatMessageResponse(
            reply=(
                "I can guide you step-by-step or run direct app commands. "
                "You can add a concept, generate related concepts, acquire images, "
                "delete concepts, or export stimuli.\n\n"
                "Examples:\n"
                "• “basketball” (adds concept and starts wizard)\n"
                "• “generate related concepts for basketball”\n"
                "• “create images for basketball”\n"
                "• “list concepts”\n"
                "• “delete concept basketball”\n"
                "• “clear all related concepts”\n"
                "• “crop selected images”\n"
                "• “export task stimuli”"
            ),
            state=request.state,
            current_concept=request.current_concept,
            choices=[
                ChatChoice(label="List concepts", action="list_concepts"),
                ChatChoice(label="Add 'Albert Einstein'", action="add_sample", payload={"concept": "Albert Einstein"}),
                ChatChoice(label="Export stimuli", action="export_stimuli"),
            ],
        )

    # 2. Status Command
    if lowered in {"status", "show status"}:
        total = len(request.workspace_concepts)
        return ChatMessageResponse(
            reply=f"Workspace has {total} loaded concept(s): {', '.join(request.workspace_concepts) if total else 'None'}. Current active concept is '{request.current_concept or 'None'}'.",
            state=request.state,
            current_concept=request.current_concept,
            choices=[ChatChoice(label="Help", action="help")],
        )

    # 3. Direct App Commands via WorkflowAgent Parser
    cmd: WorkflowCommand | None = command_from_message(text)
    if cmd is not None:
        action = cmd.action
        if action == "list_concepts":
            total = len(request.workspace_concepts)
            concepts_str = ", ".join(request.workspace_concepts) if total else "Workspace is empty."
            return ChatMessageResponse(
                reply=f"Concepts ({total}): {concepts_str}",
                state=request.state,
                current_concept=request.current_concept,
                choices=[ChatChoice(label="Add concept", action="prompt_add")],
            )

        if action in {"generate_related", "create_images"}:
            target = cmd.target or request.current_concept
            if not target:
                return ChatMessageResponse(
                    reply="Please specify a concept name, e.g. 'generate related concepts for basketball'.",
                    state=request.state,
                )
            if action == "generate_related":
                return ChatMessageResponse(
                    reply=f"Use Local GGUF or a Cloud language model for '{target}'?",
                    state="awaiting_text_provider",
                    current_concept=target,
                    ui_action="set_concept",
                    ui_payload={"concept": target},
                    choices=[
                        ChatChoice(label="Local GGUF ($0.00)", action="select_text_provider", payload={"provider": "local"}),
                        ChatChoice(label="Google Gemini Flash", action="select_text_provider", payload={"provider": "google"}),
                        ChatChoice(label="OpenAI Structured", action="select_text_provider", payload={"provider": "openai"}),
                        ChatChoice(label="Anthropic Claude", action="select_text_provider", payload={"provider": "anthropic"}),
                    ],
                )
            return ChatMessageResponse(
                reply=f"Where should images for '{target}' come from?",
                state="awaiting_image_source",
                current_concept=target,
                choices=[
                    ChatChoice(label="Bing SafeSearch", action="select_image_source", payload={"source": "bing"}),
                    ChatChoice(label="Text Stimulus (Centered)", action="select_image_source", payload={"source": "text"}),
                    ChatChoice(label="Local Diffusers", action="select_image_source", payload={"source": "local"}),
                    ChatChoice(label="Hugging Face (Cloud)", action="select_image_source", payload={"source": "huggingface"}),
                    ChatChoice(label="OpenAI DALL-E", action="select_image_source", payload={"source": "openai"}),
                    ChatChoice(label="Google Imagen", action="select_image_source", payload={"source": "google"}),
                ],
            )

        if action == "add_related":
            return ChatMessageResponse(
                reply=f"Added related concept '{cmd.target}' to '{cmd.parent}'.",
                state=request.state,
                ui_action="add_related",
                ui_payload={"concept": cmd.parent, "related": cmd.target},
                choices=[ChatChoice(label="Acquire images", action="offer_image_sources")],
            )

        if action == "delete_concept":
            target = cmd.target
            return ChatMessageResponse(
                reply=f"Deleted concept '{target}'.",
                state="awaiting_concept",
                ui_action="delete_concept",
                ui_payload={"concept": target},
                choices=[ChatChoice(label="List concepts", action="list_concepts")],
            )

        if action in {"clear_all_related", "clear_related"}:
            return ChatMessageResponse(
                reply="Cleared related concepts.",
                state="awaiting_concept",
                ui_action="clear_all_related",
                ui_payload={},
            )

        if action in {"clear_all_images", "clear_images"}:
            return ChatMessageResponse(
                reply="Cleared all workspace images.",
                state="awaiting_concept",
                ui_action="clear_all_images",
                ui_payload={},
            )

        if action == "crop_selected":
            return ChatMessageResponse(
                reply="Running frontal face detection & auto-crop...",
                state=request.state,
                ui_action="crop_selected",
                ui_payload={},
            )

        if action == "export_stimuli":
            return ChatMessageResponse(
                reply="Exporting stimulus sets to downloads...",
                state=request.state,
                ui_action="export_stimuli",
                ui_payload={},
            )

    # 4. State-Driven Conversational Flow
    if request.state == "awaiting_text_provider":
        for provider_key in ("local", "openai", "google", "anthropic", "compatible"):
            if provider_key in lowered:
                return ChatMessageResponse(
                    reply=f"Generating related concepts for '{request.current_concept}' using {provider_key.upper()}...",
                    state="awaiting_related_review",
                    current_concept=request.current_concept,
                    ui_action="generate_concepts",
                    ui_payload={"concept": request.current_concept, "provider": provider_key},
                    choices=[
                        ChatChoice(label="Create images", action="offer_image_sources"),
                        ChatChoice(label="Regenerate", action="offer_text_providers"),
                    ],
                )

    if request.state == "awaiting_image_source":
        for source_key in ("bing", "text", "local", "huggingface", "openai", "google"):
            if source_key in lowered or (source_key == "huggingface" and "hf" in lowered) or (source_key == "text" and "word" in lowered):
                return ChatMessageResponse(
                    reply=f"Acquiring images for '{request.current_concept}' using {source_key.upper()}...",
                    state="complete",
                    current_concept=request.current_concept,
                    ui_action="acquire_images",
                    ui_payload={"source": source_key},
                    choices=[
                        ChatChoice(label="Add another concept", action="restart"),
                        ChatChoice(label="Export stimuli", action="export_stimuli"),
                    ],
                )

    # 5. Default: Treat as a new seed concept entry
    concept = concept_from_message(text)
    if concept:
        return ChatMessageResponse(
            reply=f"Added '{concept}'. How should I generate its related concepts?",
            state="awaiting_text_provider",
            current_concept=concept,
            ui_action="set_concept",
            ui_payload={"concept": concept},
            choices=[
                ChatChoice(label="Local GGUF ($0.00)", action="select_text_provider", payload={"provider": "local"}),
                ChatChoice(label="Google Gemini Flash", action="select_text_provider", payload={"provider": "google"}),
                ChatChoice(label="OpenAI Structured", action="select_text_provider", payload={"provider": "openai"}),
                ChatChoice(label="Anthropic Claude", action="select_text_provider", payload={"provider": "anthropic"}),
                ChatChoice(label="Cancel", action="restart"),
            ],
        )

    return ChatMessageResponse(
        reply="I didn't quite catch that. Type a concept name (e.g. 'baseball') or type 'help' for command examples.",
        state="awaiting_concept",
        choices=[ChatChoice(label="Help", action="help")],
    )
