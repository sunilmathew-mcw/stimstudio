"""Pure helpers used by the guided workflow assistant."""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class WorkflowCommand:
    """A deterministic command extracted from conversational text."""

    action: str
    target: str = ''
    parent: str = ''
    new_name: str = ''


def _clean_target(value: str) -> str:
    return ' '.join(value.strip().strip('"\'“”‘’').split())


def concept_from_message(message: str) -> str:
    """Extract a concept from a short conversational add command."""
    value = str(message).strip()
    value = re.sub(
        r"^(?:please\s+)?(?:add|create|new)\s+(?:a\s+)?(?:concept\s+)?",
        "",
        value,
        flags=re.IGNORECASE,
    )
    return value.strip().strip('"\'“”‘’').strip()


def command_from_message(message: str) -> WorkflowCommand | None:
    """Parse supported app operations without letting an LLM execute arbitrary code."""
    text = ' '.join(str(message).strip().split())
    text = re.sub(
        r'^(?:please\s+|can you\s+|could you\s+|would you\s+|i want you to\s+)+',
        '',
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r'\s+please[?.!]?$', '', text, flags=re.IGNORECASE)
    text = text.rstrip('?.!').strip()
    lowered = text.casefold()
    exact_actions = {
        'help': 'help',
        '?': 'help',
        'status': 'status',
        'show status': 'status',
        'list concepts': 'list_concepts',
        'show concepts': 'list_concepts',
        'delete all related concepts': 'clear_all_related',
        'clear all related concepts': 'clear_all_related',
        'delete all images': 'clear_all_images',
        'clear all images': 'clear_all_images',
        'crop selected images': 'crop_selected',
        'crop images': 'crop_selected',
        'export task stimuli': 'export_stimuli',
        'export stimuli': 'export_stimuli',
        'load concepts': 'load_concepts',
        'load concept file': 'load_concepts',
        'choose local text model': 'choose_text_model',
        'choose local language model': 'choose_text_model',
        'choose local image model': 'choose_image_model',
        'choose picture library': 'choose_picture_library',
        'choose image workspace': 'choose_image_workspace',
        'choose download workspace': 'choose_image_workspace',
        'choose experiment folder': 'choose_experiment_folder',
        'delete selected image': 'delete_selected_image',
        'clear selected images': 'clear_selected_images',
        'select all concepts': 'check_all_concepts',
        'check all concepts': 'check_all_concepts',
        'deselect all concepts': 'uncheck_all_concepts',
        'uncheck all concepts': 'uncheck_all_concepts',
        'replace existing images': 'replace_images_on',
        'keep existing images': 'replace_images_off',
    }
    if lowered in exact_actions:
        return WorkflowCommand(exact_actions[lowered])

    if re.match(
        r'^(?:delete|clear|remove) all (?:the )?related concepts(?: retrieved)?$',
        text,
        flags=re.IGNORECASE,
    ):
        return WorkflowCommand('clear_all_related')
    if re.match(
        r'^(?:delete|clear|remove) all (?:the )?images$',
        text,
        flags=re.IGNORECASE,
    ):
        return WorkflowCommand('clear_all_images')

    patterns = (
        (r'^(?:list|show) related concepts(?: for)? (.+)$', 'list_related'),
        (r'^(?:generate|get) related concepts(?: for)? (.+)$', 'generate_related'),
        (r'^(?:create|generate|get) images for (.+?) from (.+)$', 'create_related_images'),
        (r'^(?:create|generate|get) images(?: for)? (.+)$', 'create_images'),
        (r'^(?:add|insert) related concept (.+?) (?:to|for|under) (.+)$', 'add_related'),
        (r'^(?:delete|remove) (?:the )?related concept (.+?) from (.+)$', 'delete_related'),
        (r'^(?:clear|delete) (?:all )?(?:the )?images for (.+?) from (.+)$', 'clear_related_images'),
        (r'^(?:clear|delete) (?:all )?(?:the )?related concepts(?: for| from) (.+)$', 'clear_related'),
        (r'^(?:clear|delete) (?:all )?(?:the )?images(?: for| from| under) (.+)$', 'clear_images'),
        (r'^(?:delete|remove) (?:the )?concept (.+)$', 'delete_concept'),
        (r'^rename related concept (.+?) to (.+?) (?:in|under|from) (.+)$', 'rename_related'),
        (r'^rename selected image to (.+)$', 'rename_selected_image'),
        (r'^rename concept (.+?) to (.+)$', 'rename_concept'),
        (r'^(?:uncheck|deselect) related concept (.+?) from (.+)$', 'uncheck_related'),
        (r'^(?:check|select) related concept (.+?) from (.+)$', 'check_related'),
        (r'^(?:uncheck|deselect) concept (.+)$', 'uncheck_concept'),
        (r'^(?:check|select) concept (.+)$', 'check_concept'),
        (r'^set images per concept to (\d+)$', 'set_image_count'),
    )
    for pattern, action in patterns:
        match = re.match(pattern, text, flags=re.IGNORECASE)
        if not match:
            continue
        values = [_clean_target(value) for value in match.groups()]
        if action in {
            'add_related', 'delete_related', 'create_related_images', 'clear_related_images',
            'check_related', 'uncheck_related',
        }:
            return WorkflowCommand(action, target=values[0], parent=values[1])
        if action == 'rename_related':
            return WorkflowCommand(
                action, target=values[0], new_name=values[1], parent=values[2]
            )
        if action == 'rename_concept':
            return WorkflowCommand(action, target=values[0], new_name=values[1])
        if action == 'rename_selected_image':
            return WorkflowCommand(action, new_name=values[0])
        return WorkflowCommand(action, target=values[0])
    return None
