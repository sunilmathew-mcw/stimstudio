"""Guided chat dock for the concept-to-image workflow."""

from __future__ import annotations

from html import escape

from qtpy import QtCore, QtWidgets
from qtpy.QtCore import Qt

from stimulus_forge.services.workflow_agent import (
    WorkflowCommand,
    command_from_message,
    concept_from_message,
)


class WorkflowChatDock(QtWidgets.QDockWidget):
    """A conversational, state-driven wrapper around the studio workflow."""

    def __init__(self, studio, parent=None):
        super().__init__("Workflow assistant", parent)
        self.studio = studio
        self.current_concept = ""
        self.current_related = ""
        self.state = "awaiting_concept"
        self.setObjectName("workflow_chat_dock")
        self.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
        self.setMinimumWidth(300)

        panel = QtWidgets.QWidget(self)
        layout = QtWidgets.QVBoxLayout(panel)
        layout.setContentsMargins(8, 8, 8, 8)

        heading = QtWidgets.QLabel("Concept workflow")
        heading.setStyleSheet("font-size: 15px; font-weight: 600;")
        layout.addWidget(heading)

        self.transcript = QtWidgets.QTextBrowser(panel)
        self.transcript.setOpenExternalLinks(False)
        self.transcript.setPlaceholderText("The workflow conversation appears here.")
        layout.addWidget(self.transcript, 1)

        self.activity_label = QtWidgets.QLabel("")
        self.activity_label.setWordWrap(True)
        self.activity_label.setStyleSheet("color: #666; font-size: 11px;")
        layout.addWidget(self.activity_label)

        self.choice_widget = QtWidgets.QWidget(panel)
        self.choice_layout = QtWidgets.QVBoxLayout(self.choice_widget)
        self.choice_layout.setContentsMargins(0, 0, 0, 0)
        self.choice_layout.setSpacing(5)
        layout.addWidget(self.choice_widget)

        input_row = QtWidgets.QHBoxLayout()
        self.input_edit = QtWidgets.QLineEdit(panel)
        self.input_edit.setPlaceholderText("Enter a concept, for example “Formula 1”…")
        self.input_edit.returnPressed.connect(self._submit_text)
        self.send_button = QtWidgets.QPushButton("Send", panel)
        self.send_button.clicked.connect(self._submit_text)
        input_row.addWidget(self.input_edit, 1)
        input_row.addWidget(self.send_button)
        layout.addLayout(input_row)

        self.setWidget(panel)
        self._assistant(
            "Tell me a concept to add. I’ll guide you through related concepts "
            "and image creation."
        )

        studio.c.related_concepts_done.connect(self._related_concepts_finished)
        studio.image_workflow_signals.finished.connect(self._images_finished)
        studio.image_workflow_signals.failed.connect(self._images_failed)
        studio.c.progress.connect(self._progress_changed)

    def _append_message(self, speaker: str, text: str, color: str) -> None:
        self.transcript.append(
            f'<div style="margin: 5px 0;"><b style="color:{color}">'
            f"{escape(speaker)}</b><br>{escape(str(text))}</div>"
        )
        scrollbar = self.transcript.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def _assistant(self, text: str) -> None:
        self._append_message("Assistant", text, "#1565c0")

    def _user(self, text: str) -> None:
        self._append_message("You", text, "#2e7d32")

    def _clear_choices(self) -> None:
        while self.choice_layout.count():
            item = self.choice_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

    def _choices(self, choices) -> None:
        self._clear_choices()
        for label, callback in choices:
            button = QtWidgets.QPushButton(label, self.choice_widget)
            button.setCursor(Qt.PointingHandCursor)
            button.clicked.connect(callback)
            self.choice_layout.addWidget(button)

    @staticmethod
    def _concept_from_message(message: str) -> str:
        return concept_from_message(message)

    def _submit_text(self) -> None:
        message = self.input_edit.text().strip()
        if not message:
            return
        self.input_edit.clear()
        self._user(message)

        lowered = message.casefold()
        if self.state == 'awaiting_confirmation':
            if lowered in {'confirm', 'yes', 'y'}:
                self._run_pending_action()
            elif lowered in {'cancel', 'no', 'n'}:
                self._cancel_chat_action()
            else:
                self._assistant('Type “confirm” or “cancel”, or use the buttons below.')
            return
        if lowered in {"restart", "start over", "reset"}:
            self._restart()
            return
        if lowered == "pause":
            if self.studio._concept_run_active:
                self.studio.toggle_related_concepts_pause()
            elif self.studio._image_run_active:
                self.studio.toggle_image_pause()
            else:
                self._assistant("There is no active job to pause.")
            return
        if lowered == "resume":
            if self.studio._concept_run_active and not self.studio._concept_pause_event.is_set():
                self.studio.toggle_related_concepts_pause()
            elif self.studio._image_run_active and not self.studio._image_pause_event.is_set():
                self.studio.toggle_image_pause()
            else:
                self._assistant("There is no paused job to resume.")
            return
        if lowered in {"stop", "cancel"}:
            if self.studio._concept_run_active:
                self.studio.stop_related_concepts()
            elif self.studio._image_run_active:
                self.studio.stop_image_generation()
            else:
                self._assistant("There is no active job to stop.")
            return

        command = command_from_message(message)
        if command is not None:
            self._execute_app_command(command)
            return

        if self.state == "awaiting_count":
            try:
                count = int(message)
            except ValueError:
                self._assistant("Enter a whole number of images per concept.")
                return
            self._start_images_with_count(count)
            return

        if self.studio._concept_run_active or self.studio._image_run_active:
            self._assistant("Let the current job finish, or type “stop” before adding another concept.")
            return
        unrecognized_action = any(
            word in lowered.split()
            for word in {
                'delete', 'remove', 'clear', 'rename', 'list', 'show', 'select',
                'deselect', 'check', 'uncheck', 'export', 'crop', 'load', 'choose',
                'set', 'generate',
            }
        ) or (
            'create' in lowered.split()
            and 'concept' not in lowered.split()
        )
        if unrecognized_action:
            self._assistant(
                'I could not determine the exact target safely. Type “help” for examples, '
                'and include both the item and its parent concept when needed.'
            )
            return
        self._accept_concept(message)

    def _execute_app_command(self, command: WorkflowCommand) -> None:
        """Dispatch a validated command to operations shared with the tree UI."""
        action = command.action
        if action == 'help':
            self._show_help()
            return
        if action == 'status':
            if self.studio._concept_run_active:
                status = 'Related-concept generation is running.'
            elif self.studio._image_run_active:
                status = 'Image creation is running.'
            else:
                status = 'No background job is running.'
            self._assistant(
                f'{status} The workspace contains {len(self.studio.patient_concepts)} concepts.'
            )
            return
        if action == 'list_concepts':
            concepts = self.studio.patient_concepts
            self._assistant(
                'Concepts: ' + ', '.join(concepts)
                if concepts else 'The current workspace has no concepts.'
            )
            return
        if action == 'list_related':
            concept = self.studio._resolve_concept_key(command.target)
            if concept is None:
                self._assistant(f'I could not find the concept “{command.target}”.')
                return
            values = self.studio.related_concepts_dict.get(concept, [])
            names = [value.get('char') or value.get('name') for value in values]
            self._assistant(
                f'Related concepts for “{concept}”: ' + ', '.join(names)
                if names else f'“{concept}” has no related concepts.'
            )
            return
        if action in {'generate_related', 'create_images'}:
            concept = self.studio._resolve_concept_key(command.target)
            if concept is None:
                self._assistant(f'I could not find the concept “{command.target}”.')
                return
            self.current_concept = concept
            self.current_related = ''
            self.studio.select_concept(concept)
            if action == 'generate_related':
                self._offer_text_provider_types()
            elif not self.studio.related_concepts_dict.get(concept):
                self._assistant(
                    f'“{concept}” has no related concepts yet. Generate them first.'
                )
            else:
                self._offer_image_sources()
            return
        if action == 'create_related_images':
            concept, value = self.studio._resolve_related_value(
                command.parent, command.target
            )
            if value is None:
                self._assistant(
                    f'I could not find “{command.target}” under “{command.parent}”.'
                )
                return
            self.current_concept = concept
            self.current_related = value.get('char') or value.get('name')
            self.studio.select_related_concept(concept, self.current_related)
            self._offer_image_sources()
            return
        if action == 'add_related':
            concept = self.studio._resolve_concept_key(command.parent)
            if concept is None:
                self._assistant(f'I could not find the concept “{command.parent}”.')
                return
            if self.studio.add_related_concepts_to(concept, command.target):
                self._assistant(f'Added related concept “{command.target}” to “{concept}”.')
            else:
                self._assistant(f'Could not add “{command.target}” to “{concept}”. It may already exist.')
            return
        if action == 'delete_concept':
            concept = self.studio._resolve_concept_key(command.target)
            if concept is None:
                self._assistant(f'I could not find the concept “{command.target}”.')
                return
            self._confirm_chat_action(
                f'Delete “{concept}”, all related concepts, and all images?',
                lambda: self.studio.delete_parent_concept(concept, confirm=False),
                f'Deleted concept “{concept}”.',
            )
            return
        if action == 'delete_related':
            concept, value = self.studio._resolve_related_value(
                command.parent, command.target
            )
            if value is None:
                self._assistant(
                    f'I could not find “{command.target}” under “{command.parent}”.'
                )
                return
            display = value.get('char') or value.get('name')
            self._confirm_chat_action(
                f'Delete “{display}” from “{concept}” and delete its images?',
                lambda: self.studio.delete_related_concept_named(
                    concept, display, confirm=False
                ),
                f'Deleted related concept “{display}” from “{concept}”.',
            )
            return
        if action in {'clear_all_related', 'clear_related'}:
            concepts = None
            label = 'all loaded concepts'
            if action == 'clear_related':
                concept = self.studio._resolve_concept_key(command.target)
                if concept is None:
                    self._assistant(f'I could not find “{command.target}”.')
                    return
                concepts = [concept]
                label = f'“{concept}”'
            self._confirm_chat_action(
                f'Clear related concepts for {label} and delete their images?',
                lambda: self.studio.clear_related_concepts_for(
                    concepts, confirm=False
                ),
                f'Cleared related concepts for {label}.',
            )
            return
        if action in {'clear_all_images', 'clear_images'}:
            concept = None
            label = 'the workspace'
            if action == 'clear_images':
                concept = self.studio._resolve_concept_key(command.target)
                if concept is None:
                    self._assistant(f'I could not find “{command.target}”.')
                    return
                label = f'“{concept}”'
            self._confirm_chat_action(
                f'Permanently delete all images under {label}?',
                lambda: self.studio.clear_images_for(
                    concept=concept, confirm=False
                ),
                f'Deleted images under {label}.',
            )
            return
        if action == 'clear_related_images':
            concept, value = self.studio._resolve_related_value(
                command.parent, command.target
            )
            if value is None:
                self._assistant(
                    f'I could not find “{command.target}” under “{command.parent}”.'
                )
                return
            display = value.get('char') or value.get('name')
            self._confirm_chat_action(
                f'Permanently delete all images for “{display}” under “{concept}”?',
                lambda: self.studio.clear_images_for(
                    concept=concept, related=display, confirm=False
                ),
                f'Deleted images for “{display}” under “{concept}”.',
            )
            return
        if action == 'rename_concept':
            if self.studio.rename_parent_concept(command.target, command.new_name):
                self.current_concept = command.new_name.casefold()
                self._assistant(
                    f'Renamed “{command.target}” to “{self.current_concept}”.'
                )
            else:
                self._assistant('I could not rename that concept. Check the names and try again.')
            return
        if action == 'rename_related':
            if self.studio.rename_related_concept_named(
                command.parent, command.target, command.new_name
            ):
                self._assistant(
                    f'Renamed “{command.target}” to “{command.new_name}” under '
                    f'“{command.parent}”.'
                )
            else:
                self._assistant('I could not rename that related concept.')
            return
        if action in {'check_concept', 'uncheck_concept'}:
            checked = action == 'check_concept'
            if self.studio.set_concept_checked(command.target, checked):
                self._assistant(
                    f'{"Selected" if checked else "Deselected"} “{command.target}”.'
                )
            else:
                self._assistant(f'I could not find “{command.target}”.')
            return
        if action in {'check_all_concepts', 'uncheck_all_concepts'}:
            checked = action == 'check_all_concepts'
            if self.studio.set_all_concepts_checked(checked):
                self._assistant(
                    'Selected all concepts.' if checked else 'Deselected all concepts.'
                )
            else:
                self._assistant('There is no loaded concept workspace.')
            return
        if action in {'check_related', 'uncheck_related'}:
            checked = action == 'check_related'
            if self.studio.set_related_concept_checked(
                command.parent, command.target, checked
            ):
                self._assistant(
                    f'{"Selected" if checked else "Deselected"} “{command.target}” '
                    f'under “{command.parent}”.'
                )
            else:
                self._assistant(
                    f'I could not find “{command.target}” under “{command.parent}”.'
                )
            return
        if action == 'delete_selected_image':
            _, item = self.studio._current_concept_item()
            if item is None or not item.text().lower().endswith(
                ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp')
            ):
                self._assistant('Select an image in the concept tree first.')
                return
            name = item.text()
            self._confirm_chat_action(
                f'Permanently delete the selected image “{name}”?',
                lambda: self.studio.delete_image(confirm=False),
                f'Deleted image “{name}”.',
            )
            return
        if action == 'rename_selected_image':
            if self.studio.rename_selected_image(command.new_name):
                self._assistant(f'Renamed the selected image to “{command.new_name}”.')
            else:
                self._assistant(
                    'I could not rename the selected image. Select an image and include '
                    'a supported filename extension.'
                )
            return
        if action == 'clear_selected_images':
            self._assistant('I opened the existing confirmation for the selected tree item.')
            self.studio.clear_images()
            return
        if action == 'crop_selected':
            self.studio.crop_selected()
            self._assistant('Finished the crop-selected operation.')
            return
        if action == 'export_stimuli':
            self.studio.create_task_stimuli()
            self._assistant('Finished the export operation. Check the status log for details.')
            return
        if action == 'load_concepts':
            self.studio.choose_concepts_file()
            return
        if action == 'choose_text_model':
            self.studio.browse_llm_model()
            return
        if action == 'choose_image_model':
            self._assistant('Choose whether the local image model is a file or folder.')
            self._choices(
                [
                    ('Model file…', self.studio.browse_local_image_model_file),
                    ('Model folder…', self.studio.browse_local_image_model_folder),
                    ('Cancel', self._cancel_chat_action),
                ]
            )
            return
        if action in {
            'choose_picture_library', 'choose_image_workspace',
            'choose_experiment_folder',
        }:
            window = self.parentWidget()
            method_names = {
                'choose_picture_library': 'choose_picture_library',
                'choose_image_workspace': 'choose_download_workspace',
                'choose_experiment_folder': 'choose_experiment_folder',
            }
            method = getattr(window, method_names[action], None)
            if method is not None:
                method()
            else:
                self._assistant('That folder chooser is unavailable in this window.')
            return
        if action == 'set_image_count':
            count = int(command.target)
            spin = self.studio.images_per_concept_spin
            if spin.minimum() <= count <= spin.maximum():
                spin.setValue(count)
                self._assistant(f'Set images per concept to {count}.')
            else:
                self._assistant(
                    f'Choose a count from {spin.minimum()} to {spin.maximum()}.'
                )
            return
        if action in {'replace_images_on', 'replace_images_off'}:
            enabled = action == 'replace_images_on'
            self.studio.replace_existing_checkbox.setChecked(enabled)
            self._assistant(
                'Existing images will be replaced.' if enabled
                else 'Existing images will be kept.'
            )
            return
        self._assistant('I recognized the request but that action is not available here yet.')

    def _confirm_chat_action(self, prompt: str, callback, success_message: str) -> None:
        """Require explicit chat confirmation before destructive operations."""
        self.state = 'awaiting_confirmation'
        self._pending_callback = callback
        self._pending_success_message = success_message
        self._assistant(prompt)
        self._choices(
            [
                ('Confirm', self._run_pending_action),
                ('Cancel', self._cancel_chat_action),
            ]
        )

    def _run_pending_action(self) -> None:
        self._clear_choices()
        callback = getattr(self, '_pending_callback', lambda: False)
        success_message = getattr(
            self, '_pending_success_message', 'The action completed.'
        )
        try:
            completed = bool(callback())
        except Exception as exc:
            completed = False
            self._assistant(f'The action failed: {exc}')
        self._assistant(
            success_message if completed
            else 'Nothing was changed. The target may no longer exist.'
        )
        self._pending_callback = None
        self._pending_success_message = ''
        self.state = 'awaiting_concept'

    def _cancel_chat_action(self) -> None:
        self._clear_choices()
        self._pending_callback = None
        self._pending_success_message = ''
        self.state = 'awaiting_concept'
        self._assistant('Cancelled. Nothing was changed.')

    def _show_help(self) -> None:
        self._assistant(
            'You can add a concept; generate related concepts or images; list concepts; '
            'delete or rename a concept; delete one related concept; clear related concepts '
            'or images; select or deselect concepts; delete the selected image; crop selected '
            'images; load a concept file; or export task stimuli. Examples: “delete concept '
            'basketball”, “delete related concept NBA from basketball”, “clear all related '
            'concepts”, and “create images for basketball”.'
        )

    def _accept_concept(self, message: str) -> None:
        concept = self._concept_from_message(message)
        if not concept:
            self._assistant("Please enter a non-empty concept.")
            return
        concept, added = self.studio.add_concept(concept)
        self.current_concept = concept
        self.current_related = ''
        self.state = "awaiting_text_provider"
        self._assistant(
            f'{"Added" if added else "Selected existing concept"} “{concept}”. '
            "How should I generate its related concepts?"
        )
        self._choices(
            [
                ("Local GGUF", lambda: self._select_text_provider("local")),
                ("Cloud LLM…", self._offer_cloud_providers),
                ("Not now", self._restart),
            ]
        )

    def _offer_cloud_providers(self) -> None:
        self._assistant("Choose the configured cloud text provider.")
        self._choices(
            [
                ("OpenAI", lambda: self._select_text_provider("openai")),
                ("Anthropic Claude", lambda: self._select_text_provider("anthropic")),
                ("Google Gemini", lambda: self._select_text_provider("google")),
                ("Hosted / compatible", lambda: self._select_text_provider("compatible")),
                ("Back", self._offer_text_provider_types),
            ]
        )

    def _offer_text_provider_types(self) -> None:
        self.state = "awaiting_text_provider"
        self._assistant("Use a local model or a cloud language model?")
        self._choices(
            [
                ("Local GGUF", lambda: self._select_text_provider("local")),
                ("Cloud LLM…", self._offer_cloud_providers),
                ("Not now", self._restart),
            ]
        )

    def _select_text_provider(self, provider: str) -> None:
        combo = self.studio.concept_provider_combo
        index = combo.findData(provider)
        if index < 0:
            self._assistant(f"The {provider} provider is not available.")
            return
        combo.setCurrentIndex(index)
        self._clear_choices()
        self.state = "generating_related"
        self._assistant(
            f"Generating related concepts for “{self.current_concept}” using "
            f"{combo.currentText()}…"
        )
        if not self.studio.start_related_concepts([self.current_concept]):
            self.state = "awaiting_text_provider"
            self._assistant(
                "I couldn’t start that provider. Check its model and credentials in step 2, "
                "then choose again."
            )
            self._choices(
                [
                    ("Local GGUF", lambda: self._select_text_provider("local")),
                    ("Cloud LLM…", self._offer_cloud_providers),
                ]
            )

    @QtCore.Slot()
    def _related_concepts_finished(self) -> None:
        if self.state != "generating_related" or not self.current_concept:
            return
        related = self.studio.related_concepts_dict.get(self.current_concept, [])
        self.activity_label.clear()
        if not related:
            self.state = "awaiting_text_provider"
            self._assistant("No usable related concepts were produced. Try another text provider.")
            self._choices(
                [
                    ("Local GGUF", lambda: self._select_text_provider("local")),
                    ("Cloud LLM…", self._offer_cloud_providers),
                    ("Start over", self._restart),
                ]
            )
            return
        self.state = "awaiting_related_review"
        self._assistant(
            f"I found {len(related)} related concepts. They are checked in the concept tree; "
            "uncheck anything you do not want before continuing."
        )
        self._choices(
            [
                ("Create images", self._offer_image_sources),
                ("Show in concept tree", self._show_current_concept),
                ("Regenerate", self._offer_text_providers_again),
                ("Finish here", self._restart),
            ]
        )

    def _show_current_concept(self) -> None:
        self.studio.select_concept(self.current_concept)
        self._assistant("The concept is selected in the tree. Review the checked associations, then continue.")

    def _offer_text_providers_again(self) -> None:
        self.state = "awaiting_text_provider"
        self._assistant("Choose a text provider for regeneration.")
        self._choices(
            [
                ("Local GGUF", lambda: self._select_text_provider("local")),
                ("Cloud LLM…", self._offer_cloud_providers),
            ]
        )

    def _offer_image_sources(self) -> None:
        self.state = "awaiting_image_source"
        self._assistant("Where should the images come from?")
        self._choices(
            [
                ("Bing — strict SafeSearch", lambda: self._select_image_source("bing")),
                ("Text stimulus (concept centered)", lambda: self._select_image_source("text")),
                ("Hugging Face (Serverless Cloud)", lambda: self._select_image_source("huggingface")),
                ("OpenAI cloud images", lambda: self._select_image_source("cloud")),
                ("Google Imagen cloud images", lambda: self._select_image_source("google")),
                ("Local image model", lambda: self._select_image_source("local")),
                ("Back to review", self._return_to_review),
            ]
        )

    def _return_to_review(self) -> None:
        self.state = "awaiting_related_review"
        self._assistant("Review the checked associations in the concept tree.")
        self._choices(
            [
                ("Create images", self._offer_image_sources),
                ("Show in concept tree", self._show_current_concept),
                ("Regenerate", self._offer_text_providers_again),
                ("Finish here", self._restart),
            ]
        )

    def _select_image_source(self, source: str) -> None:
        combo = self.studio.image_source_combo
        index = combo.findData(source)
        if index >= 0:
            combo.setCurrentIndex(index)
        self.pending_image_source = source
        current = self.studio.images_per_concept_spin.value()
        self.state = "awaiting_count"
        self._assistant(
            "How many images should I create for the parent and each checked related concept?"
        )
        self._choices(
            [
                (f"Use current setting ({current})", lambda: self._start_images_with_count(current)),
                ("5 each", lambda: self._start_images_with_count(5)),
                ("10 each", lambda: self._start_images_with_count(10)),
                ("Back", self._offer_image_sources),
            ]
        )

    def _start_images_with_count(self, count: int) -> None:
        spin = self.studio.images_per_concept_spin
        if count < spin.minimum() or count > spin.maximum():
            self._assistant(
                f"Choose a count from {spin.minimum()} to {spin.maximum()}."
            )
            return
        spin.setValue(count)
        self._clear_choices()
        self.state = "starting_images"
        if self.current_related:
            started = self.studio.acquire_related_concept_images(
                self.current_concept,
                self.current_related,
                self.pending_image_source,
            )
        else:
            started = self.studio.acquire_concept_images(
                self.current_concept, self.pending_image_source
            )
        if started:
            self.state = "generating_images"
            self._assistant("The image job is running. New images will appear in the tree as they finish.")
        else:
            self.state = "awaiting_image_source"
            self._assistant(
                "Image creation did not start. Check the selected model configuration or choose another source."
            )
            self._offer_image_sources()

    @QtCore.Slot(int, float, bool)
    def _images_finished(self, saved_count: int, elapsed_seconds: float, stopped: bool) -> None:
        del elapsed_seconds
        if self.state not in {"starting_images", "generating_images"}:
            return
        self.activity_label.clear()
        self.state = "complete"
        if stopped:
            self._assistant(f"Stopped after saving {saved_count} images.")
        else:
            self._assistant(f"Done—{saved_count} images were saved. Review them in the concept tree.")
        self._choices(
            [
                ("Add another concept", self._restart),
                ("Create more images", self._offer_image_sources),
            ]
        )

    @QtCore.Slot(str)
    def _images_failed(self, message: str) -> None:
        if self.state not in {"starting_images", "generating_images"}:
            return
        self.activity_label.clear()
        self.state = "awaiting_image_source"
        self._assistant(f"Image creation stopped: {message}")
        self._offer_image_sources()

    @QtCore.Slot(int, object, bool)
    def _progress_changed(self, value: int, message: object, append: bool) -> None:
        del value, append
        if self.state in {"generating_related", "starting_images", "generating_images"} and message:
            self.activity_label.setText(str(message))

    def _restart(self) -> None:
        self.current_concept = ""
        self.current_related = ""
        self.state = "awaiting_concept"
        self.activity_label.clear()
        self._clear_choices()
        self.input_edit.setPlaceholderText("Enter another concept…")
        self._assistant("What concept would you like to add next?")
        self.input_edit.setFocus()
