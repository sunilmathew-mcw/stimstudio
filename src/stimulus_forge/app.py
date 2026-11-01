"""Standalone launcher for Stimulus Forge."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from qtpy import QtCore, QtWidgets

from stimulus_forge.config import config
from stimulus_forge.ui.forge import StimulusForge
from stimulus_forge.ui.chat import WorkflowChatDock


class StandaloneSignals(QtCore.QObject):
    """Application-wide signals shared with the main studio widget."""

    log = QtCore.Signal(str)
    progress = QtCore.Signal(int, object, bool)
    image_names = QtCore.Signal(list)
    related_concepts_single_done = QtCore.Signal()
    related_concepts_done = QtCore.Signal()
    scr_config_updated = QtCore.Signal(int, object, bool)


class StimulusForgeWindow(QtWidgets.QMainWindow):
    def __init__(self, task_name: str):
        super().__init__()
        self.setWindowTitle("Stimulus Forge")
        self.resize(1500, 950)

        self.signals = StandaloneSignals(self)
        central = QtWidgets.QWidget(self)
        layout = QtWidgets.QVBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        self.setCentralWidget(central)

        self.studio = StimulusForge(
            c=self.signals,
            concepts_view=layout,
            task_name=task_name,
        )

        self.progress_bar = QtWidgets.QProgressBar(self)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setMaximumWidth(220)
        self.statusBar().addPermanentWidget(self.progress_bar)

        self.signals.log.connect(self.show_status)
        self.signals.progress.connect(self.update_progress)
        self.signals.related_concepts_single_done.connect(
            lambda: self.studio.update_related_concepts(b_llm=True)
        )
        self.signals.related_concepts_done.connect(self.studio.post_process_related_concepts)
        self.signals.image_names.connect(
            lambda names: self.show_status(f"Prepared {len(names)} stimulus names.")
        )
        self.chat_dock = WorkflowChatDock(self.studio, self)
        self.addDockWidget(QtCore.Qt.RightDockWidgetArea, self.chat_dock)
        self._build_menu()
        self.show_status("Ready")

    def _build_menu(self) -> None:
        file_menu = self.menuBar().addMenu("File")

        load_concepts = file_menu.addAction("Load concepts…")
        load_concepts.triggered.connect(self.load_concepts)

        choose_pictures = file_menu.addAction("Choose picture library…")
        choose_pictures.triggered.connect(self.choose_picture_library)

        choose_downloads = file_menu.addAction("Choose download workspace…")
        choose_downloads.triggered.connect(self.choose_download_workspace)

        choose_experiments = file_menu.addAction("Choose experiment folder…")
        choose_experiments.triggered.connect(self.choose_experiment_folder)

        file_menu.addSeparator()
        quit_action = file_menu.addAction("Quit")
        quit_action.triggered.connect(self.close)

        view_menu = self.menuBar().addMenu("View")
        chat_action = self.chat_dock.toggleViewAction()
        chat_action.setText("Workflow assistant")
        view_menu.addAction(chat_action)

    def show_status(self, message: str) -> None:
        message = str(message)
        self.statusBar().showMessage(message, 15000)
        print(message)

    @QtCore.Slot(int, object, bool)
    def update_progress(self, value: int, message: object, append: bool) -> None:
        del append
        self.progress_bar.setValue(max(0, min(100, int(value))))
        if message:
            self.show_status(str(message))

    def load_concepts(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Load concepts",
            str(Path(config["concepts_file"]).parent),
            "Concept lists (*.txt *.csv)",
        )
        if not path:
            return
        try:
            concepts = self.studio.load_concepts_file(path)
            config["concepts_file"] = os.path.abspath(path)
            self.show_status(f"Loaded {len(concepts)} concepts from {path}")
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Could not load concepts", str(exc))

    def choose_picture_library(self) -> None:
        path = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Choose picture library", self.studio.pics_main_folder
        )
        if not path:
            return
        self.studio.pics_main_folder = os.path.abspath(path)
        config["pics_main_folder"] = self.studio.pics_main_folder
        self.studio.update_pics_info(self.studio.pics_main_folder)
        self.show_status(f"Using picture library {self.studio.pics_main_folder}")

    def choose_download_workspace(self) -> None:
        path = QtWidgets.QFileDialog.getExistingDirectory(
            self,
            "Choose image download workspace",
            self.studio.task_images_downloader.task_images_path,
        )
        if not path:
            return
        path = os.path.abspath(path)
        self.studio.task_images_downloader.task_images_path = path
        config["task_images_folder"] = path
        self.show_status(f"Using download workspace {path}")

    def choose_experiment_folder(self) -> None:
        path = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Choose experiment folder", self.studio.exp_folder
        )
        if not path:
            return
        path = os.path.abspath(path)
        self.studio.exp_folder = path
        config["experiment_folder"] = path
        self.show_status(f"Using experiment folder {path}")

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Stimulus Forge as a standalone app.")
    parser.add_argument(
        "--task",
        choices=tuple(config["task_config"]),
        default="dynamic_scr",
        help="Initial task mode (default: dynamic_scr).",
    )
    parser.add_argument("--concepts", help="Concept list (.txt, .csv, or .xlsx).")
    parser.add_argument("--pictures", help="Existing reference picture library.")
    parser.add_argument("--task-images", help="Workspace for downloaded/cropped images.")
    parser.add_argument("--experiments", help="Experiment output folder.")
    return parser


def main(argv: list[str] | None = None) -> int:
    raw_args = list(sys.argv[1:] if argv is None else argv)
    if "--cli" in raw_args:
        from stimulus_forge.cli import run_cli
        cli_args = [a for a in raw_args if a != "--cli"]
        return run_cli(cli_args)

    args = build_parser().parse_args(argv)

    overrides = {
        "concepts_file": args.concepts,
        "pics_main_folder": args.pictures,
        "task_images_folder": args.task_images,
        "experiment_folder": args.experiments,
    }
    for key, value in overrides.items():
        if value:
            config[key] = os.path.abspath(os.path.expanduser(value))

    for key in ("pics_main_folder", "task_images_folder", "experiment_folder"):
        Path(config[key]).mkdir(parents=True, exist_ok=True)

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    app.setApplicationName("Stimulus Forge")
    window = StimulusForgeWindow(task_name=args.task)
    if args.concepts:
        window.studio.load_concepts_file(config["concepts_file"])
    window.show()
    runner = getattr(app, "exec", None) or app.exec_
    return int(runner())


if __name__ == "__main__":
    raise SystemExit(main())
