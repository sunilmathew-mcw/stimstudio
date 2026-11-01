"""
Author: Sunil Mathew
Date: 29 June 2024
Stimulus Forge
Get images from the internet, provides tools to crop, resize, and save images
Uses LLM to get related concepts
"""
import os
import importlib.util
import glob
import hashlib
import time
import threading
import shutil
import traceback
import cv2
import imageio
import pyqtgraph as pg
import pandas as pd
# Face detection is optional in the standalone application.  Keeping the
# import optional lets users review and manually crop images without pulling
# in TensorFlow and its large transitive dependency set.
try:
    from mtcnn import MTCNN
except Exception:
    MTCNN = None
from qtpy import QtWidgets, QtCore, QtGui
from qtpy.QtCore import Qt

import concurrent.futures

from stimulus_forge.config import config
from stimulus_forge.domain.rules.categ_localiz import CategLocalizInfo
from stimulus_forge.domain.rules.dynamic_scr import DynamicScrInfo
from stimulus_forge.domain.rules.task_info_base import TaskInfoBase
from stimulus_forge.domain.screening import SubscreeningConfig
from stimulus_forge.services.concept_generation import RelatedConceptGenerator
from stimulus_forge.services.concept_loader import load_concepts, workspace_name_for
from stimulus_forge.services.google_images import GoogleImageService
from stimulus_forge.services.huggingface_images import HuggingFaceImageService
from stimulus_forge.services.image_workspace import TaskImagesDownloader
from stimulus_forge.services.local_images import LocalImageService
from stimulus_forge.services.openai_images import OpenAIImageService
from stimulus_forge.services.text_images import TextImageService
from stimulus_forge.utils.time import get_elapsed_time, get_time_hrs_mins_secs_ms


IMAGE_PATH_ROLE = int(Qt.UserRole) + 10
IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp')


class ImageWorkflowSignals(QtCore.QObject):
    """Move image-worker notifications safely onto the Qt UI thread."""

    image_saved = QtCore.Signal(object, str)
    progress = QtCore.Signal(int, str)
    finished = QtCore.Signal(int, float, bool)
    failed = QtCore.Signal(str)

    def __init__(self, owner):
        super().__init__()
        self.image_saved.connect(owner._on_image_saved, Qt.QueuedConnection)
        self.progress.connect(owner._on_image_progress, Qt.QueuedConnection)
        self.finished.connect(owner._on_image_generation_finished, Qt.QueuedConnection)
        self.failed.connect(owner._on_image_generation_failed, Qt.QueuedConnection)


class ConceptWorkflowSignals(QtCore.QObject):
    """Move related-concept progress safely onto the Qt UI thread."""

    progress = QtCore.Signal(int, str)

    def __init__(self, owner):
        super().__init__()
        self.progress.connect(owner._on_concept_progress, Qt.QueuedConnection)


class StimulusForge:

    def __new__(cls, *args, **kwargs):
        if not hasattr(cls, 'instance'):
            cls.instance = super(StimulusForge, cls).__new__(cls)
        return cls.instance

    def __init__(self, c=None, concepts_view=None, task_name='gaps'):
        self.c = c
        self.img_size = 160
        self.crop_win_size = 200
        self.subject_id = None
        self.task_name = task_name
        self.task_folder = None # will be set when experiment folder is created
        self.exp_folder = config['experiment_folder']
        self.block_params = config["task_config"][task_name]["block_params"]
        self.exp_params = config["task_config"][task_name]["exp_params"]
        self.force_replace = False
        self.image_source = config.get('image_source', 'bing')
        self.local_image_model_path = os.environ.get(
            'STIMULUS_FORGE_LOCAL_IMAGE_MODEL',
            config.get('local_image_model', ''),
        )
        self.concept_provider = os.environ.get(
            'STIMULUS_FORGE_CONCEPT_PROVIDER',
            config.get('concept_provider', 'local'),
        )
        # Migrate the former automatic-provider setting to the new explicit default.
        if self.concept_provider == 'auto':
            self.concept_provider = 'local'
        self.concept_models = {
            'openai': os.environ.get('STIMULUS_FORGE_CONCEPT_MODEL', config.get('concept_model', 'gpt-5.6-luna')),
            'anthropic': os.environ.get('STIMULUS_FORGE_ANTHROPIC_MODEL', config.get('anthropic_concept_model', 'claude-sonnet-4-6')),
            'google': os.environ.get('STIMULUS_FORGE_GOOGLE_MODEL', config.get('google_concept_model', 'gemini-3.8-flash')),
            'compatible': os.environ.get('STIMULUS_FORGE_COMPAT_MODEL', config.get('compatible_concept_model', '')),
        }
        self.concept_model = self.concept_models.get(
            self.concept_provider, self.concept_models['openai']
        )
        self.concept_api_keys = {
            provider: ''
            for provider in ('openai', 'anthropic', 'google', 'compatible')
        }
        if os.environ.get('OPENAI_API_KEY'):
            self.concept_api_keys['openai'] = os.environ.get('OPENAI_API_KEY', '').strip()
        if os.environ.get('GEMINI_API_KEY') or os.environ.get('GOOGLE_API_KEY'):
            self.concept_api_keys['google'] = (
                os.environ.get('GEMINI_API_KEY') or os.environ.get('GOOGLE_API_KEY', '')
            ).strip()
        self.concept_api_key = self.concept_api_keys.get(self.concept_provider, '')
        self.concept_base_url = os.environ.get(
            'STIMULUS_FORGE_COMPAT_BASE_URL',
            config.get('compatible_base_url', ''),
        )
        configured_llm_model = os.environ.get(
            'STIMULUS_FORGE_LLM_MODEL', config.get('llm_model', '')
        )
        self.llm_model_path = (
            os.path.abspath(os.path.expanduser(configured_llm_model))
            if configured_llm_model else ''
        )
        self.local_gpu_layers = int(
            os.environ.get(
                'STIMULUS_FORGE_LLM_GPU_LAYERS',
                str(config.get('llm_gpu_layers', -1)),
            )
        )
        # self.popular_concepts = ['obama', 'trump', 'biden', 'putin', 'willsmith', 'tomcruise', 
        #                          'angelinajolie', 'scarlettjohansson', 'bradpitt', 'leonardodicaprio', 
        #                          'jenniferlopez', 'jenniferaniston', 'meganfox', 'spiderman', 'batman',
        #                          'superman', 'wonderwoman', 'hulk', 'thor', 'captainamerica', 'ironman'
        #                          'einstein', 'eddiemurphy', 'jimcarrey', 'jackiechan', 'brucelee', 'jetli',
        #                          'arnoldschwarzenegger', 'sylvesterstallone', 'clinteastwood', 'tomhanks'
        #                          'christmas', 'halloween', 'thanksgiving', 'valentinesday', 'easter', 'newyear']
        self.popular_concepts = self.init_popular_concepts()

        self.concepts_available = []
        self.downloaded_concepts = []
        self.patient_concepts = []
        self.related_concepts_dict = {}
        self.scr_config = {}
        self.concept_generator = self._new_concept_generator()
        self.openai_image_service = OpenAIImageService(
            model=os.environ.get(
                'STIMULUS_FORGE_IMAGE_MODEL',
                config.get('image_model', 'gpt-image-2'),
            ),
            quality=os.environ.get(
                'STIMULUS_FORGE_IMAGE_QUALITY',
                config.get('image_quality', 'low'),
            ),
            api_key=self.concept_api_keys.get('openai', ''),
        )
        self.google_image_service = GoogleImageService(
            model=os.environ.get(
                'STIMULUS_FORGE_GOOGLE_IMAGE_MODEL',
                config.get('google_image_model', 'imagen-3.0-generate-002'),
            ),
            api_key=self.concept_api_keys.get('google', ''),
        )
        self.local_image_service = LocalImageService(
            model_path=self.local_image_model_path
        )
        self.huggingface_image_service = HuggingFaceImageService(
            model=os.environ.get(
                'STIMULUS_FORGE_HF_IMAGE_MODEL',
                config.get('huggingface_image_model', 'black-forest-labs/FLUX.1-schnell'),
            ),
            api_key=self.concept_api_keys.get('huggingface', ''),
        )
        self.text_image_service = TextImageService()
        self.init_facenet()
        self.task_images_downloader = TaskImagesDownloader()
        self._image_run_active = False
        self._image_run_action = 'Creating'
        self._image_run_subject = ''
        self.image_workflow_signals = ImageWorkflowSignals(self)
        self._image_pause_event = threading.Event()
        self._image_pause_event.set()
        self._image_stop_event = threading.Event()
        self._concept_run_active = False
        self._concept_pause_event = threading.Event()
        self._concept_pause_event.set()
        self._concept_stop_event = threading.Event()
        self.concept_workflow_signals = ConceptWorkflowSignals(self)
        if concepts_view is not None:
            self.concepts_view = concepts_view
            self.init_concepts_layout()

        self.pics_main_folder = config['pics_main_folder']

        # if os.path.exists(pics_xlsx_file_path):
        #     self.load_concepts_pics_info(pics_xlsx_file_path)
        if os.path.exists(self.pics_main_folder):
            self.update_pics_info(self.pics_main_folder)

    def init_facenet(self):
        self.mtcnn = None
        if MTCNN is None:
            if self.c is not None and hasattr(self.c, 'log'):
                self.c.log.emit('Optional face detection is unavailable; manual crop controls remain available.')
            return
        try:
            self.mtcnn = MTCNN()
        except Exception as exc:
            if self.c is not None and hasattr(self.c, 'log'):
                self.c.log.emit(f'Face detector could not be initialized: {exc}')
        # self.resnet = InceptionResnetV1(pretrained='vggface2').eval()

    def _new_concept_generator(self):
        """Build a fresh generator when the user changes provider or GGUF."""
        return RelatedConceptGenerator(
            provider=self.concept_provider,
            openai_model=self.concept_model,
            local_model_path=self.llm_model_path,
            local_gpu_layers=self.local_gpu_layers,
            api_key=self.concept_api_key,
            base_url=self.concept_base_url,
        )

    def init_popular_concepts(self):
        """
        Initialize a list of popular concepts using the images in must_be folder
        """
        popular_concepts = []
        must_be_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), '../', 'assets', 'Pics', 'must_be'))
        if os.path.exists(must_be_folder):
            for img_file in glob.glob(os.path.join(must_be_folder, '*.*')):
                img_name = os.path.basename(img_file).split('.')[0]
                img_name = img_name.split('_')[0]
                popular_concepts.extend(os.path.basename(img_name).split("~"))
        popular_concepts = list(set(popular_concepts))
        return popular_concepts
    
    def update_pics_info(self, main_pics_path):
        """
        Update pictures info from the main pictures folder

        Args:
        	main_pics_path: Path to the main pictures folder
        """
        
        self.parent_concepts = []
        self.concepts_available = []
        self.img_paths = []
        face_imgs = glob.glob(os.path.join(main_pics_path, 'Faces', '**', '*.*'), recursive=True)
        self.img_paths.extend(face_imgs)
        for img in face_imgs:
            img_name = os.path.basename(img).split('.')[0]
            self.parent_concepts.append(img_name.split('~')[0])
            self.concepts_available.append(img_name.split('~'))
        imgs = glob.glob(os.path.join(main_pics_path, 'Non faces', '**', '*.*'), recursive=True)
        self.img_paths.extend(imgs)
        self.img_paths = list(set(self.img_paths))
        for img in imgs:
            img_name = os.path.basename(img).split('.')[0]
            self.concepts_available.append(img_name.split('~'))

        self.concepts_available = [item.split('_')[0] for sublist in self.concepts_available for item in sublist]
        self.concepts_available = list(set(self.concepts_available))
        print(f"Concept count: {len(self.concepts_available)}")

    def load_concepts_pics_info(self, file_path):
        """
        Load pictures info from an Excel file

        Args:
        	file_path: Path to the Excel file
        """
        self.concepts_pics_info = pd.read_excel(file_path)
        self.concepts_available = self.concepts_pics_info['Name'].values
        # some of them may contain more than one concept separated by ~
        self.concepts_available = [x.split('~') for x in self.concepts_available]
        self.concepts_available = [item for sublist in self.concepts_available for item in sublist]
        self.concepts_available = list(set(self.concepts_available))
        print(f"Concept count: {len(self.concepts_available)}")
        print(self.concepts_pics_info.head())

    def init_concepts_layout(self):
        if not hasattr(self, 'concepts_layout'):
            self.concepts_layout = True
            self.concepts_widget = QtWidgets.QWidget()
            self.lyt_concepts = QtWidgets.QVBoxLayout()
            self.lyt_concepts.setContentsMargins(8, 8, 8, 8)
            self.concepts_widget.setLayout(self.lyt_concepts)

            # Left: load + concept tree. Right: tracker + controls + viewer.
            self.hsplitter_concept_imgs = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
            self.lyt_concepts.addWidget(self.hsplitter_concept_imgs, 1)

            self.concepts_left_panel = QtWidgets.QWidget()
            self.concepts_left_layout = QtWidgets.QVBoxLayout(
                self.concepts_left_panel
            )
            self.concepts_left_layout.setContentsMargins(0, 0, 4, 0)

            self.concepts_right_panel = QtWidgets.QWidget()
            self.concepts_right_layout = QtWidgets.QVBoxLayout(
                self.concepts_right_panel
            )
            self.concepts_right_layout.setContentsMargins(4, 0, 0, 0)

            self.workflow_tracker_widget = self._build_workflow_tracker()
            self.concepts_right_layout.addWidget(self.workflow_tracker_widget)
            self.status_lbl = QtWidgets.QLabel('Load a concept list to begin')
            self.status_lbl.setWordWrap(True)
            self.status_lbl.setStyleSheet('color: #555; padding: 2px 4px;')
            self.concepts_right_layout.addWidget(self.status_lbl)

            self.workflow_controls_layout = QtWidgets.QHBoxLayout()
            self.workflow_controls_layout.setContentsMargins(0, 0, 0, 0)
            self.workflow_controls_layout.setSpacing(8)
            self.concepts_right_layout.addLayout(self.workflow_controls_layout)

            self.init_workflow_controls()

            self.related_concepts_tree_view = QtWidgets.QTreeView()
            self.related_concepts_tree_view.setMinimumWidth(270)
            self.related_concepts_tree_view.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
            self.related_concepts_tree_view.customContextMenuRequested.connect(self.concepts_tree_view_context_menu)
            self.related_concepts_tree_view.clicked.connect(self.concepts_tree_view_clicked)
            self.concepts_left_layout.addWidget(self.related_concepts_tree_view, 1)

            self.glw_concepts = pg.GraphicsLayoutWidget()
            self.glw_concepts.setContentsMargins(0, 0, 0, 0)
            self.glw_concepts.ci.layout.setSpacing(0)

            self.concepts_scroll = QtWidgets.QScrollArea()
            self.concepts_scroll.setWidgetResizable(True)
            self.concepts_scroll.setWidget(self.glw_concepts)
            self.concepts_scroll.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOn)
            self.concepts_scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
            self.concepts_right_layout.addWidget(self.concepts_scroll, 1)

            self.hsplitter_concept_imgs.addWidget(self.concepts_left_panel)
            self.hsplitter_concept_imgs.addWidget(self.concepts_right_panel)
            self.hsplitter_concept_imgs.setStretchFactor(0, 1)
            self.hsplitter_concept_imgs.setStretchFactor(1, 3)
            self.hsplitter_concept_imgs.setSizes([360, 1140])

            self.concepts_view.addWidget(self.concepts_widget)
            self._set_workflow_tracker(completed=0, active=1)

    def _build_workflow_tracker(self):
        """Create a compact, Domino's-style three-stage workflow tracker."""
        tracker = QtWidgets.QFrame()
        tracker.setObjectName('workflow_tracker')
        tracker.setFrameShape(QtWidgets.QFrame.StyledPanel)
        layout = QtWidgets.QHBoxLayout(tracker)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(6)
        self.workflow_step_bars = []
        self.workflow_step_labels = self.workflow_step_bars
        self._workflow_step_progress = {1: 0, 2: 0, 3: 0}
        for number, title in (
            (1, 'Concepts loaded'),
            (2, 'Related concepts'),
            (3, 'Images created'),
        ):
            progress = QtWidgets.QProgressBar(tracker)
            progress.setRange(0, 100)
            progress.setValue(0)
            progress.setFormat(f'{number}. {title} — %p%')
            progress.setAlignment(Qt.AlignCenter)
            progress.setMinimumHeight(34)
            progress.setTextVisible(True)
            self.workflow_step_bars.append(progress)
            layout.addWidget(progress, 1)
            if number < 3:
                arrow = QtWidgets.QLabel('›')
                arrow.setAlignment(Qt.AlignCenter)
                arrow.setStyleSheet('font-size: 22px; color: #777;')
                layout.addWidget(arrow)
        return tracker

    def _set_workflow_tracker(self, completed=None, active=None, reset=False):
        """Render each workflow stage as blue-to-green progress."""
        if completed is not None:
            self._workflow_completed_stage = (
                completed
                if reset
                else max(getattr(self, '_workflow_completed_stage', 0), completed)
            )
        completed = getattr(self, '_workflow_completed_stage', 0)
        self._workflow_active_stage = active
        for stage, progress_bar in enumerate(self.workflow_step_bars, start=1):
            if stage <= completed:
                self._workflow_step_progress[stage] = 100
                value = 100
                background, foreground = '#1565c0', 'white'
            elif stage == active:
                value = self._workflow_step_progress.get(stage, 0)
                background, foreground = '#1565c0', 'white'
            else:
                if reset:
                    self._workflow_step_progress[stage] = 0
                value = self._workflow_step_progress.get(stage, 0)
                background, foreground = '#e6e8eb', '#555'
            progress_bar.setValue(value)
            progress_bar.setStyleSheet(
                'QProgressBar {'
                f'background-color: {background}; color: {foreground}; '
                'border: none; border-radius: 6px; padding: 0; '
                'text-align: center; font-weight: 600;'
                '}'
                'QProgressBar::chunk {'
                'background-color: #2e7d32; border-radius: 6px;'
                '}'
            )

    def _set_workflow_step_progress(self, stage, percentage):
        """Grow one active stage from blue to green."""
        percentage = max(0, min(100, int(percentage)))
        self._workflow_step_progress[stage] = percentage
        self._set_workflow_tracker(
            completed=max(0, stage - 1), active=stage, reset=True
        )

    def _sync_workflow_tracker(self, active=None):
        """Derive completed stages from the current concepts and workspace files."""
        completed = 1 if self.patient_concepts else 0
        if any(self.related_concepts_dict.values()):
            completed = 2
        workspace = self._workspace_root() if self.subject_id else ''
        if workspace and os.path.isdir(workspace):
            has_images = any(
                filename.lower().endswith(IMAGE_EXTENSIONS)
                for _, _, filenames in os.walk(workspace)
                for filename in filenames
            )
            if has_images:
                completed = 3
        if active is None and completed < 3:
            active = completed + 1
        self._set_workflow_tracker(completed, active, reset=True)

    def init_workflow_controls(self):
        """Build a clear, explicit load → relate → image workflow."""
        load_group = QtWidgets.QGroupBox('Concept list')
        self.load_concepts_group = load_group
        load_layout = QtWidgets.QVBoxLayout(load_group)
        self.concepts_file_path_edit = QtWidgets.QLineEdit(load_group)
        self.concepts_file_path_edit.setReadOnly(True)
        self.concepts_file_path_edit.setPlaceholderText(
            'Choose a .txt or .csv file containing concepts'
        )
        self.load_concepts_button = QtWidgets.QPushButton('Choose concept file…')
        self.load_concepts_button.setObjectName('btn_load_concepts')
        self.load_concepts_button.clicked.connect(self.choose_concepts_file)
        load_layout.addWidget(self.concepts_file_path_edit, 1)
        load_layout.addWidget(self.load_concepts_button)
        load_help = QtWidgets.QLabel(
            'Checkboxes choose what to process. Right-click to clear, rename, or delete.'
        )
        load_help.setWordWrap(True)
        load_help.setStyleSheet('color: #666; font-size: 11px;')
        load_layout.addWidget(load_help)
        self.concepts_left_layout.addWidget(load_group)

        related_group = QtWidgets.QGroupBox('2. Generate related concepts')
        self.related_controls_group = related_group
        related_layout = QtWidgets.QGridLayout(related_group)
        related_layout.setHorizontalSpacing(6)
        related_layout.addWidget(QtWidgets.QLabel('Text model'), 0, 0)
        self.concept_provider_combo = QtWidgets.QComboBox(related_group)
        self.concept_provider_combo.setSizeAdjustPolicy(
            QtWidgets.QComboBox.AdjustToMinimumContentsLengthWithIcon
        )
        self.concept_provider_combo.setMinimumContentsLength(12)
        self.concept_provider_combo.setObjectName('cmb_concept_provider')
        self.concept_provider_combo.setToolTip(
            'Choose the language-model provider used to generate related concepts.'
        )
        self.concept_provider_combo.addItem('Local GGUF', 'local')
        self.concept_provider_combo.addItem('OpenAI API', 'openai')
        self.concept_provider_combo.addItem('Anthropic Claude API', 'anthropic')
        self.concept_provider_combo.addItem('Google Gemini API', 'google')
        self.concept_provider_combo.addItem(
            'Hosted Llama / OpenAI-compatible API', 'compatible'
        )
        provider_index = self.concept_provider_combo.findData(self.concept_provider)
        self.concept_provider_combo.setCurrentIndex(max(0, provider_index))
        self.concept_provider_combo.currentIndexChanged.connect(
            self.update_concept_provider
        )
        related_layout.addWidget(self.concept_provider_combo, 0, 1)

        related_layout.addWidget(QtWidgets.QLabel('Cloud model'), 1, 0)
        self.concept_model_edit = QtWidgets.QComboBox(related_group)
        self.concept_model_edit.setEditable(True)
        self.concept_model_edit.setInsertPolicy(QtWidgets.QComboBox.NoInsert)
        if self.concept_model_edit.lineEdit():
            self.concept_model_edit.lineEdit().setPlaceholderText('Provider model ID')
        self.concept_model_edit.setToolTip(
            'Select from available cloud models or enter a custom model identifier. Not used with Local GGUF.'
        )
        self.concept_model_edit.text = self.concept_model_edit.currentText
        self.concept_model_edit.setText = self.concept_model_edit.setEditText
        self._populate_concept_model_presets(self.concept_provider)
        self.concept_model_edit.currentTextChanged.connect(
            self._on_concept_model_text_changed
        )
        related_layout.addWidget(self.concept_model_edit, 1, 1, 1, 2)

        related_layout.addWidget(QtWidgets.QLabel('API key'), 2, 0)
        self.concept_api_key_edit = QtWidgets.QLineEdit(related_group)
        self.concept_api_key_edit.setEchoMode(QtWidgets.QLineEdit.Password)
        self.concept_api_key_edit.setPlaceholderText(
            'Optional here when already set in the provider environment variable'
        )
        self.concept_api_key_edit.setToolTip(
            'API key for the selected cloud provider. Leave blank when it is already set '
            'in the provider environment variable.'
        )
        self.concept_api_key_edit.editingFinished.connect(
            self.update_cloud_concept_settings
        )
        self.concept_api_key_edit.textChanged.connect(
            self._on_concept_api_key_text_changed
        )
        related_layout.addWidget(self.concept_api_key_edit, 2, 1, 1, 2)

        base_url_label = QtWidgets.QLabel('Compatible URL')
        base_url_tooltip = (
            'Optional: Leave blank for official OpenAI. Only needed if using a custom '
            'or self-hosted OpenAI-compatible endpoint (e.g., Ollama http://localhost:11434/v1, LM Studio, or vLLM).'
        )
        base_url_label.setToolTip(base_url_tooltip)
        related_layout.addWidget(base_url_label, 3, 0)
        self.concept_base_url_edit = QtWidgets.QLineEdit(
            self.concept_base_url, related_group
        )
        self.concept_base_url_edit.setPlaceholderText(
            'Leave blank for OpenAI (or e.g. http://localhost:11434/v1)'
        )
        self.concept_base_url_edit.setToolTip(base_url_tooltip)
        self.concept_base_url_edit.editingFinished.connect(
            self.update_cloud_concept_settings
        )
        related_layout.addWidget(self.concept_base_url_edit, 3, 1, 1, 2)

        related_layout.addWidget(QtWidgets.QLabel('Local GGUF'), 4, 0)
        self.llm_model_path_edit = QtWidgets.QLineEdit(
            self.llm_model_path, related_group
        )
        self.llm_model_path_edit.setReadOnly(True)
        self.llm_model_path_edit.setToolTip(self.llm_model_path)
        self.llm_model_path_edit.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.llm_model_path_edit.setCursorPosition(len(self.llm_model_path))
        self.browse_llm_model_button = QtWidgets.QPushButton('Browse…', related_group)
        self.browse_llm_model_button.setObjectName('btn_browse_llm_model')
        self.browse_llm_model_button.setToolTip(
            'Choose a local .gguf language-model file. The full path appears when you hover.'
        )
        self.browse_llm_model_button.clicked.connect(self.browse_llm_model)
        related_layout.addWidget(self.llm_model_path_edit, 4, 1)
        related_layout.addWidget(self.browse_llm_model_button, 4, 2)

        self.generate_related_button = QtWidgets.QPushButton(
            'Generate related concepts', related_group
        )
        self.generate_related_button.setObjectName('btn_generate_related')
        self.generate_related_button.setToolTip(
            'Generate related concepts for every checked concept in the list.'
        )
        self.generate_related_button.setEnabled(False)
        self.generate_related_button.clicked.connect(self.start_related_concepts)
        self.pause_related_button = QtWidgets.QPushButton('Pause', related_group)
        self.pause_related_button.setToolTip(
            'Pause before the next concept; the current model request will finish.'
        )
        self.pause_related_button.setEnabled(False)
        self.pause_related_button.clicked.connect(self.toggle_related_concepts_pause)
        self.stop_related_button = QtWidgets.QPushButton('Stop', related_group)
        self.stop_related_button.setToolTip(
            'Stop after the current concept and keep completed results.'
        )
        self.stop_related_button.setEnabled(False)
        self.stop_related_button.clicked.connect(self.stop_related_concepts)
        self.related_action_stack = QtWidgets.QStackedWidget(related_group)
        related_generate_page = QtWidgets.QWidget()
        related_generate_layout = QtWidgets.QHBoxLayout(related_generate_page)
        related_generate_layout.setContentsMargins(0, 0, 0, 0)
        related_generate_layout.addWidget(self.generate_related_button)
        related_running_page = QtWidgets.QWidget()
        related_running_layout = QtWidgets.QHBoxLayout(related_running_page)
        related_running_layout.setContentsMargins(0, 0, 0, 0)
        related_running_layout.addWidget(self.pause_related_button)
        related_running_layout.addWidget(self.stop_related_button)
        self.related_action_stack.addWidget(related_generate_page)
        self.related_action_stack.addWidget(related_running_page)
        related_layout.addWidget(self.related_action_stack, 5, 0, 1, 3)
        related_layout.setColumnStretch(1, 1)
        self.workflow_controls_layout.addWidget(related_group, 1)
        self._refresh_llm_model_controls()

        image_group = QtWidgets.QGroupBox('3. Create images')
        self.image_controls_group = image_group
        image_layout = QtWidgets.QGridLayout(image_group)
        image_layout.setHorizontalSpacing(6)
        image_layout.addWidget(QtWidgets.QLabel('Image source'), 0, 0)
        self.image_source_combo = QtWidgets.QComboBox(image_group)
        self.image_source_combo.setSizeAdjustPolicy(
            QtWidgets.QComboBox.AdjustToMinimumContentsLengthWithIcon
        )
        self.image_source_combo.setMinimumContentsLength(12)
        self.image_source_combo.setObjectName('cmb_image_source')
        self.image_source_combo.setToolTip(
            'Choose whether images come from Bing, an OpenAI cloud model, or a local '
            'Diffusers-compatible model.'
        )
        self.image_source_combo.addItem('Bing search (strict SafeSearch)', 'bing')
        self.image_source_combo.addItem('Text stimulus (concept centered)', 'text')
        self.image_source_combo.addItem('Hugging Face Serverless (Cloud)', 'huggingface')
        self.image_source_combo.addItem('OpenAI cloud image model', 'cloud')
        self.image_source_combo.addItem('Google Imagen cloud image model', 'google')
        self.image_source_combo.addItem('Local image model', 'local')
        normalized_source = 'cloud' if self.image_source == 'generate' else self.image_source
        self.image_source = normalized_source
        source_index = self.image_source_combo.findData(normalized_source)
        self.image_source_combo.setCurrentIndex(max(0, source_index))
        self.image_source_combo.currentIndexChanged.connect(self.update_image_source)
        image_layout.addWidget(self.image_source_combo, 0, 1, 1, 3)

        image_layout.addWidget(QtWidgets.QLabel('Images per concept'), 1, 0)
        self.images_per_concept_spin = QtWidgets.QSpinBox(image_group)
        self.images_per_concept_spin.setObjectName('spb_download_count')
        self.images_per_concept_spin.setToolTip(
            'Number of images to download or generate for each checked related concept.'
        )
        download_settings = config.get('concepts_spb_params', {}).get(
            'download_count', [5, 1, 100, 1]
        )
        self.images_per_concept_spin.setRange(
            int(download_settings[1]), int(download_settings[2])
        )
        self.images_per_concept_spin.setValue(int(download_settings[0]))
        image_layout.addWidget(self.images_per_concept_spin, 1, 1)

        image_layout.addWidget(QtWidgets.QLabel('Local model'), 2, 0)
        self.local_image_model_edit = QtWidgets.QLineEdit(
            self.local_image_model_path, image_group
        )
        self.local_image_model_edit.setReadOnly(True)
        self.local_image_model_edit.setPlaceholderText(
            'Choose a Diffusers model folder or checkpoint'
        )
        self.local_image_model_edit.setToolTip(self.local_image_model_path)
        self.local_image_model_edit.setAccessibleDescription(
            'Full local image-model path is shown in the tooltip.'
        )
        self.local_image_model_edit.setCursorPosition(0)
        image_layout.addWidget(self.local_image_model_edit, 2, 1, 1, 3)
        local_model_buttons = QtWidgets.QHBoxLayout()
        self.browse_local_image_file_button = QtWidgets.QPushButton('File…')
        self.browse_local_image_file_button.setMaximumWidth(64)
        self.browse_local_image_file_button.setToolTip(
            'Choose a single Diffusers-compatible .safetensors, .ckpt, or .bin checkpoint.'
        )
        self.browse_local_image_file_button.clicked.connect(
            self.browse_local_image_model_file
        )
        self.browse_local_image_folder_button = QtWidgets.QPushButton('Folder…')
        self.browse_local_image_folder_button.setMaximumWidth(72)
        self.browse_local_image_folder_button.setToolTip(
            'Choose a local Diffusers model directory.'
        )
        self.browse_local_image_folder_button.clicked.connect(
            self.browse_local_image_model_folder
        )
        local_model_buttons.addWidget(self.browse_local_image_file_button)
        local_model_buttons.addWidget(self.browse_local_image_folder_button)
        local_model_buttons.addStretch(1)
        image_layout.addLayout(local_model_buttons, 3, 1, 1, 3)

        self.create_images_button = QtWidgets.QPushButton('Create images', image_group)
        self.create_images_button.setObjectName('btn_create_images')
        self.create_images_button.setToolTip(
            'Create images for every checked related concept using the selected source.'
        )
        self.create_images_button.setEnabled(False)
        self.create_images_button.clicked.connect(self.acquire_images_ext)
        self.pause_images_button = QtWidgets.QPushButton('Pause', image_group)
        self.pause_images_button.setToolTip(
            'Pause at the next image boundary; the current render/request will finish.'
        )
        self.pause_images_button.setEnabled(False)
        self.pause_images_button.clicked.connect(self.toggle_image_pause)
        self.stop_images_button = QtWidgets.QPushButton('Stop', image_group)
        self.stop_images_button.setToolTip(
            'Stop at the next image boundary and keep images already saved.'
        )
        self.stop_images_button.setEnabled(False)
        self.stop_images_button.clicked.connect(self.stop_image_generation)
        self.image_action_stack = QtWidgets.QStackedWidget(image_group)
        image_generate_page = QtWidgets.QWidget()
        image_generate_layout = QtWidgets.QHBoxLayout(image_generate_page)
        image_generate_layout.setContentsMargins(0, 0, 0, 0)
        image_generate_layout.addWidget(self.create_images_button)
        image_running_page = QtWidgets.QWidget()
        image_running_layout = QtWidgets.QHBoxLayout(image_running_page)
        image_running_layout.setContentsMargins(0, 0, 0, 0)
        image_running_layout.addWidget(self.pause_images_button)
        image_running_layout.addWidget(self.stop_images_button)
        self.image_action_stack.addWidget(image_generate_page)
        self.image_action_stack.addWidget(image_running_page)
        self.image_eta_label = QtWidgets.QLabel('Images appear below as they finish.')
        self.image_elapsed_timer = QtCore.QTimer(image_group)
        self.image_elapsed_timer.setInterval(1000)
        self.image_elapsed_timer.timeout.connect(self._refresh_image_elapsed_status)
        self.image_eta_label.setWordWrap(True)
        image_layout.addWidget(self.image_eta_label, 4, 0, 1, 4)
        image_layout.addWidget(self.image_action_stack, 5, 0, 1, 4)
        image_layout.setColumnStretch(1, 1)
        self.workflow_controls_layout.addWidget(image_group, 1)
        self._refresh_local_image_controls()

        options_layout = QtWidgets.QHBoxLayout()
        self.replace_existing_checkbox = QtWidgets.QCheckBox('Replace existing images')
        self.replace_existing_checkbox.setChecked(self.force_replace)
        self.replace_existing_checkbox.setToolTip(
            'Delete existing images in this workspace before starting the new image batch.'
        )
        self.replace_existing_checkbox.toggled.connect(
            lambda checked: setattr(self, 'force_replace', checked)
        )
        options_layout.addWidget(self.replace_existing_checkbox)
        options_layout.addStretch(1)
        self.crop_images_button = QtWidgets.QPushButton('Crop selected images')
        self.crop_images_button.setEnabled(False)
        self.crop_images_button.clicked.connect(self.crop_selected)
        options_layout.addWidget(self.crop_images_button)
        self.export_stimuli_button = QtWidgets.QPushButton('Export task stimuli')
        self.export_stimuli_button.setEnabled(False)
        self.export_stimuli_button.clicked.connect(self.create_task_stimuli)
        options_layout.addWidget(self.export_stimuli_button)
        self.concepts_right_layout.addLayout(options_layout)

    def choose_concepts_file(self):
        """Prompt for a concept list and show parsing errors in the UI."""
        default_dir = os.path.dirname(config.get('concepts_file', ''))
        if not default_dir or not os.path.exists(default_dir):
            default_dir = os.path.expanduser('~')
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self.concepts_widget,
            'Choose a concept list',
            default_dir,
            'Concept files (*.txt *.csv *.xlsx *.xls *.docx *.json *.tsv *.md);;All files (*)',
        )
        if not path:
            return
        try:
            self.load_concepts_file(path)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(
                self.concepts_widget, 'Could not load concepts', str(exc)
            )

    def start_related_concepts(self, concepts=None):
        """Validate the chosen text backend before starting step 2."""
        if self._image_run_active:
            self.c.log.emit('Stop image creation before generating related concepts.')
            return False
        if self._concept_run_active:
            self.c.log.emit('Related-concept generation is already running.')
            return False
        self.update_cloud_concept_settings()
        provider = self.concept_generator.selected_provider
        if provider == 'local' and not os.path.isfile(self.llm_model_path):
            QtWidgets.QMessageBox.critical(
                self.concepts_widget,
                'Local GGUF model not found',
                'Choose an existing GGUF model in step 2 before generating concepts.',
            )
            return
        if provider == 'local' and importlib.util.find_spec('llama_cpp') is None:
            QtWidgets.QMessageBox.critical(
                self.concepts_widget,
                'Local LLM dependencies are missing',
                'Install them with: pip install -e ".[llm]"',
            )
            return
        key_variables = {
            'openai': ('OPENAI_API_KEY',),
            'anthropic': ('ANTHROPIC_API_KEY',),
            'google': ('GEMINI_API_KEY', 'GOOGLE_API_KEY'),
        }
        dependency_modules = {
            'openai': 'openai',
            'anthropic': 'anthropic',
            'google': 'google.genai',
            'compatible': 'openai',
        }
        if provider in dependency_modules:
            try:
                dependency_available = (
                    importlib.util.find_spec(dependency_modules[provider]) is not None
                )
            except ModuleNotFoundError:
                dependency_available = False
            if not dependency_available:
                QtWidgets.QMessageBox.critical(
                    self.concepts_widget,
                    'Cloud dependencies are missing',
                    'Install them with: pip install -e ".[cloud]"',
                )
                return False
            has_key = bool(self.concept_api_key) or any(
                os.getenv(name) for name in key_variables.get(provider, ())
            )
            if provider != 'compatible' and not has_key:
                QtWidgets.QMessageBox.critical(
                    self.concepts_widget,
                    'API key required',
                    f'Enter a {provider.title()} API key or set '
                    f'{" or ".join(key_variables[provider])}.',
                )
                return False
            if not self.concept_model:
                QtWidgets.QMessageBox.critical(
                    self.concepts_widget,
                    'Cloud model required',
                    'Enter the model ID supplied by your API provider.',
                )
                return False
            if provider == 'compatible' and not self.concept_base_url:
                QtWidgets.QMessageBox.critical(
                    self.concepts_widget,
                    'Compatible API URL required',
                    'Enter the provider base URL, normally ending in /v1.',
                )
                return False
        self._concept_run_targets = (
            list(dict.fromkeys(concepts))
            if concepts is not None
            else self.get_selected_concepts()
        )
        if not self._concept_run_targets:
            self.c.log.emit('Select at least one concept before processing.')
            return False
        self._concept_total_expected = len(self._concept_run_targets)
        self._concept_completed_count = 0
        self._concept_run_active = True
        self._concept_stop_event.clear()
        self._concept_pause_event.set()
        self._set_concept_controls_running(True)
        self._set_workflow_tracker(
            completed=1, active=2, reset=True
        )
        self._on_concept_progress(
            0,
            f'Getting related concepts for {self._concept_total_expected} concepts…',
        )
        self.process_related_concepts_threaded()
        return True

    def _on_concept_progress(self, percentage, message):
        """Show related-concept progress in workflow stage 2."""
        self._set_workflow_step_progress(2, percentage)
        if hasattr(self, 'status_lbl'):
            self.status_lbl.setText(message)

    def _set_concept_controls_running(self, running):
        self.related_action_stack.setCurrentIndex(1 if running else 0)
        self.generate_related_button.setEnabled(
            not running and bool(self.patient_concepts)
        )
        self.load_concepts_button.setEnabled(not running)
        self.create_images_button.setEnabled(
            not running and any(self.related_concepts_dict.values())
        )
        self.concept_provider_combo.setEnabled(not running)
        self.llm_model_path_edit.setEnabled(not running)
        self.browse_llm_model_button.setEnabled(not running)
        self.concept_model_edit.setEnabled(not running)
        self.concept_api_key_edit.setEnabled(not running)
        self.concept_base_url_edit.setEnabled(not running)
        self.pause_related_button.setEnabled(running)
        self.stop_related_button.setEnabled(running)
        if not running:
            self.pause_related_button.setText('Pause')
            self._refresh_llm_model_controls()

    def toggle_related_concepts_pause(self):
        if not self._concept_run_active:
            return False
        if self._concept_pause_event.is_set():
            self._concept_pause_event.clear()
            self.pause_related_button.setText('Resume')
            message = 'Related-concept generation will pause after the current concept.'
        else:
            self._concept_pause_event.set()
            self.pause_related_button.setText('Pause')
            message = 'Related-concept generation resumed.'
        self.c.log.emit(message)
        self.status_lbl.setText(message)
        return True

    def stop_related_concepts(self):
        if not self._concept_run_active:
            return False
        self._concept_stop_event.set()
        self._concept_pause_event.set()
        self.pause_related_button.setEnabled(False)
        self.stop_related_button.setEnabled(False)
        message = 'Stopping related-concept generation after the current concept…'
        self.c.log.emit(message)
        self.status_lbl.setText(message)
        return True

    def _wait_for_concept_permission(self):
        while not self._concept_stop_event.is_set():
            if self._concept_pause_event.wait(timeout=0.1):
                return not self._concept_stop_event.is_set()
        return False

    def _populate_concept_model_presets(self, provider: str):
        """Populate cloud model dropdown with recommended models for the chosen provider."""
        presets = {
            'google': [
                'gemini-3.8-flash',
                'gemini-3.8-pro',
                'gemini-2.5-flash',
                'gemini-2.5-pro',
            ],
            'openai': [
                'gpt-5.6-luna',
                'gpt-4o',
                'gpt-4o-mini',
            ],
            'anthropic': [
                'claude-sonnet-4-6',
                'claude-3-5-sonnet-20241022',
                'claude-3-5-haiku-20241022',
            ],
            'compatible': [
                os.getenv('STIMULUS_FORGE_COMPAT_MODEL', 'meta-llama/llama-3.3-70b-instruct'),
            ],
        }
        models = presets.get(provider, [])
        if hasattr(self, 'concept_model_edit') and hasattr(self.concept_model_edit, 'clear'):
            self.concept_model_edit.blockSignals(True)
            self.concept_model_edit.clear()
            for m in models:
                self.concept_model_edit.addItem(m)
            current = self.concept_models.get(provider, '')
            if current and self.concept_model_edit.findText(current) == -1:
                self.concept_model_edit.insertItem(0, current)
            self.concept_model_edit.setEditText(current or (models[0] if models else ''))
            self.concept_model_edit.blockSignals(False)

    def _on_concept_model_text_changed(self, text: str):
        self.update_cloud_concept_settings()

    def update_concept_provider(self):
        """Apply the provider selected in the related-concept LLM control."""
        provider = self.concept_provider_combo.currentData()
        if provider not in {
            'local', 'openai', 'anthropic', 'google', 'compatible'
        }:
            return
        if hasattr(self, 'concept_model_edit'):
            self.concept_models[self.concept_provider] = (
                self.concept_model_edit.text().strip()
            )
            if self.concept_provider in self.concept_api_keys:
                self.concept_api_keys[self.concept_provider] = (
                    self.concept_api_key_edit.text().strip()
                )
        self._sync_openai_image_service_key()
        self._sync_google_image_service_key()
        self.concept_provider = provider
        self._populate_concept_model_presets(provider)
        self.concept_model = self.concept_models.get(provider, '')
        self.concept_api_key = self.concept_api_keys.get(provider, '')
        self.concept_model_edit.setText(self.concept_model)
        self.concept_api_key_edit.setText(self.concept_api_key)
        config['concept_provider'] = provider
        self.concept_generator = self._new_concept_generator()
        self._refresh_llm_model_controls()
        if self.c is not None and hasattr(self.c, 'log'):
            effective = self.concept_generator.selected_provider
            self.c.log.emit(
                f'Related-concept LLM set to {provider} (using {effective}).'
            )

    def _sync_openai_image_service_key(self):
        """Ensure OpenAI image service has the latest OpenAI API key from the UI or env."""
        if hasattr(self, 'concept_api_key_edit') and self.concept_provider == 'openai':
            current_text = self.concept_api_key_edit.text().strip()
            if current_text:
                self.concept_api_keys['openai'] = current_text
                self.concept_api_key = current_text
        openai_key = self.concept_api_keys.get('openai', '').strip()
        if not openai_key and self.concept_provider == 'openai':
            openai_key = self.concept_api_key.strip()
        if not openai_key:
            openai_key = os.environ.get('OPENAI_API_KEY', '').strip()

        if hasattr(self, 'openai_image_service'):
            self.openai_image_service.api_key = openai_key or None
        if openai_key:
            os.environ['OPENAI_API_KEY'] = openai_key

    def _sync_google_image_service_key(self):
        """Ensure Google image service has the latest Google API key from the UI or env."""
        if hasattr(self, 'concept_api_key_edit') and self.concept_provider == 'google':
            current_text = self.concept_api_key_edit.text().strip()
            if current_text:
                self.concept_api_keys['google'] = current_text
                self.concept_api_key = current_text
        google_key = self.concept_api_keys.get('google', '').strip()
        if not google_key and self.concept_provider == 'google':
            google_key = self.concept_api_key.strip()
        if not google_key:
            google_key = (os.environ.get('GEMINI_API_KEY', '') or os.environ.get('GOOGLE_API_KEY', '')).strip()

        if hasattr(self, 'google_image_service'):
            self.google_image_service.api_key = google_key or None
        if google_key:
            os.environ['GEMINI_API_KEY'] = google_key

    def _on_concept_api_key_text_changed(self, text):
        """Immediately propagate key updates when editing cloud settings."""
        cleaned = text.strip()
        if self.concept_provider == 'openai':
            self.concept_api_keys['openai'] = cleaned
            self.concept_api_key = cleaned
            if hasattr(self, 'openai_image_service'):
                self.openai_image_service.api_key = cleaned or None
            if cleaned:
                os.environ['OPENAI_API_KEY'] = cleaned
            elif not os.environ.get('OPENAI_API_KEY'):
                if hasattr(self, 'openai_image_service'):
                    self.openai_image_service.api_key = None
        elif self.concept_provider == 'google':
            self.concept_api_keys['google'] = cleaned
            self.concept_api_key = cleaned
            if hasattr(self, 'google_image_service'):
                self.google_image_service.api_key = cleaned or None
            if cleaned:
                os.environ['GEMINI_API_KEY'] = cleaned
            elif not os.environ.get('GEMINI_API_KEY') and not os.environ.get('GOOGLE_API_KEY'):
                if hasattr(self, 'google_image_service'):
                    self.google_image_service.api_key = None

    def update_cloud_concept_settings(self):
        """Apply model/API fields without persisting secret credentials."""
        self.concept_model = self.concept_model_edit.text().strip()
        self.concept_models[self.concept_provider] = self.concept_model
        self.concept_api_key = self.concept_api_key_edit.text().strip()
        if self.concept_provider in self.concept_api_keys:
            self.concept_api_keys[self.concept_provider] = self.concept_api_key
        self.concept_base_url = self.concept_base_url_edit.text().strip()
        if self.concept_provider == 'openai':
            config['concept_model'] = self.concept_model
        elif self.concept_provider != 'local':
            config[f'{self.concept_provider}_concept_model'] = self.concept_model
        config['compatible_base_url'] = self.concept_base_url
        self.concept_generator = self._new_concept_generator()
        self._sync_openai_image_service_key()
        self._sync_google_image_service_key()

    def _refresh_llm_model_controls(self):
        """Enable local-model controls whenever the provider may use them."""
        local_available = self.concept_provider == 'local'
        self.llm_model_path_edit.setEnabled(local_available)
        self.browse_llm_model_button.setEnabled(local_available)
        cloud_available = self.concept_provider != 'local'
        self.concept_model_edit.setEnabled(cloud_available)
        self.concept_api_key_edit.setEnabled(
            self.concept_provider in self.concept_api_keys
        )
        self.concept_base_url_edit.setEnabled(
            self.concept_provider == 'compatible'
        )

    def browse_llm_model(self):
        """Choose a local GGUF and make it active for subsequent requests."""
        initial_path = self.llm_model_path or str(os.path.expanduser('~'))
        model_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self.concepts_widget,
            'Choose a GGUF language model',
            initial_path,
            'GGUF models (*.gguf);;All files (*)',
        )
        if not model_path:
            return

        model_name = os.path.basename(model_path)
        shard_match = re.search(r"-(\d{5})-of-(\d{5}\.gguf)$", model_name)
        if shard_match and shard_match.group(1) != "00001":
            first_shard_name = re.sub(r"-\d{5}-of-(\d{5}\.gguf)$", r"-00001-of-\1", model_name)
            first_shard_path = os.path.join(os.path.dirname(model_path), first_shard_name)
            if os.path.isfile(first_shard_path):
                model_path = first_shard_path

        self.llm_model_path = os.path.abspath(model_path)
        config['llm_model'] = self.llm_model_path
        self.llm_model_path_edit.setText(self.llm_model_path)
        self.llm_model_path_edit.setToolTip(self.llm_model_path)
        self.llm_model_path_edit.setCursorPosition(len(self.llm_model_path))
        self.concept_generator = self._new_concept_generator()
        if self.c is not None and hasattr(self.c, 'log'):
            self.c.log.emit(f'Using local GGUF model: {self.llm_model_path}')

    def update_image_source(self):
        """Apply the selected image acquisition backend."""
        source = self.image_source_combo.currentData()
        if source not in {'bing', 'cloud', 'google', 'local'}:
            return
        self.image_source = source
        config['image_source'] = source
        self._refresh_local_image_controls()

    def _refresh_local_image_controls(self):
        local_selected = self.image_source == 'local'
        self.local_image_model_edit.setEnabled(local_selected)
        self.browse_local_image_file_button.setEnabled(local_selected)
        self.browse_local_image_folder_button.setEnabled(local_selected)

    def _set_local_image_model(self, model_path):
        self.local_image_model_path = os.path.abspath(model_path)
        config['local_image_model'] = self.local_image_model_path
        self.local_image_model_edit.setText(self.local_image_model_path)
        self.local_image_model_edit.setToolTip(self.local_image_model_path)
        self.local_image_model_edit.setCursorPosition(0)
        self.local_image_service = LocalImageService(
            model_path=self.local_image_model_path
        )
        self.huggingface_image_service = HuggingFaceImageService(
            model=os.environ.get(
                'STIMULUS_FORGE_HF_IMAGE_MODEL',
                config.get('huggingface_image_model', 'black-forest-labs/FLUX.1-schnell'),
            ),
            api_key=self.concept_api_keys.get('huggingface', ''),
        )
        self.text_image_service = TextImageService()
        if self.c is not None and hasattr(self.c, 'log'):
            self.c.log.emit(
                f'Using local image model: {self.local_image_model_path}'
            )

    def browse_local_image_model_file(self):
        """Choose a single-file Diffusers checkpoint."""
        initial_path = self.local_image_model_path or str(os.path.expanduser('~'))
        model_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self.concepts_widget,
            'Choose a local image-model checkpoint',
            initial_path,
            'Model checkpoints (*.safetensors *.ckpt *.bin);;All files (*)',
        )
        if model_path:
            self._set_local_image_model(model_path)

    def browse_local_image_model_folder(self):
        """Choose a local Diffusers model directory."""
        initial_path = self.local_image_model_path or str(os.path.expanduser('~'))
        if os.path.isfile(initial_path):
            initial_path = os.path.dirname(initial_path)
        model_path = QtWidgets.QFileDialog.getExistingDirectory(
            self.concepts_widget,
            'Choose a local Diffusers model folder',
            initial_path,
        )
        if model_path:
            self._set_local_image_model(model_path)

    def enable_workspace_actions(self, enabled=True):
        """Enable actions in workflow order after a concept list is loaded."""
        self.generate_related_button.setEnabled(enabled)
        has_related = enabled and any(self.related_concepts_dict.values())
        self.create_images_button.setEnabled(has_related)
        self.crop_images_button.setEnabled(enabled)
        self.export_stimuli_button.setEnabled(enabled)

    def download_imgs_ext(self):
        """Backward-compatible entry point using the selected image source."""
        return self.acquire_images_ext()

    def _confirm_image_acquisition(self, source, image_count, replace_existing=False):
        source = 'cloud' if source == 'generate' else source
        source_names = {
            'bing': 'Bing',
            'cloud': 'OpenAI cloud',
            'google': 'Google Imagen',
            'huggingface': 'Hugging Face',
            'local': 'the local image model',
            'text': 'Text stimulus generation',
        }
        source_name = source_names.get(source, source)
        msg = QtWidgets.QMessageBox(self.concepts_widget)
        msg.setIcon(QtWidgets.QMessageBox.Warning)
        msg.setText(f'Create up to {image_count} images using {source_name}?')
        details = []
        if source == 'cloud':
            details.append('This uses the OpenAI API and may incur usage charges.')
        elif source == 'bing':
            details.append(
                'Bing strict SafeSearch is enabled. Review downloaded images before use.'
            )
        elif source == 'text':
            details.append(
                'High-contrast images with centered concept text will be generated locally.'
            )
        else:
            details.append(
                'Local models are not moderated automatically. Review every generated image.'
            )
        if replace_existing:
            details.append('Existing images in this workspace will be replaced.')
        msg.setInformativeText(' '.join(details))
        msg.setWindowTitle('Create Images')
        msg.setStandardButtons(QtWidgets.QMessageBox.Ok | QtWidgets.QMessageBox.Cancel)
        msg.setDefaultButton(QtWidgets.QMessageBox.Cancel)
        return msg.exec_() == QtWidgets.QMessageBox.Ok

    def acquire_images_ext(self, source=None):
        """Validate a batch, then run it without blocking the Qt event loop."""
        if self._concept_run_active:
            self.c.log.emit('Stop related-concept generation before creating images.')
            return False
        if self._image_run_active:
            self.c.log.emit('An image batch is already running.')
            return False
        source = source or self.image_source
        source = 'cloud' if source == 'generate' else source
        if source not in {'bing', 'cloud', 'google', 'huggingface', 'local', 'text'}:
            raise ValueError(f'Unknown image source: {source}')
        if source == 'huggingface':
            if not self.huggingface_image_service.is_available:
                message = 'Set HF_TOKEN or enter your Hugging Face API token before generating images.'
                self.c.log.emit(message)
                QtWidgets.QMessageBox.critical(
                    self.concepts_widget, 'Hugging Face is not configured', message
                )
                return False
        if source == 'cloud':
            self._sync_openai_image_service_key()
            if not self.openai_image_service.is_available:
                message = self._openai_image_configuration_error() or (
                    'OpenAI image generation needs an API key and cloud dependencies.'
                )
                self.c.log.emit(message)
                QtWidgets.QMessageBox.critical(
                    self.concepts_widget, 'OpenAI is not configured', message
                )
                return False
        if source == 'google':
            self._sync_google_image_service_key()
            if not self.google_image_service.is_available:
                message = self._google_image_configuration_error() or (
                    'Google Imagen generation needs an API key and cloud dependencies.'
                )
                self.c.log.emit(message)
                QtWidgets.QMessageBox.critical(
                    self.concepts_widget, 'Google Imagen is not configured', message
                )
                return False
        if source == 'local' and not self.local_image_service.is_available:
            message = self._local_image_configuration_error()
            self.c.log.emit(message)
            QtWidgets.QMessageBox.critical(
                self.concepts_widget, 'Local image model is not configured', message
            )
            return False
        download_count = self.concepts_widget.findChild(QtWidgets.QSpinBox, 'spb_download_count').value()
        curr_patient = self.subject_id
        output_dir_base = os.path.abspath(os.path.join(self.task_images_downloader.task_images_path, curr_patient))

        tasks = []
        for concept, related_concepts in self.get_selected_image_targets().items():
            concept_tasks = self.prepare_concept_download_tasks(concept, related_concepts, output_dir_base, download_count)
            tasks.extend(concept_tasks)        

        if not tasks:
            self.c.log.emit('No images need to be created for the selected concepts.')
            return True

        replace_existing = self.force_replace and os.path.exists(output_dir_base)
        if not self._confirm_image_acquisition(
            source, sum(task['limit'] for task in tasks), replace_existing
        ):
            return False
        if replace_existing:
            shutil.rmtree(output_dir_base)

        action = 'Downloading' if source == 'bing' else 'Generating'
        self._start_image_tasks(tasks, source, action)
        return True

    def acquire_concept_images(self, concept, source=None):
        """Create images only for one parent concept and its checked associations."""
        if self._concept_run_active or self._image_run_active:
            self.c.log.emit('Wait for the current workflow job to finish.')
            return False
        source = source or self.image_source
        source = 'cloud' if source == 'generate' else source
        if source not in {'bing', 'cloud', 'google', 'huggingface', 'local', 'text'}:
            return False
        if source == 'huggingface':
            if not self.huggingface_image_service.is_available:
                message = 'Set HF_TOKEN or enter your Hugging Face API token before generating images.'
                self.c.log.emit(message)
                QtWidgets.QMessageBox.critical(
                    self.concepts_widget, 'Hugging Face is not configured', message
                )
                return False
        if source == 'cloud':
            self._sync_openai_image_service_key()
            if not self.openai_image_service.is_available:
                message = self._openai_image_configuration_error() or (
                    'OpenAI image generation needs an API key and cloud dependencies.'
                )
                self.c.log.emit(message)
                QtWidgets.QMessageBox.critical(
                    self.concepts_widget, 'OpenAI is not configured', message
                )
                return False
        if source == 'google':
            self._sync_google_image_service_key()
            if not self.google_image_service.is_available:
                message = self._google_image_configuration_error() or (
                    'Google Imagen generation needs an API key and cloud dependencies.'
                )
                self.c.log.emit(message)
                QtWidgets.QMessageBox.critical(
                    self.concepts_widget, 'Google Imagen is not configured', message
                )
                return False
        if source == 'local' and not self.local_image_service.is_available:
            message = self._local_image_configuration_error()
            self.c.log.emit(message)
            QtWidgets.QMessageBox.critical(
                self.concepts_widget, 'Local image model is not configured', message
            )
            return False

        targets = self.get_selected_image_targets().get(concept)
        if targets is None:
            self.c.log.emit(f'Select “{concept}” in the concept tree first.')
            return False
        count = self.images_per_concept_spin.value()
        output_root = os.path.abspath(
            os.path.join(self.task_images_downloader.task_images_path, self.subject_id)
        )
        tasks = self.prepare_concept_download_tasks(
            concept, targets, output_root, count
        )
        if not tasks:
            self.c.log.emit(f'No images need to be created for {concept}.')
            return False
        replace_existing = self.force_replace and os.path.exists(
            os.path.join(output_root, concept)
        )
        if not self._confirm_image_acquisition(
            source, sum(int(task['limit']) for task in tasks), replace_existing
        ):
            return False
        if replace_existing:
            shutil.rmtree(os.path.join(output_root, concept))
        action = 'Downloading' if source == 'bing' else 'Generating'
        return self._start_image_tasks(tasks, source, action)

    def _start_image_tasks(self, tasks, source, action):
        """Start a complete image batch on a daemon coordinator thread."""
        if self._concept_run_active:
            self.c.log.emit('Stop related-concept generation before creating images.')
            return False
        if self._image_run_active or not tasks:
            return False
        self._image_run_active = True
        self._image_run_action = action
        self._image_run_subject = str(self.subject_id or '')
        self._image_run_started = time.time()
        self._image_total_expected = sum(int(task['limit']) for task in tasks)
        self._image_saved_ui_count = 0
        self._image_stop_event.clear()
        self._image_pause_event.set()
        self._set_workflow_tracker(
            completed=2, active=3, reset=True
        )
        self._set_image_controls_enabled(False)
        self._on_image_progress(
            0, f'{action} 0/{self._image_total_expected} images | estimating time…'
        )
        if hasattr(self, 'image_elapsed_timer'):
            self.image_elapsed_timer.start()
        self.c.log.emit(f'{action} images for {self._image_run_subject} in the background…')
        worker = threading.Thread(
            target=self._image_worker,
            args=(tasks, source),
            name='stimstudio-image-batch',
            daemon=True,
        )
        self._image_worker_thread = worker
        worker.start()
        return True

    def _image_worker(self, tasks, source):
        start_time = time.time()
        try:
            saved = self.run_image_tasks_in_parallel(
                tasks, source=source, start_time=start_time
            )
            self.image_workflow_signals.finished.emit(
                saved,
                time.time() - start_time,
                self._image_stop_event.is_set(),
            )
        except Exception as exc:
            self.image_workflow_signals.failed.emit(str(exc))

    def _set_image_controls_enabled(self, enabled):
        if hasattr(self, 'image_action_stack'):
            self.image_action_stack.setCurrentIndex(0 if enabled else 1)
        for control_name in (
            'create_images_button',
            'generate_related_button',
            'load_concepts_button',
            'image_source_combo',
            'images_per_concept_spin',
            'browse_local_image_file_button',
            'browse_local_image_folder_button',
            'replace_existing_checkbox',
        ):
            control = getattr(self, control_name, None)
            if control is not None:
                control.setEnabled(enabled)
        if hasattr(self, 'pause_images_button'):
            self.pause_images_button.setEnabled(not enabled)
            self.stop_images_button.setEnabled(not enabled)
            if enabled:
                self.pause_images_button.setText('Pause')
        if enabled:
            self._refresh_local_image_controls()
            self.generate_related_button.setEnabled(bool(self.patient_concepts))
            self.create_images_button.setEnabled(
                bool(self.related_concepts_dict) and any(
                    self.related_concepts_dict.values()
                )
            )

    def toggle_image_pause(self):
        if not self._image_run_active:
            return False
        if self._image_pause_event.is_set():
            self._image_pause_event.clear()
            self.pause_images_button.setText('Resume')
            message = 'Image creation will pause at the next image boundary.'
        else:
            self._image_pause_event.set()
            self.pause_images_button.setText('Pause')
            message = 'Image creation resumed.'
        self.c.log.emit(message)
        self.status_lbl.setText(message)
        return True

    def stop_image_generation(self):
        if not self._image_run_active:
            return False
        self._image_stop_event.set()
        self._image_pause_event.set()
        self.pause_images_button.setEnabled(False)
        self.stop_images_button.setEnabled(False)
        message = 'Stopping image creation at the next image boundary…'
        self.c.log.emit(message)
        self.status_lbl.setText(message)
        return True

    def _wait_for_image_permission(self):
        while not self._image_stop_event.is_set():
            if self._image_pause_event.wait(timeout=0.1):
                return not self._image_stop_event.is_set()
        return False

    def _on_image_progress(self, percentage, message):
        self._set_workflow_step_progress(3, percentage)
        if hasattr(self, 'image_eta_label'):
            self.image_eta_label.setText(message)
        self.c.progress.emit(percentage, message, False)

    def _refresh_image_elapsed_status(self):
        """Keep elapsed time moving even while the first image is still rendering."""
        if not self._image_run_active:
            return
        elapsed = max(time.time() - self._image_run_started, 0.001)
        completed = self._image_saved_ui_count
        total = self._image_total_expected
        hrs, mins, secs, _ = get_time_hrs_mins_secs_ms(seconds=elapsed)
        if completed:
            remaining = max(total - completed, 0) * elapsed / completed
            hrs_r, mins_r, secs_r, _ = get_time_hrs_mins_secs_ms(seconds=remaining)
            estimate = f'About {hrs_r:02.0f}:{mins_r:02.0f}:{secs_r:02.0f} remaining'
        else:
            estimate = 'Estimating remaining time after the first image'
        percentage = min(99, int(completed / total * 100)) if total else 0
        self._on_image_progress(
            percentage,
            f'{self._image_run_action} {completed}/{total} images | '
            f'Elapsed {hrs:02.0f}:{mins:02.0f}:{secs:02.0f} | {estimate}',
        )

    def _on_image_saved(self, task, file_path):
        """Add and preview each image immediately after it reaches disk."""
        if not os.path.isfile(file_path) or not hasattr(self, 'related_concepts_model'):
            return
        root = self.related_concepts_model.item(0, 0)
        if root is None:
            return
        concept_item = next(
            (
                root.child(row)
                for row in range(root.rowCount())
                if root.child(row).text() == task['concept']
            ),
            None,
        )
        if concept_item is None:
            concept_item = QtGui.QStandardItem(task['concept'])
            concept_item.setCheckable(True)
            concept_item.setCheckState(Qt.Checked)
            root.appendRow(concept_item)

        display_name = task.get('related_concept') or task.get('name') or ''
        parent_item = concept_item
        if display_name:
            parent_item = next(
                (
                    concept_item.child(row)
                    for row in range(concept_item.rowCount())
                    if concept_item.child(row).text() == display_name
                ),
                None,
            )
            if parent_item is None:
                parent_item = QtGui.QStandardItem(display_name)
                parent_item.setCheckable(True)
                parent_item.setCheckState(Qt.Checked)
                concept_item.appendRow(parent_item)

        existing_paths = {
            parent_item.child(row).data(IMAGE_PATH_ROLE)
            for row in range(parent_item.rowCount())
        }
        if file_path in existing_paths:
            return
        self._image_saved_ui_count += 1
        image_item = QtGui.QStandardItem(os.path.basename(file_path))
        image_item.setCheckable(True)
        image_item.setCheckState(Qt.Checked)
        image_item.setData(file_path, IMAGE_PATH_ROLE)
        parent_item.appendRow(image_item)
        self.related_concepts_tree_view.expand(concept_item.index())
        self.related_concepts_tree_view.expand(parent_item.index())
        self.related_concepts_tree_view.setCurrentIndex(image_item.index())
        try:
            # Keep automatic previews lightweight; face analysis can run when the
            # user explicitly clicks the image.
            self.show_image(file_path=file_path, item=image_item, detect_faces=False)
        except Exception as exc:
            self.c.log.emit(f'Image was saved but could not be previewed: {exc}')

    def _on_image_generation_finished(self, saved_count, elapsed_seconds, stopped):
        self._image_run_active = False
        if hasattr(self, 'image_elapsed_timer'):
            self.image_elapsed_timer.stop()
        self._set_image_controls_enabled(True)
        hrs, mins, secs, _ = get_time_hrs_mins_secs_ms(seconds=elapsed_seconds)
        outcome = 'stopped' if stopped else 'complete'
        message = (
            f'{self._image_run_action} {outcome}: {saved_count} images saved in '
            f'{hrs:02.0f}:{mins:02.0f}:{secs:02.0f}'
        )
        percentage = (
            min(99, int(saved_count / self._image_total_expected * 100))
            if stopped and self._image_total_expected else 100
        )
        self._on_image_progress(percentage, message)
        self.c.log.emit(message)
        if stopped:
            self._workflow_step_progress[3] = percentage
            self._set_workflow_tracker(completed=2, active=3, reset=True)
        else:
            self._set_workflow_tracker(completed=3, active=None, reset=True)
        if hasattr(self, 'status_lbl'):
            self.status_lbl.setText(message)

    def _on_image_generation_failed(self, message):
        self._image_run_active = False
        if hasattr(self, 'image_elapsed_timer'):
            self.image_elapsed_timer.stop()
        self._set_image_controls_enabled(True)
        display = f'Image creation stopped: {message}'
        self._on_image_progress(0, display)
        self.c.log.emit(display)
        self._sync_workflow_tracker()
        QtWidgets.QMessageBox.critical(
            self.concepts_widget, 'Could not create images', display
        )

    def _bing_image_validator(self):
        if not config.get('moderate_bing_images', True):
            return None
        self._sync_openai_image_service_key()
        if not self.openai_image_service.is_available:
            return None
        return self.openai_image_service.is_sexually_safe

    def _openai_image_configuration_error(self):
        """Explain precisely why OpenAI image generation cannot start."""
        self._sync_openai_image_service_key()
        if not self.openai_image_service.is_configured:
            return (
                'OpenAI image generation requires an API key. Please enter your key in '
                'Step 2 (Related Concepts) under "API Key" or set the OPENAI_API_KEY environment variable.'
            )
        if importlib.util.find_spec('openai') is None:
            return (
                'OpenAI package is missing. Install cloud dependencies with: '
                'pip install -e ".[cloud]"'
            )
        return None

    def _google_image_configuration_error(self):
        """Explain precisely why Google image generation cannot start."""
        self._sync_google_image_service_key()
        if not self.google_image_service.is_configured:
            return (
                'Google Imagen requires a Gemini/Google API key. Please enter your key in '
                'Step 2 (Related Concepts) under "API Key" (with Google selected) or set GEMINI_API_KEY.'
            )
        if importlib.util.find_spec('google.genai') is None:
            return (
                'Google GenAI package is missing. Install cloud dependencies with: '
                'pip install -e ".[cloud]"'
            )
        return None

    def _local_image_configuration_error(self):
        """Explain precisely why local image generation cannot start."""
        if not self.local_image_service.is_configured:
            return 'Choose a local image model file or folder in step 3.'
        missing = self.local_image_service.missing_dependencies
        if missing:
            return (
                f'Missing local image dependencies: {", ".join(missing)}. '
                'Install them in the Python environment that launches Stimulus Forge with: '
                'python -m pip install -e ".[local-images]"'
            )
        return None

    def image_task(
        self,
        task,
        source,
        on_image=None,
        seen_images=None,
        seen_images_lock=None,
    ):
        try:
            if source in {'generate', 'cloud'}:
                self._sync_openai_image_service_key()
                paths = self.openai_image_service.generate(
                    concept=task['concept'],
                    related_concept=task['related_concept'] or '',
                    name=task['name'] or '',
                    description=task.get('description', ''),
                    limit=task['limit'],
                    output_dir=task['output_dir'],
                    on_image=on_image,
                    should_continue=self._wait_for_image_permission,
                )
            elif source == 'google':
                self._sync_google_image_service_key()
                paths = self.google_image_service.generate(
                    concept=task['concept'],
                    related_concept=task['related_concept'] or '',
                    name=task['name'] or '',
                    description=task.get('description', ''),
                    limit=task['limit'],
                    output_dir=task['output_dir'],
                    on_image=on_image,
                    should_continue=self._wait_for_image_permission,
                )
            elif source == 'local':
                paths = self.local_image_service.generate(
                    concept=task['concept'],
                    related_concept=task['related_concept'] or '',
                    name=task['name'] or '',
                    description=task.get('description', ''),
                    limit=task['limit'],
                    output_dir=task['output_dir'],
                    on_image=on_image,
                    should_continue=self._wait_for_image_permission,
                )
            elif source == 'huggingface':
                paths = self.huggingface_image_service.generate(
                    concept=task['concept'],
                    related_concept=task['related_concept'] or '',
                    name=task['name'] or '',
                    description=task.get('description', ''),
                    limit=task['limit'],
                    output_dir=task['output_dir'],
                    on_image=on_image,
                    should_continue=self._wait_for_image_permission,
                )
            elif source == 'text':
                paths = self.text_image_service.generate(
                    concept=task['concept'],
                    related_concept=task['related_concept'] or '',
                    name=task['name'] or '',
                    description=task.get('description', ''),
                    limit=task['limit'],
                    output_dir=task['output_dir'],
                    on_image=on_image,
                    should_continue=self._wait_for_image_permission,
                )
            else:
                paths = self.task_images_downloader.download(
                    concept=task['concept'],
                    related_concept=task['related_concept'] or '',
                    name=task['name'] or '',
                    limit=task['limit'],
                    output_dir=task['output_dir'],
                    safe_search='strict',
                    image_validator=self._bing_image_validator(),
                    on_image=on_image,
                    should_continue=self._wait_for_image_permission,
                    filter='photo',
                    force_replace=False,
                    verbose=True,
                    seen_images=seen_images,
                    seen_images_lock=seen_images_lock,
                    context=task.get('context', ''),
                )
            return paths
        except Exception as exc:
            target = task['related_concept'] or task['name'] or task['concept']
            self.c.log.emit(f'Could not create images for {target}: {exc}')
            print(traceback.format_exc())
            return []

    def prepare_concept_download_tasks(self, concept, related_concepts, output_dir_base, download_count):
        tasks = []
        ctx = getattr(self, "concept_excerpts", {}).get(concept, "")
        # If the main concept is not already downloaded, add a task for it
        if concept not in self.concepts_available:
            output_dir = os.path.join(output_dir_base, concept)
            tasks.append({
                'concept': concept,
                'related_concept': None,
                'name': None,
                'description': '',
                'context': ctx,
                'limit': download_count,
                'output_dir': output_dir
            })
        for related_concept in related_concepts:
            try:
                related_concept_name = related_concept['char'].replace(' ', '').lower()
                if related_concept_name == "":
                    related_concept_name = related_concept['name'].replace(' ', '').lower()
                if related_concept_name not in self.concepts_available:
                    output_dir = os.path.join(output_dir_base, concept, related_concept_name)
                    tasks.append({
                        'concept': concept,
                        'related_concept': related_concept['char'],
                        'name': related_concept['name'],
                        'description': related_concept['description'],
                        'context': ctx,
                        'limit': download_count,
                        'output_dir': output_dir
                    })
            except Exception:
                print(traceback.format_exc())
        return tasks

    def run_image_tasks_in_parallel(self, tasks, source, start_time=None):
        if start_time is None:
            start_time = time.time()
        if not tasks:
            return 0

        saved_count = 0
        total_images = sum(int(task['limit']) for task in tasks)
        count_lock = threading.Lock()
        batch_seen_images = set()
        batch_seen_images_lock = threading.Lock()
        if source == 'bing':
            output_paths = [
                os.path.abspath(
                    os.path.dirname(task['output_dir'])
                    if task.get('related_concept') or task.get('name')
                    else task['output_dir']
                )
                for task in tasks
            ]
            scan_root = os.path.commonpath(output_paths)
            if os.path.isfile(scan_root):
                scan_root = os.path.dirname(scan_root)
            if os.path.isdir(scan_root):
                for root, _, filenames in os.walk(scan_root):
                    for filename in filenames:
                        if not filename.lower().endswith(IMAGE_EXTENSIONS):
                            continue
                        try:
                            with open(os.path.join(root, filename), 'rb') as image_file:
                                batch_seen_images.add(
                                    hashlib.sha256(image_file.read()).hexdigest()
                                )
                        except OSError:
                            continue

        def image_saved(task, path):
            nonlocal saved_count
            with count_lock:
                saved_count += 1
                current_count = saved_count
            elapsed = max(time.time() - start_time, 0.001)
            remaining_images = max(total_images - current_count, 0)
            remaining_time = remaining_images / (current_count / elapsed)
            hrs, mins, secs, _ = get_time_hrs_mins_secs_ms(seconds=elapsed)
            hrs_r, mins_r, secs_r, _ = get_time_hrs_mins_secs_ms(
                seconds=remaining_time
            )
            percentage = min(99, int(current_count / total_images * 100))
            message = (
                f'{self._image_run_action} {current_count}/{total_images} images | '
                f'Elapsed {hrs:02.0f}:{mins:02.0f}:{secs:02.0f} | '
                f'About {hrs_r:02.0f}:{mins_r:02.0f}:{secs_r:02.0f} remaining'
            )
            self.image_workflow_signals.image_saved.emit(task, str(path))
            self.image_workflow_signals.progress.emit(percentage, message)

        def run_task(task):
            if not self._wait_for_image_permission():
                return []
            return self.image_task(
                task,
                source,
                lambda path, task=task: image_saved(task, path),
                seen_images=batch_seen_images if source == 'bing' else None,
                seen_images_lock=(
                    batch_seen_images_lock if source == 'bing' else None
                ),
            )

        max_workers = 1 if source == 'local' else (2 if source in {'generate', 'cloud'} else 6)
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(run_task, task): task
                for task in tasks
            }
            for future in concurrent.futures.as_completed(futures):
                future.result()
        return saved_count

    def run_download_tasks_in_parallel(self, tasks, start_time=None):
        """Backward-compatible wrapper for older integrations."""
        return self.run_image_tasks_in_parallel(tasks, 'bing', start_time)

    def color_tree_item(self, concept, color, check=False):
        """
        Color the concept tree item
        """
        concept_item = self.related_concepts_model.findItems(concept, QtCore.Qt.MatchExactly | QtCore.Qt.MatchRecursive)
        if concept_item:
            concept_item[0].setBackground(QtGui.QBrush(QtGui.QColor(color)))
            concept_item[0].setCheckState(QtCore.Qt.Checked if check else QtCore.Qt.Unchecked)
        # Update the tree view
        self.related_concepts_tree_view.setModel(self.related_concepts_model)
    
    def download_img(self):
        """Download images for the current tree item with strict Bing filtering."""
        return self.acquire_current_item_images('bing')

    def generate_img(self):
        """Generate images for the current tree item with OpenAI."""
        return self.acquire_current_item_images('generate')

    def acquire_current_item_images(self, source):
        source = 'cloud' if source == 'generate' else source
        if source == 'cloud':
            self._sync_openai_image_service_key()
            if not self.openai_image_service.is_available:
                message = self._openai_image_configuration_error() or (
                    'OpenAI image generation needs an API key and cloud dependencies.'
                )
                self.c.log.emit(message)
                QtWidgets.QMessageBox.critical(
                    self.concepts_widget, 'OpenAI is not configured', message
                )
                return False
        if source == 'google':
            self._sync_google_image_service_key()
            if not self.google_image_service.is_available:
                message = self._google_image_configuration_error() or (
                    'Google Imagen generation needs an API key and cloud dependencies.'
                )
                self.c.log.emit(message)
                QtWidgets.QMessageBox.critical(
                    self.concepts_widget, 'Google Imagen is not configured', message
                )
                return False
        if source == 'local' and not self.local_image_service.is_available:
            message = self._local_image_configuration_error()
            self.c.log.emit(message)
            QtWidgets.QMessageBox.critical(
                self.concepts_widget, 'Local image model is not configured', message
            )
            return False
        curr_patient = self.subject_id
        index = self.related_concepts_tree_view.currentIndex()
        item = self.related_concepts_model.itemFromIndex(index)
        if item is None:
            return False
        if item.text().lower().endswith(IMAGE_EXTENSIONS):
            return False
        elif item.text() == curr_patient:
            return self.acquire_images_ext(source)
        elif item.parent().text() == curr_patient:
            concept = item.text()
            related_concept = ""
            name = ""
            description = ""
            limit = self.concepts_widget.findChild(
                QtWidgets.QSpinBox, 'spb_download_count'
            ).value()
            output_dir = os.path.abspath(os.path.join(self.task_images_downloader.task_images_path, curr_patient, concept))
        elif item.parent().parent().text() == curr_patient:
            concept = item.parent().text()
            display_name = item.text()
            metadata = next(
                (
                    value
                    for value in self.related_concepts_dict.get(concept, [])
                    if (value.get('char') or value.get('name')) == display_name
                ),
                {},
            )
            related_concept = metadata.get('char', '')
            name = metadata.get('name', display_name)
            description = metadata.get('description', '')
            folder_name = display_name.replace(' ', '').lower()
            output_dir = os.path.abspath(
                os.path.join(
                    self.task_images_downloader.task_images_path,
                    curr_patient,
                    concept,
                    folder_name,
                )
            )
            limit = self.images_per_concept_spin.value()
        else:
            return False

        replace_existing = self.force_replace and os.path.exists(output_dir)
        if not self._confirm_image_acquisition(source, limit, replace_existing):
            return False
        if replace_existing:
            shutil.rmtree(output_dir)
        task = {
            'concept': concept,
            'related_concept': related_concept,
            'name': name,
            'description': description,
            'limit': limit,
            'output_dir': output_dir,
        }
        action = 'Downloading' if source == 'bing' else 'Generating'
        return self._start_image_tasks([task], source, action)
    
    def rename_concept(self):
        """
        Rename concept
        """
        curr_patient = self.subject_id
        index = self.related_concepts_tree_view.currentIndex()
        item = self.related_concepts_model.itemFromIndex(index)

        if item.parent().text() == curr_patient:
            concept = item.text()
            new_name, ok = QtWidgets.QInputDialog.getText(self.concepts_widget, 'Rename Concept', 'Enter new name:', text=concept)
            if ok and new_name:
                return self.rename_parent_concept(concept, new_name)
        elif item.parent().parent().text() == curr_patient:
            related_concept = item.text()
            concept = item.parent().text()
            new_name, ok = QtWidgets.QInputDialog.getText(self.concepts_widget, 'Rename Concept', 'Enter new name:', text=related_concept)
            if ok and new_name:
                return self.rename_related_concept_named(
                    concept, related_concept, new_name
                )
        return False

    def add_related_concept_dialog(self, concept=None):
        """Prompt user to type in a related concept for the selected concept."""
        curr_patient = getattr(self, 'subject_id', None)
        if not concept:
            _, item = self._current_concept_item()
            if item is None:
                return False
            if item.parent() is not None and item.parent().text() == curr_patient:
                concept = item.text()
            elif item.parent() is not None and item.parent().parent() is not None and item.parent().parent().text() == curr_patient:
                concept = item.parent().text()
            else:
                return False

        parent_widget = getattr(self, 'concepts_widget', None)
        new_concept, ok = QtWidgets.QInputDialog.getText(
            parent_widget,
            f'Add Related Concept for "{concept}"',
            'Enter related concept (comma-separated for multiple):',
        )
        if not ok or not new_concept or not str(new_concept).strip():
            return False

        return self.add_related_concepts_to(concept, str(new_concept).strip())

    def add_related_concepts_to(self, concept, raw_names):
        """Add manually typed related concepts to a parent concept."""
        if not concept:
            return False

        if concept not in self.related_concepts_dict:
            self.related_concepts_dict[concept] = []

        if isinstance(raw_names, str):
            names = [n.strip() for n in raw_names.split(',') if n.strip()]
        elif isinstance(raw_names, (list, tuple)):
            names = [str(n).strip() for n in raw_names if str(n).strip()]
        else:
            return False

        if not names:
            return False

        existing_names = {
            (item.get('name') or '').strip().lower()
            for item in self.related_concepts_dict[concept]
        }
        existing_chars = {
            (item.get('char') or '').strip().lower()
            for item in self.related_concepts_dict[concept]
            if item.get('char')
        }

        added = 0
        for name in names:
            lower_name = name.lower()
            if lower_name in existing_names or lower_name in existing_chars or lower_name == concept.lower():
                continue
            item = {
                'name': name,
                'char': '',
                'description': 'Manually added association',
            }
            self.related_concepts_dict[concept].append(item)
            existing_names.add(lower_name)
            added += 1

        if added > 0:
            if hasattr(self, 'update_related_concepts'):
                self.update_related_concepts(b_llm=True)
            if hasattr(self, '_sync_workflow_tracker'):
                self._sync_workflow_tracker()
            if hasattr(self, 'create_images_button'):
                self.create_images_button.setEnabled(True)
            if hasattr(self, 'c') and hasattr(self.c, 'log'):
                self.c.log.emit(f'Added {added} related concept(s) to "{concept}".')
            return True
        return False
    
    def rename_img(self):
        """
        Rename image
        """
        index = self.related_concepts_tree_view.currentIndex()
        item = self.related_concepts_model.itemFromIndex(index)

        if item.text().lower().endswith(IMAGE_EXTENSIONS):
            img = item.text()
            file_path = item.data(IMAGE_PATH_ROLE)
            if not file_path:
                return False
            new_name, ok = QtWidgets.QInputDialog.getText(self.concepts_widget, 'Rename Image', 'Enter new name:', text=img)
            if ok and new_name and new_name != img:
                new_file_path = os.path.abspath(
                    os.path.join(os.path.dirname(file_path), new_name)
                )
                os.rename(file_path, new_file_path)
                item.setText(new_name)
                item.setData(new_file_path, IMAGE_PATH_ROLE)
                if getattr(self, 'file_path', None) == file_path:
                    self.file_path = new_file_path
                return True
        return False

    def _current_concept_item(self):
        index = self.related_concepts_tree_view.currentIndex()
        return index, self.related_concepts_model.itemFromIndex(index)

    def _confirm_removal(self, title, message):
        return QtWidgets.QMessageBox.question(
            self.concepts_widget,
            title,
            message,
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.Cancel,
            QtWidgets.QMessageBox.Cancel,
        ) == QtWidgets.QMessageBox.Yes

    def _workspace_root(self):
        return os.path.abspath(
            os.path.join(
                self.task_images_downloader.task_images_path,
                str(self.subject_id),
            )
        )

    def _remove_workspace_path(self, path):
        """Remove one resolved workspace descendant, never the workspace itself."""
        workspace = self._workspace_root()
        target = os.path.abspath(path)
        if target == workspace or os.path.commonpath([workspace, target]) != workspace:
            raise ValueError('Refusing to remove a path outside the image workspace.')
        if os.path.isfile(target):
            os.remove(target)
        elif os.path.isdir(target):
            shutil.rmtree(target)

    def delete_image(self, confirm=True):
        """Delete the selected image from disk and the tree after confirmation."""
        if self._image_run_active:
            self.c.log.emit('Wait for image creation to finish before deleting images.')
            return False
        index, item = self._current_concept_item()
        if item is None or not item.text().lower().endswith(IMAGE_EXTENSIONS):
            return False
        file_path = item.data(IMAGE_PATH_ROLE)
        if not file_path:
            return False
        if confirm and not self._confirm_removal(
            'Delete image', f'Delete “{item.text()}” permanently?'
        ):
            return False
        self._remove_workspace_path(file_path)
        self.related_concepts_model.removeRow(index.row(), index.parent())
        if getattr(self, 'file_path', None) == file_path:
            self.clear_image()
        self._sync_workflow_tracker()
        self.c.log.emit(f'Deleted image {file_path}')
        return True

    def delete_related_concept(self):
        """Delete one generated related concept and its image folder."""
        if self._image_run_active:
            self.c.log.emit('Wait for image creation to finish before deleting concepts.')
            return False
        index, item = self._current_concept_item()
        if item is None or item.parent() is None or item.parent().parent() is None:
            return False
        if item.parent().parent().text() != self.subject_id:
            return False
        concept = item.parent().text()
        display_name = item.text()
        if not self._confirm_removal(
            'Delete related concept',
            f'Delete “{display_name}” and all of its images permanently?',
        ):
            return False
        folder_name = display_name.replace(' ', '').lower()
        folder = os.path.join(self._workspace_root(), concept, folder_name)
        self._remove_workspace_path(folder)
        self.related_concepts_dict[concept] = [
            value
            for value in self.related_concepts_dict.get(concept, [])
            if (value.get('char') or value.get('name')) != display_name
        ]
        self.related_concepts_model.removeRow(index.row(), index.parent())
        self.clear_image()
        self._sync_workflow_tracker()
        self.c.log.emit(f'Deleted related concept {display_name} from {concept}')
        return True

    def clear_related_concepts(self, all_concepts=False):
        """Clear related concepts while retaining the loaded parent concepts."""
        if self._image_run_active:
            self.c.log.emit('Wait for image creation to finish before clearing concepts.')
            return False
        _, item = self._current_concept_item()
        if item is None:
            return False
        if all_concepts:
            concepts = list(self.related_concepts_dict)
            label = 'all loaded concepts'
        elif item.parent() is not None and item.parent().text() == self.subject_id:
            concepts = [item.text()]
            label = f'“{item.text()}”'
        else:
            return False
        if not self._confirm_removal(
            'Clear related concepts',
            f'Clear related concepts for {label}? Their downloaded/generated images '
            'will also be permanently deleted.',
        ):
            return False
        for concept in concepts:
            concept_folder = os.path.join(self._workspace_root(), concept)
            if os.path.isdir(concept_folder):
                for name in os.listdir(concept_folder):
                    child_path = os.path.join(concept_folder, name)
                    if os.path.isdir(child_path):
                        self._remove_workspace_path(child_path)
            self.related_concepts_dict[concept] = []
        self.update_related_concepts(b_llm=True)
        self.clear_image()
        self._sync_workflow_tracker()
        self.c.log.emit(f'Cleared related concepts for {label}')
        return True

    def clear_images(self):
        """Delete all images beneath the selected workspace/concept item."""
        if self._image_run_active:
            self.c.log.emit('Wait for image creation to finish before clearing images.')
            return False
        _, item = self._current_concept_item()
        if item is None:
            return False
        if item.text() == self.subject_id:
            directory = self._workspace_root()
            label = 'the workspace'
        elif item.parent() is not None and item.parent().text() == self.subject_id:
            directory = os.path.join(self._workspace_root(), item.text())
            label = f'“{item.text()}”'
        elif item.parent() is not None and item.parent().parent() is not None:
            directory = os.path.join(
                self._workspace_root(),
                item.parent().text(),
                item.text().replace(' ', '').lower(),
            )
            label = f'“{item.text()}”'
        else:
            return False
        image_paths = [
            os.path.join(root, filename)
            for root, _, filenames in os.walk(directory)
            for filename in filenames
            if filename.lower().endswith(IMAGE_EXTENSIONS)
        ] if os.path.isdir(directory) else []
        if not image_paths:
            self.c.log.emit(f'No images found under {label}.')
            return True
        if not self._confirm_removal(
            'Clear images',
            f'Permanently delete {len(image_paths)} images under {label}?',
        ):
            return False
        for file_path in image_paths:
            self._remove_workspace_path(file_path)
        self.update_related_concepts(b_llm=True)
        self.clear_image()
        self._sync_workflow_tracker()
        self.c.log.emit(f'Deleted {len(image_paths)} images under {label}')
        return True

    def delete_img(self):
        """Backward-compatible name for deleting the selected image."""
        return self.delete_image()

    def delete_all(self):
        """
        Delete all downloaded images for the current patient
        """
        # Delete everything inside the task images path
        curr_patient = self.subject_id
        folder_path = os.path.abspath(os.path.join(self.task_images_downloader.task_images_path, curr_patient))
        if os.path.exists(folder_path):
            # Show a warning message box
            msg = QtWidgets.QMessageBox()
            msg.setIcon(QtWidgets.QMessageBox.Warning)
            msg.setText(f'Are you sure you want to delete all images downloaded for {curr_patient}?')
            msg.setInformativeText("This action cannot be undone.")
            msg.setWindowTitle("Delete All Images")
            msg.setStandardButtons(QtWidgets.QMessageBox.Ok | QtWidgets.QMessageBox.Cancel)
            msg.setDefaultButton(QtWidgets.QMessageBox.Cancel)
            ret = msg.exec_()
            if ret == QtWidgets.QMessageBox.Cancel:
                return
            try:
                shutil.rmtree(folder_path)
                curr_patient_item = self.related_concepts_model.findItems(curr_patient, QtCore.Qt.MatchExactly | QtCore.Qt.MatchRecursive)
                if curr_patient_item:
                    # remove all the children
                    for i in range(curr_patient_item[0].rowCount()):
                        curr_patient_item[0].removeRow(i)
            except:
                print(f"[Error] Deleting {folder_path} failed.")
                print(traceback.format_exc())

    def create_task_stimuli(self):
        """
        Create the task stimuli folder if it doesn't exist
        """
        
        if not os.path.exists(self.curr_patient_path):
            os.makedirs(self.curr_patient_path)
        if self.task_name.lower().startswith('dynamic_scr'):
            task_stim_folder = os.path.abspath(os.path.join(self.curr_patient_path, 'stimuli', 'dynamic_scr'))
            if not os.path.exists(task_stim_folder):
                os.makedirs(task_stim_folder)
            if os.path.exists(task_stim_folder):
                custom_pics_path = os.path.abspath(os.path.join(task_stim_folder, 'custom_pics')) 
                if not os.path.exists(custom_pics_path):           
                    os.makedirs(custom_pics_path, exist_ok=True)
                
                    pics_now_path = os.path.abspath(os.path.join(task_stim_folder, 'pics_now'))
                    if not os.path.exists(pics_now_path):
                        os.makedirs(pics_now_path, exist_ok=True)
                    self.create_dynamic_scr_stimuli()
                else:
                    # ask if user wants to replace the existing custom pics
                    msg = QtWidgets.QMessageBox()
                    msg.setIcon(QtWidgets.QMessageBox.Warning)
                    msg.setText(f'Custom pics folder already exists for {self.subject_id}.')
                    msg.setInformativeText("Do you want to replace it? (This will delete existing custom pics)")
                    msg.setWindowTitle("Custom Pics Exists")
                    msg.setStandardButtons(QtWidgets.QMessageBox.Ok | QtWidgets.QMessageBox.Cancel)
                    msg.setDefaultButton(QtWidgets.QMessageBox.Cancel)
                    ret = msg.exec_()
                    if ret == QtWidgets.QMessageBox.Ok:
                        # User wants to replace the existing custom pics
                        shutil.rmtree(custom_pics_path)
                        os.makedirs(custom_pics_path, exist_ok=True)
                        self.create_dynamic_scr_stimuli()
        elif self.task_name.lower().startswith('categ_localiz'):
            # No processing needed, just copy the categlocaliz folder to patient folder
            categlocaliz_folder = os.path.abspath(os.path.join(self.pics_main_folder, 'categ_localiz'))
            # if it exists, copy to patient folder
            if os.path.exists(categlocaliz_folder):
                task_stim_folder = os.path.abspath(os.path.join(self.curr_patient_path, 'stimuli', 'categ_localiz'))
                if not os.path.exists(task_stim_folder):
                    shutil.copytree(categlocaliz_folder, task_stim_folder)
                    self.c.log.emit(f'Copied categlocaliz folder to {task_stim_folder}.')
                custom_pics_path = os.path.abspath(os.path.join(task_stim_folder, 'custom_pics'))
        else:
            return

        
        self.create_subscreening_1_config(custom_pics_path=custom_pics_path)
        self.update_task_stims_folder_status()  

    def check_if_dynamic_scr_processed(self):
        """
        Check if related concepts were retrieved and images downloaded already
        """
        bOK = False
        crops_path = os.path.abspath(os.path.join(self.task_images_downloader.task_images_path, self.subject_id, 'crops'))
        if os.path.exists(crops_path):
            if os.listdir(crops_path):
                bOK = True
        else:
            # show a non blocking message box with OK button only
            msg = QtWidgets.QMessageBox()
            msg.setIcon(QtWidgets.QMessageBox.Warning)
            msg.setText("No images found")
            msg.setInformativeText(f'Please press the process button, to retrieve related concepts, download images and crop them.')
            msg.setWindowTitle("Warning")
            msg.setStandardButtons(QtWidgets.QMessageBox.Ok)
            msg.setDefaultButton(QtWidgets.QMessageBox.Ok)
            msg.exec_()
        return bOK

    def create_dynamic_scr_stimuli(self):
        """
        Create the dynamic SCR stimuli folder structure
        """
        bOK = False
        bOK = self.check_if_dynamic_scr_processed()
        if not bOK:
            return
        custom_pics_count = 0
        pics_now_count = 0
        copied_paths = []
        self.update_crops_ids()
        self.c.progress.emit(0, 'Updating custom pics, pics now...', False)
        self.c.log.emit('Updating custom pics, pics now...')
        crops_path = os.path.abspath(os.path.join(self.task_images_downloader.task_images_path, self.subject_id, 'crops'))
        task_folder = os.path.abspath(os.path.join(self.curr_patient_path, 'stimuli', 'dynamic_scr'))
        custom_pics_path = os.path.abspath(os.path.join(task_folder, 'custom_pics'))            
        pics_now_path = os.path.abspath(os.path.join(task_folder, 'pics_now'))
        b1_n_pics = self.block_params["block1"]["n_pics"]
        for concept in self.related_concepts_dict:
            if concept in self.concepts_available:
                # Copy the images already available in the Pics\Faces, Pics\Non faces to the custom pics folder
                for img_path in self.img_paths:
                    img_name = os.path.basename(img_path).split('.')[0]
                    if concept in img_name and img_name.endswith('_1'):
                        try:
                            shutil.copy(img_path, os.path.abspath(os.path.join(custom_pics_path, os.path.basename(img_path))))
                            copied_paths.append(img_path)
                            custom_pics_count += 1
                            if custom_pics_count >= b1_n_pics:
                                break
                        except:
                            print(f"[Error] Could not copy {img_path} to {custom_pics_path}.")
                    elif concept in img_name:
                        try:
                            shutil.copy(img_path, os.path.abspath(os.path.join(pics_now_path, os.path.basename(img_path))))
                            pics_now_count += 1
                        except:
                            print(f"[Error] Could not copy {img_path} to {pics_now_path}.")
            progress_perc = int(custom_pics_count / b1_n_pics * 100)
            self.c.progress.emit(progress_perc, f'Updating custom pics {custom_pics_count}, pics now {pics_now_count} using images from database', False)
        self.c.log.emit(f'Copied {custom_pics_count} custom pics, {pics_now_count} pics now using images from database.')
        db_custom_pics_count = custom_pics_count
        db_pics_now_count = pics_now_count
        # Copy the cropped images to the custom pics folder
        for root, dirs, files in os.walk(crops_path):
            for img in files:
                img_name = img.split('.')[0]
                if img_name.endswith('_1'):
                    try:
                        shutil.copy(os.path.abspath(os.path.join(root, img)), os.path.abspath(os.path.join(custom_pics_path, img)))
                        copied_paths.append(os.path.abspath(os.path.join(root, img)))
                        custom_pics_count += 1
                        if custom_pics_count >= b1_n_pics:
                            break
                    except:
                        print(f"[Error] Could not copy {img} to {custom_pics_path}.")
                else:
                    try:
                        shutil.copy(os.path.abspath(os.path.join(root, img)), os.path.abspath(os.path.join(pics_now_path, img)))
                        pics_now_count += 1
                    except:
                        print(f"[Error] Could not copy {img} to {pics_now_path}.")
            progress_perc = int(custom_pics_count / b1_n_pics * 100)
            self.c.progress.emit(progress_perc, f'Updating custom pics {custom_pics_count}, pics now {pics_now_count} using downloaded images', False)
        self.c.log.emit(f'Copied {custom_pics_count - db_custom_pics_count} custom pics, {pics_now_count - db_pics_now_count} pics now using downloaded images.')
        unused_concepts = list(set(self.popular_concepts) - set(self.patient_concepts))
        used_concepts = []
        if custom_pics_count < b1_n_pics:
            self.c.log.emit(f'Not enough custom pics found, looking for more in the database. {custom_pics_count} custom pics found so far.')
            for concept in unused_concepts:
                for img_path in self.img_paths:
                    if img_path in copied_paths:
                        continue
                    img_name = os.path.basename(img_path).split('.')[0]
                    if concept in img_name and img_name.endswith('_1'):
                        try:
                            shutil.copy(img_path, os.path.abspath(os.path.join(custom_pics_path, os.path.basename(img_path))))
                            copied_paths.append(img_path)
                            custom_pics_count += 1
                            if custom_pics_count >= b1_n_pics:
                                break
                        except:
                            print(f"[Error] Could not copy {img_path} to {custom_pics_path}.")
                    elif concept in img_name:
                        try:
                            shutil.copy(img_path, os.path.abspath(os.path.join(pics_now_path, os.path.basename(img_path))))
                            pics_now_count += 1
                        except:
                            print(f"[Error] Could not copy {img_path} to {pics_now_path}.")
                progress_perc = int(custom_pics_count / b1_n_pics * 100)
                self.c.progress.emit(progress_perc, f'Updating custom pics, pics now...', False)
                if custom_pics_count >= b1_n_pics:
                    break

        if custom_pics_count < b1_n_pics:
            unused_concepts = list(set(self.concepts_available) - set(self.patient_concepts) - set(used_concepts))
            for concept in unused_concepts:
                for img_path in self.img_paths:
                    if img_path in copied_paths:
                        continue
                    img_name = os.path.basename(img_path).split('.')[0]
                    if concept in img_name and img_name.endswith('_1'):
                        try:
                            shutil.copy(img_path, os.path.abspath(os.path.join(custom_pics_path, os.path.basename(img_path))))
                            copied_paths.append(img_path)
                            custom_pics_count += 1
                            if custom_pics_count >= b1_n_pics:
                                break
                        except:
                            print(f"[Error] Could not copy {img_path} to {custom_pics_path}.")
                    elif concept in img_name:
                        try:
                            shutil.copy(img_path, os.path.abspath(os.path.join(pics_now_path, os.path.basename(img_path))))
                            pics_now_count += 1
                        except:
                            print(f"[Error] Could not copy {img_path} to {pics_now_path}.")
                progress_perc = int(custom_pics_count / b1_n_pics * 100)
                self.c.progress.emit(progress_perc, f'Updating custom pics, pics now...', False)
                if custom_pics_count >= b1_n_pics:
                    break
        copied_count = len(os.listdir(custom_pics_path))
        if copied_count < b1_n_pics:
            self.c.progress.emit(100, f'Could not find enough custom pics for {self.subject_id}. {copied_count} custom pics.', False)
            self.c.log.emit(f'Could not find enough custom pics for {self.subject_id}. {copied_count} images copied.')
        else:
            self.c.progress.emit(100, f'Updating custom pics, pics now completed. {copied_count} custom pics.', False)
            self.c.log.emit(f'Updating custom pics, pics now completed. {copied_count} images copied.')

    def create_subscreening_1_config(self, custom_pics_path):
        """
        Create subscreening_1 config file
        """
        custom_pics_list = glob.glob(os.path.join(custom_pics_path, '**', '*.*'), recursive=True)
        # custom pics are used for subscreening 1
        custom_pics_list = [os.path.basename(x) for x in custom_pics_list]
        if self.task_name.lower().startswith('dynamic_scr'):
            self.task_info = DynamicScrInfo(block_params=self.block_params, 
                                            exp_params=self.exp_params, custom_pics_path=custom_pics_path)
        elif self.task_name.lower().startswith('categ_localiz'):
            self.task_info = CategLocalizInfo(block_params=self.block_params, 
                                               exp_params=self.exp_params, custom_pics_path=custom_pics_path)
            
        # Get sub table with rows of custom pics only
        all_stims_list = self.task_info.img_info_table['name'].tolist()
        self.c.image_names.emit(all_stims_list)
        stim_list = self.task_info.img_info_table[self.task_info.img_info_table['name'].str.lower().isin([x.lower() for x in custom_pics_list])]
        subscreening_1 = SubscreeningConfig(id=1, task_name=self.task_name, stim_list=stim_list, 
                                            n_rep=self.block_params["block1"]["n_rep"], 
                                            isi=self.exp_params["isi"], 
                                            max_trials=self.exp_params["max_trials"],
                                            sort_n_rep=self.block_params["block1"]["sort_n_rep"],
                                            b_manual_select=self.block_params["block1"]["manual_select"],)
        self.scr_config['subscreening_1'] = subscreening_1

        self.c.scr_config_updated.emit(1, self.scr_config, True)

    def update_stim_trial_count(self, subscr_idx, seq_idx):
        """
        Update the stim trial count in the task_info table
        """
        if hasattr(self, 'task_info'):
            prev_subscreening = self.scr_config.get(f'subscreening_{subscr_idx}', None)
            prev_subscreening.last_valid_seq = seq_idx
            self.task_info.update_stim_trial_count(prev_subscreening=prev_subscreening, seq_idx=seq_idx)
            self.task_info.save_task_info(save_folder=self.task_folder)

    def update_next_subscreening_stimuli(self, scr_id, scr_ranked_stims, keep_stims, discarded_stimuli):
        """
        Called by responses when a raster is marked as keep or discard
        """
        
        if scr_id == 1:
            # save task info to file
            self.task_info.save_task_info(save_folder=self.task_folder)
        else:
            b_last_subscreening = scr_id == len(self.block_params)
            prev_subscreening_config = self.scr_config.get(f'subscreening_{scr_id-1}', None)
            # get stimuli by executing rules on selected stims, move protected stims to the top of the list followed by responsive stims
            nxt_scr_stim_list, extra_stims, removed_stims = self.task_info.get_nxt_scr_stim_list(scr_id=scr_id, scr_selected_stims=keep_stims, 
                                                                     discarded_stimuli=discarded_stimuli, scr_responsive_stims=scr_ranked_stims, 
                                                                     prev_subscreening=prev_subscreening_config)
            # get nxt_scr_stim_list from task_info_table
            stim_list = self.task_info.img_info_table[self.task_info.img_info_table['name'].isin(nxt_scr_stim_list)]
            # reorder rows in stim_list to match the order in nxt_scr_stim_list
            stim_list = stim_list.set_index('name').loc[nxt_scr_stim_list].reset_index()
            # keep only up to NPICS[scr_id-1] stims
            stim_list = stim_list.head(self.block_params[f"block{scr_id}"]["n_pics"])

            subscreening = SubscreeningConfig(id=scr_id, task_name=self.task_name, stim_list=stim_list, 
                                            n_rep=self.block_params[f"block{scr_id}"]["n_rep"], 
                                            isi=self.exp_params["isi"], 
                                            max_trials=self.exp_params["max_trials"],
                                            sort_n_rep=self.block_params[f"block{scr_id}"]["sort_n_rep"],
                                            b_manual_select=self.block_params[f"block{scr_id}"]["manual_select"],
                                            b_last=b_last_subscreening)
            self.scr_config[f'subscreening_{scr_id}'] = subscreening
            # self.c.log.emit(self.task_info.subscr_sel_info_txt['subscreening_'+str(scr_id-1)])
        # self.c.scr_config_updated.emit(scr_id, self.scr_config, False)
        return nxt_scr_stim_list, extra_stims, removed_stims

    def create_subscreening_config(self, scr_id, 
                                   scr_selected_stims, 
                                   discarded_stimuli,
                                   responsive_stims,
                                   prev_subscreening_config, update_table=True):
        """
        Create subscreening config file
        """
        
        if scr_id == 1:
            if update_table:
                # save task info to file
                self.task_info.save_task_info(save_folder=self.task_folder)
            nxt_scr_stim_list, extra_stims, removed_stims = [], [], []
        else:
            b_last_subscreening = scr_id == len(self.block_params)
            # get stimuli by executing rules on selected stims, move protected stims to the top of the list followed by responsive stims
            nxt_scr_stim_list, extra_stims, removed_stims = self.task_info.get_nxt_scr_stim_list(scr_id=scr_id, scr_selected_stims=scr_selected_stims, 
                                                                     discarded_stimuli=discarded_stimuli, scr_responsive_stims=responsive_stims, 
                                                                     prev_subscreening=prev_subscreening_config, update_table=update_table)
            
            # get nxt_scr_stim_list from task_info_table
            stim_list = self.task_info.img_info_table[self.task_info.img_info_table['name'].isin(nxt_scr_stim_list)]
            # reorder rows in stim_list to match the order in nxt_scr_stim_list
            stim_list = stim_list.set_index('name').loc[nxt_scr_stim_list].reset_index()
            # keep only up to NPICS[scr_id-1] stims
            stim_list = stim_list.head(self.block_params[f"block{scr_id}"]["n_pics"])

            subscreening = SubscreeningConfig(id=scr_id, task_name=self.task_name, stim_list=stim_list, 
                                            n_rep=self.block_params[f"block{scr_id}"]["n_rep"], 
                                            isi=self.exp_params["isi"], 
                                            max_trials=self.exp_params["max_trials"],
                                            sort_n_rep=self.block_params[f"block{scr_id}"]["sort_n_rep"],
                                            b_manual_select=self.block_params[f"block{scr_id}"]["manual_select"],
                                            b_last=b_last_subscreening)
            self.scr_config[f'subscreening_{scr_id}'] = subscreening
            self.c.log.emit(self.task_info.subscr_sel_info_txt['subscreening_'+str(scr_id-1)])
        if update_table:
            self.c.scr_config_updated.emit(scr_id, self.scr_config, True)

        return nxt_scr_stim_list, extra_stims, removed_stims
    
    def load_img_info_table(self, folder_path):
        """
        Load the img_info_table from file
        """
        try:
            if not hasattr(self, 'task_info'):
                self.task_info = TaskInfoBase.load_task_info(load_folder=folder_path)
        except:
            print(f"[Error] Loading task info from {folder_path} failed.")
            print(traceback.format_exc())

    def save_task_info(self):
        """
        Save task info to file
        """
        if hasattr(self, 'task_info'):
            self.task_info.save_task_info(save_folder=self.task_folder)

    def update_crops_ids(self):
        """
        Rename images to have _1, _2, _3, .. 
        for example: img_1.jpg, img_3.jpg, img_5.jpg will be renamed to img_1.jpg, img_2.jpg, img_3.jpg
        """
        curr_patient = self.subject_id
        crops_path = os.path.abspath(os.path.join(self.task_images_downloader.task_images_path, curr_patient, 'crops'))
        for root, dirs, files in os.walk(crops_path):
            # Only process image files (jpg, jpeg, png, bmp, gif)
            img_files = [f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif'))]
            img_files.sort()  # Optional: sort for consistent renaming
            # check if there is a need to rename
            for idx, img in enumerate(img_files):
                if not img.endswith(f'_{idx+1}{os.path.splitext(img)[1]}'):
                    img_base, img_ext = os.path.splitext(img)
                    img_name = img_base.split('_')[0]
                    new_img_name = f'{img_name}_{idx+1}{img_ext}'
                    src = os.path.abspath(os.path.join(root, img))
                    dst = os.path.abspath(os.path.join(root, new_img_name))
                    if os.path.exists(dst):
                        os.remove(dst)
                    os.rename(src, dst)

    def load_concepts_file(self, file_path):
        """Load a concept list and refresh the working tree."""
        from stimulus_forge.services.questionnaire_intelligence import parse_questionnaire
        q_res = parse_questionnaire(file_path, subject_id=self.subject_id)
        if q_res.is_questionnaire and q_res.selected_concepts:
            concepts = q_res.selected_concepts
            self.concept_excerpts = getattr(q_res, "selected_excerpts", {})
            self.subject_id = q_res.selected_subject or workspace_name_for(file_path)
        else:
            concepts = load_concepts(file_path, subject_id=self.subject_id)
            self.concept_excerpts = {}
            self.subject_id = workspace_name_for(file_path, self.subject_id)
        self.curr_patient_path = os.path.abspath(
            os.path.join(self.exp_folder, self.subject_id)
        )
        self.related_concepts_dict = {concept: [] for concept in concepts}
        self.patient_concepts = concepts
        if hasattr(self, 'concepts_file_path_edit'):
            self.concepts_file_path_edit.setText(os.path.abspath(file_path))
            self.concepts_file_path_edit.setToolTip(os.path.abspath(file_path))
        self.enable_workspace_actions(True)

        if self.task_name.lower().startswith('dynamic_scr'):
            self.update_related_concepts()
        elif self.task_name.lower().startswith('categ_localiz'):
            self.update_categ_localiz_concepts()

        self.clear_image()
        self.status_lbl.setText(
            f"Loaded {len(concepts)} concepts from {os.path.basename(file_path)}"
        )
        self._sync_workflow_tracker(active=2)
        return concepts

    def add_concept(self, concept):
        """Add a normalized concept to the current in-memory workspace."""
        normalized = ' '.join(str(concept).split()).casefold()
        if not normalized:
            raise ValueError('Concept cannot be empty.')
        if not self.subject_id:
            self.subject_id = 'chat_concepts'
            self.curr_patient_path = os.path.abspath(
                os.path.join(self.exp_folder, self.subject_id)
            )
            if hasattr(self, 'concepts_file_path_edit'):
                self.concepts_file_path_edit.setText('Chat workspace (not saved)')
                self.concepts_file_path_edit.setToolTip(
                    'Concepts added in chat are kept in the current session.'
                )
        added = normalized not in self.related_concepts_dict
        if added:
            self.patient_concepts.append(normalized)
            self.related_concepts_dict[normalized] = []
        self.enable_workspace_actions(True)
        self.update_related_concepts(
            b_llm=any(self.related_concepts_dict.values())
        )
        self._sync_workflow_tracker(active=2)
        self.select_concept(normalized)
        return normalized, added

    def select_concept(self, concept):
        """Select and reveal a parent concept in the tree."""
        if not hasattr(self, 'related_concepts_model'):
            return False
        matches = self.related_concepts_model.findItems(
            concept, Qt.MatchExactly | Qt.MatchRecursive
        )
        if not matches:
            return False
        item = matches[0]
        self.related_concepts_tree_view.setCurrentIndex(item.index())
        self.related_concepts_tree_view.scrollTo(item.index())
        self.related_concepts_tree_view.expand(item.index())
        return True

    def _resolve_concept_key(self, concept):
        """Return the canonical in-memory key for a case-insensitive concept name."""
        wanted = ' '.join(str(concept).split()).casefold()
        return next(
            (key for key in self.related_concepts_dict if key.casefold() == wanted),
            None,
        )

    def _resolve_related_value(self, concept, related):
        concept_key = self._resolve_concept_key(concept)
        if concept_key is None:
            return None, None
        wanted = ' '.join(str(related).split()).casefold()
        value = next(
            (
                item
                for item in self.related_concepts_dict.get(concept_key, [])
                if (item.get('char') or item.get('name', '')).casefold() == wanted
            ),
            None,
        )
        return concept_key, value

    def delete_parent_concept(self, concept, confirm=True):
        """Delete one parent concept, its related metadata, and workspace images."""
        if self._concept_run_active or self._image_run_active:
            self.c.log.emit('Wait for the current workflow job before deleting concepts.')
            return False
        concept_key = self._resolve_concept_key(concept)
        if concept_key is None:
            return False
        if confirm and not self._confirm_removal(
            'Delete concept',
            f'Delete “{concept_key}”, its related concepts, and all of its images?',
        ):
            return False
        folder = os.path.join(self._workspace_root(), concept_key)
        if os.path.exists(folder):
            self._remove_workspace_path(folder)
        self.related_concepts_dict.pop(concept_key, None)
        self.patient_concepts = [
            value for value in self.patient_concepts if value != concept_key
        ]
        self.update_related_concepts(b_llm=True)
        self.clear_image()
        self.enable_workspace_actions(bool(self.patient_concepts))
        self._sync_workflow_tracker()
        self.c.log.emit(f'Deleted concept {concept_key}')
        return True

    def delete_related_concept_named(self, concept, related, confirm=True):
        """Delete an explicitly named related concept without relying on tree selection."""
        if self._concept_run_active or self._image_run_active:
            self.c.log.emit('Wait for the current workflow job before deleting concepts.')
            return False
        concept_key, value = self._resolve_related_value(concept, related)
        if value is None:
            return False
        display_name = value.get('char') or value.get('name')
        if confirm and not self._confirm_removal(
            'Delete related concept',
            f'Delete “{display_name}” from “{concept_key}” and delete its images?',
        ):
            return False
        folder = os.path.join(
            self._workspace_root(),
            concept_key,
            display_name.replace(' ', '').lower(),
        )
        if os.path.exists(folder):
            self._remove_workspace_path(folder)
        self.related_concepts_dict[concept_key] = [
            item
            for item in self.related_concepts_dict[concept_key]
            if item is not value
        ]
        self.update_related_concepts(b_llm=True)
        self.clear_image()
        self._sync_workflow_tracker()
        self.c.log.emit(f'Deleted related concept {display_name} from {concept_key}')
        return True

    def clear_related_concepts_for(self, concepts=None, confirm=True):
        """Clear related concepts for explicit parents, or all parents when omitted."""
        if self._concept_run_active or self._image_run_active:
            self.c.log.emit('Wait for the current workflow job before clearing concepts.')
            return False
        if concepts is None:
            resolved = list(self.related_concepts_dict)
        else:
            resolved = [self._resolve_concept_key(value) for value in concepts]
            resolved = [value for value in resolved if value is not None]
        if not resolved:
            return False
        label = 'all loaded concepts' if concepts is None else ', '.join(resolved)
        if confirm and not self._confirm_removal(
            'Clear related concepts',
            f'Clear related concepts for {label}? Their images will also be deleted.',
        ):
            return False
        for concept_key in resolved:
            concept_folder = os.path.join(self._workspace_root(), concept_key)
            if os.path.isdir(concept_folder):
                for name in os.listdir(concept_folder):
                    child_path = os.path.join(concept_folder, name)
                    if os.path.isdir(child_path):
                        self._remove_workspace_path(child_path)
            self.related_concepts_dict[concept_key] = []
        self.update_related_concepts(b_llm=True)
        self.clear_image()
        self._sync_workflow_tracker()
        self.c.log.emit(f'Cleared related concepts for {label}')
        return True

    def clear_images_for(self, concept=None, related=None, confirm=True):
        """Delete images under an explicit workspace, concept, or association."""
        if self._image_run_active:
            self.c.log.emit('Wait for image creation to finish before clearing images.')
            return False
        if concept is None:
            directory = self._workspace_root()
            label = 'the workspace'
        else:
            concept_key = self._resolve_concept_key(concept)
            if concept_key is None:
                return False
            if related:
                _, value = self._resolve_related_value(concept_key, related)
                if value is None:
                    return False
                display_name = value.get('char') or value.get('name')
                directory = os.path.join(
                    self._workspace_root(),
                    concept_key,
                    display_name.replace(' ', '').lower(),
                )
                label = f'“{display_name}” under “{concept_key}”'
            else:
                directory = os.path.join(self._workspace_root(), concept_key)
                label = f'“{concept_key}”'
        image_paths = [
            os.path.join(root, filename)
            for root, _, filenames in os.walk(directory)
            for filename in filenames
            if filename.lower().endswith(IMAGE_EXTENSIONS)
        ] if os.path.isdir(directory) else []
        if not image_paths:
            self.c.log.emit(f'No images found under {label}.')
            return True
        if confirm and not self._confirm_removal(
            'Clear images',
            f'Permanently delete {len(image_paths)} images under {label}?',
        ):
            return False
        for file_path in image_paths:
            self._remove_workspace_path(file_path)
        self.update_related_concepts(b_llm=True)
        self.clear_image()
        self._sync_workflow_tracker()
        self.c.log.emit(f'Deleted {len(image_paths)} images under {label}')
        return True

    def rename_parent_concept(self, concept, new_name):
        """Rename a parent concept in metadata, the tree, and its workspace folder."""
        if self._concept_run_active or self._image_run_active:
            return False
        concept_key = self._resolve_concept_key(concept)
        normalized = ' '.join(str(new_name).split()).casefold()
        if concept_key is None or not normalized:
            return False
        existing = self._resolve_concept_key(normalized)
        if existing is not None and existing != concept_key:
            return False
        old_folder = os.path.join(self._workspace_root(), concept_key)
        new_folder = os.path.join(self._workspace_root(), normalized)
        if os.path.exists(old_folder) and old_folder != new_folder:
            if os.path.exists(new_folder):
                return False
            os.rename(old_folder, new_folder)
        values = self.related_concepts_dict.pop(concept_key)
        insert_index = self.patient_concepts.index(concept_key)
        self.patient_concepts[insert_index] = normalized
        # Rebuild in patient order so renaming does not move the concept visually.
        rebuilt = {
            key: (values if key == normalized else self.related_concepts_dict[key])
            for key in self.patient_concepts
        }
        self.related_concepts_dict = rebuilt
        self.update_related_concepts(b_llm=True)
        self.select_concept(normalized)
        self.c.log.emit(f'Renamed concept {concept_key} to {normalized}')
        return True

    def rename_related_concept_named(self, concept, related, new_name):
        """Rename one related concept and its corresponding image folder."""
        if self._concept_run_active or self._image_run_active:
            return False
        concept_key, value = self._resolve_related_value(concept, related)
        new_display = ' '.join(str(new_name).split()).strip()
        if value is None or not new_display:
            return False
        old_display = value.get('char') or value.get('name')
        duplicate = any(
            (item.get('char') or item.get('name', '')).casefold()
            == new_display.casefold()
            for item in self.related_concepts_dict[concept_key]
            if item is not value
        )
        if duplicate:
            return False
        old_folder = os.path.join(
            self._workspace_root(), concept_key, old_display.replace(' ', '').lower()
        )
        new_folder = os.path.join(
            self._workspace_root(), concept_key, new_display.replace(' ', '').lower()
        )
        if os.path.exists(old_folder) and old_folder != new_folder:
            if os.path.exists(new_folder):
                return False
            os.rename(old_folder, new_folder)
        if value.get('char'):
            value['char'] = new_display
        else:
            value['name'] = new_display
        self.update_related_concepts(b_llm=True)
        self.c.log.emit(
            f'Renamed related concept {old_display} to {new_display} under {concept_key}'
        )
        return True

    def set_concept_checked(self, concept, checked):
        """Check or uncheck a parent concept and its descendants."""
        concept_key = self._resolve_concept_key(concept)
        if concept_key is None or not self.select_concept(concept_key):
            return False
        _, item = self._current_concept_item()
        item.setCheckState(Qt.Checked if checked else Qt.Unchecked)
        return True

    def set_all_concepts_checked(self, checked):
        """Check or uncheck the complete concept workspace."""
        if not hasattr(self, 'related_concepts_model'):
            return False
        root = self.related_concepts_model.item(0, 0)
        if root is None:
            return False
        root.setCheckState(Qt.Checked if checked else Qt.Unchecked)
        return True

    def select_related_concept(self, concept, related):
        """Select one related concept in the tree by explicit parent and name."""
        concept_key, value = self._resolve_related_value(concept, related)
        if value is None:
            return False
        display = value.get('char') or value.get('name')
        if not self.select_concept(concept_key):
            return False
        _, parent_item = self._current_concept_item()
        for row in range(parent_item.rowCount()):
            item = parent_item.child(row)
            if item.text().casefold() == display.casefold():
                self.related_concepts_tree_view.setCurrentIndex(item.index())
                self.related_concepts_tree_view.scrollTo(item.index())
                self.related_concepts_tree_view.expand(item.index())
                return True
        return False

    def set_related_concept_checked(self, concept, related, checked):
        """Check or uncheck one related concept and its image descendants."""
        if not self.select_related_concept(concept, related):
            return False
        _, item = self._current_concept_item()
        item.setCheckState(Qt.Checked if checked else Qt.Unchecked)
        return True

    def acquire_related_concept_images(self, concept, related, source=None):
        """Create images for exactly one explicitly named related concept."""
        if self._concept_run_active or self._image_run_active:
            return False
        concept_key, value = self._resolve_related_value(concept, related)
        if value is None:
            return False
        source = source or self.image_source
        source = 'cloud' if source == 'generate' else source
        if source == 'cloud':
            self._sync_openai_image_service_key()
            if not self.openai_image_service.is_available:
                message = self._openai_image_configuration_error() or (
                    'OpenAI image generation is not configured.'
                )
                self.c.log.emit(message)
                QtWidgets.QMessageBox.critical(
                    self.concepts_widget, 'OpenAI is not configured', message
                )
                return False
        if source == 'google':
            self._sync_google_image_service_key()
            if not self.google_image_service.is_available:
                message = self._google_image_configuration_error() or (
                    'Google Imagen generation is not configured.'
                )
                self.c.log.emit(message)
                QtWidgets.QMessageBox.critical(
                    self.concepts_widget, 'Google Imagen is not configured', message
                )
                return False
        if source == 'huggingface':
            if not self.huggingface_image_service.is_available:
                message = 'Set HF_TOKEN or enter your Hugging Face API token before generating images.'
                self.c.log.emit(message)
                QtWidgets.QMessageBox.critical(
                    self.concepts_widget, 'Hugging Face is not configured', message
                )
                return False
        if source == 'local' and not self.local_image_service.is_available:
            message = self._local_image_configuration_error()
            self.c.log.emit(message)
            QtWidgets.QMessageBox.critical(
                self.concepts_widget, 'Local image model is not configured', message
            )
            return False
        output_root = os.path.join(
            self.task_images_downloader.task_images_path, self.subject_id
        )
        tasks = self.prepare_concept_download_tasks(
            concept_key,
            [value],
            os.path.abspath(output_root),
            self.images_per_concept_spin.value(),
        )
        # prepare_concept_download_tasks includes the parent; retain only the association.
        tasks = [task for task in tasks if task.get('related_concept') or task.get('name')]
        if not tasks:
            return False
        if not self._confirm_image_acquisition(
            source, sum(int(task['limit']) for task in tasks), False
        ):
            return False
        action = 'Downloading' if source == 'bing' else 'Generating'
        return self._start_image_tasks(tasks, source, action)

    def rename_selected_image(self, new_name):
        """Rename the currently selected image without opening an input dialog."""
        _, item = self._current_concept_item()
        if item is None or not item.text().lower().endswith(IMAGE_EXTENSIONS):
            return False
        file_path = item.data(IMAGE_PATH_ROLE)
        new_name = os.path.basename(str(new_name).strip())
        if not file_path or not new_name or not new_name.lower().endswith(IMAGE_EXTENSIONS):
            return False
        new_path = os.path.join(os.path.dirname(file_path), new_name)
        if os.path.exists(new_path):
            return False
        os.rename(file_path, new_path)
        item.setText(new_name)
        item.setData(new_path, IMAGE_PATH_ROLE)
        if getattr(self, 'file_path', None) == file_path:
            self.file_path = new_path
        return True

    def concepts_tree_view_context_menu(self, position):
        """
        Handle right click event on the concepts tree view
        """
        curr_patient = self.subject_id
        # Get the selected item
        index = self.related_concepts_tree_view.indexAt(position)
        if not index.isValid():
            return
        self.related_concepts_tree_view.setCurrentIndex(index)
        item = self.related_concepts_model.itemFromIndex(index)
        if item is None:
            return
        menu = QtWidgets.QMenu()
        source_labels = {
            'bing': 'Bing',
            'cloud': 'OpenAI cloud',
            'google': 'Google Imagen',
            'huggingface': 'Hugging Face',
            'local': 'local image model',
            'text': 'Text stimulus (centered)',
        }
        source_label = source_labels.get(self.image_source, self.image_source)
        if item.text().lower().endswith(IMAGE_EXTENSIONS):
            menu.addAction("Rename", self.rename_img)
            menu.addAction("Delete image…", self.delete_image)
        elif item.text() == curr_patient:
            menu.addAction(
                f"Create all images using {source_label}",
                lambda: self.acquire_current_item_images(self.image_source),
            )
            menu.addSeparator()
            menu.addAction(
                "Clear all related concepts…",
                lambda: self.clear_related_concepts(all_concepts=True),
            )
            menu.addAction("Clear all images…", self.clear_images)
        elif item.parent() is not None and item.parent().text() == curr_patient:
            menu.addAction("Get related concepts", lambda: self.get_related_concepts_context(item.text()))
            menu.addAction("Add related concept…", lambda: self.add_related_concept_dialog(item.text()))
            menu.addAction(
                f"Create images using {source_label}",
                lambda: self.acquire_current_item_images(self.image_source),
            )
            menu.addAction("Clear related concepts…", self.clear_related_concepts)
            menu.addAction("Clear images…", self.clear_images)
            menu.addAction("Rename", self.rename_concept)
            menu.addAction(
                "Delete concept…",
                lambda: self.delete_parent_concept(item.text()),
            )
        else:
            parent_concept = item.parent().text() if item.parent() else None
            menu.addAction("Add related concept…", lambda: self.add_related_concept_dialog(parent_concept))
            menu.addAction(
                f"Create images using {source_label}",
                lambda: self.acquire_current_item_images(self.image_source),
            )
            menu.addAction("Clear images…", self.clear_images)
            menu.addAction("Rename", self.rename_concept)
            menu.addAction("Delete related concept…", self.delete_related_concept)
        menu.exec_(self.related_concepts_tree_view.viewport().mapToGlobal(position))

    def concepts_tree_view_clicked(self):
        """
        Handle click event on the concepts tree view
        """
        # Get the selected item
        index = self.related_concepts_tree_view.currentIndex()
        item = self.related_concepts_model.itemFromIndex(index)

        if item is None:
            return
        
        # Get the selected item text
        concept = item.text()

        curr_patient = self.subject_id
        if item.text().lower().endswith(IMAGE_EXTENSIONS):
            file_path = item.data(IMAGE_PATH_ROLE)
            if not file_path:
                related_concept = item.parent().text().replace(' ', '').lower()
                concept = item.parent().parent().text()
                img = item.text()
                if self.task_name.lower().startswith('categ_localiz'):
                    categ = img.split('-')[0]
                    file_path = os.path.abspath(os.path.join(self.curr_patient_path, 'stimuli', self.task_name, categ, img))
                else:
                    file_path = os.path.abspath(os.path.join(self.task_images_downloader.task_images_path, curr_patient, concept, related_concept, img))
            if os.path.exists(file_path):
                self.show_image(file_path=file_path, item=item)
            else:
                print(f"[Error] {file_path} does not exist.")

    def clear_image(self):
        """
        Clear the image from the concepts view
        """
        # Clear the layout
        self.glw_concepts.clear()
        if hasattr(self, 'img_info'):
            self.img_info.setText('')
    
    def show_image(self, file_path, item, detect_faces=True):
        """
        Show the image in the concepts view

        Args:
        	file_path: Path to the image
        """
        self.clear_image()

        self.file_path = file_path

        # Add the image to the layout
        # imageio returns arrays as rows × columns (height × width).  Pyqtgraph
        # defaults to column-major, which transposes/rotates rectangular images.
        img = pg.ImageItem(axisOrder='row-major')
        imgData = imageio.imread(file_path)
        img.setImage(imgData)

        lyt_img = self.glw_concepts.addLayout(row=0, col=0)

        vb = lyt_img.addViewBox(row=0, col=0, lockAspect=True, invertY=True)
        vb.addItem(img)
        vb.autoRange()

        if not hasattr(self, 'img_info'):
            self.img_info = QtWidgets.QLabel()
            self.lyt_concepts.addWidget(self.img_info)
        
        self.img_info.setText(f'{imgData.shape[1]} x {imgData.shape[0]}')

        if self.task_name.lower().startswith('categ_localiz'):
            return

        if not detect_faces:
            return

        # if imgData.shape[0] != imgData.shape[1] or imgData.shape[0] > 1000:
        if item.data() is not None:
            faces_dict = item.data()
            if isinstance(faces_dict, dict):
                face_images = []
                for face_idx, face in faces_dict.items():
                    x1, y1, x2, y2 = face['x1'], face['y1'], face['x2'], face['y2']
                    prob = face['prob']
                    text = pg.TextItem(text=f'{prob:.2f} WxH:{x2-x1:.0f}x{y2-y1:.0f}', color='b', anchor=(0, 0), fill=(255, 255, 255, 100))
                    vb.addItem(text)
                    text.setPos(x1, y1+10)

                    crop_win = pg.RectROI([x1, y1], [x2-x1, y2-y1], scaleSnap=True, removable=True,
                                          maxBounds=pg.QtCore.QRectF(0, 0, imgData.shape[1], imgData.shape[0]),   
                                          pen=pg.mkPen('g', width=1, style=QtCore.Qt.DashDotDotLine))
                    vb.addItem(crop_win)
                    crop_win.addScaleHandle([1, 1], [0, 0], lockAspect=True)
                    crop_win.sigRegionChanged.connect(lambda: self.update_crop(crop_win, text, prob, face_idx, faces_dict, file_path))
                    crop_win.sigRemoveRequested.connect(lambda: self.remove_roi(crop_win, text, faces_dict, face_idx))

                    # crop_data = imgData[y1:y2, x1:x2]
                    # crop_data = cv2.resize(crop_data, (224, 224))
                    # crop_data_uri = self.image_to_base64_data_uri(img=crop_data)
                    # response = self.llama_model.create_chat_completion(
                    #     messages = [
                    #         {
                    #             "role": "user",
                    #             "content": [
                    #                 {"type" : "text", "text": "Who or what is this in one or two words?"},
                    #                 {"type": "image_url", "image_url": {"url": crop_data_uri} }

                    #             ]
                    #         }
                    #     ]   
                    # )
                    # print(response["choices"][0]["message"]['content'])
                    # response_txt = pg.TextItem(text=response["choices"][0]["message"]['content'], color='b', anchor=(0, 0), fill=(255, 255, 255, 150))
                    # response_txt.setPos(x1, y2+10)
                    # vb.addItem(response_txt)

        else:
            self.detect_face_data(item, imgData, file_path, b_online=True, vb=vb)

        # mouse move event
        # vb.scene().sigMouseMoved.connect(self.mouse_moved)

        # mouse click event
        # vb.scene().sigMouseClicked.connect(self.mouse_clicked)

    def detect_face_data_facenet_mtcnn(self, item, img_data, file_path, b_online=False, vb=None):
        """
        Detect faces in the image and set the face data for each check item
        """
        if self.mtcnn is None or not hasattr(self.mtcnn, 'detect'):
            return self.detect_face_data(item, img_data, file_path, b_online=b_online, vb=vb)
        if len(img_data.shape) > 2 and img_data.shape[2] == 4:
            img_data = cv2.cvtColor(img_data, cv2.COLOR_BGRA2BGR)
        elif len(img_data.shape) == 2:
            img_data = cv2.cvtColor(img_data, cv2.COLOR_GRAY2BGR)
        boxes, probs = self.mtcnn.detect(img_data)
        faces_dict = {}
        face_idx = 0
        short_dim = max(1, min(img_data.shape[0], img_data.shape[1]))
        min_raw_edge = max(80, int(short_dim * 0.08))
        min_raw_area = max(80 * 80, int(((short_dim * short_dim) * (1.0 / 25.0)) / (1.6 ** 2)))
        if boxes is not None:
            valid_pairs = []
            for box, prob in zip(boxes, probs):
                if prob < 0.90:
                    continue
                x1, y1, x2, y2 = box.astype(int)
                raw_w, raw_h = max(0, x2 - x1), max(0, y2 - y1)
                if raw_w < min_raw_edge or raw_h < min_raw_edge:
                    continue
                if (raw_w * raw_h) < min_raw_area:
                    continue
                valid_pairs.append((box, float(prob), raw_w * raw_h))
            if valid_pairs:
                max_raw_area = max(p[2] for p in valid_pairs)
                valid_pairs = [p for p in valid_pairs if p[2] >= max_raw_area * 0.35]
                valid_pairs.sort(key=lambda p: p[2] * p[1], reverse=True)
            for box, prob, _ in valid_pairs:
                if face_idx >= 10:
                    break
                b_square = False
                x1, y1, x2, y2 = box.astype(int)
                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)
                # expand the box by 10% and make it square
                box_size = max(x2 - x1, y2 - y1)
                if box_size + 0.6 * box_size < img_data.shape[0] and \
                box_size + 0.6 * box_size < img_data.shape[1]:
                    box_size = int(box_size + 0.6 * box_size)
                hw = int(box_size / 2)
                x1 = center_x - hw
                x2 = center_x + hw
                y1 = center_y - hw
                y2 = center_y + hw

                x1 = max(0, x1)
                x2 = min(img_data.shape[1], x2)
                if x1 == 0 and box_size < img_data.shape[1]:
                    x2 = box_size
                elif x1 > 0 and x2 == img_data.shape[1] and x2 - box_size > 0:
                    x1 = x2 - box_size
                y1 = max(0, y1)
                y2 = min(img_data.shape[0], y2)
                if y1 == 0 and box_size < img_data.shape[0]:
                    y2 = box_size
                elif y1 > 0 and y2 == img_data.shape[0] and y2 - box_size > 0:
                    y1 = y2 - box_size
                    
                if x2 - x1 < self.img_size or y2 - y1 < self.img_size:
                    # colr = 'y'
                    continue
                if (x2 -x1) - (y2 - y1) == 0:
                    colr = 'g'
                    b_square = True
                    faces_dict[face_idx] = {'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2, 'prob': prob, 'valid': True}
                else:
                    # colr = 'r'
                    # print(f'Invalid crop window for {os.path.basename(file_path)}: {x2 - x1}, {y2 - y1}')
                    continue
                
                if b_online:
                    self.add_roi(vb, img_data, x1, y1, x2, y2, prob, face_idx, faces_dict, file_path)

                face_idx += 1

            item.setData(faces_dict)
        if len(faces_dict) == 0:
            faces_dict[face_idx] = {'x1': 0, 'y1': 0, 'x2': self.crop_win_size, 'y2': self.crop_win_size, 'prob': 0.0, 'valid': False}
            if b_online:
                self.add_roi(vb, img_data, 0, 0, self.crop_win_size, self.crop_win_size, 0.0, 0, faces_dict, file_path)
            item.setData(faces_dict)

    def detect_face_data(self, item, img_data, file_path, b_online=False, vb=None):
        """
        Detect faces in the image and set the face data for each check item
        """
        if self.mtcnn is None:
            crop_size = min(self.crop_win_size, img_data.shape[0], img_data.shape[1])
            faces_dict = {
                0: {
                    'x1': 0,
                    'y1': 0,
                    'x2': crop_size,
                    'y2': crop_size,
                    'prob': 0.0,
                    'valid': False,
                }
            }
            item.setData(faces_dict)
            if b_online and vb is not None:
                self.add_roi(
                    vb,
                    img_data,
                    0,
                    0,
                    crop_size,
                    crop_size,
                    0.0,
                    0,
                    faces_dict,
                    file_path,
                )
            return
        if len(img_data.shape) > 2 and img_data.shape[2] == 4:
            img_data = cv2.cvtColor(img_data, cv2.COLOR_BGRA2BGR)
        elif len(img_data.shape) == 2:
            img_data = cv2.cvtColor(img_data, cv2.COLOR_GRAY2BGR)
        faces = self.mtcnn.detect_faces(img_data)
        faces_dict = {}
        face_idx = 0
        short_dim = max(1, min(img_data.shape[0], img_data.shape[1]))
        min_raw_edge = max(80, int(short_dim * 0.08))
        min_raw_area = max(80 * 80, int(((short_dim * short_dim) * (1.0 / 25.0)) / (1.6 ** 2)))
        if faces is not None:
            valid_faces = []
            for face in faces:
                prob = float(face.get('confidence', 0.0))
                box = face.get('box', [0, 0, 0, 0])
                if prob < 0.90 or len(box) < 4:
                    continue
                _, _, w, h = box
                if w < min_raw_edge or h < min_raw_edge:
                    continue
                if (w * h) < min_raw_area:
                    continue
                valid_faces.append((face, w * h * prob, w * h))
            if valid_faces:
                max_raw_area = max(f[2] for f in valid_faces)
                valid_faces = [f for f in valid_faces if f[2] >= max_raw_area * 0.35]
                valid_faces.sort(key=lambda f: f[1], reverse=True)
            for face, _, _ in valid_faces:
                if face_idx >= 10:
                    break
                b_square = False
                prob = face['confidence']
                box = face['box']
                x1, y1, w, h = box
                x2 = x1 + w
                y2 = y1 + h
                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)
                # expand the box by 10% and make it square
                box_size = max(x2 - x1, y2 - y1)
                if box_size + 0.6 * box_size < img_data.shape[0] and \
                box_size + 0.6 * box_size < img_data.shape[1]:
                    box_size = int(box_size + 0.6 * box_size)
                hw = int(box_size / 2)
                x1 = center_x - hw
                x2 = center_x + hw
                y1 = center_y - hw
                y2 = center_y + hw

                x1 = max(0, x1)
                x2 = min(img_data.shape[1], x2)
                if x1 == 0 and box_size < img_data.shape[1]:
                    x2 = box_size
                elif x1 > 0 and x2 == img_data.shape[1] and x2 - box_size > 0:
                    x1 = x2 - box_size
                y1 = max(0, y1)
                y2 = min(img_data.shape[0], y2)
                if y1 == 0 and box_size < img_data.shape[0]:
                    y2 = box_size
                elif y1 > 0 and y2 == img_data.shape[0] and y2 - box_size > 0:
                    y1 = y2 - box_size
                    
                if x2 - x1 < self.img_size or y2 - y1 < self.img_size:
                    # colr = 'y'
                    continue
                if (x2 -x1) - (y2 - y1) == 0:
                    colr = 'g'
                    b_square = True
                    faces_dict[face_idx] = {'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2, 'prob': prob, 'valid': True}
                else:
                    # colr = 'r'
                    # print(f'Invalid crop window for {os.path.basename(file_path)}: {x2 - x1}, {y2 - y1}')
                    continue
                
                if b_online:
                    self.add_roi(vb, img_data, x1, y1, x2, y2, prob, face_idx, faces_dict, file_path)

                face_idx += 1

            item.setData(faces_dict)
        if len(faces_dict) == 0:
            faces_dict[face_idx] = {'x1': 0, 'y1': 0, 'x2': self.crop_win_size, 'y2': self.crop_win_size, 'prob': 0.0, 'valid': False}
            if b_online:
                self.add_roi(vb, img_data, 0, 0, self.crop_win_size, self.crop_win_size, 0.0, 0, faces_dict, file_path)
            item.setData(faces_dict)

    def add_roi(self, vb, img_data, x1, y1, x2, y2, prob, face_idx, faces_dict, file_path):
        # Add probability text
        text = pg.TextItem(text=f'{prob:.2f} WxH:{x2-x1:.0f}x{y2-y1:.0f}', 
                           color='b', anchor=(0, 0), fill=(255, 255, 255, 150))
        vb.addItem(text)
        text.setPos(x1, y2+10)

        crop_win = pg.RectROI([x1, y1], [x2-x1, y2-y1], scaleSnap=True, removable=True,
                                maxBounds=pg.QtCore.QRectF(0, 0, img_data.shape[1], img_data.shape[0]),
                                pen=pg.mkPen('g', width=1, style=QtCore.Qt.DashDotDotLine))
        crop_win.addScaleHandle([1, 1], [0, 0], lockAspect=True)
        vb.addItem(crop_win)
        crop_win.sigRegionChanged.connect(lambda: self.update_crop(crop_win, text, 
                                                                   prob, face_idx, 
                                                                   faces_dict, file_path))
        
        crop_win.sigRemoveRequested.connect(lambda: self.remove_roi(crop_win, text,
                                                                        faces_dict, face_idx))

    def remove_roi(self, crop_win, text, faces_dict, face_idx):
        vb = crop_win.getViewBox()
        if vb is not None:
            vb.removeItem(crop_win)

        vb = text.getViewBox()
        if vb is not None:
            vb.removeItem(text)

        faces_dict.pop(face_idx, None)

    def update_crop(self, crop_win, text, prob, face_idx, faces_dict, file_path):
        x1, y1 = crop_win.pos()
        x1, y1 = int(x1), int(y1)
        x2, y2 = crop_win.size()
        x2, y2 = int(x2), int(y2)
        x2 += x1
        y2 += y1
        if x2 - x1 < 160 or y2 - y1 < 160:
            colr = 'y'
        elif (x2 -x1) - (y2 - y1) < 0.1:
            colr = 'g'
            faces_dict[face_idx] = {'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2, 'prob': prob, 'valid': True}
        else:
            colr = 'r'
            print(f'Invalid crop window for {os.path.basename(file_path)}: {x2 - x1}, {y2 - y1}')
        crop_win.setPen(pg.mkPen(colr, width=1, style=QtCore.Qt.DashDotDotLine))
        text.setPos(x1, y2+10)
        text.setText(f'{prob:.2f} WxH:{x2-x1:.0f}x{y2-y1:.0f}')

    def crop_selected(self):
        """
        Crop the checked images in concept tree view (parallelized)
        """
        import concurrent.futures
        curr_patient = self.subject_id
        root_item = self.related_concepts_model.invisibleRootItem()
        crop_tasks = []
        tree_updates = []
        if root_item.hasChildren():
            for i in range(root_item.rowCount()):
                patient_item = root_item.child(i)
                if patient_item.text() == curr_patient:
                    crops_item = QtGui.QStandardItem('crops')
                    patient_item.appendRow(crops_item)
                    if patient_item.hasChildren():
                        for j in range(patient_item.rowCount()):
                            concept_item = patient_item.child(j)
                            if concept_item.checkState() == QtCore.Qt.Checked or \
                               concept_item.checkState() == QtCore.Qt.PartiallyChecked:
                                concept = concept_item.text()
                                crops_concept_item = QtGui.QStandardItem(concept)
                                crops_item.appendRow(crops_concept_item)
                                if concept_item.hasChildren():
                                    for k in range(concept_item.rowCount()):
                                        related_concept_item = concept_item.child(k)
                                        if related_concept_item.checkState() == QtCore.Qt.Checked or \
                                           related_concept_item.checkState() == QtCore.Qt.PartiallyChecked:
                                            related_concept = related_concept_item.text().replace(' ', '').lower()
                                            crops_related_concept_item = QtGui.QStandardItem(related_concept)
                                            crops_concept_item.appendRow(crops_related_concept_item)
                                            if related_concept_item.hasChildren():
                                                for l in range(related_concept_item.rowCount()):
                                                    img_item = related_concept_item.child(l)
                                                    if img_item.checkState() == QtCore.Qt.Checked:
                                                        img = img_item.text()
                                                        if img.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp')):
                                                            file_path = os.path.abspath(os.path.join(self.task_images_downloader.task_images_path, curr_patient, concept, related_concept, img))
                                                            crop_file_path = os.path.abspath(os.path.join(self.task_images_downloader.task_images_path, curr_patient, 'crops', concept, related_concept, img))
                                                            if os.path.exists(file_path):
                                                                crop_tasks.append((img_item, file_path, crop_file_path, concept, related_concept, crops_related_concept_item))

        def crop_worker(args):
            img_item, file_path, crop_file_path, concept, related_concept, crops_related_concept_item = args
            try:
                import imageio
                img_data = imageio.imread(file_path)
                if img_item.data() is not None:
                    faces_dict = img_item.data()
                else:
                    self.detect_face_data(item=img_item, img_data=img_data, file_path=file_path, b_online=False)
                    faces_dict = img_item.data()
                result = []
                if isinstance(faces_dict, dict):
                    for face_idx, face in faces_dict.items():
                        if face['valid']:
                            crop_file_name = self.crop_save_image(img_data=img_data,
                                                                  x1=face['x1'], x2=face['x2'],
                                                                  y1=face['y1'], y2=face['y2'],
                                                                  file_path=crop_file_path, face_idx=face_idx)
                            result.append((crops_related_concept_item, crop_file_name))
                return result
            except Exception:
                print(f"[Error] Crop failed for {file_path}.")
                print(traceback.format_exc())
                return []

        # Run crop tasks in parallel
        results = []
        complete_count = 0
        total_count = len(crop_tasks)
        crop_start_time = time.time()
        if crop_tasks:
            with concurrent.futures.ThreadPoolExecutor(max_workers=None, thread_name_prefix="Crop") as executor:
                futures = [executor.submit(crop_worker, args) for args in crop_tasks]
                for future in concurrent.futures.as_completed(futures):
                    res = future.result()
                    results.extend(res)

                    complete_count += 1
                    progress_perc = int(complete_count / total_count * 100)
                    elapsed_time = time.time() - crop_start_time
                    est_total_time = elapsed_time / complete_count * total_count
                    est_remaining_time = est_total_time - elapsed_time
                    est_remaining_time_str = time.strftime("%H:%M:%S", time.gmtime(est_remaining_time))
                    self.c.progress.emit(progress_perc, f'Cropping images... Est. remaining time: {est_remaining_time_str}', False)
        self.c.progress.emit(100, f'Cropping images completed.', False)

        # Update tree with results
        for crops_related_concept_item, crop_file_name in results:
            crop_img_item = QtGui.QStandardItem(crop_file_name)
            crops_related_concept_item.appendRow(crop_img_item)

    def get_selected_concepts(self):
        """
        Get the selected concepts from the tree view
        """
        selected_concepts = []
        root_item = self.related_concepts_model.invisibleRootItem()
        if root_item.hasChildren():
            for i in range(root_item.rowCount()):
                patient_item = root_item.child(i)
                if patient_item.checkState() == QtCore.Qt.Checked or \
                   patient_item.checkState() == QtCore.Qt.PartiallyChecked:
                    # selected_concepts.append(patient_item.text())
                    if patient_item.hasChildren():
                        for j in range(patient_item.rowCount()):
                            concept_item = patient_item.child(j)
                            if concept_item.checkState() == QtCore.Qt.Checked or \
                               concept_item.checkState() == QtCore.Qt.PartiallyChecked:
                                selected_concepts.append(concept_item.text())
                                # if concept_item.hasChildren():
                                #     for k in range(concept_item.rowCount()):
                                #         related_concept_item = concept_item.child(k)
                                #         if related_concept_item.checkState() == QtCore.Qt.Checked or \
                                #            related_concept_item.checkState() == QtCore.Qt.PartiallyChecked:
                                #             selected_concepts.append(related_concept_item.text())
                                #             if related_concept_item.hasChildren():
                                #                 for l in range(related_concept_item.rowCount()):
                                #                     img_item = related_concept_item.child(l)
                                #                     if img_item.checkState() == QtCore.Qt.Checked:
                                #                         selected_concepts.append(img_item.text())
        print(selected_concepts)
        return selected_concepts

    def get_selected_image_targets(self):
        """Return checked parent concepts and only their checked related concepts."""
        targets = {}
        root_item = self.related_concepts_model.invisibleRootItem()
        if not root_item.hasChildren():
            return targets

        for root_index in range(root_item.rowCount()):
            workspace_item = root_item.child(root_index)
            if workspace_item.checkState() == Qt.Unchecked:
                continue
            for concept_index in range(workspace_item.rowCount()):
                concept_item = workspace_item.child(concept_index)
                if concept_item.checkState() == Qt.Unchecked:
                    continue
                concept = concept_item.text()
                if concept not in self.related_concepts_dict:
                    continue
                selected_names = {
                    concept_item.child(index).text()
                    for index in range(concept_item.rowCount())
                    if concept_item.child(index).checkState() != Qt.Unchecked
                    and not concept_item.child(index).text().lower().endswith(
                        ('.jpg', '.jpeg', '.png', '.gif', '.bmp')
                    )
                }
                targets[concept] = [
                    value
                    for value in self.related_concepts_dict.get(concept, [])
                    if (value.get('char') or value.get('name')) in selected_names
                ]
        return targets
        
    def mouse_clicked(self, evt):
        """
        Handle mouse click event

        Args:
        	pos: Position of the mouse
        """
        vb = evt.scene()
        if not hasattr(self, 'lbl_pixel_value'):
            self.lbl_pixel_value = QtWidgets.QLabel()
            self.lyt_concepts.addWidget(self.lbl_pixel_value)
        # Get the position of the mouse
        pos = evt.scenePos()
        if vb.sceneBoundingRect().contains(pos):
            mouse_point = vb.mapSceneToView(pos)
        else:
            return

        # Get the pixel value at the mouse position
        img = vb.addedItems[0]
        imgData = img.image
        if imgData is not None:
            x = int(mouse_point.x())
            y = int(mouse_point.y())
            if x >= 0 and x < imgData.shape[1] and y >= 0 and y < imgData.shape[0]:
                # Draw square crop window with mouse cursor at the center (white dotted lines)
                crop_window_size = self.crop_win_size
                x1 = x - crop_window_size // 2
                x2 = x + crop_window_size // 2
                y1 = y - crop_window_size // 2
                y2 = y + crop_window_size // 2
                
                self.crop_window.setData([x1, x1, x2, x2, x1], 
                                         [y1, y2, y2, y1, y1], 
                                         pen=pg.mkPen('g', width=1, style=QtCore.Qt.DashDotDotLine))
                vb.addItem(self.crop_window)

                # Crop and save the image
                self.crop_save_image(imgData, x1, x2, y1, y2)
        
    def crop_save_image(self, img_data, x1, x2, y1, y2, file_path=None, face_idx=0):
        """
        Crop and save the image
        """
        if file_path is None and hasattr(self, 'file_path'):
            crop_file_path = os.path.splitext(self.file_path)[0] + '_crop.png'
        elif file_path is not None:
            # append crop to the file name
            if face_idx > 0:
                crop_file_path = os.path.splitext(file_path)[0] + f'_{face_idx}.png'
            else:
                crop_file_path = os.path.splitext(file_path)[0] + '.png'
            crop_file_name = os.path.basename(crop_file_path)
        else:
            print("[Error] No file path provided, cropping skipped.")
            return
        crop_dir = os.path.dirname(crop_file_path)
        os.makedirs(crop_dir, exist_ok=True)
        # Crop the image and save it
        # convert to pixel coordinates
        x1 = max(0, x1)
        x2 = min(img_data.shape[1], x2)
        y1 = max(0, y1)
        y2 = min(img_data.shape[0], y2)
        cropData = img_data[y1:y2, x1:x2]
        
        # Save it as 160x160
        cropData = cv2.resize(cropData, (self.img_size, self.img_size))
        imageio.imwrite(crop_file_path, cropData, format='png')

        return crop_file_name

    def mouse_moved(self, pos):
        """
        Handle mouse move event

        Args:
        	pos: Position of the mouse
        """
        vb = pos.scene()
        if not hasattr(self, 'lbl_pixel_value'):
            self.lbl_pixel_value = QtWidgets.QLabel()
            self.lyt_concepts.addWidget(self.lbl_pixel_value)
        # Get the position of the mouse
        if vb.sceneBoundingRect().contains(pos):
            mouse_point = vb.mapSceneToView(pos)
        else:
            return

        # Get the pixel value at the mouse position
        img = vb.addedItems[0]
        imgData = img.image
        if imgData is not None:
            x = int(mouse_point.x())
            y = int(mouse_point.y())
            if x >= 0 and x < imgData.shape[1] and y >= 0 and y < imgData.shape[0]:
                pixel_value = imgData[y, x]
                self.lbl_pixel_value.setText(f'Mouse: [{x},{y}] Pixel Value: {pixel_value}')

                # Draw square crop window with mouse cursor at the center (white dotted lines)
                crop_window_size = self.crop_win_size
                x1 = x - crop_window_size // 2
                x2 = x + crop_window_size // 2
                y1 = y - crop_window_size // 2
                y2 = y + crop_window_size // 2

                if x1 < 0 or x2 >= imgData.shape[1] or y1 < 0 or y2 >= imgData.shape[0]:
                    col = 'y'
                else:
                    col = 'b'
                
                self.crop_window.setData([x1, x1, x2, x2, x1], 
                                         [y1, y2, y2, y1, y1], 
                                         pen=pg.mkPen(col, width=1, style=QtCore.Qt.DashDotDotLine))
                vb.addItem(self.crop_window, ignoreBounds=True)

    def process_related_concepts_threaded(self):
        try:
            proc_thread = threading.Thread(
                target=self.get_related_concepts,
                name='stimstudio-related-concepts',
                daemon=True,
            )
            self._concept_worker_thread = proc_thread
            proc_thread.start()
        except:
            print(traceback.format_exc())

    def get_related_concepts(self):
        """
        Get schema-validated related concepts for the checked concepts.
        """
        completed = 0
        start_time = time.time()
        try:
            new_concepts = list(self._concept_run_targets)
            tot_concepts = len(new_concepts)
            provider = self.concept_generator.selected_provider
            self.c.log.emit(f'Using {provider} related-concept generation.')
            self.c.progress.emit(0, f'Getting related concepts for {tot_concepts} concepts...', False)
            self.c.log.emit(f'Getting related concepts for {tot_concepts} concepts...')
            for concept in new_concepts:
                if not self._wait_for_concept_permission():
                    break
                self.get_related_concepts_ext(concept)
                completed += 1
                self._concept_completed_count = completed
                progress_percentage = completed / tot_concepts * 100
                hrs, mins, secs, _ = get_elapsed_time(start_time=start_time)
                elapsed = hrs * 3600 + mins * 60 + secs
                remaining_time = elapsed / completed * (tot_concepts - completed)
                hrs_rem, mins_rem, secs_rem, _ = get_time_hrs_mins_secs_ms(
                    seconds=remaining_time
                )
                progress_message = (
                    f'{completed}/{tot_concepts} related concepts retrieved | '
                    f'Elapsed {hrs:02.0f}:{mins:02.0f}:{secs:02.0f} | '
                    f'Remaining {hrs_rem:02.0f}:{mins_rem:02.0f}:{secs_rem:02.0f}'
                )
                self.c.progress.emit(int(progress_percentage), progress_message, False)
                self.concept_workflow_signals.progress.emit(
                    int(progress_percentage), progress_message
                )
            hrs, mins, secs, ms = get_elapsed_time(start_time=start_time)
            outcome = 'stopped after' if self._concept_stop_event.is_set() else 'processed'
            message = (
                f'Related-concept generation {outcome} {completed} concepts in '
                f'{hrs:.0f} hours, {mins:.0f} minutes, {secs:.0f} seconds'
            )
            final_progress = (
                int(completed / tot_concepts * 100)
                if self._concept_stop_event.is_set() else 100
            )
            self.c.progress.emit(final_progress, message, False)
            self.c.log.emit(message)
        except Exception:
            print(traceback.format_exc())
        finally:
            self.c.related_concepts_done.emit()

    def get_related_concepts_context(self, concept):
        """
        Get related concepts for a concept using context

        Args:
        	concept: Concept
        """
        try:
            self.get_related_concepts_ext(concept)
            self.update_related_concepts(b_llm=True)
        except:
            print(traceback.format_exc())

    def get_related_concepts_ext(self, concept):
        """
        Get related concepts for a concept

        Args:
        	concept: Concept
        """
        try:
            self.c.progress.emit(0, f'Getting related concepts for {concept} ...', False)
            count = int(config.get('related_concept_count', 10))
            ctx = getattr(self, "concept_excerpts", {}).get(concept, "")
            self.related_concepts_dict[concept] = self.concept_generator.generate(
                concept, count=count, context=ctx
            )
            self.c.progress.emit(100, f'Getting related concepts for {concept} ... Done', False)
            self.c.related_concepts_single_done.emit()
            return True
        except Exception as exc:
            self.related_concepts_dict[concept] = []
            self.c.log.emit(f'Error getting related concepts for {concept}: {exc}')
            print(traceback.format_exc())
            return False

    def post_process_related_concepts(self):
        """Finish step 2 without unexpectedly starting later workflow steps."""
        self._concept_run_active = False
        self._set_concept_controls_running(False)
        self.update_related_concepts(b_llm=True)
        total = getattr(self, '_concept_total_expected', 0)
        completed = getattr(self, '_concept_completed_count', 0)
        percentage = int(completed / total * 100) if total else 0
        if self._concept_stop_event.is_set() and percentage < 100:
            self._workflow_step_progress[2] = percentage
            self._set_workflow_tracker(completed=1, active=2, reset=True)
        else:
            self._sync_workflow_tracker()
        if hasattr(self, 'create_images_button'):
            self.create_images_button.setEnabled(
                any(self.related_concepts_dict.values())
            )

    @staticmethod
    def _append_image_items(parent_item, directory):
        if not os.path.isdir(directory):
            return
        for filename in sorted(os.listdir(directory)):
            file_path = os.path.abspath(os.path.join(directory, filename))
            if not os.path.isfile(file_path) or not filename.lower().endswith(
                IMAGE_EXTENSIONS
            ):
                continue
            image_item = QtGui.QStandardItem(filename)
            image_item.setCheckable(True)
            image_item.setCheckState(Qt.Checked)
            image_item.setData(file_path, IMAGE_PATH_ROLE)
            parent_item.appendRow(image_item)

    def update_related_concepts(self, b_llm=False):
        """
        Add related concepts to the related concepts tree view
        """
        curr_patient = self.subject_id
        # Clear the related concepts tree view
        self.related_concepts_tree_view.setModel(None)

        self.related_concepts_model = QtGui.QStandardItemModel()
        self.related_concepts_tree_view.setModel(self.related_concepts_model)
        self.related_concepts_tree_view.setRootIsDecorated(False)
        self.related_concepts_tree_view.setAlternatingRowColors(True)
        self.related_concepts_tree_view.setHeaderHidden(True)
        self.related_concepts_model.itemChanged.connect(self.check_uncheck_items)

        # Create a root item to check/uncheck all
        root_item = QtGui.QStandardItem(curr_patient)
        root_item.setCheckable(True)
        root_item.setCheckState(Qt.Checked)
        self.related_concepts_model.appendRow(root_item)

        if b_llm:
            for concept in self.related_concepts_dict:
                item = QtGui.QStandardItem(concept)
                item.setCheckable(True)
                item.setCheckState(Qt.Checked)
                if concept.replace(" ", "") in self.concepts_available:
                    item.setBackground(QtGui.QBrush(QtGui.QColor(0, 255, 0)))
                    item.setCheckState(Qt.Unchecked)
                concept_path = os.path.join(
                    self.task_images_downloader.task_images_path,
                    curr_patient,
                    concept,
                )
                self._append_image_items(item, concept_path)
                for key in self.related_concepts_dict[concept]:
                    try:
                        if key['char'] == '':
                            child = QtGui.QStandardItem(key['name'])
                            folder_name = key['name'].replace(" ", "").lower()
                        else:
                            child = QtGui.QStandardItem(key['char'])
                            folder_name = key['char'].replace(" ", "").lower()
                        
                        child.setToolTip(key['description'])
                        child.setCheckable(True)
                        child.setCheckState(Qt.Checked)
                        item.appendRow(child)
                        
                        path = os.path.join(self.task_images_downloader.task_images_path, curr_patient, concept, folder_name)
                        self._append_image_items(child, path)
                    except:
                        print(f"[Error] {key} of {concept} not formatted correctly. Skipping.")

                root_item.appendRow(item)
        else:
            for concept in self.related_concepts_dict:
                item = QtGui.QStandardItem(concept)
                item.setCheckable(True)
                item.setCheckState(Qt.Checked)
                if concept.replace(' ', '') in self.concepts_available:
                    # Color the concept if it is in the concepts_all list
                    item.setBackground(QtGui.QBrush(QtGui.QColor(0, 255, 0)))
                    item.setCheckState(Qt.Unchecked)
                path = os.path.join(self.task_images_downloader.task_images_path, curr_patient, concept)
                if os.path.exists(path):
                    self._append_image_items(item, path)
                    related_concepts = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
                    for related_concept in related_concepts:
                        child = QtGui.QStandardItem(related_concept)
                        child.setCheckable(True)
                        child.setCheckState(Qt.Checked)
                        item.appendRow(child)
                        if related_concept in self.concepts_available:
                            # Color the related concept if it is in the concepts_all list
                            child.setBackground(QtGui.QBrush(QtGui.QColor(0, 255, 0)))
                        # Get the images for the related concept
                        path = os.path.join(self.task_images_downloader.task_images_path, curr_patient, concept, related_concept)
                        self._append_image_items(child, path)

                root_item.appendRow(item)

        self.related_concepts_tree_view.expandAll()

    def update_categ_localiz_concepts(self):
        """
        Update the concepts for categ_localiz task
        """
        curr_patient = self.subject_id
        # Clear the related concepts tree view
        self.related_concepts_tree_view.setModel(None)
        # remove context menu
        self.related_concepts_tree_view.setContextMenuPolicy(Qt.NoContextMenu)

        self.related_concepts_model = QtGui.QStandardItemModel()
        self.related_concepts_tree_view.setModel(self.related_concepts_model)
        self.related_concepts_tree_view.setRootIsDecorated(False)
        self.related_concepts_tree_view.setAlternatingRowColors(True)
        self.related_concepts_tree_view.setHeaderHidden(True)
        self.related_concepts_model.itemChanged.connect(self.check_uncheck_items)

        # Create a root item to check/uncheck all
        root_item = QtGui.QStandardItem(curr_patient)
        root_item.setCheckable(True)
        root_item.setCheckState(Qt.Checked)
        self.related_concepts_model.appendRow(root_item)

        task_folder = os.path.join(config['experiment_folder'], curr_patient, 'stimuli', 'categ_localiz')
        if not os.path.exists(task_folder):
            self.create_task_stimuli()
        if os.path.exists(task_folder):
            categ_folders = [d for d in os.listdir(task_folder) if os.path.isdir(os.path.join(task_folder, d)) and d != 'custom_pics']
            for categ in categ_folders:
                item = QtGui.QStandardItem(categ)
                item.setCheckable(True)
                item.setCheckState(Qt.Checked)
                if categ.replace(' ', '') in self.concepts_available:
                    # Color the concept if it is in the concepts_all list
                    item.setBackground(QtGui.QBrush(QtGui.QColor(0, 255, 0)))
                    item.setCheckState(Qt.Unchecked)
                path = os.path.join(task_folder, categ)
                if os.path.exists(path):
                    imgs = os.listdir(path)
                    for img in imgs:
                        if img.lower().endswith(('.png', '.jpg', '.jpeg')):
                            img_item = QtGui.QStandardItem(img)
                            img_item.setCheckable(True)
                            img_item.setCheckState(Qt.Checked)
                            item.appendRow(img_item)

                root_item.appendRow(item)

        self.related_concepts_tree_view.expandAll()

    def update_task_stims_folder_status(self):
        """
        Check if the task folder is already created and has the custom pics and pics now folders
        """
        curr_patient = self.subject_id
        

        if self.task_name.startswith('dynamic_scr'):
            task_stim_folder = os.path.join(config['experiment_folder'], curr_patient, 'stimuli', 'dynamic_scr')
            self.update_dynamic_scr_folder_status(task_stim_folder)
        elif self.task_name.startswith('categ_localiz'):
            task_stim_folder = os.path.join(config['experiment_folder'], curr_patient, 'stimuli', 'categ_localiz')
            self.update_categ_localiz_folder_status(task_stim_folder)
        else:
            self.status_lbl.setStyleSheet("QLabel { background-color : yellow; color : black; }")
            self.status_lbl.setText(f'Task folder: {task_stim_folder} (Task not recognized for folder check)')

    def update_dynamic_scr_folder_status(self, task_folder):
        """
        Check if the dynamic SCR folder is already created and has the custom pics and pics now folders
        """

        b_custom_pics = False
        b_pics_now = False
        custom_pics_count = 0
        pics_now_count = 0
        if os.path.exists(task_folder):
            custom_pics_folder = os.path.join(task_folder, 'custom_pics')
            pics_now_folder = os.path.join(task_folder, 'pics_now')
            if os.path.exists(custom_pics_folder):
                b_custom_pics = True
                custom_pics = [f for f in os.listdir(custom_pics_folder) if os.path.isfile(os.path.join(custom_pics_folder, f)) and f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                custom_pics_count = len(custom_pics)
                missing_custom_pics = self.check_custom_pics(custom_pics)
            if os.path.exists(pics_now_folder):
                b_pics_now = True
                pics_now_count = len([f for f in os.listdir(pics_now_folder) if os.path.isfile(os.path.join(pics_now_folder, f)) and f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        
        if b_custom_pics and b_pics_now:
            self.status_lbl.setStyleSheet("QLabel { color : green; }")
            status_msg = f'Task folder: {task_folder} | Custom pics: {custom_pics_count} | Pics now: {pics_now_count}'
            if missing_custom_pics:
                status_msg += f' | Missing custom pics count: {len(missing_custom_pics)}'
            self.status_lbl.setText(status_msg)
        else:
            self.status_lbl.setStyleSheet("QLabel { background-color : yellow; color : black; }")
            self.status_lbl.setText(f'Task folder: {task_folder} | Custom pics: {custom_pics_count} | Pics now: {pics_now_count} (Please create custom_pics and pics_now folders)')

    def update_categ_localiz_folder_status(self, task_folder):
        """
        Check if the categ_localiz folder is already created and has the custom pics folder
        """

        b_custom_pics = False
        custom_pics_count = 0
        categ_folders = {}
        if os.path.exists(task_folder):
            custom_pics_folder = os.path.join(task_folder, 'custom_pics')
            if os.path.exists(custom_pics_folder):
                b_custom_pics = True
                custom_pics = glob.glob(os.path.join(custom_pics_folder, '**', '*.*'), recursive=True)
                custom_pics_count = len(custom_pics)
            
            # add all other folders in the task folder as categories
            for item in os.listdir(task_folder):
                item_path = os.path.join(task_folder, item)
                if os.path.isdir(item_path) and item != 'custom_pics':
                    categ_folders[item] = len([f for f in os.listdir(item_path) if os.path.isfile(os.path.join(item_path, f)) and f.lower().endswith(('.png', '.jpg', '.jpeg'))])
                categ_info_str = ', '.join([f'{k}: {v}' for k, v in categ_folders.items()])

        if b_custom_pics:
            self.status_lbl.setStyleSheet("QLabel { color : green; }")
            status_msg = f'Task folder: {task_folder} | Custom pics: {custom_pics_count}'
            if categ_info_str:
                status_msg += f' | Categories: {categ_info_str}'
            self.status_lbl.setText(status_msg)
        else:
            self.status_lbl.setStyleSheet("QLabel { background-color : yellow; color : black; }")
            self.status_lbl.setText(f'Task folder: {task_folder} | Custom pics: {custom_pics_count} (Please create custom_pics folder)')

    def check_custom_pics(self, custom_pics):
        """
        Check if the custom pics are in the related concepts tree view
        """
        missing_custom_pics = []
        custom_pics_bases = [os.path.splitext(x)[0] for x in custom_pics]
        root_item = self.related_concepts_model.invisibleRootItem()
        if root_item.hasChildren():
            for i in range(root_item.rowCount()):
                patient_item = root_item.child(i)
                if patient_item.text() == self.subject_id:
                    if patient_item.hasChildren():
                        for j in range(patient_item.rowCount()):
                            concept_item = patient_item.child(j)
                            if concept_item.hasChildren():
                                for k in range(concept_item.rowCount()):
                                    related_concept_item = concept_item.child(k)
                                    if related_concept_item.hasChildren():
                                        for l in range(related_concept_item.rowCount()):
                                            img_item = related_concept_item.child(l)
                                            img_name = img_item.text()
                                            # skip if does not end with _1
                                            
                                            img_base = os.path.splitext(img_name)[0]
                                            if not img_base.endswith('_1'):
                                                continue
                                            if img_base in custom_pics_bases:
                                                custom_pics_bases.remove(img_base)
                                                # img_item.setBackground(QtGui.QBrush(QtGui.QColor(0, 255, 0)))
                                                # img_item.setForeground(QtGui.QBrush(QtGui.QColor(0, 0, 0)))
                                                # img_item.setCheckState(Qt.Checked)
                                            else:
                                                missing_custom_pics.append(img_name)
                                                img_item.setBackground(QtGui.QBrush(QtGui.QColor(255, 255, 0))) # yellow for missing
                                                img_item.setForeground(QtGui.QBrush(QtGui.QColor(0, 0, 0)))

        return missing_custom_pics

    def check_parent(self, parent):
        if not parent:
            return
        checked_count = 0
        total_count = parent.rowCount()
        for row in range(total_count):
            if parent.child(row).checkState() != Qt.Unchecked:
                checked_count += 1
        if checked_count == 0:
            parent.setCheckState(Qt.Unchecked)
        elif checked_count == total_count:
            parent.setCheckState(Qt.Checked)
        else:
            parent.setCheckState(Qt.PartiallyChecked)
        self.check_parent(parent.parent())

    def check_uncheck_items(self, item, update_selection=True):
        self.related_concepts_model.itemChanged.disconnect(self.check_uncheck_items)
        
        if item.hasChildren():
            self.check_children(item, item.checkState())
        
        self.check_parent(item.parent())
        
        self.related_concepts_model.itemChanged.connect(self.check_uncheck_items)

    def check_children(self, parent, check_state):
        for row in range(parent.rowCount()):
            child = parent.child(row)
            child.setCheckState(check_state)
            if child.hasChildren():
                self.check_children(child, check_state)

    def update_exp_config(self, task_name):
        self.task_name = task_name

        self.block_params = config["task_config"][task_name]["block_params"]
        self.exp_params = config["task_config"][task_name]["exp_params"]

        if task_name.startswith('gaps'):
            self.clear_image()
            return

        if hasattr(self, 'curr_patient_path'):
            if self.task_name.lower().startswith('dynamic_scr'):
                self.update_related_concepts()
            elif self.task_name.lower().startswith('categ_localiz'):
                self.update_categ_localiz_concepts()

if __name__ == "__main__":
    app = QtWidgets.QApplication([])
    concepts_view = QtWidgets.QVBoxLayout()
    stimulus_forge = StimulusForge(concepts_view=concepts_view)
    stimulus_forge.concepts_widget.show()
    app.exec_()
