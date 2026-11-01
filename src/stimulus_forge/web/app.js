/**
 * Stimulus Forge Web Client Application Logic
 * Enterprise Multimodal Architecture & Concept Tree Controller
 */

document.addEventListener("DOMContentLoaded", () => {
  // ==========================================
  // 1. Reactive State & Tree Data Store
  // ==========================================
  let activeSeedConcept = "";
  let selectedFilterConcept = ""; // Filter gallery by concept or related association
  let selectedTreeTarget = { type: "concept", concept: "", association: null };
  let currentLightboxImage = null; // { img: object, concept: string }
  let contextMenuTarget = null; // { type, concept, relItem, img }
  let chatState = "awaiting_concept";
  let chatCurrentConcept = "";
  let showFaceRoi = true;
  let showFaceRoiLabels = localStorage.getItem("sf_show_face_roi_labels") !== "0";
  let activeAcquisitionAbort = null;
  let acquisitionCancelled = false;
  const faceRoiCache = new Map(); // path -> { faces: [], count: number, detector: string }

  /**
   * Concept Tree Model (mirrors Desktop QTreeView)
   * {
   *   [seedConcept]: {
   *     checked: boolean,
   *     expanded: boolean,
   *     sourceMethod: "manual" | "deterministic" | "agentic_rag" | "generated",
   *     excerpt: string,
   *     related: Array<{ name: string, char: string, description: string, checked: boolean, expanded?: boolean }>,
   *     images: Array<{ url: string, filename: string, source: string, related_concept: string, checked: boolean }>
   *   }
   * }
   */
  const workspaceTree = {};

  // ==========================================
  // 2. DOM Elements
  // ==========================================
  // System Health
  const backendStatusPill = document.getElementById("backend-status");
  const backendStatusText = document.getElementById("backend-status-text");
  const hardwareText = document.getElementById("hardware-text");

  // Collapsible Settings Panel
  const workflowPanel = document.getElementById("workflow-panel");
  const workflowPanelHeader = document.getElementById("workflow-panel-header");
  const btnToggleWorkflow = document.getElementById("btn-toggle-workflow");
  const workflowToggleHint = document.getElementById("workflow-toggle-hint");
  const workflowSummaryBadges = document.getElementById("workflow-summary-badges");
  const summaryChipInference = document.getElementById("summary-chip-inference");
  const summaryTextInference = document.getElementById("summary-text-inference");
  const summaryChipImages = document.getElementById("summary-chip-images");
  const summaryTextImages = document.getElementById("summary-text-images");
  const summaryChipFace = document.getElementById("summary-chip-face");
  const summaryTextFace = document.getElementById("summary-text-face");
  const summaryChipCrop = document.getElementById("summary-chip-crop");
  const summaryTextCrop = document.getElementById("summary-text-crop");
  const summaryChipCache = document.getElementById("summary-chip-cache");
  const summaryTextCache = document.getElementById("summary-text-cache");

  // Load Concepts Input & File Upload
  const conceptInput = document.getElementById("concept-input");
  const btnLoadConcepts = document.getElementById("btn-load-concepts");
  const conceptFileInput = document.getElementById("concept-file-input");
  const btnUploadFile = document.getElementById("btn-upload-file");
  const fileUploadStatus = document.getElementById("file-upload-status");
  const fileHint = document.getElementById("file-hint");
  const subjectSelectorRow = document.getElementById("subject-selector-row");
  const subjectSelect = document.getElementById("subject-select");
  const subjectInput = document.getElementById("subject-input");
  const btnToggleSubjectMode = document.getElementById("btn-toggle-subject-mode");

  let lastUploadedQuestionnaireData = null;

  function isSubjectDropdownActive() {
    return subjectSelect && subjectSelect.style.display !== "none";
  }

  function getActiveSubjectId() {
    if (isSubjectDropdownActive() && subjectSelect.value) {
      const val = subjectSelect.value.trim();
      if (val && val !== "all" && val !== "__custom__") {
        return val;
      }
    }
    if (subjectInput && subjectInput.value.trim()) {
      return subjectInput.value.trim();
    }
    return "SUBJ_TEST_001";
  }

  // Related Concepts Settings
  const providerSelect = document.getElementById("provider-select");
  const maxConceptsInput = document.getElementById("max-concepts-input");
  const localModelGroup = document.getElementById("local-model-group");
  const localModelPathInput = document.getElementById("local-model-path-input");
  const btnBrowseModel = document.getElementById("btn-browse-model");
  const detectedModelsSelect = document.getElementById("detected-models-select");
  const btnScanModels = document.getElementById("btn-scan-models");
  const localModelValidation = document.getElementById("local-model-validation");

  const cloudModelGroup = document.getElementById("cloud-model-group");
  const cloudModelLabel = document.getElementById("cloud-model-label");
  const cloudModelSelect = document.getElementById("cloud-model-select");
  const btnRefreshCloudModels = document.getElementById("btn-refresh-cloud-models");
  const customCloudModelRow = document.getElementById("custom-cloud-model-row");
  const customCloudModelInput = document.getElementById("custom-cloud-model-input");

  const DEFAULT_CLOUD_MODELS = {
    google: [
      { id: "gemini-3.8-flash", label: "Gemini 3.8 Flash (Fast & Low Cost — Default)" },
      { id: "gemini-3.8-pro", label: "Gemini 3.8 Pro (Advanced Reasoning)" },
      { id: "gemini-2.5-flash", label: "Gemini 2.5 Flash (Balanced)" },
      { id: "gemini-2.5-pro", label: "Gemini 2.5 Pro (Deep Intelligence)" },
    ],
    openai: [
      { id: "gpt-5.6-luna", label: "GPT-5.6 Luna (Default)" },
      { id: "gpt-4o", label: "GPT-4o (Omni Multimodal)" },
      { id: "gpt-4o-mini", label: "GPT-4o Mini (Fast & Lightweight)" },
    ],
    anthropic: [
      { id: "claude-sonnet-4-6", label: "Claude Sonnet 4.6 (Default)" },
      { id: "claude-3-5-sonnet-20241022", label: "Claude 3.5 Sonnet (High Accuracy)" },
      { id: "claude-3-5-haiku-20241022", label: "Claude 3.5 Haiku (Fast & Efficient)" },
    ],
    compatible: [
      { id: "meta-llama/llama-3.3-70b-instruct", label: "Llama 3.3 70B Instruct" },
    ],
  };

  const providerModelsCache = {};

  function populateCloudModels(provider) {
    if (!cloudModelSelect) return;
    const previousSelected = cloudModelSelect.value;
    cloudModelSelect.innerHTML = "";

    const models = (providerModelsCache[provider] && providerModelsCache[provider].length > 0)
      ? providerModelsCache[provider].map(m => typeof m === "string" ? { id: m, label: m } : m)
      : (DEFAULT_CLOUD_MODELS[provider] || []);

    models.forEach(m => {
      const opt = document.createElement("option");
      opt.value = m.id;
      opt.textContent = m.label;
      cloudModelSelect.appendChild(opt);
    });

    const customOpt = document.createElement("option");
    customOpt.value = "__custom__";
    customOpt.textContent = "Custom model ID...";
    cloudModelSelect.appendChild(customOpt);

    if (previousSelected && Array.from(cloudModelSelect.options).some(o => o.value === previousSelected)) {
      cloudModelSelect.value = previousSelected;
    } else if (models.length > 0) {
      cloudModelSelect.value = models[0].id;
    }

    if (customCloudModelRow) {
      customCloudModelRow.style.display = cloudModelSelect.value === "__custom__" ? "block" : "none";
    }
  }

  function getSelectedCloudModel() {
    if (!cloudModelSelect || (providerSelect && providerSelect.value === "local")) return "";
    if (cloudModelSelect.value === "__custom__") {
      return customCloudModelInput ? customCloudModelInput.value.trim() : "";
    }
    return cloudModelSelect.value;
  }

  async function fetchProviderModels() {
    try {
      const res = await fetch("/api/v1/providers");
      if (!res.ok) return;
      const providers = await res.json();
      providers.forEach(p => {
        if (p.available_models && p.available_models.length > 0) {
          providerModelsCache[p.id] = p.available_models.map(id => ({
            id,
            label: id === p.default_model ? `${id} (Default)` : id
          }));
        }
      });
      const currentProv = providerSelect ? providerSelect.value : "local";
      if (currentProv !== "local") {
        populateCloudModels(currentProv);
        updateSettingsSummary();
      }
    } catch (_) {}
  }

  const cloudCredentialsDetails = document.getElementById("cloud-credentials-details");
  const apiKeyInput = document.getElementById("api-key-input");
  const baseUrlInput = document.getElementById("base-url-input");
  const cacheToggle = document.getElementById("cache-toggle");
  const btnClearConceptCache = document.getElementById("btn-clear-concept-cache");

  // Image Settings
  const imageSourceSelect = document.getElementById("image-source-select");
  const imagesPerConceptSelect = document.getElementById("images-per-concept");
  const downloadFolderInput = document.getElementById("download-folder-input");
  const btnBrowseDownloadFolder = document.getElementById("btn-browse-download-folder");
  const downloadFolderStatus = document.getElementById("download-folder-status");
  const subsettingsFaceCard = document.getElementById("subsettings-face-card");
  const subsettingsCropCard = document.getElementById("subsettings-crop-card");
  const detectFacesToggle = document.getElementById("detect-faces-toggle");
  const detectFacesStatusBadge = document.getElementById("detect-faces-status-badge");
  const maxFacesInput = document.getElementById("max-faces-input");
  const showFaceRoiLabelsToggle = document.getElementById("show-face-roi-labels-toggle");
  const cropFacesToggle = document.getElementById("crop-faces-toggle");
  const cropFacesStatusBadge = document.getElementById("crop-faces-status-badge");
  const targetWidthInput = document.getElementById("target-width-input");
  const targetHeightInput = document.getElementById("target-height-input");

  // Generate Controls in Concept Tree Panel
  const genRelatedToggle = document.getElementById("gen-related-toggle");
  const genImagesToggle = document.getElementById("gen-images-toggle");
  const btnGenerate = document.getElementById("btn-generate");
  const generateSpinner = document.getElementById("generate-spinner");
  const btnStopAcquire = document.getElementById("btn-stop-acquire");
  const acquisitionPlanHint = document.getElementById("acquisition-plan-hint");
  const generationStatus = acquisitionPlanHint;

  // Bing Options & Safety Exclusions
  const bingOptionsDetails = document.getElementById("bing-options-details");
  const bingFilterSelect = document.getElementById("bing-filter-select");
  const bingSafesearchSelect = document.getElementById("bing-safesearch-select");
  const bingExclusionsInput = document.getElementById("bing-exclusions-input");
  const btnResetExclusions = document.getElementById("btn-reset-exclusions");
  const DEFAULT_EXCLUSIONS = "-porn -pornography -xxx -nude -nudity -sex -adult";

  // Local Diffusion Model
  const localImageModelGroup = document.getElementById("local-image-model-group");
  const localImageModelPathInput = document.getElementById("local-image-model-path-input");
  const btnBrowseImageModel = document.getElementById("btn-browse-image-model");
  const localImageModelValidation = document.getElementById("local-image-model-validation");

  // Concept Tree Panel
  const treeContainer = document.getElementById("concept-tree-container");
  const treeTotalBadge = document.getElementById("tree-total-badge");
  const treeFilterInput = document.getElementById("tree-filter-input");
  const treeSelectAllCheckbox = document.getElementById("tree-select-all-checkbox");
  const btnToggleExpandTree = document.getElementById("btn-toggle-expand-tree");
  const btnCheckAllTree = document.getElementById("btn-check-all-tree");
  const btnUncheckAllTree = document.getElementById("btn-uncheck-all-tree");
  const btnClearTree = document.getElementById("btn-clear-tree");
  const btnExpandAllTree = document.getElementById("btn-expand-all-tree");
  const btnCollapseAllTree = document.getElementById("btn-collapse-all-tree");

  // Workspace & Image Gallery
  const galleryTitle = document.getElementById("gallery-title");
  const gallerySubtitle = document.getElementById("gallery-subtitle");
  const galleryGrid = document.getElementById("gallery-grid");
  const imageCountBadge = document.getElementById("image-count-badge");
  const toggleFaceRoi = document.getElementById("toggle-face-roi");
  const btnCropSelected = document.getElementById("btn-crop-selected");
  const btnExportJson = document.getElementById("btn-export-json");
  const btnExportCsv = document.getElementById("btn-export-csv");

  // Lightbox Modal
  const imageLightboxModal = document.getElementById("image-lightbox-modal");
  const lightboxImg = document.getElementById("lightbox-img");
  const lightboxRoiCanvas = document.getElementById("lightbox-roi-canvas");
  const lightboxFaceRoiToggle = document.getElementById("lightbox-face-roi-toggle");
  const lightboxTitle = document.getElementById("lightbox-title");
  const lightboxMeta = document.getElementById("lightbox-meta");
  const lightboxFaceStatus = document.getElementById("lightbox-face-status");
  const btnLightboxCropFace = document.getElementById("btn-lightbox-crop-face");
  const btnCloseLightbox = document.getElementById("btn-close-lightbox");
  const btnLightboxDelete = document.getElementById("btn-lightbox-delete");

  // Help & How-To Modal
  const btnOpenHelp = document.getElementById("btn-open-help");
  const helpGuideModal = document.getElementById("help-guide-modal");
  const btnCloseHelp = document.getElementById("btn-close-help");
  const btnCloseHelpFooter = document.getElementById("btn-close-help-footer");
  const helpTabBtns = document.querySelectorAll(".help-tab-btn");
  const helpTabPanes = document.querySelectorAll(".help-tab-pane");

  // Tree Context Menu
  const treeContextMenu = document.getElementById("tree-context-menu");
  const ctxAcquire = document.getElementById("ctx-acquire");
  const ctxAcquireText = document.getElementById("ctx-acquire-text");
  const ctxGenerate = document.getElementById("ctx-generate");
  const ctxAddRelated = document.getElementById("ctx-add-related");
  const ctxEditContext = document.getElementById("ctx-edit-context");
  const ctxClearImages = document.getElementById("ctx-clear-images");
  const ctxDelete = document.getElementById("ctx-delete");

  // Add Related Concept Modal
  const addRelatedModal = document.getElementById("add-related-modal");
  const addRelatedSubtitle = document.getElementById("add-related-subtitle");
  const addRelatedForm = document.getElementById("add-related-form");
  const addRelatedInput = document.getElementById("add-related-input");
  const btnCloseAddRelated = document.getElementById("btn-close-add-related");
  const btnCancelAddRelated = document.getElementById("btn-cancel-add-related");
  let addRelatedTargetConcept = null;

  // Edit Context / Category Modal
  const editContextModal = document.getElementById("edit-context-modal");
  const editContextSubtitle = document.getElementById("edit-context-subtitle");
  const editContextForm = document.getElementById("edit-context-form");
  const editContextInput = document.getElementById("edit-context-input");
  const btnCloseEditContext = document.getElementById("btn-close-edit-context");
  const btnCancelEditContext = document.getElementById("btn-cancel-edit-context");
  const editContextBackdrop = document.getElementById("edit-context-backdrop");
  let editContextTargetConcept = null;

  // Telemetry & FinOps
  const metricLatency = document.getElementById("metric-latency");
  const metricTokens = document.getElementById("metric-tokens");
  const metricCost = document.getElementById("metric-cost");
  const metricSavings = document.getElementById("metric-savings");
  const metricCacheHit = document.getElementById("metric-cache-hit");
  const btnResetTelemetry = document.getElementById("btn-reset-telemetry");

  // Workflow Assistant Chat
  const btnToggleChat = document.getElementById("btn-toggle-chat");
  const chatDock = document.getElementById("chat-dock");
  const btnCloseChat = document.getElementById("btn-close-chat");
  const btnClearChat = document.getElementById("btn-clear-chat");
  const chatTranscript = document.getElementById("chat-transcript");
  const chatChoicesContainer = document.getElementById("chat-choices-container");
  const chatInputForm = document.getElementById("chat-input-form");
  const chatInputText = document.getElementById("chat-input-text");

  // ==========================================
  // Collapsible Settings Panel Controller (Collapsed by default)
  // ==========================================
  function toggleWorkflowPanel(forceCollapse = null) {
    if (!workflowPanel) return;
    const willCollapse = forceCollapse !== null ? forceCollapse : !workflowPanel.classList.contains("collapsed");
    if (willCollapse) {
      workflowPanel.classList.add("collapsed");
      if (workflowToggleHint) workflowToggleHint.textContent = "Click to expand";
      if (btnToggleWorkflow) btnToggleWorkflow.textContent = "Expand Settings";
      localStorage.setItem("sf_settings_collapsed", "1");
    } else {
      workflowPanel.classList.remove("collapsed");
      if (workflowToggleHint) workflowToggleHint.textContent = "Click to collapse";
      if (btnToggleWorkflow) btnToggleWorkflow.textContent = "Collapse Settings";
      localStorage.setItem("sf_settings_collapsed", "0");
    }
    updateSettingsSummary();
  }

  if (workflowPanelHeader) {
    workflowPanelHeader.addEventListener("click", (e) => {
      if (e.target === btnToggleWorkflow || btnToggleWorkflow?.contains(e.target)) return;
      toggleWorkflowPanel();
    });
  }

  if (btnToggleWorkflow) {
    btnToggleWorkflow.addEventListener("click", (e) => {
      e.stopPropagation();
      toggleWorkflowPanel();
    });
  }

  if (summaryChipCache) {
    summaryChipCache.addEventListener("click", (e) => {
      e.stopPropagation();
      if (cacheToggle) {
        cacheToggle.checked = true;
        cacheToggle.dispatchEvent(new Event("change"));
      }
    });
  }

  // Collapsed by default
  toggleWorkflowPanel(true);

  // ==========================================
  // 3. Tree Management & Synchronization
  // ==========================================

  function ensureTreeNode(conceptName, sourceMethod = "manual", excerpt = "") {
    const key = conceptName.trim();
    if (!key) return null;
    if (!workspaceTree[key]) {
      workspaceTree[key] = {
        checked: true,
        expanded: true,
        sourceMethod: sourceMethod,
        excerpt: excerpt || "",
        related: [],
        images: [],
      };
    }
    return workspaceTree[key];
  }

  function populateTreeWithConcepts(concepts, sourceMethod = "manual", excerpts = {}) {
    // 1. Completely clear existing tree nodes from workspaceTree
    Object.keys(workspaceTree).forEach((k) => delete workspaceTree[k]);

    // 2. Clear filter input so all newly loaded concepts are immediately visible
    if (treeFilterInput) {
      treeFilterInput.value = "";
    }

    // 3. Populate new concept nodes
    concepts.forEach((c) => {
      const excerpt = excerpts[c] || "";
      ensureTreeNode(c, sourceMethod, excerpt);
    });

    // 4. Update active selections
    const first = concepts.length > 0 ? concepts[0] : "";
    activeSeedConcept = first;
    selectedFilterConcept = first;
    selectedTreeTarget = { type: "concept", concept: first, association: null };
    if (conceptInput) {
      conceptInput.value = first;
    }

    // 5. Reset scroll to top
    if (treeContainer) {
      treeContainer.scrollTop = 0;
    }

    renderConceptTree();
    updateGeneratePlan();
    updateAcquisitionPlan();
    renderGallery();
  }

  function setSubjectMode(mode) {
    if (mode === "dropdown" && subjectSelect && subjectSelect.options.length > 0) {
      if (subjectInput) subjectInput.style.display = "none";
      if (subjectSelect) subjectSelect.style.display = "block";
      if (btnToggleSubjectMode) {
        btnToggleSubjectMode.style.display = "inline-flex";
        btnToggleSubjectMode.title = "Switch to text input (type custom subject ID)";
      }
    } else {
      if (subjectSelect) subjectSelect.style.display = "none";
      if (subjectInput) subjectInput.style.display = "block";
      const hasMultiple = Boolean(lastUploadedQuestionnaireData?.subjects?.length > 1);
      if (btnToggleSubjectMode) {
        btnToggleSubjectMode.style.display = hasMultiple ? "inline-flex" : "none";
        btnToggleSubjectMode.title = "Switch to questionnaire subjects dropdown";
      }
    }
  }

  function switchSubject(selSub) {
    if (!selSub || selSub === "__custom__") return;
    if (subjectInput && subjectInput.value !== selSub) {
      subjectInput.value = selSub;
    }
    if (subjectSelect && subjectSelect.value !== selSub) {
      for (let i = 0; i < subjectSelect.options.length; i++) {
        if (subjectSelect.options[i].value === selSub) {
          subjectSelect.selectedIndex = i;
          break;
        }
      }
    }

    if (!lastUploadedQuestionnaireData) return;
    const data = lastUploadedQuestionnaireData;

    let subConcepts = [];
    let subExcerpts = {};

    if (selSub === "all") {
      subConcepts = data.all_concepts || [];
      subExcerpts = data.all_excerpts || {};
      if (subConcepts.length === 0) {
        const seen = new Set();
        Object.values(data.concepts_by_subject || {}).forEach((arr) => {
          arr.forEach((c) => {
            if (!seen.has(c)) {
              seen.add(c);
              subConcepts.push(c);
            }
          });
        });
      }
    } else if (data.concepts_by_subject && data.concepts_by_subject[selSub]) {
      subConcepts = data.concepts_by_subject[selSub];
      subExcerpts = (data.excerpts_by_subject && data.excerpts_by_subject[selSub]) || data.rag_excerpts || {};
    }

    if (subConcepts.length > 0) {
      populateTreeWithConcepts(subConcepts, data.method || "questionnaire_rag", subExcerpts);

      if (fileUploadStatus) {
        fileUploadStatus.textContent = `✓ Loaded ${subConcepts.length} concepts (${selSub})`;
        fileUploadStatus.className = "badge-status valid";
      }
      if (fileHint) {
        fileHint.textContent = `Loaded ${subConcepts.length} concepts for ${selSub}`;
      }
      if (generationStatus) {
        generationStatus.textContent = `Ready: ${subConcepts.length} concepts loaded in tree. Click "Generate" to start.`;
      }
    }
  }

  if (subjectSelect) {
    subjectSelect.addEventListener("change", () => {
      if (subjectSelect.value === "__custom__") {
        setSubjectMode("input");
        if (subjectInput) {
          subjectInput.value = "";
          subjectInput.placeholder = "Enter custom subject ID…";
          subjectInput.focus();
        }
        return;
      }
      switchSubject(subjectSelect.value);
    });
  }
  if (subjectInput) {
    subjectInput.addEventListener("change", () => {
      switchSubject(subjectInput.value.trim());
    });
    subjectInput.addEventListener("input", () => {
      const val = subjectInput.value.trim();
      if (lastUploadedQuestionnaireData && lastUploadedQuestionnaireData.concepts_by_subject && lastUploadedQuestionnaireData.concepts_by_subject[val]) {
        switchSubject(val);
      }
    });
  }
  if (btnToggleSubjectMode) {
    btnToggleSubjectMode.addEventListener("click", () => {
      const isDropdown = subjectSelect && subjectSelect.style.display !== "none";
      setSubjectMode(isDropdown ? "input" : "dropdown");
      if (isDropdown && subjectInput) {
        subjectInput.focus();
      }
    });
  }

  function getCheckedConcepts() {
    const checked = Object.keys(workspaceTree).filter((k) => workspaceTree[k].checked !== false);
    if (checked.length > 0) return checked;

    // Fallback to manual concept input
    const raw = conceptInput.value.trim();
    if (raw) {
      const parsed = raw.split(/[\n,]+/).map((s) => s.trim()).filter(Boolean);
      parsed.forEach((c) => ensureTreeNode(c, "manual"));
      return parsed;
    }
    return [];
  }

  function updateGeneratePlan() {
    updateAcquisitionPlan();
  }

  function getTargetDimensions() {
    const rawW = targetWidthInput ? parseInt(targetWidthInput.value, 10) : 160;
    const rawH = targetHeightInput ? parseInt(targetHeightInput.value, 10) : 160;
    const width = Math.min(4096, Math.max(32, Number.isFinite(rawW) && rawW > 0 ? rawW : 160));
    const height = Math.min(4096, Math.max(32, Number.isFinite(rawH) && rawH > 0 ? rawH : 160));
    return { width, height };
  }

  function getImagesPerConcept() {
    const raw = imagesPerConceptSelect ? parseInt(imagesPerConceptSelect.value, 10) : 3;
    return Math.min(30, Math.max(1, Number.isFinite(raw) && raw > 0 ? raw : 3));
  }

  function syncDimensionLabels() {
    const { width, height } = getTargetDimensions();
    if (btnLightboxCropFace && !btnLightboxCropFace.disabled) {
      btnLightboxCropFace.textContent = `Crop Face (${width}×${height})`;
      btnLightboxCropFace.title = `Crop image to detected frontal face (${width}×${height})`;
    }
    updateAcquisitionPlan();
    updateSettingsSummary();
  }

  function updateAcquisitionPlan() {
    const limit = getImagesPerConcept();
    const { width, height } = getTargetDimensions();
    const concepts = Object.keys(workspaceTree);
    const doRelated = genRelatedToggle ? genRelatedToggle.checked : true;
    const doImages = genImagesToggle ? genImagesToggle.checked : true;

    let totalCheckedItems = 0;
    const checkedConcepts = concepts.filter((k) => workspaceTree[k].checked !== false);
    checkedConcepts.forEach((c) => {
      const node = workspaceTree[c];
      const rels = (node.related || []).filter((r) => r.checked !== false);
      totalCheckedItems += (rels.length > 0 ? rels.length : 1);
    });
    const totalAllImages = totalCheckedItems * limit;

    if (btnGenerate && !btnGenerate.disabled) {
      const btnText = btnGenerate.querySelector(".btn-text");
      if (btnText) {
        btnText.textContent = "Generate";
      }
    }

    if (acquisitionPlanHint && !acquisitionRunning) {
      const sizeNote = `min ${width}×${height}px`;
      if (!doRelated && !doImages) {
        acquisitionPlanHint.textContent = "Check 'Related concepts' and/or 'Images' above to generate.";
      } else if (doRelated && doImages) {
        acquisitionPlanHint.innerHTML = `Ready to generate related concepts &amp; images (${limit}/item, ${sizeNote}).`;
      } else if (doRelated) {
        acquisitionPlanHint.innerHTML = `Ready to generate related concepts for <strong>${checkedConcepts.length}</strong> checked concept${checkedConcepts.length === 1 ? "" : "s"}.`;
      } else if (totalCheckedItems > 0) {
        acquisitionPlanHint.innerHTML = `Ready to download <strong>${totalAllImages}</strong> image${totalAllImages === 1 ? "" : "s"} (${limit} per item across <strong>${totalCheckedItems}</strong> checked item${totalCheckedItems === 1 ? "" : "s"}, ${sizeNote}).`;
      } else {
        acquisitionPlanHint.textContent = `Check items in the tree to generate (${limit} images/item, ${sizeNote}).`;
      }
    }
    updateSettingsSummary();
  }

  // ==========================================
  // Dynamic Settings Summary Controller (Shows active selections when collapsed)
  // ==========================================
  function updateSettingsSummary() {
    // 1. Inference Provider & Max Concepts
    if (summaryTextInference) {
      const provider = providerSelect ? providerSelect.value : "local";
      const maxConcepts = maxConceptsInput ? Math.min(50, Math.max(1, parseInt(maxConceptsInput.value, 10) || 5)) : 5;

      let providerName = "Local Llama.cpp";
      const modelPath = localModelPathInput ? localModelPathInput.value.trim() : "";
      if (provider === "local") {
        if (modelPath) {
          const filename = modelPath.split("/").pop().replace(/\.gguf$/i, "");
          providerName = `Local: ${filename}`;
        } else {
          providerName = "Local Llama.cpp";
        }
      } else if (provider === "google") {
        const selModel = getSelectedCloudModel();
        providerName = selModel ? `Gemini (${selModel})` : "Gemini";
      } else if (provider === "openai") {
        const selModel = getSelectedCloudModel();
        providerName = selModel ? `OpenAI (${selModel})` : "OpenAI";
      } else if (provider === "anthropic") {
        const selModel = getSelectedCloudModel();
        providerName = selModel ? `Claude (${selModel})` : "Claude";
      } else if (provider === "compatible") {
        const selModel = getSelectedCloudModel();
        providerName = selModel ? `Compatible (${selModel})` : "OpenAI-Compatible";
      } else if (providerSelect && providerSelect.selectedOptions && providerSelect.selectedOptions[0]) {
        providerName = providerSelect.selectedOptions[0].text.split("—")[0].trim();
      }

      summaryTextInference.textContent = `${providerName} • ${maxConcepts} concepts`;
      if (summaryChipInference) {
        summaryChipInference.title = `Inference Provider: ${providerName}\nMax Concepts: ${maxConcepts}${modelPath ? `\nModel: ${modelPath}` : ""}\nClick to configure in Settings`;
      }
    }

    // 2. Image Source & Count
    if (summaryTextImages) {
      const src = imageSourceSelect ? imageSourceSelect.value : "bing";
      const count = getImagesPerConcept();
      let srcName = "Bing Search";
      let extraDetail = "";

      if (src === "wikimedia") {
        srcName = "Wikimedia";
      } else if (src === "duckduckgo") {
        srcName = "DuckDuckGo";
      } else if (src === "openverse") {
        srcName = "Openverse";
      } else if (src === "bing") {
        const filterVal = bingFilterSelect ? bingFilterSelect.value : "";
        if (filterVal) {
          const filterText = bingFilterSelect.selectedOptions && bingFilterSelect.selectedOptions[0] ? bingFilterSelect.selectedOptions[0].text : filterVal;
          srcName = `Bing (${filterText})`;
          extraDetail = `\nStyle Filter: ${filterText}`;
        } else {
          srcName = "Bing Search";
        }
      } else if (src === "openai") {
        srcName = "OpenAI Images";
      } else if (src === "local") {
        const diffPath = localImageModelPathInput ? localImageModelPathInput.value.trim() : "";
        if (diffPath) {
          const diffFile = diffPath.split("/").pop().replace(/\.safetensors$/i, "");
          srcName = `Local SD: ${diffFile}`;
          extraDetail = `\nCheckpoint: ${diffPath}`;
        } else {
          srcName = "Local Diffusion";
        }
      } else if (imageSourceSelect && imageSourceSelect.selectedOptions && imageSourceSelect.selectedOptions[0]) {
        srcName = imageSourceSelect.selectedOptions[0].text.split("(")[0].trim();
      }

      summaryTextImages.textContent = `${srcName} • ${count}/item`;
      const customFolder = getDownloadFolderPath();
      if (customFolder) {
        extraDetail += `\nSave Folder: ${shortenPath(customFolder, 3)}`;
      }
      if (summaryChipImages) {
        summaryChipImages.title = `Image Source: ${srcName}\nImages per concept: ${count}${extraDetail}\nClick to configure in Settings`;
      }
    }

    // 3. Face Detection
    if (summaryTextFace && summaryChipFace) {
      const faceEnabled = detectFacesToggle ? detectFacesToggle.checked : true;
      const maxFaces = getMaxFaceCount();
      if (faceEnabled) {
        summaryTextFace.textContent = "Face Detection: On";
        summaryChipFace.className = "settings-summary-chip chip-active";
        summaryChipFace.title = `Face Detection: Enabled (detects up to ${maxFaces} frontal faces, labels: ${showFaceRoiLabels ? "On" : "Off"})\nClick to configure in Settings`;
      } else {
        summaryTextFace.textContent = "Face Detection: Off";
        summaryChipFace.className = "settings-summary-chip chip-muted";
        summaryChipFace.title = "Face Detection: Disabled\nClick to configure in Settings";
      }
    }

    // 4. Auto-Crop & Dimensions
    if (summaryTextCrop && summaryChipCrop) {
      const cropEnabled = cropFacesToggle ? cropFacesToggle.checked : false;
      const { width, height } = getTargetDimensions();
      if (cropEnabled) {
        summaryTextCrop.textContent = `Crop: ${width}x${height}`;
        summaryChipCrop.className = "settings-summary-chip chip-active";
        summaryChipCrop.title = `Auto-Crop: Enabled (${width} × ${height} px)\nClick to configure in Settings`;
      } else {
        summaryTextCrop.textContent = "Crop: Off";
        summaryChipCrop.className = "settings-summary-chip chip-muted";
        summaryChipCrop.title = `Auto-Crop: Disabled (web images saved at original aspect ratio, min ${width} × ${height} px)\nClick to configure in Settings`;
      }
    }

    // 5. Semantic Cache Warning (shown only when turned off)
    if (summaryChipCache && summaryTextCache) {
      const cacheEnabled = cacheToggle ? cacheToggle.checked : true;
      if (!cacheEnabled) {
        summaryChipCache.classList.remove("hidden");
        summaryChipCache.style.display = "inline-flex";
        summaryTextCache.textContent = "Cache: Off";
        summaryChipCache.title = "Semantic LRU Cache: Disabled (identical concepts will be re-queried from provider)\nClick to re-enable cache";
      } else {
        summaryChipCache.classList.add("hidden");
        summaryChipCache.style.display = "none";
      }
    }
  }

  if (genRelatedToggle) {
    genRelatedToggle.addEventListener("change", updateAcquisitionPlan);
  }
  if (genImagesToggle) {
    genImagesToggle.addEventListener("change", updateAcquisitionPlan);
  }

  // ==========================================
  // Face ROI Detection & Drawing Helper
  // ==========================================
  function getMaxFaceCount() {
    const raw = maxFacesInput ? parseInt(maxFacesInput.value, 10) : 5;
    return Math.min(10, Math.max(1, Number.isFinite(raw) ? raw : 5));
  }

  async function getFaceRoisForImage(imgObj) {
    if (!imgObj) return { faces: [], count: 0 };
    const key = imgObj.local_path || imgObj.url || imgObj.filename || "";
    if (faceRoiCache.has(key)) {
      return faceRoiCache.get(key);
    }
    try {
      const param = encodeURIComponent(key);
      const maxFaces = getMaxFaceCount();
      const { width, height } = getTargetDimensions();
      const res = await fetch(`/api/v1/images/detect-face?path=${param}&max_faces=${maxFaces}&min_face_size=80&target_width=${width}&target_height=${height}`);
      if (!res.ok) {
        throw new Error(`Face detection failed: HTTP ${res.status}`);
      }
      const data = await res.json();
      faceRoiCache.set(key, data);
      return data;
    } catch (err) {
      console.warn("Face detection error for", key, err);
      const fallback = { faces: [], count: 0, detector: "none" };
      faceRoiCache.set(key, fallback);
      return fallback;
    }
  }

  function drawFaceRoi(canvas, imgEl, faces, isVisible = true) {
    if (!canvas || !imgEl) return;
    const ctx = canvas.getContext("2d");
    const displayWidth = imgEl.clientWidth;
    const displayHeight = imgEl.clientHeight;
    const naturalWidth = imgEl.naturalWidth || displayWidth;
    const naturalHeight = imgEl.naturalHeight || displayHeight;

    if (!displayWidth || !displayHeight || !naturalWidth || !naturalHeight) {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      return;
    }

    const dpr = window.devicePixelRatio || 1;
    canvas.width = displayWidth * dpr;
    canvas.height = displayHeight * dpr;
    canvas.style.width = `${displayWidth}px`;
    canvas.style.height = `${displayHeight}px`;

    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, displayWidth, displayHeight);

    if (!isVisible || !faces || faces.length === 0) {
      return;
    }

    // Determine scale and offset based on computed object-fit
    const fit = window.getComputedStyle(imgEl).objectFit || "cover";
    const imgAspect = naturalWidth / naturalHeight;
    const containerAspect = displayWidth / displayHeight;

    let renderWidth = displayWidth;
    let renderHeight = displayHeight;
    let offsetX = 0;
    let offsetY = 0;

    if (fit === "cover") {
      if (imgAspect > containerAspect) {
        // Image is wider than container: height fills container, sides cropped
        renderHeight = displayHeight;
        renderWidth = displayHeight * imgAspect;
        offsetX = (displayWidth - renderWidth) / 2;
      } else {
        // Image is taller than container: width fills container, top/bottom cropped
        renderWidth = displayWidth;
        renderHeight = displayWidth / imgAspect;
        offsetY = (displayHeight - renderHeight) / 2;
      }
    } else if (fit === "contain") {
      if (imgAspect > containerAspect) {
        // Wider than container: width fills container, top/bottom letterboxed
        renderWidth = displayWidth;
        renderHeight = displayWidth / imgAspect;
        offsetY = (displayHeight - renderHeight) / 2;
      } else {
        // Taller than container: height fills container, left/right pillarboxed
        renderHeight = displayHeight;
        renderWidth = displayHeight * imgAspect;
        offsetX = (displayWidth - renderWidth) / 2;
      }
    }

    const scaleX = renderWidth / naturalWidth;
    const scaleY = renderHeight / naturalHeight;

    // Clip to canvas bounds so ROI does not bleed outside thumbnail rounded corners
    ctx.save();
    ctx.beginPath();
    ctx.rect(0, 0, displayWidth, displayHeight);
    ctx.clip();

    faces.forEach((f, idx) => {
      // Use desktop-style square crop ROI if present
      const crop = f.crop_box;
      const box = crop || f;
      const x = offsetX + (box.x * scaleX);
      const y = offsetY + (box.y * scaleY);
      const w = box.w * scaleX;
      const h = box.h * scaleY;

      // Draw emerald dashed crop outline (mirroring desktop Qt DashDotDotLine)
      ctx.save();
      ctx.setLineDash([5, 4]);
      ctx.strokeStyle = "#10b981";
      ctx.lineWidth = Math.max(1.5, Math.round(displayWidth / 120));
      ctx.strokeRect(x, y, w, h);

      // Light translucent fill inside crop ROI
      ctx.fillStyle = "rgba(16, 185, 129, 0.14)";
      ctx.fillRect(x, y, w, h);

      // Draw corner accent handles like desktop RectROI
      const handleSize = Math.max(4, Math.round(displayWidth / 60));
      ctx.setLineDash([]);
      ctx.fillStyle = "#10b981";
      ctx.fillRect(x, y, handleSize, handleSize);
      ctx.fillRect(x + w - handleSize, y, handleSize, handleSize);
      ctx.fillRect(x, y + h - handleSize, handleSize, handleSize);
      ctx.fillRect(x + w - handleSize, y + h - handleSize, handleSize);
      ctx.restore();

      // Tag header (probability and dimensions, matching desktop: "0.98 WxH:240x240")
      if (showFaceRoiLabels) {
        const confPercent = f.confidence ? Math.round(f.confidence * 100) : 95;
        const tagText = faces.length > 1
          ? `Face ${idx + 1} (${confPercent}%) ${Math.round(box.w)}×${Math.round(box.h)}`
          : `Face ${confPercent}% ${Math.round(box.w)}×${Math.round(box.h)}`;
        ctx.font = "bold 10px system-ui, -apple-system, sans-serif";
        const textWidth = ctx.measureText(tagText).width + 8;
        const tagY = y > 20 ? y - 17 : Math.min(displayHeight - 16, y + h + 2);

        ctx.fillStyle = "#10b981";
        ctx.fillRect(x, tagY, textWidth, 16);

        ctx.fillStyle = "#ffffff";
        ctx.fillText(tagText, x + 4, tagY + 12);
      }
    });

    ctx.restore();
  }

  // ==========================================
  // Lightbox Modal Controller & Face ROI Hooks
  // ==========================================
  async function updateLightboxRoi() {
    if (!lightboxRoiCanvas || !lightboxImg || !currentLightboxImage) return;
    const isVisible = lightboxFaceRoiToggle ? lightboxFaceRoiToggle.checked : showFaceRoi;
    if (!isVisible) {
      drawFaceRoi(lightboxRoiCanvas, lightboxImg, [], false);
      if (lightboxFaceStatus) lightboxFaceStatus.textContent = "";
      return;
    }
    if (lightboxFaceStatus) {
      lightboxFaceStatus.className = "badge-status";
      lightboxFaceStatus.textContent = "Scanning faces…";
    }
    try {
      const data = await getFaceRoisForImage(currentLightboxImage.img);
      if (!currentLightboxImage) return;
      drawFaceRoi(lightboxRoiCanvas, lightboxImg, data.faces, isVisible);
      if (lightboxFaceStatus) {
        if (data.count > 0) {
          lightboxFaceStatus.className = "badge-status valid";
          lightboxFaceStatus.textContent = `✓ ${data.count} face${data.count > 1 ? "s" : ""} detected (${(data.detector || "MTCNN").toUpperCase()})`;
        } else {
          lightboxFaceStatus.className = "badge-status";
          lightboxFaceStatus.textContent = "No face detected";
        }
      }
    } catch {
      if (lightboxFaceStatus) {
        lightboxFaceStatus.className = "badge-status";
        lightboxFaceStatus.textContent = "Face detector ready";
      }
    }
  }

  function openLightbox(img, conceptName) {
    if (!imageLightboxModal || !img) return;
    currentLightboxImage = { img, concept: conceptName };
    if (lightboxImg) lightboxImg.src = img.url || "";
    if (lightboxTitle) lightboxTitle.textContent = img.related_concept || img.concept || img.filename || "Stimulus Image";
    if (lightboxMeta) {
      lightboxMeta.textContent = `${img.filename || "image"} • Source: ${(img.source || "bing").toUpperCase()}${conceptName ? ` • Concept: ${conceptName}` : ""}`;
    }
    if (lightboxFaceRoiToggle) {
      lightboxFaceRoiToggle.checked = showFaceRoi;
    }
    if (lightboxFaceStatus) {
      lightboxFaceStatus.textContent = "";
      lightboxFaceStatus.className = "badge-status";
    }
    imageLightboxModal.classList.remove("hidden");
    if (lightboxImg && lightboxImg.complete) {
      updateLightboxRoi();
    }
  }

  function closeLightbox() {
    if (!imageLightboxModal) return;
    imageLightboxModal.classList.add("hidden");
    if (lightboxRoiCanvas) {
      const ctx = lightboxRoiCanvas.getContext("2d");
      ctx.clearRect(0, 0, lightboxRoiCanvas.width, lightboxRoiCanvas.height);
    }
    if (lightboxImg) lightboxImg.src = "";
    currentLightboxImage = null;
  }

  if (lightboxImg) {
    lightboxImg.onload = () => {
      updateLightboxRoi();
    };
  }

  // Checkbox synchronization for Face Detection & ROI
  function setFaceDetectionEnabled(enabled) {
    showFaceRoi = !!enabled;
    if (detectFacesToggle) detectFacesToggle.checked = showFaceRoi;
    if (toggleFaceRoi) toggleFaceRoi.checked = showFaceRoi;
    if (lightboxFaceRoiToggle) lightboxFaceRoiToggle.checked = showFaceRoi;
    if (subsettingsFaceCard) subsettingsFaceCard.classList.toggle("active", showFaceRoi);
    if (detectFacesStatusBadge) {
      detectFacesStatusBadge.textContent = showFaceRoi ? "On" : "Off";
      detectFacesStatusBadge.classList.toggle("valid", showFaceRoi);
      detectFacesStatusBadge.classList.toggle("muted", !showFaceRoi);
    }
    if (maxFacesInput) maxFacesInput.disabled = !showFaceRoi;
    if (showFaceRoiLabelsToggle) showFaceRoiLabelsToggle.disabled = !showFaceRoi;
    renderGallery();
    if (currentLightboxImage) {
      updateLightboxRoi();
    }
    updateSettingsSummary();
  }

  function updateCropCardState() {
    const cropEnabled = cropFacesToggle ? cropFacesToggle.checked : false;
    if (subsettingsCropCard) subsettingsCropCard.classList.toggle("active", cropEnabled);
    if (cropFacesStatusBadge) {
      cropFacesStatusBadge.textContent = cropEnabled ? "On" : "Off";
      cropFacesStatusBadge.classList.toggle("valid", cropEnabled);
      cropFacesStatusBadge.classList.toggle("muted", !cropEnabled);
    }
  }

  // Restore saved max faces and ROI label toggle
  const savedMaxFaces = parseInt(localStorage.getItem("sf_max_faces"), 10);
  if (maxFacesInput && Number.isFinite(savedMaxFaces) && savedMaxFaces >= 1 && savedMaxFaces <= 10) {
    maxFacesInput.value = savedMaxFaces;
  }
  if (showFaceRoiLabelsToggle) {
    showFaceRoiLabelsToggle.checked = showFaceRoiLabels;
    showFaceRoiLabelsToggle.addEventListener("change", () => {
      showFaceRoiLabels = showFaceRoiLabelsToggle.checked;
      localStorage.setItem("sf_show_face_roi_labels", showFaceRoiLabels ? "1" : "0");
      renderGallery();
      if (currentLightboxImage) {
        updateLightboxRoi();
      }
      updateSettingsSummary();
    });
  }

  if (maxFacesInput) {
    maxFacesInput.addEventListener("change", () => {
      const val = getMaxFaceCount();
      maxFacesInput.value = val;
      localStorage.setItem("sf_max_faces", val);
      faceRoiCache.clear();
      renderGallery();
      if (currentLightboxImage) {
        updateLightboxRoi();
      }
      updateSettingsSummary();
    });
  }

  if (detectFacesToggle) {
    detectFacesToggle.addEventListener("change", () => {
      setFaceDetectionEnabled(detectFacesToggle.checked);
    });
  }

  if (cropFacesToggle) {
    cropFacesToggle.addEventListener("change", () => {
      if (cropFacesToggle.checked && detectFacesToggle && !detectFacesToggle.checked) {
        setFaceDetectionEnabled(true);
      }
      updateCropCardState();
      updateAcquisitionPlan();
      updateSettingsSummary();
    });
  }

  if (targetWidthInput) {
    targetWidthInput.addEventListener("input", syncDimensionLabels);
    targetWidthInput.addEventListener("change", () => {
      const { width } = getTargetDimensions();
      targetWidthInput.value = width;
      syncDimensionLabels();
    });
  }

  if (targetHeightInput) {
    targetHeightInput.addEventListener("input", syncDimensionLabels);
    targetHeightInput.addEventListener("change", () => {
      const { height } = getTargetDimensions();
      targetHeightInput.value = height;
      syncDimensionLabels();
    });
  }

  if (toggleFaceRoi) {
    toggleFaceRoi.addEventListener("change", () => {
      setFaceDetectionEnabled(toggleFaceRoi.checked);
    });
  }

  if (lightboxFaceRoiToggle) {
    lightboxFaceRoiToggle.addEventListener("change", () => {
      setFaceDetectionEnabled(lightboxFaceRoiToggle.checked);
    });
  }

  if (btnLightboxCropFace) {
    btnLightboxCropFace.addEventListener("click", async () => {
      if (!currentLightboxImage || !currentLightboxImage.img) return;
      const img = currentLightboxImage.img;
      const { width, height } = getTargetDimensions();
      btnLightboxCropFace.disabled = true;
      btnLightboxCropFace.textContent = "Cropping…";
      try {
        const res = await fetch("/api/v1/images/crop-face", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            image_path: img.local_path || img.url || img.filename,
            target_width: width,
            target_height: height,
          }),
        });
        if (!res.ok) {
          const err = await res.json().catch(() => ({}));
          throw new Error(err.detail || "Failed to auto-crop face.");
        }
        const data = await res.json();
        if (data.status === "cropped" && data.url) {
          faceRoiCache.delete(img.local_path);
          faceRoiCache.delete(img.filename);
          faceRoiCache.delete(img.url);
          const newUrl = `${data.url.split("?")[0]}?t=${Date.now()}`;
          img.url = newUrl;
          if (data.path) img.filename = data.path.split("/").pop();
          if (lightboxImg) lightboxImg.src = newUrl;
          if (lightboxFaceStatus) {
            lightboxFaceStatus.className = "badge-status valid";
            lightboxFaceStatus.textContent = `✓ Face cropped (${width}×${height})`;
          }
          renderConceptTree();
          renderGallery();
        } else {
          alert(data.message || "No frontal face detected to crop.");
        }
      } catch (err) {
        alert(`Face crop error: ${err.message}`);
      } finally {
        btnLightboxCropFace.disabled = false;
        syncDimensionLabels();
      }
    });
  }

  if (btnCloseLightbox) {
    btnCloseLightbox.addEventListener("click", closeLightbox);
  }

  if (imageLightboxModal) {
    imageLightboxModal.addEventListener("click", (e) => {
      if (e.target === imageLightboxModal) {
        closeLightbox();
      }
    });
  }

  // ==========================================
  // Help & User Guide Controller
  // ==========================================
  function openHelpGuide(tabName = "quickstart") {
    if (!helpGuideModal) return;
    helpGuideModal.classList.remove("hidden");
    switchHelpTab(tabName);
  }

  function closeHelpGuide() {
    if (!helpGuideModal) return;
    helpGuideModal.classList.add("hidden");
  }

  function switchHelpTab(tabName) {
    helpTabBtns.forEach((btn) => {
      btn.classList.toggle("active", btn.dataset.tab === tabName);
    });
    helpTabPanes.forEach((pane) => {
      pane.classList.toggle("active", pane.id === `pane-${tabName}`);
    });
  }

  if (btnOpenHelp) {
    btnOpenHelp.addEventListener("click", () => openHelpGuide());
  }

  if (btnCloseHelp) {
    btnCloseHelp.addEventListener("click", closeHelpGuide);
  }

  if (btnCloseHelpFooter) {
    btnCloseHelpFooter.addEventListener("click", closeHelpGuide);
  }

  if (helpGuideModal) {
    helpGuideModal.addEventListener("click", (e) => {
      if (e.target === helpGuideModal) {
        closeHelpGuide();
      }
    });
  }

  helpTabBtns.forEach((btn) => {
    btn.addEventListener("click", () => {
      const tab = btn.dataset.tab;
      if (tab) switchHelpTab(tab);
    });
  });

  window.addEventListener("keydown", (e) => {
    if (e.key === "Escape") {
      closeLightbox();
      closeContextMenu();
      closeHelpGuide();
      closeAddRelatedModal();
    }
  });

  if (btnLightboxDelete) {
    btnLightboxDelete.addEventListener("click", () => {
      if (!currentLightboxImage || !currentLightboxImage.img || !currentLightboxImage.concept) return;
      const conceptName = currentLightboxImage.concept;
      const node = workspaceTree[conceptName];
      if (node && node.images) {
        node.images = node.images.filter((i) => i !== currentLightboxImage.img);
      }
      closeLightbox();
      renderConceptTree();
      renderGallery();
      updateAcquisitionPlan();
    });
  }

  // ==========================================
  // Add Related Concept Modal & Logic
  // ==========================================
  function openAddRelatedModal(concept) {
    if (!concept) return;
    addRelatedTargetConcept = concept;
    if (addRelatedSubtitle) {
      addRelatedSubtitle.textContent = `Add custom related concepts to "${concept}"`;
    }
    if (addRelatedInput) {
      addRelatedInput.value = "";
    }
    if (addRelatedModal) {
      addRelatedModal.classList.remove("hidden");
      setTimeout(() => {
        if (addRelatedInput) addRelatedInput.focus();
      }, 50);
    }
  }

  function closeAddRelatedModal() {
    if (addRelatedModal) {
      addRelatedModal.classList.add("hidden");
    }
    addRelatedTargetConcept = null;
  }

  function addRelatedConcepts(concept, rawInput) {
    if (!concept || !rawInput) return 0;
    const node = ensureTreeNode(concept, "manual");
    if (!node) return 0;
    if (!Array.isArray(node.related)) {
      node.related = [];
    }

    const items = rawInput
      .split(/[,;\n]/)
      .map((s) => s.trim())
      .filter((s) => s.length > 0);

    let addedCount = 0;
    let lastAddedName = null;
    items.forEach((name) => {
      // Don't add parent concept itself or duplicate associations
      if (name.toLowerCase() === concept.toLowerCase()) return;
      const alreadyExists = node.related.some(
        (r) => r.name.toLowerCase() === name.toLowerCase() || (r.char && r.char.toLowerCase() === name.toLowerCase())
      );
      if (alreadyExists) return;

      node.related.push({
        name: name,
        char: "",
        description: "Manually added association",
        checked: true,
        expanded: false,
      });
      addedCount++;
      lastAddedName = name;
    });

    if (addedCount > 0) {
      node.expanded = true; // Auto-expand parent node to reveal new items
      if (lastAddedName) {
        selectedTreeTarget = { type: "association", concept: concept, association: lastAddedName };
        activeSeedConcept = concept;
        selectedFilterConcept = lastAddedName;
      }
      renderConceptTree();
      renderGallery();
      updateGeneratePlan();
      updateAcquisitionPlan();
      setStatus(`Added ${addedCount} related concept${addedCount > 1 ? "s" : ""} to "${concept}".`);
    }

    return addedCount;
  }

  if (btnCloseAddRelated) {
    btnCloseAddRelated.addEventListener("click", closeAddRelatedModal);
  }
  if (btnCancelAddRelated) {
    btnCancelAddRelated.addEventListener("click", closeAddRelatedModal);
  }
  if (addRelatedModal) {
    addRelatedModal.addEventListener("click", (e) => {
      if (e.target === addRelatedModal) {
        closeAddRelatedModal();
      }
    });
  }
  if (addRelatedForm) {
    addRelatedForm.addEventListener("submit", (e) => {
      e.preventDefault();
      if (!addRelatedTargetConcept) {
        closeAddRelatedModal();
        return;
      }
      const val = addRelatedInput ? addRelatedInput.value.trim() : "";
      if (val) {
        addRelatedConcepts(addRelatedTargetConcept, val);
      }
      closeAddRelatedModal();
    });
  }

  // ==========================================
  // Edit Context / Category Modal & Logic
  // ==========================================
  function openEditContextModal(concept) {
    if (!concept) return;
    editContextTargetConcept = concept;
    const node = workspaceTree[concept];
    const currentExcerpt = node ? (node.excerpt || "") : "";
    if (editContextSubtitle) {
      editContextSubtitle.textContent = `Set category or disambiguation hint for "${concept}"`;
    }
    if (editContextInput) {
      editContextInput.value = currentExcerpt;
    }
    if (editContextModal) {
      editContextModal.classList.remove("hidden");
      setTimeout(() => {
        if (editContextInput) {
          editContextInput.focus();
          editContextInput.select();
        }
      }, 50);
    }
  }

  function closeEditContextModal() {
    if (editContextModal) {
      editContextModal.classList.add("hidden");
    }
    editContextTargetConcept = null;
  }

  if (btnCloseEditContext) {
    btnCloseEditContext.addEventListener("click", closeEditContextModal);
  }
  if (btnCancelEditContext) {
    btnCancelEditContext.addEventListener("click", closeEditContextModal);
  }
  if (editContextBackdrop) {
    editContextBackdrop.addEventListener("click", closeEditContextModal);
  }
  if (editContextModal) {
    editContextModal.addEventListener("click", (e) => {
      if (e.target === editContextModal) {
        closeEditContextModal();
      }
    });
  }
  if (editContextForm) {
    editContextForm.addEventListener("submit", (e) => {
      e.preventDefault();
      if (!editContextTargetConcept) {
        closeEditContextModal();
        return;
      }
      const val = editContextInput ? editContextInput.value.trim() : "";
      const node = ensureTreeNode(editContextTargetConcept, "manual");
      if (node) {
        node.excerpt = val;
      }
      closeEditContextModal();
      renderConceptTree();
      setStatus(`Updated context for "${editContextTargetConcept}" to "${val || '(empty)'}".`);
    });
  }

  // ==========================================
  // Tree Context Menu Controller
  // ==========================================
  function openContextMenu(e, type, concept, relItem = null, img = null) {
    e.preventDefault();
    e.stopPropagation();
    if (!treeContextMenu) return;
    contextMenuTarget = { type, concept, relItem, img };

    // Dynamic visibility & labels based on target type
    if (ctxAcquire) ctxAcquire.style.display = type === "image" ? "none" : "";
    if (ctxAcquireText) ctxAcquireText.style.display = type === "image" ? "none" : "";
    if (ctxGenerate) {
      ctxGenerate.style.display = type === "concept" ? "" : "none";
    }
    if (ctxEditContext) {
      ctxEditContext.style.display = type === "concept" ? "" : "none";
    }
    if (ctxAddRelated) {
      if (type === "concept") {
        ctxAddRelated.style.display = "";
        ctxAddRelated.textContent = "➕ Add Related Concept…";
      } else if (type === "association") {
        ctxAddRelated.style.display = "";
        ctxAddRelated.textContent = `➕ Add Related Concept to "${concept}"…`;
      } else {
        ctxAddRelated.style.display = "none";
      }
    }
    if (ctxClearImages) {
      ctxClearImages.style.display = type === "image" ? "none" : "";
    }
    if (ctxDelete) {
      if (type === "image") {
        ctxDelete.textContent = "🗑 Delete Image";
      } else if (type === "association" && relItem) {
        ctxDelete.textContent = `🗑 Delete "${relItem.name}"`;
      } else {
        ctxDelete.textContent = `🗑 Delete "${concept}"`;
      }
    }

    const x = Math.min(e.clientX, window.innerWidth - 240);
    const y = Math.min(e.clientY, window.innerHeight - 260);
    treeContextMenu.style.left = `${Math.max(8, x)}px`;
    treeContextMenu.style.top = `${Math.max(8, y)}px`;
    treeContextMenu.classList.remove("hidden");
  }

  function closeContextMenu() {
    if (treeContextMenu && !treeContextMenu.classList.contains("hidden")) {
      treeContextMenu.classList.add("hidden");
      contextMenuTarget = null;
    }
  }

  window.addEventListener("click", closeContextMenu);

  if (treeContextMenu) {
    treeContextMenu.querySelectorAll(".context-menu-item").forEach((item) => {
      item.addEventListener("click", (e) => {
        e.stopPropagation();
        const action = item.getAttribute("data-action");
        if (!contextMenuTarget) {
          closeContextMenu();
          return;
        }

        const { type, concept, relItem, img } = contextMenuTarget;
        closeContextMenu();

        if (action === "add-related") {
          openAddRelatedModal(concept);
        } else if (action === "edit-context") {
          openEditContextModal(concept);
        } else if (action === "acquire") {
          if (type === "association" && relItem) {
            selectedTreeTarget = { type: "association", concept: concept, association: relItem.name };
            activeSeedConcept = concept;
            selectedFilterConcept = relItem.name;
          } else {
            selectedTreeTarget = { type: "concept", concept: concept, association: null };
            activeSeedConcept = concept;
            selectedFilterConcept = concept;
          }
          updateAcquisitionPlan();
          renderConceptTree();
          acquireForTarget(concept, type === "association" ? relItem : null);
        } else if (action === "acquire-text") {
          if (type === "association" && relItem) {
            selectedTreeTarget = { type: "association", concept: concept, association: relItem.name };
            activeSeedConcept = concept;
            selectedFilterConcept = relItem.name;
          } else {
            selectedTreeTarget = { type: "concept", concept: concept, association: null };
            activeSeedConcept = concept;
            selectedFilterConcept = concept;
          }
          updateAcquisitionPlan();
          renderConceptTree();
          acquireForTarget(concept, type === "association" ? relItem : null, "text");
        } else if (action === "generate") {
          activeSeedConcept = concept;
          selectedTreeTarget = { type: "concept", concept: concept, association: null };
          conceptInput.value = concept;
          updateAcquisitionPlan();
          renderConceptTree();
          runGenerateRelatedConcepts([concept]);
        } else if (action === "clear-images") {
          const node = workspaceTree[concept];
          if (node && node.images) {
            if (type === "association" && relItem) {
              node.images = node.images.filter(
                (i) => !i.related_concept || i.related_concept.toLowerCase() !== relItem.name.toLowerCase()
              );
            } else {
              node.images = [];
            }
          }
          renderConceptTree();
          renderGallery();
          updateAcquisitionPlan();
        } else if (action === "delete") {
          if (type === "image" && img) {
            const node = workspaceTree[concept];
            if (node && node.images) {
              node.images = node.images.filter((i) => i !== img);
            }
          } else if (type === "association" && relItem) {
            const node = workspaceTree[concept];
            if (node) {
              node.related = (node.related || []).filter((r) => r !== relItem);
              node.images = (node.images || []).filter(
                (i) => !i.related_concept || i.related_concept.toLowerCase() !== relItem.name.toLowerCase()
              );
            }
          } else if (type === "concept") {
            delete workspaceTree[concept];
            if (activeSeedConcept.toLowerCase() === concept.toLowerCase()) {
              const remaining = Object.keys(workspaceTree);
              activeSeedConcept = remaining.length > 0 ? remaining[0] : "";
              selectedFilterConcept = activeSeedConcept;
              conceptInput.value = activeSeedConcept;
              selectedTreeTarget = { type: "concept", concept: activeSeedConcept, association: null };
            }
          }
          renderConceptTree();
          renderGallery();
          updateAcquisitionPlan();
        }
      });
    });
  }

  function syncTreeHeaderControls() {
    const concepts = Object.keys(workspaceTree);
    if (treeSelectAllCheckbox) {
      if (concepts.length === 0) {
        treeSelectAllCheckbox.checked = false;
        treeSelectAllCheckbox.indeterminate = false;
        treeSelectAllCheckbox.disabled = true;
      } else {
        treeSelectAllCheckbox.disabled = false;
        let totalItems = 0;
        let checkedItems = 0;
        concepts.forEach((c) => {
          const node = workspaceTree[c];
          totalItems++;
          if (node.checked !== false) checkedItems++;
          (node.related || []).forEach((r) => {
            totalItems++;
            if (r.checked !== false) checkedItems++;
          });
        });
        if (checkedItems === 0) {
          treeSelectAllCheckbox.checked = false;
          treeSelectAllCheckbox.indeterminate = false;
        } else if (checkedItems === totalItems) {
          treeSelectAllCheckbox.checked = true;
          treeSelectAllCheckbox.indeterminate = false;
        } else {
          treeSelectAllCheckbox.checked = false;
          treeSelectAllCheckbox.indeterminate = true;
        }
      }
    }

    if (btnToggleExpandTree) {
      if (concepts.length === 0) {
        btnToggleExpandTree.disabled = true;
      } else {
        btnToggleExpandTree.disabled = false;
        const anyExpanded = concepts.some((c) => workspaceTree[c].expanded);
        const icon = btnToggleExpandTree.querySelector(".expand-toggle-icon");
        const text = btnToggleExpandTree.querySelector(".expand-toggle-text");
        if (icon) icon.textContent = anyExpanded ? "⤡" : "⤢";
        if (text) text.textContent = anyExpanded ? "Collapse" : "Expand";
        btnToggleExpandTree.title = anyExpanded ? "Collapse all tree nodes" : "Expand all tree nodes";
      }
    }

    if (btnClearTree) {
      btnClearTree.disabled = (concepts.length === 0);
    }
  }

  /**
   * Render the hierarchical concept tree matching the desktop QTreeView.
   * Level 1: Concept
   * Level 2: Associated entities
   * Level 3: Visual Stimuli images
   */
  function renderConceptTree() {
    const filter = (treeFilterInput ? treeFilterInput.value.trim().toLowerCase() : "");
    const concepts = Object.keys(workspaceTree);
    const totalCount = concepts.length;

    if (treeTotalBadge) {
      treeTotalBadge.textContent = totalCount;
    }

    syncTreeHeaderControls();

    const currentScroll = treeContainer ? treeContainer.scrollTop : 0;

    if (totalCount === 0) {
      treeContainer.innerHTML = `
        <div class="tree-empty-state" id="tree-empty-state">
          <p>No concepts loaded.</p>
          <p class="text-xs text-muted">Upload a concept file (.txt, .csv, .xlsx, .docx) or type a seed concept above.</p>
        </div>
      `;
      updateGeneratePlan();
      updateAcquisitionPlan();
      renderGallery();
      return;
    }

    treeContainer.innerHTML = "";

    concepts.forEach((concept) => {
      const node = workspaceTree[concept];
      const relatedCount = (node.related || []).length;
      const imageCount = (node.images || []).length;

      // Filter logic
      const matchesConcept = concept.toLowerCase().includes(filter);
      const matchesChild = (node.related || []).some((r) =>
        r.name.toLowerCase().includes(filter) || (r.char && r.char.toLowerCase().includes(filter))
      );
      if (filter && !matchesConcept && !matchesChild) {
        return;
      }

      const isConceptSelected = (
        selectedTreeTarget.type === "concept" &&
        selectedTreeTarget.concept.toLowerCase() === concept.toLowerCase()
      );

      // Parent container
      const parentEl = document.createElement("div");
      parentEl.className = `tree-node-parent ${imageCount > 0 ? "has-images" : ""}`;

      // Parent Row
      const rowEl = document.createElement("div");
      rowEl.className = `tree-node-row ${isConceptSelected ? "active" : ""}`;

      // Level 1 Chevron
      const chevron = document.createElement("span");
      chevron.className = `tree-chevron ${node.expanded ? "expanded" : ""}`;
      chevron.textContent = "▶";
      chevron.title = node.expanded ? "Collapse" : "Expand";
      chevron.onclick = (e) => {
        e.stopPropagation();
        node.expanded = !node.expanded;
        renderConceptTree();
      };

      // Checkbox (cascades to all associations and images)
      const checkbox = document.createElement("input");
      checkbox.type = "checkbox";
      checkbox.checked = !!node.checked;
      checkbox.title = "Select for generation & image acquisition";
      checkbox.onclick = (e) => {
        e.stopPropagation();
        node.checked = checkbox.checked;
        (node.related || []).forEach((r) => { r.checked = checkbox.checked; });
        (node.images || []).forEach((img) => { img.checked = checkbox.checked; });
        updateGeneratePlan();
        updateAcquisitionPlan();
        renderConceptTree();
      };

      // Label
      const label = document.createElement("span");
      label.className = "tree-node-label";
      label.textContent = concept;
      if (node.excerpt) {
        const ctxBadge = document.createElement("span");
        ctxBadge.className = "badge-status rag text-xs";
        ctxBadge.style.marginLeft = "0.35rem";
        ctxBadge.style.cursor = "pointer";
        const shortHint = node.excerpt.length > 25 ? `${node.excerpt.slice(0, 23)}…` : node.excerpt;
        ctxBadge.textContent = `🏷️ ${shortHint}`;
        ctxBadge.title = `Context: "${node.excerpt}" (click to edit)`;
        ctxBadge.onclick = (e) => {
          e.stopPropagation();
          openEditContextModal(concept);
        };
        label.appendChild(ctxBadge);
      } else if (node.sourceMethod === "agentic_rag" || node.sourceMethod === "questionnaire_rag") {
        const ragTag = document.createElement("span");
        ragTag.className = "badge-status rag text-xs";
        ragTag.style.marginLeft = "0.35rem";
        ragTag.style.cursor = "pointer";
        ragTag.textContent = node.sourceMethod === "questionnaire_rag" ? "Survey" : "RAG";
        ragTag.title = "Extracted concept (click to add context / category)";
        ragTag.onclick = (e) => {
          e.stopPropagation();
          openEditContextModal(concept);
        };
        label.appendChild(ragTag);
      }

      // Count badge
      const countBadge = document.createElement("span");
      countBadge.className = `badge-count ${imageCount > 0 ? "success" : ""}`;
      countBadge.textContent = `${relatedCount} rel • ${imageCount} img`;

      // Actions container
      const actionsContainer = document.createElement("div");
      actionsContainer.className = "tree-node-actions";

      // Quick Acquire for this concept
      const quickAcquireBtn = document.createElement("button");
      quickAcquireBtn.type = "button";
      quickAcquireBtn.className = "tree-action-btn";
      quickAcquireBtn.textContent = "📸";
      quickAcquireBtn.title = `Acquire images for "${concept}"`;
      quickAcquireBtn.onclick = (e) => {
        e.stopPropagation();
        selectedTreeTarget = { type: "concept", concept: concept, association: null };
        activeSeedConcept = concept;
        selectedFilterConcept = concept;
        updateAcquisitionPlan();
        renderConceptTree();
        acquireForTarget(concept);
      };
      actionsContainer.appendChild(quickAcquireBtn);

      // Edit Context / Category Quick Button
      const editContextBtn = document.createElement("button");
      editContextBtn.type = "button";
      editContextBtn.className = "tree-action-btn";
      editContextBtn.textContent = "🏷️";
      editContextBtn.title = node.excerpt ? `Context: "${node.excerpt}" (click to edit)` : `Set context / category for "${concept}"`;
      editContextBtn.onclick = (e) => {
        e.stopPropagation();
        openEditContextModal(concept);
      };
      actionsContainer.appendChild(editContextBtn);

      // Add Related Concept Quick Button
      const addRelBtn = document.createElement("button");
      addRelBtn.type = "button";
      addRelBtn.className = "tree-action-btn";
      addRelBtn.textContent = "➕";
      addRelBtn.title = `Add related concept to "${concept}"`;
      addRelBtn.onclick = (e) => {
        e.stopPropagation();
        openAddRelatedModal(concept);
      };
      actionsContainer.appendChild(addRelBtn);

      // Delete Button
      const delBtn = document.createElement("button");
      delBtn.type = "button";
      delBtn.className = "tree-node-delete";
      delBtn.textContent = "✕";
      delBtn.title = `Remove "${concept}" from tree`;
      delBtn.onclick = (e) => {
        e.stopPropagation();
        delete workspaceTree[concept];
        if (activeSeedConcept.toLowerCase() === concept.toLowerCase()) {
          const remaining = Object.keys(workspaceTree);
          activeSeedConcept = remaining.length > 0 ? remaining[0] : "";
          selectedFilterConcept = activeSeedConcept;
          conceptInput.value = activeSeedConcept;
          selectedTreeTarget = { type: "concept", concept: activeSeedConcept, association: null };
        }
        renderConceptTree();
        renderGallery();
        updateAcquisitionPlan();
      };
      actionsContainer.appendChild(delBtn);

      // Row Click Selection
      rowEl.onclick = () => {
        selectedTreeTarget = { type: "concept", concept: concept, association: null };
        activeSeedConcept = concept;
        selectedFilterConcept = concept;
        conceptInput.value = concept;
        updateAcquisitionPlan();
        renderConceptTree();
        renderGallery();
      };

      // Row Double Click Expand/Collapse
      rowEl.ondblclick = (e) => {
        e.stopPropagation();
        node.expanded = !node.expanded;
        renderConceptTree();
      };

      // Row Right Click Context Menu
      rowEl.oncontextmenu = (e) => {
        openContextMenu(e, "concept", concept);
      };

      rowEl.appendChild(chevron);
      rowEl.appendChild(checkbox);
      rowEl.appendChild(label);
      rowEl.appendChild(countBadge);
      rowEl.appendChild(actionsContainer);
      parentEl.appendChild(rowEl);

      // Children Container (Level 2 Associations & Level 3 Images)
      if (node.expanded && (relatedCount > 0 || imageCount > 0)) {
        const childrenContainer = document.createElement("div");
        childrenContainer.className = "tree-children-container";

        (node.related || []).forEach((relItem) => {
          if (filter && !matchesConcept && !relItem.name.toLowerCase().includes(filter)) {
            return;
          }

          const isAssocSelected = (
            selectedTreeTarget.type === "association" &&
            selectedTreeTarget.concept.toLowerCase() === concept.toLowerCase() &&
            selectedTreeTarget.association &&
            selectedTreeTarget.association.toLowerCase() === relItem.name.toLowerCase()
          );

          const childRow = document.createElement("div");
          childRow.className = `tree-child-row ${isAssocSelected ? "active" : ""}`;

          // Filter images belonging to this association
          const relImages = (node.images || []).filter((img) =>
            img.related_concept && img.related_concept.toLowerCase() === relItem.name.toLowerCase()
          );

          // Level 2 Chevron for expanding Level 3 Images
          if (relImages.length > 0) {
            const relChevron = document.createElement("span");
            relChevron.className = `tree-chevron ${relItem.expanded ? "expanded" : ""}`;
            relChevron.textContent = "▶";
            relChevron.title = relItem.expanded ? "Collapse images" : `Expand ${relImages.length} images`;
            relChevron.onclick = (e) => {
              e.stopPropagation();
              relItem.expanded = !relItem.expanded;
              renderConceptTree();
            };
            childRow.appendChild(relChevron);
          } else {
            const relSpacer = document.createElement("span");
            relSpacer.style.width = "16px";
            relSpacer.style.height = "16px";
            relSpacer.style.display = "inline-block";
            relSpacer.style.flexShrink = "0";
            childRow.appendChild(relSpacer);
          }

          // Association Checkbox
          const childCheck = document.createElement("input");
          childCheck.type = "checkbox";
          childCheck.checked = (relItem.checked !== false);
          childCheck.onclick = (e) => {
            e.stopPropagation();
            relItem.checked = childCheck.checked;
            updateAcquisitionPlan();
          };

          const childName = document.createElement("span");
          childName.className = "tree-child-name";
          childName.textContent = relItem.name;

          childRow.appendChild(childCheck);
          childRow.appendChild(childName);

          if (relItem.char) {
            const childRole = document.createElement("span");
            childRole.className = "tree-child-role";
            childRole.textContent = relItem.char;
            childRow.appendChild(childRole);
          }

          // Green badge if images acquired
          if (relImages.length > 0) {
            const imgBadge = document.createElement("span");
            imgBadge.className = "badge-count success";
            imgBadge.style.fontSize = "0.62rem";
            imgBadge.textContent = `✓ ${relImages.length} img`;
            childRow.appendChild(imgBadge);
          }

          // Association Actions (Quick acquire, delete)
          const assocActions = document.createElement("div");
          assocActions.className = "tree-node-actions";

          const assocQuickBtn = document.createElement("button");
          assocQuickBtn.type = "button";
          assocQuickBtn.className = "tree-action-btn";
          assocQuickBtn.textContent = "📸";
          assocQuickBtn.title = `Acquire images for "${relItem.name}"`;
          assocQuickBtn.onclick = (e) => {
            e.stopPropagation();
            selectedTreeTarget = { type: "association", concept: concept, association: relItem.name };
            activeSeedConcept = concept;
            selectedFilterConcept = relItem.name;
            updateAcquisitionPlan();
            renderConceptTree();
            acquireForTarget(concept, relItem);
          };
          assocActions.appendChild(assocQuickBtn);

          const assocDelBtn = document.createElement("button");
          assocDelBtn.type = "button";
          assocDelBtn.className = "tree-node-delete";
          assocDelBtn.textContent = "✕";
          assocDelBtn.title = `Remove "${relItem.name}"`;
          assocDelBtn.onclick = (e) => {
            e.stopPropagation();
            node.related = (node.related || []).filter((r) => r !== relItem);
            node.images = (node.images || []).filter(
              (img) => !img.related_concept || img.related_concept.toLowerCase() !== relItem.name.toLowerCase()
            );
            renderConceptTree();
            renderGallery();
            updateAcquisitionPlan();
          };
          assocActions.appendChild(assocDelBtn);

          childRow.appendChild(assocActions);

          // Association Row Click: Select & Filter Gallery
          childRow.onclick = (e) => {
            e.stopPropagation();
            selectedTreeTarget = { type: "association", concept: concept, association: relItem.name };
            activeSeedConcept = concept;
            selectedFilterConcept = relItem.name;
            updateAcquisitionPlan();
            renderConceptTree();
            renderGallery();
          };

          // Association Double Click: Toggle Level 3 Image expansion
          childRow.ondblclick = (e) => {
            e.stopPropagation();
            if (relImages.length > 0) {
              relItem.expanded = !relItem.expanded;
              renderConceptTree();
            }
          };

          // Association Right Click Context Menu
          childRow.oncontextmenu = (e) => {
            openContextMenu(e, "association", concept, relItem);
          };

          childrenContainer.appendChild(childRow);

          // Level 3: Images Nested under Association
          if (relItem.expanded && relImages.length > 0) {
            const imageContainer = document.createElement("div");
            imageContainer.className = "tree-image-container";

            relImages.forEach((img) => {
              const isImgSelected = (
                selectedTreeTarget.type === "image" &&
                selectedTreeTarget.img === img
              );
              const imgRow = document.createElement("div");
              imgRow.className = `tree-image-row ${isImgSelected ? "active" : ""}`;

              const thumb = document.createElement("img");
              thumb.className = "tree-image-thumb";
              thumb.src = img.url || "";
              thumb.alt = img.filename || "stimulus";
              thumb.loading = "lazy";

              const imgName = document.createElement("span");
              imgName.className = "tree-image-name";
              imgName.textContent = img.filename || "image";

              const srcBadge = document.createElement("span");
              srcBadge.className = "tree-image-src-badge";
              srcBadge.textContent = (img.source || "SRC").toUpperCase();

              const delImgBtn = document.createElement("button");
              delImgBtn.type = "button";
              delImgBtn.className = "tree-node-delete";
              delImgBtn.textContent = "✕";
              delImgBtn.title = "Delete image";
              delImgBtn.onclick = (e) => {
                e.stopPropagation();
                node.images = (node.images || []).filter((i) => i !== img);
                updateAcquisitionPlan();
                renderConceptTree();
                renderGallery();
              };

              imgRow.appendChild(thumb);
              imgRow.appendChild(imgName);
              imgRow.appendChild(srcBadge);
              imgRow.appendChild(delImgBtn);

              // Single click selects image as gallery target
              imgRow.onclick = (e) => {
                e.stopPropagation();
                selectedTreeTarget = {
                  type: "image",
                  concept: concept,
                  association: relItem ? relItem.name : null,
                  img: img,
                };
                activeSeedConcept = concept;
                selectedFilterConcept = relItem ? relItem.name : concept;
                renderConceptTree();
                renderGallery();
              };

              // Double click opens Lightbox viewer
              imgRow.ondblclick = (e) => {
                e.stopPropagation();
                openLightbox(img, concept);
              };

              imgRow.oncontextmenu = (e) => {
                openContextMenu(e, "image", concept, relItem, img);
              };

              imageContainer.appendChild(imgRow);
            });

            childrenContainer.appendChild(imageContainer);
          }
        });

        // Direct images under concept (when no related concepts exist or unassociated images)
        const unassociatedImages = (node.images || []).filter(
          (img) => !img.related_concept || !(node.related || []).some((r) => r.name.toLowerCase() === img.related_concept.toLowerCase())
        );

        if (unassociatedImages.length > 0) {
          const directContainer = document.createElement("div");
          directContainer.className = "tree-image-container";
          unassociatedImages.forEach((img) => {
            const isImgSelected = (
              selectedTreeTarget.type === "image" &&
              selectedTreeTarget.img === img
            );
            const imgRow = document.createElement("div");
            imgRow.className = `tree-image-row ${isImgSelected ? "active" : ""}`;

            const thumb = document.createElement("img");
            thumb.className = "tree-image-thumb";
            thumb.src = img.url || "";
            thumb.alt = img.filename || "stimulus";
            thumb.loading = "lazy";

            const imgName = document.createElement("span");
            imgName.className = "tree-image-name";
            imgName.textContent = img.filename || "image";

            const srcBadge = document.createElement("span");
            srcBadge.className = "tree-image-src-badge";
            srcBadge.textContent = (img.source || "SRC").toUpperCase();

            const delImgBtn = document.createElement("button");
            delImgBtn.type = "button";
            delImgBtn.className = "tree-node-delete";
            delImgBtn.textContent = "✕";
            delImgBtn.title = "Delete image";
            delImgBtn.onclick = (e) => {
              e.stopPropagation();
              node.images = (node.images || []).filter((i) => i !== img);
              updateAcquisitionPlan();
              renderConceptTree();
              renderGallery();
            };

            imgRow.appendChild(thumb);
            imgRow.appendChild(imgName);
            imgRow.appendChild(srcBadge);
            imgRow.appendChild(delImgBtn);

            // Single click selects image as gallery target
            imgRow.onclick = (e) => {
              e.stopPropagation();
              selectedTreeTarget = {
                type: "image",
                concept: concept,
                association: null,
                img: img,
              };
              activeSeedConcept = concept;
              selectedFilterConcept = concept;
              renderConceptTree();
              renderGallery();
            };

            // Double click opens Lightbox viewer
            imgRow.ondblclick = (e) => {
              e.stopPropagation();
              openLightbox(img, concept);
            };

            imgRow.oncontextmenu = (e) => {
              openContextMenu(e, "image", concept, null, img);
            };

            directContainer.appendChild(imgRow);
          });
          childrenContainer.appendChild(directContainer);
        }

        parentEl.appendChild(childrenContainer);
      }

      treeContainer.appendChild(parentEl);
    });

    if (treeContainer) {
      treeContainer.scrollTop = currentScroll;
    }

    syncTreeHeaderControls();
    updateGeneratePlan();
    updateAcquisitionPlan();
    renderGallery();
  }

  // ==========================================
  // 4. Image Workspace Gallery Rendering
  // ==========================================

  function renderGallery() {
    galleryGrid.innerHTML = "";

    // Gather images strictly based on what is selected in the concept tree
    let imagesToDisplay = [];
    let currentConceptName = "";

    if (selectedTreeTarget && selectedTreeTarget.concept) {
      currentConceptName = selectedTreeTarget.concept;
      const node = workspaceTree[currentConceptName];

      if (selectedTreeTarget.type === "image" && selectedTreeTarget.img) {
        imagesToDisplay = [selectedTreeTarget.img];
        const assocLabel = selectedTreeTarget.association || currentConceptName;
        galleryTitle.textContent = selectedTreeTarget.img.filename || `Selected Image for "${assocLabel}"`;
        gallerySubtitle.textContent = `Showing single selected image for "${assocLabel}" (${currentConceptName}).`;
      } else if (selectedTreeTarget.type === "association" && selectedTreeTarget.association) {
        const assocName = selectedTreeTarget.association.toLowerCase();
        imagesToDisplay = (node ? node.images || [] : []).filter((img) =>
          img.related_concept && img.related_concept.toLowerCase() === assocName
        );
        galleryTitle.textContent = `Stimulus Images for "${selectedTreeTarget.association}"`;
        gallerySubtitle.textContent = `Showing stimuli strictly for association "${selectedTreeTarget.association}" (${currentConceptName}).`;
      } else {
        // Concept level selection
        imagesToDisplay = node ? (node.images || []) : [];
        galleryTitle.textContent = `Stimulus Images for "${currentConceptName}"`;
        gallerySubtitle.textContent = `Showing all stimuli strictly for concept "${currentConceptName}".`;
      }
    } else {
      // Nothing selected in tree
      imageCountBadge.textContent = "0 images";
      galleryTitle.textContent = "Stimulus Image Workspace";
      gallerySubtitle.textContent = "Select a concept, association, or image in the tree to view its stimuli.";
      galleryGrid.innerHTML = `
        <div style="grid-column: 1 / -1; padding: 3rem 1rem; text-align: center; color: var(--text-secondary);">
          <p>No concept or item selected in the tree.</p>
          <p class="text-xs text-muted" style="margin-top: 0.25rem;">Click any concept, association, or image on the left to view its images.</p>
        </div>
      `;
      return;
    }

    imageCountBadge.textContent = `${imagesToDisplay.length} image${imagesToDisplay.length === 1 ? "" : "s"}`;
    const hasAnyImages = Object.values(workspaceTree).some((n) => (n.images || []).length > 0);
    if (btnCropSelected) btnCropSelected.disabled = !hasAnyImages;
    if (btnExportJson) btnExportJson.disabled = !hasAnyImages;
    if (btnExportCsv) btnExportCsv.disabled = !hasAnyImages;

    if (imagesToDisplay.length === 0) {
      galleryGrid.innerHTML = `
        <div style="grid-column: 1 / -1; padding: 3rem 1rem; text-align: center; color: var(--text-secondary);">
          <p>No stimulus images found for this selection.</p>
          <p class="text-xs text-muted" style="margin-top: 0.25rem;">Click "Generate" above or click the 📸 icon on an item in the tree to acquire images.</p>
        </div>
      `;
      return;
    }

    const isSingleImage = (imagesToDisplay.length === 1);

    imagesToDisplay.forEach((img) => {
      const isSelected = (
        (selectedTreeTarget && selectedTreeTarget.type === "image" && selectedTreeTarget.img === img) ||
        isSingleImage
      );

      const item = document.createElement("div");
      item.className = `gallery-item ${isSelected ? "selected-card" : ""}`;
      item.style.position = "relative";
      item.style.cursor = "pointer";

      const imageEl = document.createElement("img");
      imageEl.src = img.url || "";
      imageEl.alt = img.filename || "stimulus image";
      imageEl.loading = "lazy";

      // Canvas overlay for Face ROI
      const canvasEl = document.createElement("canvas");
      canvasEl.className = "gallery-roi-canvas";

      const badge = document.createElement("span");
      badge.className = "gallery-badge";
      badge.textContent = (img.source || "IMG").toUpperCase();

      if (img.related_concept) {
        const titleBadge = document.createElement("span");
        titleBadge.className = "gallery-badge";
        titleBadge.style.left = "auto";
        titleBadge.style.right = "0.45rem";
        titleBadge.style.color = "var(--text-primary)";
        titleBadge.textContent = img.related_concept;
        item.appendChild(titleBadge);
      }

      // View button on thumbnail - opens popup modal
      const zoomBtn = document.createElement("button");
      zoomBtn.type = "button";
      zoomBtn.className = "gallery-view-btn";
      zoomBtn.innerHTML = "<span>🔍</span><span>View</span>";
      zoomBtn.title = "View enlarged stimulus image in popup";
      zoomBtn.onclick = (e) => {
        e.stopPropagation();
        openLightbox(img, img.concept || currentConceptName);
      };
      item.appendChild(zoomBtn);

      // Render Face ROI on ALL thumbnails when showFaceRoi is selected
      const updateCardRoi = async () => {
        if (!showFaceRoi) {
          const ctx = canvasEl.getContext("2d");
          ctx.clearRect(0, 0, canvasEl.width, canvasEl.height);
          return;
        }
        try {
          const data = await getFaceRoisForImage(img);
          if (data && data.faces) {
            drawFaceRoi(canvasEl, imageEl, data.faces, showFaceRoi);
          }
        } catch (e) {
          console.debug("Error drawing card face ROI:", e);
        }
      };

      imageEl.onload = () => {
        updateCardRoi();
      };

      // Clicking anywhere on thumbnail opens popup modal as earlier
      item.onclick = () => {
        openLightbox(img, img.concept || currentConceptName);
      };

      item.ondblclick = () => {
        openLightbox(img, img.concept || currentConceptName);
      };

      item.appendChild(imageEl);
      item.appendChild(canvasEl);
      item.appendChild(badge);
      galleryGrid.appendChild(item);

      if (imageEl.complete) {
        updateCardRoi();
      }
    });
  }

  // ==========================================
  // 5. Tree Control Actions
  // ==========================================
  if (treeSelectAllCheckbox) {
    treeSelectAllCheckbox.addEventListener("change", () => {
      const shouldCheck = treeSelectAllCheckbox.checked;
      Object.keys(workspaceTree).forEach((k) => {
        workspaceTree[k].checked = shouldCheck;
        (workspaceTree[k].related || []).forEach((r) => { r.checked = shouldCheck; });
        (workspaceTree[k].images || []).forEach((img) => { img.checked = shouldCheck; });
      });
      updateAcquisitionPlan();
      renderConceptTree();
    });
  }

  if (btnToggleExpandTree) {
    btnToggleExpandTree.addEventListener("click", () => {
      const concepts = Object.keys(workspaceTree);
      if (concepts.length === 0) return;
      const anyExpanded = concepts.some((k) => workspaceTree[k].expanded);
      const targetState = !anyExpanded;
      concepts.forEach((k) => {
        workspaceTree[k].expanded = targetState;
        (workspaceTree[k].related || []).forEach((r) => { r.expanded = targetState; });
      });
      renderConceptTree();
    });
  }

  if (btnClearTree) {
    btnClearTree.addEventListener("click", () => {
      if (confirm("Clear all concepts from the tree?")) {
        Object.keys(workspaceTree).forEach((k) => delete workspaceTree[k]);
        activeSeedConcept = "";
        selectedFilterConcept = "";
        conceptInput.value = "";
        selectedTreeTarget = { type: "concept", concept: "", association: null };
        updateAcquisitionPlan();
        renderConceptTree();
      }
    });
  }

  // Backwards compatibility for legacy buttons if present in DOM
  if (btnCheckAllTree) {
    btnCheckAllTree.addEventListener("click", () => {
      Object.keys(workspaceTree).forEach((k) => {
        workspaceTree[k].checked = true;
        (workspaceTree[k].related || []).forEach((r) => { r.checked = true; });
        (workspaceTree[k].images || []).forEach((img) => { img.checked = true; });
      });
      updateAcquisitionPlan();
      renderConceptTree();
    });
  }

  if (btnUncheckAllTree) {
    btnUncheckAllTree.addEventListener("click", () => {
      Object.keys(workspaceTree).forEach((k) => {
        workspaceTree[k].checked = false;
        (workspaceTree[k].related || []).forEach((r) => { r.checked = false; });
        (workspaceTree[k].images || []).forEach((img) => { img.checked = false; });
      });
      updateAcquisitionPlan();
      renderConceptTree();
    });
  }

  if (btnExpandAllTree) {
    btnExpandAllTree.addEventListener("click", () => {
      Object.keys(workspaceTree).forEach((k) => {
        workspaceTree[k].expanded = true;
        (workspaceTree[k].related || []).forEach((r) => { r.expanded = true; });
      });
      renderConceptTree();
    });
  }

  if (btnCollapseAllTree) {
    btnCollapseAllTree.addEventListener("click", () => {
      Object.keys(workspaceTree).forEach((k) => {
        workspaceTree[k].expanded = false;
        (workspaceTree[k].related || []).forEach((r) => { r.expanded = false; });
      });
      renderConceptTree();
    });
  }

  if (imagesPerConceptSelect) {
    imagesPerConceptSelect.addEventListener("input", () => {
      updateAcquisitionPlan();
    });
    imagesPerConceptSelect.addEventListener("change", () => {
      imagesPerConceptSelect.value = getImagesPerConcept();
      updateAcquisitionPlan();
    });
  }

  if (maxConceptsInput) {
    maxConceptsInput.addEventListener("input", () => {
      updateSettingsSummary();
    });
    maxConceptsInput.addEventListener("change", () => {
      const raw = parseInt(maxConceptsInput.value, 10);
      maxConceptsInput.value = Math.min(50, Math.max(1, Number.isFinite(raw) && raw > 0 ? raw : 5));
      updateSettingsSummary();
    });
  }

  if (treeFilterInput) {
    treeFilterInput.addEventListener("input", () => {
      renderConceptTree();
    });
  }

  // Function to parse comma-separated concepts and load into tree
  function loadEnteredConcepts() {
    const raw = conceptInput ? conceptInput.value.trim() : "";
    if (!raw) {
      if (fileUploadStatus) {
        fileUploadStatus.textContent = "✗ Please enter at least one concept";
        fileUploadStatus.className = "badge-status invalid";
      }
      return;
    }
    const parsed = raw.split(/[\n,]+/).map((s) => s.trim()).filter(Boolean);
    if (parsed.length === 0) return;

    const loadedNames = [];
    parsed.forEach((rawItem) => {
      // Support 'concept (context)' or 'concept [context]' or 'concept: context'
      const m = rawItem.match(/^(.*?)\s*[\(\[]([^\)\]]+)[\)\]]$/) || rawItem.match(/^(.*?)\s*:\s*(.+)$/);
      if (m && m[1].trim() && m[2].trim()) {
        const cName = m[1].trim();
        const ctx = m[2].trim();
        ensureTreeNode(cName, "manual", ctx);
        loadedNames.push(cName);
      } else {
        ensureTreeNode(rawItem, "manual");
        loadedNames.push(rawItem);
      }
    });

    const firstLoaded = loadedNames[0] || parsed[0];
    activeSeedConcept = firstLoaded;
    selectedFilterConcept = firstLoaded;
    selectedTreeTarget = { type: "concept", concept: firstLoaded, association: null };

    if (fileUploadStatus) {
      fileUploadStatus.textContent = `✓ Loaded ${parsed.length} concept${parsed.length === 1 ? "" : "s"}`;
      fileUploadStatus.className = "badge-status valid";
    }
    if (fileHint) {
      fileHint.textContent = `Loaded ${parsed.length} concept${parsed.length === 1 ? "" : "s"}: ${parsed.join(", ")}`;
    }
    if (generationStatus) {
      generationStatus.textContent = `Ready: ${Object.keys(workspaceTree).length} concept(s) in tree. Click "Generate" to start.`;
    }

    renderConceptTree();
    updateAcquisitionPlan();
    renderGallery();
    refreshTelemetry();
  }

  if (btnLoadConcepts) {
    btnLoadConcepts.addEventListener("click", loadEnteredConcepts);
  }

  conceptInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter") {
      e.preventDefault();
      loadEnteredConcepts();
    }
  });

  conceptInput.addEventListener("change", () => {
    loadEnteredConcepts();
  });

  // ==========================================
  // 6. Concept File Upload & Extraction
  // ==========================================
  if (btnUploadFile && conceptFileInput) {
    btnUploadFile.addEventListener("click", () => {
      conceptFileInput.value = "";
      conceptFileInput.click();
    });

    conceptFileInput.addEventListener("click", () => {
      conceptFileInput.value = "";
    });

    conceptFileInput.addEventListener("change", async (e) => {
      const file = e.target.files && e.target.files[0];
      if (!file) return;

      btnUploadFile.disabled = true;
      btnUploadFile.textContent = "Processing…";
      fileUploadStatus.textContent = "Uploading & analyzing...";
      fileUploadStatus.className = "badge-status";
      if (fileHint) fileHint.textContent = `Analyzing ${file.name} for concept entities...`;

      try {
        const formData = new FormData();
        formData.append("file", file);
        formData.append("provider", providerSelect ? providerSelect.value : "local");

        const activeSub = getActiveSubjectId();
        if (activeSub) {
          formData.append("subject_id", activeSub);
        }

        if (providerSelect && providerSelect.value === "local") {
          const customPath = localModelPathInput ? localModelPathInput.value.trim() : "";
          if (customPath) formData.append("local_model_path", customPath);
        } else {
          const selectedModel = getSelectedCloudModel();
          if (selectedModel) formData.append("model", selectedModel);
        }

        const apiKey = apiKeyInput ? apiKeyInput.value.trim() : "";
        if (apiKey) formData.append("api_key", apiKey);

        const baseUrl = baseUrlInput ? baseUrlInput.value.trim() : "";
        if (baseUrl) formData.append("base_url", baseUrl);

        const res = await fetch("/api/v1/concepts/upload", {
          method: "POST",
          body: formData,
        });

        if (!res.ok) {
          const errData = await res.json().catch(() => ({}));
          throw new Error(errData.detail || "Failed to parse concepts from file.");
        }

        const data = await res.json();
        const extracted = data.concepts || [];

        if (extracted.length === 0) {
          throw new Error("No concepts found in the uploaded file.");
        }

        if (data.method === "questionnaire_rag") {
          fileUploadStatus.textContent = `✓ Extracted ${extracted.length} concepts for ${data.selected_subject || "Subject"} (Questionnaire RAG)`;
          fileUploadStatus.className = "badge-status rag";
        } else if (data.method === "agentic_rag") {
          fileUploadStatus.textContent = `✓ Extracted ${extracted.length} concepts (Agentic RAG)`;
          fileUploadStatus.className = "badge-status rag";
        } else {
          fileUploadStatus.textContent = `✓ Loaded ${extracted.length} concepts`;
          fileUploadStatus.className = "badge-status valid";
        }

        if (fileHint) {
          fileHint.textContent = data.message || `Loaded ${extracted.length} concepts from ${file.name}`;
        }

        lastUploadedQuestionnaireData = data;

        // If questionnaire contains multiple subjects, convert text area to dropdown
        if (data.subjects && data.subjects.length > 1) {
          if (subjectSelect) {
            subjectSelect.innerHTML = "";
            data.subjects.forEach((s) => {
              const opt = document.createElement("option");
              opt.value = s;
              const count = (data.concepts_by_subject && data.concepts_by_subject[s]) ? data.concepts_by_subject[s].length : 0;
              opt.textContent = `Subject ${s} (${count} concepts)`;
              if (s === data.selected_subject) opt.selected = true;
              subjectSelect.appendChild(opt);
            });

            const allOpt = document.createElement("option");
            allOpt.value = "all";
            const allCount = (data.all_concepts && data.all_concepts.length) || Object.values(data.concepts_by_subject || {}).flat().length;
            allOpt.textContent = `All Subjects (${allCount} concepts)`;
            if (data.selected_subject === "all") allOpt.selected = true;
            subjectSelect.appendChild(allOpt);

            const customOpt = document.createElement("option");
            customOpt.value = "__custom__";
            customOpt.textContent = "✏️ Enter custom subject ID…";
            subjectSelect.appendChild(customOpt);

            const targetSub = data.selected_subject || data.subjects[0];
            subjectSelect.value = targetSub;
            if (subjectInput) subjectInput.value = targetSub;
          }
          setSubjectMode("dropdown");
        } else {
          // Single subject or general concept file: remain/convert back to text area
          const targetSub = (data.subjects && data.subjects.length === 1) ? data.subjects[0] : (subjectInput ? subjectInput.value.trim() : "SUBJ_TEST_001");
          if (subjectInput) {
            subjectInput.value = targetSub || "SUBJ_TEST_001";
          }
          setSubjectMode("input");
        }

        // Populate initial extracted concepts into the tree
        let initialConcepts = extracted;
        let initialExcerpts = data.rag_excerpts || {};
        if (data.selected_subject && data.concepts_by_subject && data.concepts_by_subject[data.selected_subject]) {
          initialConcepts = data.concepts_by_subject[data.selected_subject];
          initialExcerpts = (data.excerpts_by_subject && data.excerpts_by_subject[data.selected_subject]) || data.rag_excerpts || {};
        }
        populateTreeWithConcepts(initialConcepts, data.method || "deterministic", initialExcerpts);

        if (generationStatus) {
          generationStatus.textContent = `Ready: ${initialConcepts.length} concepts loaded in tree. Click "Generate" to start.`;
        }

      } catch (err) {
        fileUploadStatus.textContent = `✗ ${err.message}`;
        fileUploadStatus.className = "badge-status invalid";
        alert(`File Ingestion Error: ${err.message}`);
      } finally {
        btnUploadFile.disabled = false;
        btnUploadFile.textContent = "📂 File…";
        conceptFileInput.value = "";
      }
    });
  }

  // ==========================================
  // 7. Related Concepts Generation & Image Acquisition Engine
  // ==========================================
  let acquisitionRunning = false;

  if (btnClearConceptCache) {
    btnClearConceptCache.addEventListener("click", async () => {
      const origText = btnClearConceptCache.textContent;
      btnClearConceptCache.disabled = true;
      btnClearConceptCache.textContent = "Clearing…";
      try {
        const res = await fetch("/api/v1/concepts/cache/clear", { method: "POST" });
        const data = await res.json().catch(() => ({}));
        if (!res.ok) {
          throw new Error(data.detail || `HTTP ${res.status}`);
        }
        if (generationStatus) {
          generationStatus.textContent = data.message || "Concept cache cleared.";
        }
        await refreshTelemetry();
        btnClearConceptCache.textContent = "Cleared ✓";
        setTimeout(() => {
          btnClearConceptCache.textContent = origText;
          btnClearConceptCache.disabled = false;
        }, 1200);
      } catch (err) {
        console.error("Failed to clear concept cache:", err);
        if (generationStatus) {
          generationStatus.textContent = `Failed to clear cache: ${err.message}`;
        }
        btnClearConceptCache.textContent = origText;
        btnClearConceptCache.disabled = false;
      }
    });
  }

  async function runGenerateRelatedConcepts(customConcepts = null) {
    const targetConcepts = customConcepts || getCheckedConcepts();

    if (targetConcepts.length === 0) {
      alert("Please load a concept file or enter at least one concept.");
      return false;
    }

    const btnText = btnGenerate ? btnGenerate.querySelector(".btn-text") : null;
    if (btnGenerate) btnGenerate.disabled = true;
    if (generateSpinner) generateSpinner.classList.remove("hidden");
    if (btnStopAcquire) {
      btnStopAcquire.classList.remove("hidden");
      btnStopAcquire.disabled = false;
    }

    acquisitionRunning = true;
    acquisitionCancelled = false;
    activeAcquisitionAbort = new AbortController();

    let completed = 0;
    let cachedCount = 0;
    let failed = 0;

    try {
      for (let i = 0; i < targetConcepts.length; i++) {
        if (acquisitionCancelled) {
          if (generationStatus) generationStatus.textContent = "⏹ Generation stopped by user.";
          break;
        }
        const conceptName = targetConcepts[i];

        if (btnText) {
          btnText.textContent = `Concepts (${i + 1}/${targetConcepts.length}): ${conceptName}…`;
        }
        if (generationStatus) {
          generationStatus.textContent = `Getting related concepts for "${conceptName}" (${i + 1} of ${targetConcepts.length})…`;
        }

        try {
          const maxConceptsVal = Math.min(50, Math.max(1, parseInt(maxConceptsInput ? maxConceptsInput.value : "5", 10) || 5));
          const existingNode = workspaceTree[conceptName];
          const conceptExcerpt = (existingNode && existingNode.excerpt) ? existingNode.excerpt : "";
          const payload = {
            concept: conceptName,
            count: maxConceptsVal,
            provider: providerSelect.value,
            use_cache: cacheToggle.checked,
            context: conceptExcerpt,
          };

          if (providerSelect.value === "local") {
            const customPath = localModelPathInput.value.trim();
            if (customPath) payload.local_model_path = customPath;
          } else {
            const selectedModel = getSelectedCloudModel();
            if (selectedModel) payload.model = selectedModel;
          }

          const apiKey = apiKeyInput.value.trim();
          if (apiKey) payload.api_key = apiKey;

          const baseUrl = baseUrlInput.value.trim();
          if (baseUrl) payload.base_url = baseUrl;

          const res = await fetch("/api/v1/concepts/generate", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload),
            signal: activeAcquisitionAbort.signal,
          });

          if (!res.ok) {
            const errData = await res.json().catch(() => ({}));
            throw new Error(errData.detail || `Generation failed for "${conceptName}"`);
          }

          const data = await res.json();

          // Update tree for this concept
          const node = ensureTreeNode(conceptName, "generated");
          node.related = (data.concepts || []).map((c) => ({ ...c, checked: true }));
          node.expanded = true; // Auto-expand to show new associations
          completed++;
          if (data.cached) {
            cachedCount++;
          }

          // Live refresh of the tree so user watches each branch populate
          renderConceptTree();
          await refreshTelemetry();

        } catch (err) {
          if (err.name === "AbortError" || acquisitionCancelled) {
            if (generationStatus) generationStatus.textContent = "⏹ Generation stopped by user.";
            break;
          }
          console.error(`Generation failed for "${conceptName}":`, err);
          failed++;
          if (targetConcepts.length === 1) {
            alert(`Error: ${err.message}`);
          }
        }
      }
    } finally {
      acquisitionRunning = false;
      activeAcquisitionAbort = null;
      if (btnGenerate) btnGenerate.disabled = false;
      if (btnText) btnText.textContent = "Generate";
      if (generateSpinner) generateSpinner.classList.add("hidden");
      if (btnStopAcquire) {
        btnStopAcquire.classList.add("hidden");
        btnStopAcquire.disabled = false;
      }
    }

    if (!acquisitionCancelled && generationStatus) {
      const sourceNote = cachedCount > 0 ? ` (${cachedCount} from cache)` : " (freshly generated)";
      generationStatus.textContent = `✓ Generated related concepts for ${completed} of ${targetConcepts.length} concepts${sourceNote}${failed > 0 ? ` (${failed} failed)` : ""}.`;
    }

    if (btnExportJson) btnExportJson.disabled = false;
    if (btnExportCsv) btnExportCsv.disabled = false;
    return !acquisitionCancelled && completed > 0;
  }

  function tagImages(images, conceptName, assocName) {
    return (images || []).map((img) => ({
      ...img,
      concept: conceptName,
      related_concept: assocName,
      checked: true,
    }));
  }

  // Acquire images for one item. Returns the number of images added.
  async function acquireForAssociationItem(conceptName, item, limit, source, signal = null) {
    const node = workspaceTree[conceptName];
    if (!node) return 0;
    if (!node.images) node.images = [];

    const isConceptLevel = !item;
    const assocName = isConceptLevel ? conceptName : item.name;
    const relatedForApi = isConceptLevel ? "" : (item.name || "");
    const charForApi = isConceptLevel ? "" : (item.char || "");
    const descForApi = isConceptLevel ? "" : (item.description || "");
    const contextForApi = (node && node.excerpt) ? node.excerpt : "";
    const { width, height } = getTargetDimensions();
    const detectFaces = detectFacesToggle ? detectFacesToggle.checked : true;
    const maxFaces = getMaxFaceCount();
    const cropFaces = cropFacesToggle ? cropFacesToggle.checked : false;
    const customOutputDir = getDownloadFolderPath();
    const activeSubjectId = getActiveSubjectId();

    let url;
    let body;
    if (["wikimedia", "duckduckgo", "openverse", "bing"].includes(source)) {
      url = "/api/v1/images/search";
      body = {
        subject_id: activeSubjectId,
        concept: conceptName,
        related_concept: relatedForApi,
        name: charForApi,
        context: contextForApi,
        limit: limit,
        provider: source,
        safe_search: "strict",
        filter: bingFilterSelect ? bingFilterSelect.value : "",
        min_width: width,
        min_height: height,
        detect_faces: detectFaces,
        max_faces: maxFaces,
        crop_faces: cropFaces,
        target_width: width,
        target_height: height,
        output_dir: customOutputDir,
      };
    } else {
      url = "/api/v1/images/generate";
      const promptText = isConceptLevel
        ? (contextForApi ? `${conceptName} (${contextForApi})` : conceptName)
        : `${assocName} clearly associated with ${conceptName}`;
      body = {
        subject_id: activeSubjectId,
        prompt: promptText,
        concept: conceptName,
        related_concept: relatedForApi,
        name: charForApi,
        description: descForApi,
        context: contextForApi,
        count: limit,
        provider: source,
        detect_faces: detectFaces,
        max_faces: maxFaces,
        crop_faces: cropFaces,
        target_width: width,
        target_height: height,
        output_dir: customOutputDir,
      };
      if (source === "text") {
        const textStyleSelect = document.getElementById("text-stim-style-select");
        const styleVal = textStyleSelect ? textStyleSelect.value : "black-on-white";
        if (styleVal === "white-on-black") {
          body.text_bg_color = "#000000";
          body.text_color = "#FFFFFF";
        } else if (styleVal === "black-on-gray") {
          body.text_bg_color = "#808080";
          body.text_color = "#000000";
        } else {
          body.text_bg_color = "#FFFFFF";
          body.text_color = "#000000";
        }
      }
      const apiKeyVal = apiKeyInput ? apiKeyInput.value.trim() : "";
      if (apiKeyVal && source !== "text") {
        body.api_key = apiKeyVal;
      }
      if (source === "local" && localImageModelPathInput) {
        body.local_model_path = localImageModelPathInput.value.trim();
      }
    }

    const res = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
      signal: signal,
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      let detail = `HTTP ${res.status}`;
      if (typeof err.detail === "string") {
        detail = err.detail;
      } else if (Array.isArray(err.detail)) {
        detail = err.detail.map((d) => (d.loc ? `${d.loc.slice(1).join('.')}: ` : '') + d.msg).join('; ');
      } else if (err.message) {
        detail = err.message;
      }
      throw new Error(detail);
    }
    const data = await res.json();
    const tagged = tagImages(data.images, conceptName, assocName);
    tagged.forEach((img) => {
      if (img.filename) faceRoiCache.delete(img.filename);
      if (img.url) faceRoiCache.delete(img.url);
    });
    node.images.push(...tagged);
    return tagged.length;
  }

  // Build tasks: checked items across tree. If none checked, falls back
  // to the currently selected tree row or all items in tree.
  function tasksForChecked() {
    const tasks = [];
    Object.keys(workspaceTree).forEach((c) => {
      const node = workspaceTree[c];
      if (node.checked === false) return;
      const rels = (node.related || []).filter((r) => r.checked !== false);
      if (rels.length > 0) rels.forEach((r) => tasks.push({ concept: c, item: r }));
      else tasks.push({ concept: c, item: null });
    });

    if (tasks.length === 0 && selectedTreeTarget && selectedTreeTarget.concept) {
      const concept = selectedTreeTarget.concept;
      const node = workspaceTree[concept];
      if (selectedTreeTarget.type === "association" && selectedTreeTarget.association) {
        const item = (node?.related || []).find((r) => r.name === selectedTreeTarget.association);
        return [{ concept, item: item || { name: selectedTreeTarget.association } }];
      } else {
        const rels = (node?.related || []).filter((r) => r.checked !== false);
        return rels.length > 0 ? rels.map((r) => ({ concept, item: r })) : [{ concept, item: null }];
      }
    }

    if (tasks.length === 0) {
      Object.keys(workspaceTree).forEach((c) => {
        const node = workspaceTree[c];
        const rels = (node.related || []);
        if (rels.length > 0) rels.forEach((r) => tasks.push({ concept: c, item: r }));
        else tasks.push({ concept: c, item: null });
      });
    }

    return tasks;
  }

  // Tasks for a single tree row (used by the row 📸 icon and right-click menu).
  function tasksForTarget(concept, relItem) {
    if (relItem) return [{ concept, item: relItem }];
    const node = workspaceTree[concept];
    if (!node) return [];
    const rels = (node.related || []).filter((r) => r.checked !== false);
    return rels.length > 0 ? rels.map((r) => ({ concept, item: r })) : [{ concept, item: null }];
  }

  function setStatus(text) {
    if (acquisitionPlanHint) acquisitionPlanHint.textContent = text;
  }

  if (btnStopAcquire) {
    btnStopAcquire.addEventListener("click", () => {
      if (!acquisitionRunning) return;
      acquisitionCancelled = true;
      if (activeAcquisitionAbort) {
        activeAcquisitionAbort.abort();
      }
      btnStopAcquire.disabled = true;
      setStatus("⏹ Stopping generation…");
    });
  }

  async function runAcquisition(tasks, overrideSource = null) {
    if (acquisitionRunning) return;
    if (!tasks || tasks.length === 0) {
      setStatus("Nothing checked — tick at least one concept or association in the tree.");
      return;
    }

    const limit = getImagesPerConcept();
    const source = overrideSource || imageSourceSelect.value;
    const btnText = btnGenerate ? btnGenerate.querySelector(".btn-text") : null;

    acquisitionRunning = true;
    acquisitionCancelled = false;
    activeAcquisitionAbort = new AbortController();

    if (btnGenerate) btnGenerate.disabled = true;
    if (btnStopAcquire) {
      btnStopAcquire.classList.remove("hidden");
      btnStopAcquire.disabled = false;
    }
    if (generateSpinner) generateSpinner.classList.remove("hidden");

    let added = 0;
    const empty = [];
    const failed = [];

    try {
      for (let i = 0; i < tasks.length; i++) {
        if (acquisitionCancelled) {
          setStatus("⏹ Acquisition stopped by user.");
          break;
        }

        const task = tasks[i];
        const label = task.item ? task.item.name : task.concept;
        if (btnText) btnText.textContent = `Images (${i + 1}/${tasks.length})…`;
        setStatus(`Searching for "${label}" (${i + 1} of ${tasks.length})…`);
        try {
          const n = await acquireForAssociationItem(
            task.concept,
            task.item,
            limit,
            source,
            activeAcquisitionAbort.signal
          );
          added += n;
          if (n === 0) empty.push(label);
          if (task.item) task.item.expanded = true;
          workspaceTree[task.concept].expanded = true;
        } catch (err) {
          if (err.name === "AbortError" || acquisitionCancelled) {
            setStatus("⏹ Acquisition stopped by user.");
            break;
          }
          console.error(`Image acquisition failed for "${label}":`, err);
          failed.push({ label, error: err.message });
        }
        renderConceptTree();
        renderGallery();
      }
    } finally {
      acquisitionRunning = false;
      activeAcquisitionAbort = null;
      if (btnGenerate) btnGenerate.disabled = false;
      if (btnText) btnText.textContent = "Generate";
      if (btnStopAcquire) {
        btnStopAcquire.classList.add("hidden");
        btnStopAcquire.disabled = false;
      }
      if (generateSpinner) generateSpinner.classList.add("hidden");
      updateAcquisitionPlan();
      renderGallery();
      if (btnExportJson) btnExportJson.disabled = false;
      if (btnExportCsv) btnExportCsv.disabled = false;
    }

    if (acquisitionCancelled) {
      const msg = `⏹ Stopped: acquired ${added} image${added === 1 ? "" : "s"}.`;
      setStatus(msg);
      if (gallerySubtitle) gallerySubtitle.textContent = msg;
    } else {
      let msg = `✓ Added ${added} image${added === 1 ? "" : "s"} for ${tasks.length - failed.length} item${tasks.length - failed.length === 1 ? "" : "s"}.`;
      if (empty.length) msg += ` No results for: ${empty.join(", ")}.`;
      if (failed.length) {
        const firstErr = failed[0].error || "";
        if (tasks.length === 1 && firstErr) {
          msg += ` Failed for ${failed[0].label}: ${firstErr}`;
        } else {
          msg += ` Failed for: ${failed.map(f => f.label).join(", ")}.`;
          if (firstErr) msg += ` (${firstErr})`;
        }
      }
      setStatus(msg);
      if (gallerySubtitle) gallerySubtitle.textContent = msg;
      if (tasks.length === 1 && failed.length === 1 && failed[0].error) {
        alert(`Image acquisition failed for "${failed[0].label}":\n\n${failed[0].error}`);
      }
    }
  }

  // Exposed for row icons and the context menu.
  function acquireForTarget(concept, relItem = null, overrideSource = null) {
    runAcquisition(tasksForTarget(concept, relItem), overrideSource);
  }

  if (btnGenerate) {
    btnGenerate.addEventListener("click", async () => {
      if (acquisitionRunning) return;
      const doRelated = genRelatedToggle ? genRelatedToggle.checked : true;
      const doImages = genImagesToggle ? genImagesToggle.checked : true;

      if (!doRelated && !doImages) {
        setStatus("Please check 'Related concepts' and/or 'Images' before clicking Generate.");
        return;
      }

      if (doRelated) {
        await runGenerateRelatedConcepts();
        if (acquisitionCancelled) return;
      }

      if (doImages) {
        await runAcquisition(tasksForChecked());
      }
    });
  }

  // ==========================================
  // 9. Export Handlers (JSON & CSV across all concepts)
  // ==========================================
  btnExportJson.addEventListener("click", () => {
    const exportData = {};
    Object.keys(workspaceTree).forEach((c) => {
      exportData[c] = {
        related_concepts: workspaceTree[c].related || [],
        images: workspaceTree[c].images || [],
      };
    });
    const blob = new Blob([JSON.stringify(exportData, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `stim_studio_workspace.json`;
    a.click();
    URL.revokeObjectURL(url);
  });

  btnExportCsv.addEventListener("click", () => {
    const rows = [["Concept", "Related Association", "Role", "Description", "Image Count"]];
    Object.keys(workspaceTree).forEach((c) => {
      const rels = workspaceTree[c].related || [];
      if (rels.length === 0) {
        rows.push([`"${c}"`, '""', '""', '""', "0"]);
      } else {
        rels.forEach((r) => {
          const imgCount = (workspaceTree[c].images || []).filter((img) =>
            img.related_concept && img.related_concept.toLowerCase() === r.name.toLowerCase()
          ).length;
          rows.push([
            `"${c.replace(/"/g, '""')}"`,
            `"${(r.name || "").replace(/"/g, '""')}"`,
            `"${(r.char || "").replace(/"/g, '""')}"`,
            `"${(r.description || "").replace(/"/g, '""')}"`,
            imgCount,
          ]);
        });
      }
    });
    const csvContent = rows.map((r) => r.join(",")).join("\n");
    const blob = new Blob([csvContent], { type: "text/csv" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `stim_studio_workspace.csv`;
    a.click();
    URL.revokeObjectURL(url);
  });

  async function cropSelectedImages() {
    const { width, height } = getTargetDimensions();
    const tasks = tasksForChecked();
    const targetImages = [];
    const seen = new Set();

    tasks.forEach((t) => {
      const node = workspaceTree[t.concept];
      if (!node) return;
      if (t.item && t.item.name) {
        (node.images || []).forEach((img) => {
          const key = img.local_path || img.filename || img.url;
          if (key && !seen.has(key) && (img.related_concept === t.item.name || img.concept === t.concept)) {
            seen.add(key);
            targetImages.push(img);
          }
        });
      } else {
        (node.images || []).forEach((img) => {
          const key = img.local_path || img.filename || img.url;
          if (key && !seen.has(key)) {
            seen.add(key);
            targetImages.push(img);
          }
        });
      }
    });

    if (targetImages.length === 0) {
      Object.values(workspaceTree).forEach((node) => {
        (node.images || []).forEach((img) => {
          const key = img.local_path || img.filename || img.url;
          if (key && !seen.has(key)) {
            seen.add(key);
            targetImages.push(img);
          }
        });
      });
    }

    if (targetImages.length === 0) {
      alert("No stimulus images found to crop.");
      return;
    }

    if (btnCropSelected) {
      btnCropSelected.disabled = true;
      btnCropSelected.textContent = "Cropping…";
    }

    let croppedCount = 0;
    try {
      for (let i = 0; i < targetImages.length; i++) {
        const img = targetImages[i];
        const key = img.local_path || img.url || img.filename;
        if (!key) continue;
        try {
          const res = await fetch("/api/v1/images/crop-face", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              image_path: key,
              target_width: width,
              target_height: height,
            }),
          });
          if (res.ok) {
            const data = await res.json();
            if (data.status === "cropped") {
              faceRoiCache.delete(key);
              faceRoiCache.delete(img.local_path);
              faceRoiCache.delete(img.filename);
              faceRoiCache.delete(img.url);
              if (data.url) {
                const newUrl = `${data.url.split("?")[0]}?t=${Date.now()}`;
                img.url = newUrl;
              }
              croppedCount++;
            }
          }
        } catch (err) {
          console.warn("Failed cropping image", key, err);
        }
      }
      renderConceptTree();
      renderGallery();
      alert(`Face auto-crop complete: ${croppedCount} of ${targetImages.length} image(s) cropped to ${width}×${height}.`);
    } finally {
      if (btnCropSelected) {
        btnCropSelected.disabled = false;
        btnCropSelected.textContent = "Crop Selected";
      }
    }
  }

  if (btnCropSelected) {
    btnCropSelected.addEventListener("click", cropSelectedImages);
  }

  // ==========================================
  // 10. Model Management, Health & Diagnostics
  // ==========================================
  const defaultLlmPath = "/Users/smathew/LLM/Qwen2.5-7B-Instruct-GGUF/qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf";
  let savedModelPath = localStorage.getItem("sf_local_model_path") || defaultLlmPath;
  if (
    savedModelPath.endsWith("qwen3-4b-instruct-2507-q4_k_m.gguf") ||
    savedModelPath.includes("qwen-image") ||
    savedModelPath.endsWith("Qwen2.5-14B-Instruct-Q4_K_M.gguf") ||
    savedModelPath.includes("-00002-of-")
  ) {
    savedModelPath = defaultLlmPath;
    localStorage.setItem("sf_local_model_path", savedModelPath);
  }
  if (savedModelPath) {
    localModelPathInput.value = savedModelPath;
    validateModelPath(savedModelPath, localModelValidation);
  }

  const defaultDiffPath = "/Users/smathew/LLM/v1-5-pruned-emaonly.safetensors";
  const savedImageModelPath = localStorage.getItem("sf_local_image_model_path") || defaultDiffPath;
  if (localImageModelPathInput) {
    localImageModelPathInput.value = savedImageModelPath;
    validateModelPath(savedImageModelPath, localImageModelValidation);
  }

  if (btnResetExclusions) {
    btnResetExclusions.addEventListener("click", () => {
      bingExclusionsInput.value = DEFAULT_EXCLUSIONS;
    });
  }

  function updateProviderVisibility() {
    const isLocal = providerSelect.value === "local";
    const isCloudImage = ["openai", "google", "huggingface"].includes(imageSourceSelect ? imageSourceSelect.value : "");
    if (localModelGroup) localModelGroup.style.display = isLocal ? "block" : "none";
    if (cloudCredentialsDetails) {
      cloudCredentialsDetails.style.display = (isLocal && !isCloudImage) ? "none" : "block";
    }
    if (cloudModelGroup) {
      cloudModelGroup.style.display = isLocal ? "none" : "block";
      if (!isLocal) {
        if (cloudModelLabel) {
          const prov = providerSelect.value;
          cloudModelLabel.textContent = prov === "google" ? "Google Gemini Model" :
                                        prov === "openai" ? "OpenAI Model" :
                                        prov === "anthropic" ? "Claude Model" : "Endpoint Model";
        }
        populateCloudModels(providerSelect.value);
      }
    }
    updateSettingsSummary();
  }

  function updateImageSourceVisibility() {
    const src = imageSourceSelect.value;
    if (localImageModelGroup) localImageModelGroup.style.display = src === "local" ? "block" : "none";
    if (bingOptionsDetails) bingOptionsDetails.style.display = src === "bing" ? "block" : "none";
    const textStimGroup = document.getElementById("text-stimulus-options-group");
    if (textStimGroup) textStimGroup.style.display = src === "text" ? "block" : "none";
    const isCloudImage = ["openai", "google", "huggingface"].includes(src);
    const isLocalProvider = providerSelect.value === "local";
    if (cloudCredentialsDetails) {
      cloudCredentialsDetails.style.display = (!isLocalProvider || isCloudImage) ? "block" : "none";
      if (isCloudImage && apiKeyInput && !apiKeyInput.value.trim()) {
        cloudCredentialsDetails.open = true;
      }
    }
    updateSettingsSummary();
  }

  providerSelect.addEventListener("change", updateProviderVisibility);
  imageSourceSelect.addEventListener("change", updateImageSourceVisibility);
  const textStimStyleSelect = document.getElementById("text-stim-style-select");
  if (textStimStyleSelect) {
    textStimStyleSelect.addEventListener("change", updateSettingsSummary);
  }
  if (cloudModelSelect) {
    cloudModelSelect.addEventListener("change", () => {
      if (customCloudModelRow) {
        customCloudModelRow.style.display = cloudModelSelect.value === "__custom__" ? "block" : "none";
      }
      updateSettingsSummary();
    });
  }
  if (customCloudModelInput) {
    customCloudModelInput.addEventListener("input", updateSettingsSummary);
  }
  if (btnRefreshCloudModels) {
    btnRefreshCloudModels.addEventListener("click", () => {
      fetchProviderModels();
    });
  }
  fetchProviderModels();
  if (bingFilterSelect) {
    bingFilterSelect.addEventListener("change", updateSettingsSummary);
  }
  if (cacheToggle) {
    cacheToggle.addEventListener("change", updateSettingsSummary);
  }
  updateProviderVisibility();
  updateImageSourceVisibility();
  updateSettingsSummary();

  async function validateModelPath(path, targetBadge) {
    if (!path || !path.trim() || !targetBadge) {
      if (targetBadge) {
        targetBadge.textContent = "";
        targetBadge.className = "badge-status";
      }
      return;
    }

    try {
      const res = await fetch("/api/v1/system/validate-model", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ path: path.trim() }),
      });
      if (!res.ok) return;
      const data = await res.json();
      if (data.valid) {
        targetBadge.textContent = `✓ Valid (${data.size_mb} MB)`;
        targetBadge.className = "badge-status valid";
      } else {
        targetBadge.textContent = `✗ Not found`;
        targetBadge.className = "badge-status invalid";
      }
    } catch {
      targetBadge.textContent = "";
    }
  }

  localModelPathInput.addEventListener("input", (e) => {
    const val = e.target.value.trim();
    localStorage.setItem("sf_local_model_path", val);
    validateModelPath(val, localModelValidation);
    updateSettingsSummary();
  });

  if (localImageModelPathInput) {
    localImageModelPathInput.addEventListener("input", (e) => {
      const val = e.target.value.trim();
      localStorage.setItem("sf_local_image_model_path", val);
      validateModelPath(val, localImageModelValidation);
      updateSettingsSummary();
    });
  }

  async function scanHostModels() {
    try {
      const res = await fetch("/api/v1/system/scan-models");
      if (!res.ok) return;
      const models = await res.json();

      detectedModelsSelect.innerHTML = '<option value="">-- Choose from detected models on machine --</option>';
      if (models.length > 0) {
        models.forEach((m) => {
          const opt = document.createElement("option");
          opt.value = m.path;
          opt.textContent = `${m.name} (${m.size_mb} MB)`;
          detectedModelsSelect.appendChild(opt);
        });

        if (!localModelPathInput.value.trim()) {
          const preferredGguf =
            models.find((m) => m.name === "qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf") ||
            models.find((m) => m.name === "Qwen2.5-14B-Instruct-Q4_K_M.gguf") ||
            models.find((m) => m.name.endsWith(".gguf") && !m.name.includes("image") && !m.name.includes("-00002-of-")) ||
            models.find((m) => m.name.endsWith(".gguf")) ||
            models[0];
          localModelPathInput.value = preferredGguf.path;
          localStorage.setItem("sf_local_model_path", preferredGguf.path);
          validateModelPath(preferredGguf.path, localModelValidation);
          updateSettingsSummary();
        }
      }
    } catch (e) {
      console.error("Model scan error", e);
    }
  }

  function normalizeGgufShardPath(p) {
    if (!p) return p;
    return p.replace(/-\d{5}-of-(\d{5}\.gguf)$/i, "-00001-of-$1");
  }

  detectedModelsSelect.addEventListener("change", (e) => {
    if (e.target.value) {
      const normalized = normalizeGgufShardPath(e.target.value);
      localModelPathInput.value = normalized;
      localStorage.setItem("sf_local_model_path", normalized);
      validateModelPath(normalized, localModelValidation);
      updateSettingsSummary();
    }
  });

  btnScanModels.addEventListener("click", scanHostModels);

  btnBrowseModel.addEventListener("click", async () => {
    btnBrowseModel.disabled = true;
    btnBrowseModel.textContent = "Opening…";
    try {
      const res = await fetch("/api/v1/system/browse-file?model_type=llm", { method: "POST" });
      const data = await res.json();
      if (data.status === "selected" && data.path) {
        const normalized = normalizeGgufShardPath(data.path);
        localModelPathInput.value = normalized;
        localStorage.setItem("sf_local_model_path", normalized);
        validateModelPath(normalized, localModelValidation);
        updateSettingsSummary();
      } else if (data.status === "unsupported") {
        alert(data.message || "Please enter the path to your GGUF file manually.");
      }
    } catch (err) {
      alert(`Could not open file picker: ${err.message}`);
    } finally {
      btnBrowseModel.disabled = false;
      btnBrowseModel.textContent = "Browse…";
    }
  });

  if (btnBrowseImageModel) {
    btnBrowseImageModel.addEventListener("click", async () => {
      btnBrowseImageModel.disabled = true;
      btnBrowseImageModel.textContent = "Opening…";
      try {
        const res = await fetch("/api/v1/system/browse-file?model_type=image", { method: "POST" });
        const data = await res.json();
        if (data.status === "selected" && data.path) {
          localImageModelPathInput.value = data.path;
          localStorage.setItem("sf_local_image_model_path", data.path);
          validateModelPath(data.path, localImageModelValidation);
          updateSettingsSummary();
        } else if (data.status === "unsupported") {
          alert(data.message || "Please enter the path to your .safetensors file manually.");
        }
      } catch (err) {
        alert(`Could not open file picker: ${err.message}`);
      } finally {
        btnBrowseImageModel.disabled = false;
        btnBrowseImageModel.textContent = "Browse…";
      }
    });
  }

  // ==========================================
  // Download Destination Folder Handling
  // ==========================================
  function shortenPath(pathStr, maxFolders = 3) {
    if (!pathStr || typeof pathStr !== "string") return "";
    const trimmed = pathStr.trim().replace(/[\\/]+$/, "");
    if (!trimmed) return "";
    const sep = trimmed.includes("\\") ? "\\" : "/";
    const parts = trimmed.split(/[\\/]+/).filter(Boolean);
    if (parts.length <= maxFolders) {
      return trimmed;
    }
    const lastParts = parts.slice(-maxFolders);
    return "..." + sep + lastParts.join(sep);
  }

  function getDownloadFolderPath() {
    if (!downloadFolderInput) return null;
    const full = downloadFolderInput.dataset.fullPath || "";
    if (full) return full.trim();
    const val = downloadFolderInput.value.trim();
    if (val && !val.startsWith("...")) return val;
    const saved = localStorage.getItem("sf_download_folder_path") || "";
    return saved.trim() || null;
  }

  function setDownloadFolderDisplay(fullPath) {
    if (!downloadFolderInput) return;
    const trimmed = (fullPath || "").trim();
    downloadFolderInput.dataset.fullPath = trimmed;
    if (trimmed) {
      downloadFolderInput.value = shortenPath(trimmed, 3);
      downloadFolderInput.title = `Full path: ${trimmed}`;
    } else {
      downloadFolderInput.value = "";
      downloadFolderInput.title = downloadFolderInput.dataset.defaultTitle || "";
    }
  }

  async function validateDownloadFolder(path) {
    if (!downloadFolderStatus) return;
    if (!path || !path.trim()) {
      downloadFolderStatus.textContent = "Default workspace";
      downloadFolderStatus.className = "badge-status";
      return;
    }
    try {
      const res = await fetch("/api/v1/system/validate-folder", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ path: path.trim() }),
      });
      if (!res.ok) return;
      const data = await res.json();
      if (data.valid) {
        downloadFolderStatus.textContent = data.exists ? "✓ Folder exists" : "✓ Will create folder";
        downloadFolderStatus.className = "badge-status valid";
        downloadFolderStatus.title = `Full path: ${data.path}`;
      } else {
        downloadFolderStatus.textContent = `✗ ${data.message || "Invalid path"}`;
        downloadFolderStatus.className = "badge-status invalid";
        downloadFolderStatus.title = data.path ? `Invalid: ${data.path}` : "";
      }
    } catch {
      downloadFolderStatus.textContent = "";
      downloadFolderStatus.className = "badge-status";
    }
  }

  async function initDownloadFolder() {
    if (!downloadFolderInput) return;
    const saved = localStorage.getItem("sf_download_folder_path");
    if (saved) {
      setDownloadFolderDisplay(saved);
      validateDownloadFolder(saved);
    } else {
      try {
        const res = await fetch("/api/v1/system/config-paths");
        if (res.ok) {
          const cfg = await res.json();
          if (cfg.current_image_dir) {
            const shortDef = shortenPath(cfg.current_image_dir, 3);
            downloadFolderInput.placeholder = `Default: ${shortDef}`;
            downloadFolderInput.dataset.defaultTitle = `Default folder: ${cfg.current_image_dir}`;
            downloadFolderInput.title = `Default folder: ${cfg.current_image_dir}`;
          }
        }
      } catch (e) {
        // ignore
      }
      validateDownloadFolder("");
    }
  }

  if (downloadFolderInput) {
    // When focused, expand to full path so the user can easily copy or edit
    downloadFolderInput.addEventListener("focus", () => {
      if (downloadFolderInput.dataset.fullPath) {
        downloadFolderInput.value = downloadFolderInput.dataset.fullPath;
      }
    });

    // When blurred, shorten to last 3 folders with dots
    downloadFolderInput.addEventListener("blur", () => {
      const full = downloadFolderInput.value.trim();
      if (full && !full.startsWith("...")) {
        downloadFolderInput.dataset.fullPath = full;
        localStorage.setItem("sf_download_folder_path", full);
      } else if (!full) {
        downloadFolderInput.dataset.fullPath = "";
        localStorage.removeItem("sf_download_folder_path");
      }
      const activePath = downloadFolderInput.dataset.fullPath || "";
      setDownloadFolderDisplay(activePath);
      validateDownloadFolder(activePath);
      updateSettingsSummary();
    });

    downloadFolderInput.addEventListener("keydown", (e) => {
      if (e.key === "Enter") {
        downloadFolderInput.blur();
      }
    });

    downloadFolderInput.addEventListener("input", (e) => {
      const val = e.target.value.trim();
      if (!val.startsWith("...")) {
        downloadFolderInput.dataset.fullPath = val;
        if (val) {
          localStorage.setItem("sf_download_folder_path", val);
        } else {
          localStorage.removeItem("sf_download_folder_path");
        }
      }
      validateDownloadFolder(downloadFolderInput.dataset.fullPath || val);
      updateSettingsSummary();
    });
  }

  if (btnBrowseDownloadFolder) {
    btnBrowseDownloadFolder.addEventListener("click", async () => {
      btnBrowseDownloadFolder.disabled = true;
      btnBrowseDownloadFolder.textContent = "Opening…";
      try {
        const res = await fetch("/api/v1/system/browse-folder", { method: "POST" });
        const data = await res.json();
        if (data.status === "selected" && data.path) {
          setDownloadFolderDisplay(data.path);
          localStorage.setItem("sf_download_folder_path", data.path);
          validateDownloadFolder(data.path);
          updateSettingsSummary();
        } else if (data.status === "unsupported") {
          alert(data.message || "Please enter the folder path manually.");
        }
      } catch (err) {
        alert(`Could not open folder picker: ${err.message}`);
      } finally {
        btnBrowseDownloadFolder.disabled = false;
        btnBrowseDownloadFolder.textContent = "Browse…";
      }
    });
  }

  async function checkHealth() {
    try {
      const res = await fetch("/api/v1/health/diagnostics");
      if (!res.ok) throw new Error("Health check failed");
      const data = await res.json();

      backendStatusPill.querySelector(".status-dot").className = "status-dot online";
      backendStatusText.textContent = `Online (v${data.version})`;
      hardwareText.textContent = data.hardware_acceleration || "CPU";
    } catch {
      backendStatusPill.querySelector(".status-dot").className = "status-dot offline";
      backendStatusText.textContent = "Offline";
      hardwareText.textContent = "API Unreachable";
    }
  }

  async function refreshTelemetry() {
    try {
      const res = await fetch("/api/v1/telemetry/stats");
      if (!res.ok) return;
      const stats = await res.json();

      metricLatency.textContent = `${stats.avg_latency_ms || 0} ms`;
      metricTokens.textContent = `${(stats.total_prompt_tokens + stats.total_completion_tokens).toLocaleString()} toks`;
      metricCost.textContent = `$${(stats.total_cost_usd || 0).toFixed(4)}`;
      metricSavings.textContent = `$${(stats.total_cost_avoided_usd || 0).toFixed(4)}`;
      metricCacheHit.textContent = `${Math.round((stats.cache_hit_ratio || 0) * 100)}%`;
    } catch (e) {
      console.error("Telemetry error", e);
    }
  }

  btnResetTelemetry.addEventListener("click", async () => {
    try {
      await fetch("/api/v1/telemetry/reset", { method: "POST" });
      await refreshTelemetry();
    } catch (e) {
      console.error(e);
    }
  });

  // ==========================================
  // 11. Workflow Assistant Chat Dock
  // ==========================================
  if (btnToggleChat && chatDock) {
    btnToggleChat.addEventListener("click", () => {
      chatDock.classList.toggle("minimized");
      if (!chatDock.classList.contains("minimized")) {
        chatInputText.focus();
      }
    });

    btnCloseChat.addEventListener("click", () => {
      chatDock.classList.add("minimized");
    });

    btnClearChat.addEventListener("click", () => {
      chatTranscript.innerHTML = "";
      chatChoicesContainer.innerHTML = "";
      chatState = "awaiting_concept";
      chatCurrentConcept = "";
      appendAssistantMessage(
        "Conversation reset. Tell me a concept to add or type “help” for command examples.",
        [
          { label: "Add 'Formula 1'", action: "add_sample", payload: { concept: "Formula 1" } },
          { label: "List concepts", action: "list_concepts" },
          { label: "Help", action: "help" },
        ]
      );
    });

    function appendMessage(speaker, text, isUser = false) {
      const msg = document.createElement("div");
      msg.className = `chat-message ${isUser ? "user" : "assistant"}`;

      const spk = document.createElement("div");
      spk.className = "chat-speaker";
      spk.textContent = speaker;

      const body = document.createElement("div");
      body.textContent = text;

      msg.appendChild(spk);
      msg.appendChild(body);
      chatTranscript.appendChild(msg);
      chatTranscript.scrollTop = chatTranscript.scrollHeight;
    }

    function appendAssistantMessage(text, choices = []) {
      appendMessage("Assistant", text, false);
      renderChatChoices(choices);
    }

    function renderChatChoices(choices) {
      chatChoicesContainer.innerHTML = "";
      if (!choices || !choices.length) return;

      choices.forEach((c) => {
        const btn = document.createElement("button");
        btn.type = "button";
        btn.className = "chat-choice-chip";
        btn.textContent = c.label;
        btn.onclick = () => {
          if (c.action === "add_sample" && c.payload && c.payload.concept) {
            sendChatMessage(c.payload.concept);
          } else if (c.action === "select_text_provider" && c.payload && c.payload.provider) {
            sendChatMessage(`Use ${c.payload.provider}`);
          } else if (c.action === "select_image_source" && c.payload && c.payload.source) {
            sendChatMessage(`Use ${c.payload.source}`);
          } else if (c.action === "offer_image_sources") {
            sendChatMessage("Create images");
          } else if (c.action === "restart") {
            sendChatMessage("Start over");
          } else if (c.action === "export_stimuli") {
            sendChatMessage("export task stimuli");
          } else if (c.action === "list_concepts") {
            sendChatMessage("list concepts");
          } else if (c.action === "help") {
            sendChatMessage("help");
          } else {
            sendChatMessage(c.label);
          }
        };
        chatChoicesContainer.appendChild(btn);
      });
    }

    async function sendChatMessage(text) {
      if (!text || !text.trim()) return;
      const cleanText = text.trim();
      appendMessage("You", cleanText, true);
      chatInputText.value = "";

      try {
        const allKnownConcepts = Array.from(
          new Set([activeSeedConcept, ...Object.keys(workspaceTree)])
        ).filter(Boolean);

        const payload = {
          message: cleanText,
          state: chatState,
          current_concept: chatCurrentConcept || activeSeedConcept,
          workspace_concepts: allKnownConcepts,
        };

        const res = await fetch("/api/v1/chat/message", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
        });

        if (!res.ok) throw new Error("Assistant response failed");
        const data = await res.json();

        chatState = data.state;
        if (data.current_concept) {
          chatCurrentConcept = data.current_concept;
        }

        appendAssistantMessage(data.reply, data.choices);

        if (data.ui_action) {
          handleChatUiAction(data.ui_action, data.ui_payload);
        }
      } catch (err) {
        appendAssistantMessage(`Error: ${err.message}`);
      }
    }

    function handleChatUiAction(action, payload) {
      if (action === "add_related" && payload.concept && payload.related) {
        addRelatedConcepts(payload.concept, payload.related);
      } else if (action === "set_concept" && payload.concept) {
        ensureTreeNode(payload.concept, "manual");
        activeSeedConcept = payload.concept;
        conceptInput.value = payload.concept;
        renderConceptTree();
      } else if (action === "generate_concepts") {
        if (payload.concept) {
          ensureTreeNode(payload.concept, "manual");
          activeSeedConcept = payload.concept;
          conceptInput.value = payload.concept;
        }
        if (payload.provider) {
          providerSelect.value = payload.provider;
          updateProviderVisibility();
        }
        runGenerateRelatedConcepts();
      } else if (action === "acquire_images") {
        if (payload.source) {
          imageSourceSelect.value = payload.source;
          updateImageSourceVisibility();
        }
        runAcquisition(tasksForChecked());
      } else if (action === "delete_concept" && payload.concept) {
        delete workspaceTree[payload.concept];
        renderConceptTree();
      } else if (action === "crop_selected") {
        cropSelectedImages();
      } else if (action === "export_stimuli") {
        btnExportCsv.click();
      }
    }

    chatInputForm.addEventListener("submit", (e) => {
      e.preventDefault();
      sendChatMessage(chatInputText.value);
    });

    appendAssistantMessage(
      "Tell me a concept to add. I’ll guide you through related concepts and image creation.",
      [
        { label: "Add 'Formula 1'", action: "add_sample", payload: { concept: "Formula 1" } },
        { label: "List concepts", action: "list_concepts" },
        { label: "Help", action: "help" },
      ]
    );
  }

  // ==========================================
  // Workspace Splitter (Horizontal Resizing)
  // ==========================================
  function initWorkspaceSplitter() {
    const splitter = document.getElementById("workspace-splitter");
    const treePanel = document.querySelector(".concept-tree-panel");
    const splitLayout = document.querySelector(".workspace-split-layout");
    if (!splitter || !treePanel || !splitLayout) return;

    const DEFAULT_WIDTH = 440;
    const MIN_TREE_WIDTH = 280;
    const MIN_MAIN_WIDTH = 320;

    // Restore saved width from localStorage if valid
    try {
      const savedWidth = parseInt(localStorage.getItem("sf_tree_panel_width") || "", 10);
      if (!isNaN(savedWidth) && savedWidth >= MIN_TREE_WIDTH) {
        const maxAllowed = Math.max(MIN_TREE_WIDTH, splitLayout.clientWidth - MIN_MAIN_WIDTH);
        treePanel.style.width = `${Math.min(savedWidth, maxAllowed)}px`;
      }
    } catch {
      // localStorage may fail in some environments
    }

    let isDragging = false;
    let startX = 0;
    let startWidth = 0;

    function onMouseDown(e) {
      if (window.innerWidth <= 1080) return; // In mobile/stacked mode, no horizontal resize
      isDragging = true;
      startX = e.clientX;
      startWidth = treePanel.getBoundingClientRect().width;
      splitter.classList.add("dragging");
      document.body.classList.add("is-resizing-splitter");

      window.addEventListener("mousemove", onMouseMove);
      window.addEventListener("mouseup", onMouseUp);
      e.preventDefault();
    }

    function onMouseMove(e) {
      if (!isDragging) return;
      const deltaX = e.clientX - startX;
      let newWidth = startWidth + deltaX;

      const containerWidth = splitLayout.clientWidth;
      const maxTreeWidth = Math.max(MIN_TREE_WIDTH, containerWidth - MIN_MAIN_WIDTH);

      if (newWidth < MIN_TREE_WIDTH) newWidth = MIN_TREE_WIDTH;
      if (newWidth > maxTreeWidth) newWidth = maxTreeWidth;

      treePanel.style.width = `${newWidth}px`;
    }

    function onMouseUp() {
      if (!isDragging) return;
      isDragging = false;
      splitter.classList.remove("dragging");
      document.body.classList.remove("is-resizing-splitter");

      window.removeEventListener("mousemove", onMouseMove);
      window.removeEventListener("mouseup", onMouseUp);

      try {
        const finalWidth = Math.round(treePanel.getBoundingClientRect().width);
        localStorage.setItem("sf_tree_panel_width", finalWidth.toString());
      } catch {}
    }

    // Double-click resets to default width
    splitter.addEventListener("dblclick", () => {
      treePanel.style.width = `${DEFAULT_WIDTH}px`;
      try {
        localStorage.setItem("sf_tree_panel_width", DEFAULT_WIDTH.toString());
      } catch {}
    });

    // Keyboard support (Left / Right arrow keys)
    splitter.addEventListener("keydown", (e) => {
      if (e.key === "ArrowLeft" || e.key === "ArrowRight") {
        e.preventDefault();
        const currentWidth = treePanel.getBoundingClientRect().width;
        const delta = e.key === "ArrowRight" ? 30 : -30;
        const containerWidth = splitLayout.clientWidth;
        const maxTreeWidth = Math.max(MIN_TREE_WIDTH, containerWidth - MIN_MAIN_WIDTH);
        const newWidth = Math.min(maxTreeWidth, Math.max(MIN_TREE_WIDTH, currentWidth + delta));
        treePanel.style.width = `${newWidth}px`;
        try {
          localStorage.setItem("sf_tree_panel_width", Math.round(newWidth).toString());
        } catch {}
      }
    });

    splitter.addEventListener("mousedown", onMouseDown);
  }

  // ==========================================
  // 12. App Initialization
  // ==========================================
  checkHealth();
  scanHostModels();
  initDownloadFolder();
  initWorkspaceSplitter();
  setFaceDetectionEnabled(detectFacesToggle ? detectFacesToggle.checked : true);
  updateCropCardState();
  refreshTelemetry();
  setInterval(checkHealth, 30000);

  const rawSeed = conceptInput ? conceptInput.value.trim() : "basketball";
  const initialSeeds = (rawSeed || "basketball").split(/[\n,]+/).map((s) => s.trim()).filter(Boolean);
  initialSeeds.forEach((s) => ensureTreeNode(s, "manual"));
  const firstSeed = initialSeeds[0] || "basketball";
  activeSeedConcept = firstSeed;
  selectedFilterConcept = firstSeed;
  selectedTreeTarget = { type: "concept", concept: firstSeed, association: null };
  updateAcquisitionPlan();
  renderConceptTree();
});
