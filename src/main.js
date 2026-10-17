/**
 * NeuroVision Studio Enterprise - Master Systems Orchestrator
 * Four-workspace Computer Vision and Explainable AI suite.
 */
import { modelService } from './ml/modelService.js';
import { objectDetector } from './ml/objectDetector.js';
import { computeGradCam, renderGradCamOverlay } from './ml/gradCam.js';
import { semanticSegmenter } from './ml/segmenter.js';
import { teachableStudio } from './ml/teachableStudio.js';
import { similarityEngine } from './ml/similarityEngine.js';
import { preprocessImage } from './ml/tensorUtils.js';

import { setupDropzone } from './ui/dropzone.js';
import { setupSampleGallery, SAMPLE_IMAGES } from './ui/sampleManager.js';
import { WebcamManager } from './ui/webcamManager.js';
import { ResultsRenderer } from './ui/resultsRenderer.js';
import { PipelineVisualizer } from './ui/pipelineVisualizer.js';
import { batchProcessor } from './ui/batchProcessor.js';
import { augmentationStudio } from './ui/augmentationStudio.js';
import { voiceSynthesizer } from './ui/voiceSynthesizer.js';
import { analyticsDashboard } from './ui/analyticsDashboard.js';
import { setupInfoModal } from './ui/infoModal.js';
import { showToast } from './ui/toast.js';

// DOM Elements: Header Controls
const pillMobilenet = document.getElementById('pillMobilenet');
const pillCustomCnn = document.getElementById('pillCustomCnn');
const activeModelBadge = document.getElementById('activeModelBadge');
const headerBackendText = document.getElementById('headerBackendText');
const specNormText = document.getElementById('specNormText');
const specBackendText = document.getElementById('specBackendText');
const btnVoiceToggle = document.getElementById('btnVoiceToggle');
const voiceStatusText = document.getElementById('voiceStatusText');
const btnPwaInstall = document.getElementById('btnPwaInstall');
const btnTriggerPwaPrompt = document.getElementById('btnTriggerPwaPrompt');

// Navigation Tabs & Four Workspaces
const studioTabBtns = document.querySelectorAll('.studio-tab-btn');
const moduleViews = document.querySelectorAll('.module-view');

// Page 1 Elements (Hub)
const tabBtnUpload = document.getElementById('tabBtnUpload');
const tabBtnWebcam = document.getElementById('tabBtnWebcam');
const uploadView = document.getElementById('uploadView');
const webcamView = document.getElementById('webcamView');
const dropzone = document.getElementById('dropzone');
const fileInput = document.getElementById('fileInput');
const btnBrowseTrigger = document.getElementById('btnBrowseTrigger');
const sampleGalleryContainer = document.getElementById('sampleGalleryContainer');
const webcamVideo = document.getElementById('webcamVideo');
const btnToggleCamera = document.getElementById('btnToggleCamera');
const btnSnapshotCamera = document.getElementById('btnSnapshotCamera');

const previewBox = document.getElementById('previewBox');
const previewPlaceholder = document.getElementById('previewPlaceholder');
const mainPreviewImg = document.getElementById('mainPreviewImg');
const previewBadgeOverlay = document.getElementById('previewBadgeOverlay');
const previewDimensionsBadge = document.getElementById('previewDimensionsBadge');
const previewSizeBadge = document.getElementById('previewSizeBadge');
const detectionCanvas = document.getElementById('detectionCanvas');
const webcamDetectionCanvas = document.getElementById('webcamDetectionCanvas');

const heroPredictionCard = document.getElementById('heroPredictionCard');
const predictedClassTitle = document.getElementById('predictedClassTitle');
const confidencePctVal = document.getElementById('confidencePctVal');
const confidenceProgressCircle = document.getElementById('confidenceProgressCircle');
const statLatencyVal = document.getElementById('statLatencyVal');
const statFpsVal = document.getElementById('statFpsVal');
const confQualitativeLevel = document.getElementById('confQualitativeLevel');
const probabilityDistributionList = document.getElementById('probabilityDistributionList');
const btnToggleAllClasses = document.getElementById('btnToggleAllClasses');

const btnRunDetection = document.getElementById('btnRunDetection');
const modeClassification = document.getElementById('modeClassification');
const modeDetection = document.getElementById('modeDetection');
const sliderDetConf = document.getElementById('sliderDetConf');
const detConfVal = document.getElementById('detConfVal');
const detectionSummaryText = document.getElementById('detectionSummaryText');
const detectedObjectsList = document.getElementById('detectedObjectsList');

const btnClearAll = document.getElementById('btnClearAll');
const btnRunInference = document.getElementById('btnRunInference');
const btnExportReport = document.getElementById('btnExportReport');
const btnCopyResult = document.getElementById('btnCopyResult');
const btnThemeToggle = document.getElementById('btnThemeToggle');

// Page 2 Elements (XAI & Semantic Lab)
const btnRunGradCam = document.getElementById('btnRunGradCam');
const gradCamCanvas = document.getElementById('gradCamCanvas');
const sliderGradCamOpacity = document.getElementById('sliderGradCamOpacity');
const gradCamOpacityVal = document.getElementById('gradCamOpacityVal');

const segModeBlur = document.getElementById('segModeBlur');
const segModeRemove = document.getElementById('segModeRemove');
const segModeChroma = document.getElementById('segModeChroma');
const sliderSegBlur = document.getElementById('sliderSegBlur');
const segBlurVal = document.getElementById('segBlurVal');
const btnApplySegmentation = document.getElementById('btnApplySegmentation');
const segmentationCanvas = document.getElementById('segmentationCanvas');
const segBlurControls = document.getElementById('segBlurControls');

const pipelinePreprocessedCanvas = document.getElementById('pipelinePreprocessedCanvas');
const pipelineEdgeCanvas = document.getElementById('pipelineEdgeCanvas');
const pipelineActivationCanvas = document.getElementById('pipelineActivationCanvas');
const pipelineDeepCanvas = document.getElementById('pipelineDeepCanvas');

// Page 3 Elements (Teachable & Similarity)
const btnTeachablePredict = document.getElementById('btnTeachablePredict');
const btnTeachableReset = document.getElementById('btnTeachableReset');
const teachableVerdictText = document.getElementById('teachableVerdictText');
let isTeachablePredicting = false;
let lastTeachableAnnouncement = '';
const btnRunSimilarity = document.getElementById('btnRunSimilarity');
const similarityResultsGrid = document.getElementById('similarityResultsGrid');

// Page 4 Elements (Batch & Adversarial)
const btnTriggerBatchSelect = document.getElementById('btnTriggerBatchSelect');
const batchFileInput = document.getElementById('batchFileInput');
const batchDropTarget = document.getElementById('batchDropTarget');
const btnExportBatchCSV = document.getElementById('btnExportBatchCSV');
const btnExportBatchJSON = document.getElementById('btnExportBatchJSON');
const batchProgressBarWrapper = document.getElementById('batchProgressBarWrapper');
const batchProgressBar = document.getElementById('batchProgressBar');
const batchProgressPctText = document.getElementById('batchProgressPctText');
const batchTableBody = document.getElementById('batchTableBody');

const sliderBright = document.getElementById('sliderBright');
const sliderContrast = document.getElementById('sliderContrast');
const sliderNoise = document.getElementById('sliderNoise');
const sliderBlur = document.getElementById('sliderBlur');
const sliderRot = document.getElementById('sliderRot');
const btnResetAugmentation = document.getElementById('btnResetAugmentation');
const augmentationCanvas = document.getElementById('augmentationCanvas');
const augVerdictBadge = document.getElementById('augVerdictBadge');

// Page 5 Elements (Telemetry & Analytics)
const btnRefreshAnalytics = document.getElementById('btnRefreshAnalytics');
const btnClearTelemetryLogs = document.getElementById('btnClearTelemetryLogs');
const analyticTotalCount = document.getElementById('analyticTotalCount');
const analyticAvgLatency = document.getElementById('analyticAvgLatency');
const analyticTopCategory = document.getElementById('analyticTopCategory');
const analyticsDistributionChart = document.getElementById('analyticsDistributionChart');
const analyticsTrendChart = document.getElementById('analyticsTrendChart');

// Application State
let currentMediaElement = null;
let currentImageMeta = null;
let isInferencing = false;
let webcamManager = null;
let lastGradCamHeatmap = null;
let deferredPrompt = null;
let inferenceMode = 'classification';

// Initialize Core Subsystems
const resultsRenderer = new ResultsRenderer({
  heroCardEl: heroPredictionCard,
  predictionTitleEl: predictedClassTitle,
  confidencePctEl: confidencePctVal,
  confidenceCircleEl: confidenceProgressCircle,
  latencyValEl: statLatencyVal,
  fpsValEl: statFpsVal,
  backendValEl: headerBackendText,
  probListEl: probabilityDistributionList,
  confLevelEl: confQualitativeLevel
});

const pipelineVisualizer = new PipelineVisualizer({
  preprocessedCanvas: pipelinePreprocessedCanvas,
  edgeMapCanvas: pipelineEdgeCanvas,
  activationMapCanvas: pipelineActivationCanvas,
  deepMapCanvas: pipelineDeepCanvas
});

/**
 * PAGE 1: Forward CNN Inference Pipeline (Memory Safe)
 */
async function runInference(mediaElement) {
  if (!mediaElement || isInferencing) return;
  if (inferenceMode === 'detection') {
    await runObjectDetection(mediaElement);
    return;
  }
  isInferencing = true;

  try {
    previewBox.classList.add('is-scanning');

    const result = await modelService.predict(mediaElement);

    // Update Result Visualizations
    resultsRenderer.render(result);

    // Update 4-Stage CNN Feature Pipeline Inspector (Page 2 Sync)
    pipelineVisualizer.update(mediaElement, modelService.customCnnModel);

    // Update Metrics Banner
    const bannerStat = document.getElementById('bannerStatLatency');
    if (bannerStat) bannerStat.textContent = `${result.latencyMs} ms`;

    // Voice Accessibility Announcement (Non-overlapping queues)
    voiceSynthesizer.announceVerdict(result.topClass, result.topConfidence, result.isOOD);

    // Async Telemetry Transmission to FastAPI Backend (Page 5 Sync)
    analyticsDashboard.logInference({
      model_name: result.modelName,
      module: 'classification',
      input_dimensions: currentImageMeta?.dimensions || '224 × 224 px',
      top_class: result.topClass,
      confidence: parseFloat(result.topConfidence) || 0,
      latency_ms: result.latencyMs,
      fps: result.fps,
      hardware_backend: result.backend,
      is_ood: result.isOOD || false
    });

  } catch (err) {
    console.error('Inference execution error:', err);
    showToast(`Inference error: ${err.message}`, 'error');
  } finally {
    isInferencing = false;
    previewBox.classList.remove('is-scanning');
  }
}

/**
 * Handle Loaded Image
 */
function handleImageLoaded(imgElement, meta) {
  currentMediaElement = imgElement;
  currentImageMeta = meta;

  previewPlaceholder.style.display = 'none';
  mainPreviewImg.style.display = 'block';
  mainPreviewImg.src = imgElement.src;

  previewBadgeOverlay.style.display = 'flex';
  previewDimensionsBadge.textContent = meta.dimensions || `${imgElement.naturalWidth} × ${imgElement.naturalHeight} px`;
  previewSizeBadge.textContent = meta.fileSize || 'Standard';

  // Run forward inference
  runInference(imgElement);

  // Sync to Adversarial Playground canvas (Page 4 Sync)
  updateAugmentationView();
}

/**
 * Four-workspace switcher
 */
function switchPage(pageId) {
  studioTabBtns.forEach(btn => {
    btn.classList.toggle('active', btn.getAttribute('data-page') === pageId);
  });
  moduleViews.forEach(view => {
    view.classList.toggle('active', view.id === pageId);
  });

  if (pageId === 'page4') {
    refreshAnalyticsView();
    updateAugmentationView();
  } else if (pageId === 'page2' && currentMediaElement) {
    pipelineVisualizer.update(currentMediaElement, modelService.customCnnModel);
  }
}

/**
 * PAGE 1: Real-Time Object Detection (COCO-SSD)
 */
async function runObjectDetection(mediaElement = currentMediaElement) {
  if (!mediaElement || isInferencing) {
    showToast('Please select or upload an image first.', 'info');
    return;
  }
  showToast('Running COCO-SSD Object Detection...', 'info');
  isInferencing = true;
  btnRunDetection.disabled = true;

  try {
    const result = await objectDetector.detect(mediaElement);
    const displayWidth = mediaElement.naturalWidth || mediaElement.videoWidth || 480;
    const displayHeight = mediaElement.naturalHeight || mediaElement.videoHeight || 360;

    const targetCanvas = mediaElement === webcamVideo ? webcamDetectionCanvas : detectionCanvas;
    objectDetector.renderBoundingBoxes(targetCanvas, result.detections, displayWidth, displayHeight);

    detectionSummaryText.textContent = `Located ${result.count} targets in ${result.latencyMs} ms (${result.fps} FPS)`;
    detectedObjectsList.innerHTML = result.detections.map((d, i) => `
      <li style="display: flex; justify-content: space-between; font-size: 0.82rem; padding: 4px 8px; background: rgba(255,255,255,0.03); border-radius: 4px;">
        <span style="font-weight: 600; color: #06b6d4;">#${i + 1} ${d.class}</span>
        <span style="font-family: var(--font-mono);">${Math.round(d.score * 100)}%</span>
      </li>
    `).join('');

    const detectedLabels = result.detections.slice(0, 3).map(item => item.class).join(', ');
    voiceSynthesizer.speak(result.count ? `Detected ${detectedLabels}` : 'No objects detected');
    showToast(`Detection complete: ${result.count} targets located`, 'success');

    // Telemetry log to FastAPI
    analyticsDashboard.logInference({
      model_name: 'COCO-SSD (MobileNetV2)',
      module: 'object_detection',
      input_dimensions: `${displayWidth} × ${displayHeight} px`,
      top_class: result.detections[0]?.class || 'None',
      confidence: result.detections[0] ? Math.round(result.detections[0].score * 100) : 0,
      latency_ms: result.latencyMs,
      inference_time_ms: result.latencyMs,
      top_label: result.detections[0]?.class || 'None',
      timestamp: new Date().toISOString(),
      fps: result.fps,
      hardware_backend: 'webgl',
      is_ood: false
    });
  } catch (err) {
    console.error('Detection error:', err);
    showToast(`Detection error: ${err.message}`, 'error');
  } finally {
    isInferencing = false;
    btnRunDetection.disabled = false;
  }
}

/**
 * PAGE 2: Grad-CAM Explainable AI Heatmap
 */
async function runGradCam() {
  if (!currentMediaElement) {
    showToast('Please select or upload an image first.', 'info');
    return;
  }
  showToast('Computing Conv2D activation gradients for Grad-CAM...', 'info');

  try {
    const tensor = preprocessImage(currentMediaElement, 'custom_cnn');
    try {
      if (!modelService.customCnnModel) throw new Error('The custom CNN LayersModel is not available.');
      lastGradCamHeatmap = computeGradCam(modelService.customCnnModel, tensor);
    } finally {
      tensor.dispose();
    }

    const opacity = (parseFloat(sliderGradCamOpacity.value) || 65) / 100;
    renderGradCamOverlay(currentMediaElement, gradCamCanvas, lastGradCamHeatmap, opacity);
    showToast('Grad-CAM Jet thermal heatmap generated!', 'success');
  } catch (err) {
    console.error('Grad-CAM error:', err);
    showToast(`Grad-CAM error: ${err.message}`, 'error');
  }
}

/**
 * PAGE 2: Semantic Segmentation Compositing
 */
function applySegmentation() {
  if (!currentMediaElement) {
    showToast('Please select or upload an image first.', 'info');
    return;
  }
  semanticSegmenter.process(currentMediaElement, segmentationCanvas);
  showToast(`Applied segmentation: ${semanticSegmenter.mode}`, 'success');
}

/**
 * PAGE 3: Visual Similarity Search (Reverse Image Lookup)
 */
async function runVisualSimilarity() {
  if (!currentMediaElement) {
    showToast('Please select or upload an image first.', 'info');
    return;
  }
  showToast('Extracting GAP feature vector & calculating cosine distance...', 'info');

  if (!similarityEngine.isIndexed) {
    await similarityEngine.buildIndex(modelService.mobilenetModel, SAMPLE_IMAGES);
  }

  const results = similarityEngine.search(modelService.mobilenetModel, currentMediaElement, 6);
  if (results.length === 0) {
    similarityResultsGrid.innerHTML = '<div style="grid-column: span 3; text-align: center; color: var(--text-dim);">No similarity matches found.</div>';
    return;
  }

  similarityResultsGrid.innerHTML = results.map(r => `
    <div class="sample-card" style="position: relative;">
      <span style="position: absolute; top: 8px; right: 8px; background: rgba(6, 182, 212, 0.85); color: #fff; font-size: 0.65rem; font-weight: 700; padding: 2px 6px; border-radius: 9999px;">
        ${r.similarityPct}% Match
      </span>
      <img src="${r.url}" alt="${r.title}" class="sample-thumb" />
      <div class="sample-caption">${r.title}</div>
    </div>
  `).join('');

  showToast('Similarity search complete!', 'success');
}

/**
 * PAGE 4: Adversarial Playground & Augmentation Updates
 */
function updateAugmentationView() {
  if (!currentMediaElement) return;
  augmentationStudio.applyAugmentation(currentMediaElement, augmentationCanvas);

  modelService.predict(augmentationCanvas).then(res => {
    augVerdictBadge.textContent = `${res.topClass} (${res.topConfidence}%) — Latency: ${res.latencyMs}ms`;
    augVerdictBadge.style.color = res.isOOD ? '#f43f5e' : '#06b6d4';
  }).catch(() => {});
}

/**
 * Analytics Dashboard & Charts Refresh
 */
async function refreshAnalyticsView() {
  const data = await analyticsDashboard.fetchAnalytics();
  analyticTotalCount.textContent = data.total_inferences;
  analyticAvgLatency.textContent = `${data.average_latency_ms} ms`;
  analyticTopCategory.textContent = data.top_classes[0]?.class_name || 'N/A';

  // Render both Chart.js charts
  analyticsDashboard.renderDistributionChart(analyticsDistributionChart, data);
  analyticsDashboard.renderLatencyTrendChart(analyticsTrendChart, data);
}

/**
 * Master Bootstrap Lifecycle
 */
async function bootstrap() {
  console.log('[NeuroVision Studio Enterprise] Initializing four-workspace Edge AI Suite...');
  setupInfoModal();

  // Avoid caching Vite source modules during development; version production shell caches.
  if ('serviceWorker' in navigator && import.meta.env.PROD) {
    navigator.serviceWorker.register('/sw.js').then(() => {
      console.log('[PWA] Service Worker active. Offline model caching enabled.');
    }).catch(err => {
      console.warn('[PWA] Service Worker registration failed:', err);
    });
  } else if ('serviceWorker' in navigator) {
    navigator.serviceWorker.getRegistrations().then(registrations => {
      registrations.forEach(registration => registration.unregister());
    });
  }

  // PWA Install Prompt Listeners
  window.addEventListener('beforeinstallprompt', (e) => {
    e.preventDefault();
    deferredPrompt = e;
    if (btnPwaInstall) btnPwaInstall.style.display = 'inline-flex';
  });

  const triggerInstall = () => {
    if (deferredPrompt) {
      deferredPrompt.prompt();
      deferredPrompt.userChoice.then(() => {
        deferredPrompt = null;
        if (btnPwaInstall) btnPwaInstall.style.display = 'none';
      });
    } else {
      showToast('PWA is already installed or offline-ready in browser.', 'info');
    }
  };
  btnPwaInstall?.addEventListener('click', triggerInstall);
  btnTriggerPwaPrompt?.addEventListener('click', triggerInstall);

  // Initialize TensorFlow.js WebGL Engine
  try {
    await modelService.initialize();
    const backend = modelService.backendName.toUpperCase();
    if (headerBackendText) headerBackendText.textContent = `${backend} 2.0 (GPU)`;
    if (specBackendText) specBackendText.textContent = `${backend} Accelerated`;
    showToast(`TensorFlow.js WebGL Engine Initialized`, 'success', 2500);
  } catch (err) {
    console.warn('Backend init warning:', err);
    headerBackendText.textContent = 'CPU Engine';
  }

  // Four-workspace navigation
  studioTabBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      const pageId = btn.getAttribute('data-page');
      switchPage(pageId);
    });
  });

  // Voice Assistant Audio Toggle (Page 1)
  btnVoiceToggle?.addEventListener('click', () => {
    const isNowActive = voiceSynthesizer.toggle();
    btnVoiceToggle.classList.toggle('active', isNowActive);
    voiceStatusText.textContent = isNowActive ? 'Voice: On' : 'Voice: Off';
    showToast(isNowActive ? 'Voice Assistant Enabled' : 'Voice Assistant Muted', 'info');
  });

  // Dropzone File Upload (Page 1)
  setupDropzone({
    dropzoneEl: dropzone,
    fileInputEl: fileInput,
    onImageSelected: ({ imgElement, fileName, fileSize, dimensions }) => {
      handleImageLoaded(imgElement, { fileName, fileSize, dimensions });
      showToast(`Loaded: ${fileName}`, 'success');
    }
  });

  btnBrowseTrigger?.addEventListener('click', (e) => {
    e.stopPropagation();
    fileInput.click();
  });

  // Preset Benchmark Gallery (Page 1)
  setupSampleGallery(sampleGalleryContainer, (sample) => {
    const img = new Image();
    img.crossOrigin = 'anonymous';
    img.onload = () => {
      handleImageLoaded(img, {
        fileName: sample.title,
        fileSize: 'Benchmark Asset',
        dimensions: `${img.naturalWidth} × ${img.naturalHeight} px`
      });
      showToast(`Loaded sample: ${sample.title}`, 'info');
    };
    img.src = sample.url;
  });

  // Live Webcam Scanner (Page 1)
  webcamManager = new WebcamManager({
    videoElement: webcamVideo,
    onFramePredict: (videoEl) => {
      runInference(videoEl);
    }
  });

  btnToggleCamera?.addEventListener('click', async () => {
    if (webcamManager.isRunning) {
      webcamManager.stopCamera();
      btnToggleCamera.innerHTML = `<span>Start Live Inference</span>`;
      btnToggleCamera.classList.remove('btn-secondary');
      btnToggleCamera.classList.add('btn-primary');
      btnSnapshotCamera.disabled = true;
    } else {
      const ok = await webcamManager.startCamera();
      if (ok) {
        btnToggleCamera.innerHTML = `<span>Stop Camera</span>`;
        btnToggleCamera.classList.remove('btn-primary');
        btnToggleCamera.classList.add('btn-secondary');
        btnSnapshotCamera.disabled = false;
        currentMediaElement = webcamVideo;
      }
    }
  });

  btnSnapshotCamera?.addEventListener('click', () => {
    if (!webcamManager.isRunning) return;
    const snapCanvas = document.createElement('canvas');
    snapCanvas.width = webcamVideo.videoWidth || 640;
    snapCanvas.height = webcamVideo.videoHeight || 480;
    const ctx = snapCanvas.getContext('2d');
    ctx.drawImage(webcamVideo, 0, 0, snapCanvas.width, snapCanvas.height);

    const img = new Image();
    img.onload = () => {
      webcamManager.stopCamera();
      btnToggleCamera.innerHTML = `<span>Start Live Inference</span>`;
      btnSnapshotCamera.disabled = true;
      switchInputTab('upload');
      handleImageLoaded(img, {
        fileName: 'Webcam Snapshot',
        fileSize: 'Live Stream Frame',
        dimensions: `${img.naturalWidth} × ${img.naturalHeight} px`
      });
      showToast('Camera frame frozen for inspection', 'success');
    };
    img.src = snapCanvas.toDataURL('image/jpeg');
  });

  function switchInputTab(tab) {
    if (tab === 'upload') {
      tabBtnUpload.classList.add('active');
      tabBtnWebcam.classList.remove('active');
      uploadView.style.display = 'block';
      webcamView.classList.remove('active');
      if (webcamManager?.isRunning) webcamManager.stopCamera();
    } else {
      tabBtnWebcam.classList.add('active');
      tabBtnUpload.classList.remove('active');
      uploadView.style.display = 'none';
      webcamView.classList.add('active');
    }
  }

  tabBtnUpload?.addEventListener('click', () => switchInputTab('upload'));
  tabBtnWebcam?.addEventListener('click', () => switchInputTab('webcam'));

  // Unified Model Switcher Pills (Page 1)
  function switchModel(modelId) {
    modelService.setActiveModel(modelId);
    pillMobilenet.classList.toggle('active', modelId === 'mobilenet');
    pillCustomCnn.classList.toggle('active', modelId === 'custom_cnn');

    if (modelId === 'mobilenet') {
      activeModelBadge.textContent = 'MobileNetV2 Active';
      if (specNormText) specNormText.textContent = '[-1.0, +1.0] Standard';
      showToast('Switched to MobileNetV2 (1000 ImageNet Classes)', 'info');
    } else {
      activeModelBadge.textContent = 'Vehicle Vision CNN Active';
      if (specNormText) specNormText.textContent = '[0.0, 1.0] Normalized';
      showToast('Switched to Vehicle Vision CNN (10 Transportation Classes)', 'info');
    }

    if (currentMediaElement) runInference(currentMediaElement);
  }

  pillMobilenet?.addEventListener('click', () => switchModel('mobilenet'));
  pillCustomCnn?.addEventListener('click', () => switchModel('custom_cnn'));

  // Hub Action Buttons
  btnRunInference?.addEventListener('click', () => {
    if (!currentMediaElement) {
      showToast('Please select or upload an image first.', 'info');
      return;
    }
    runInference(currentMediaElement);
  });

  btnClearAll?.addEventListener('click', () => {
    currentMediaElement = null;
    currentImageMeta = null;
    fileInput.value = '';
    mainPreviewImg.style.display = 'none';
    previewPlaceholder.style.display = 'flex';
    previewBadgeOverlay.style.display = 'none';
    const ctx = detectionCanvas.getContext('2d');
    ctx.clearRect(0, 0, detectionCanvas.width, detectionCanvas.height);
    resultsRenderer.clear();
    pipelineVisualizer.clear();
    showToast('Dashboard cleared.', 'info');
  });

  btnExportReport?.addEventListener('click', () => resultsRenderer.exportReport());
  btnCopyResult?.addEventListener('click', () => resultsRenderer.copyResult());

  btnToggleAllClasses?.addEventListener('click', () => {
    const isAll = resultsRenderer.toggleShowAll();
    btnToggleAllClasses.textContent = isAll ? 'Show Top 5 Only' : 'Show All Classes';
  });

  // Object Detection Wireup (Page 1)
  btnRunDetection?.addEventListener('click', runObjectDetection);
  const setInferenceMode = (mode) => {
    inferenceMode = mode;
    const detecting = mode === 'detection';
    modeClassification?.classList.toggle('active', !detecting);
    modeDetection?.classList.toggle('active', detecting);
    modeClassification?.setAttribute('aria-pressed', String(!detecting));
    modeDetection?.setAttribute('aria-pressed', String(detecting));
    btnRunDetection.style.display = detecting ? 'block' : 'none';
    btnRunInference.textContent = detecting ? 'Run Object Detection' : 'Re-Run Inference';
    detectionCanvas.getContext('2d').clearRect(0, 0, detectionCanvas.width, detectionCanvas.height);
    webcamDetectionCanvas.getContext('2d').clearRect(0, 0, webcamDetectionCanvas.width, webcamDetectionCanvas.height);
    if (currentMediaElement) runInference(currentMediaElement);
  };
  modeClassification?.addEventListener('click', () => setInferenceMode('classification'));
  modeDetection?.addEventListener('click', () => setInferenceMode('detection'));
  sliderDetConf?.addEventListener('input', (e) => {
    const val = parseInt(e.target.value, 10);
    detConfVal.textContent = `${val}%`;
    objectDetector.minConfidence = val / 100;
  });

  // Grad-CAM Wireup (Page 2)
  btnRunGradCam?.addEventListener('click', runGradCam);
  sliderGradCamOpacity?.addEventListener('input', (e) => {
    const val = parseInt(e.target.value, 10);
    gradCamOpacityVal.textContent = `${val}%`;
    if (lastGradCamHeatmap && currentMediaElement) {
      renderGradCamOverlay(currentMediaElement, gradCamCanvas, lastGradCamHeatmap, val / 100);
    }
  });

  // Semantic Segmentation Wireup (Page 2)
  segModeBlur?.addEventListener('click', () => {
    semanticSegmenter.mode = 'blur';
    segModeBlur.classList.add('active');
    segModeRemove.classList.remove('active');
    segModeChroma.classList.remove('active');
    segBlurControls.style.display = 'flex';
    applySegmentation();
  });
  segModeRemove?.addEventListener('click', () => {
    semanticSegmenter.mode = 'remove';
    segModeRemove.classList.add('active');
    segModeBlur.classList.remove('active');
    segModeChroma.classList.remove('active');
    segBlurControls.style.display = 'none';
    applySegmentation();
  });
  segModeChroma?.addEventListener('click', () => {
    semanticSegmenter.mode = 'chroma';
    segModeChroma.classList.add('active');
    segModeBlur.classList.remove('active');
    segModeRemove.classList.remove('active');
    segBlurControls.style.display = 'none';
    applySegmentation();
  });
  sliderSegBlur?.addEventListener('input', (e) => {
    const val = parseInt(e.target.value, 10);
    segBlurVal.textContent = `${val} px`;
    semanticSegmenter.blurRadius = val;
    if (semanticSegmenter.mode === 'blur') applySegmentation();
  });
  btnApplySegmentation?.addEventListener('click', applySegmentation);

  // Teachable Studio Wireup (Page 3)
  document.querySelectorAll('.btn-capture-sample').forEach(btn => {
    btn.addEventListener('click', async () => {
      if (!currentMediaElement) {
        showToast('Load an image or start camera first.', 'info');
        return;
      }
      const classId = parseInt(btn.getAttribute('data-class-id'), 10);
      try {
        const count = await teachableStudio.addExample(modelService.mobilenetModel, currentMediaElement, classId);
        document.getElementById(`teachableCount${classId}`).textContent = `${count} Samples`;
        showToast(`Captured sample for Class ${classId + 1}`, 'success');
      } catch (err) {
        showToast(`Capture error: ${err.message}`, 'error');
      }
    });
  });

  const renderTeachablePrediction = (prediction) => {
    const title = document.createElement('strong');
    title.textContent = `${prediction.className} (${prediction.confidencePct}%)`;
    title.style.color = '#06b6d4';
    teachableVerdictText.replaceChildren(title);
    prediction.breakdown.forEach(item => {
      const row = document.createElement('div');
      row.className = 'teachable-probability-row';
      const label = document.createElement('span');
      label.textContent = item.name;
      const percent = document.createElement('span');
      percent.textContent = `${item.confidencePct}%`;
      const track = document.createElement('div');
      track.className = 'prob-track';
      const fill = document.createElement('div');
      fill.className = 'prob-fill';
      fill.style.width = `${item.confidencePct}%`;
      track.appendChild(fill);
      row.append(label, percent, track);
      teachableVerdictText.appendChild(row);
    });
  };

  const predictTeachableFrame = async () => {
    if (!isTeachablePredicting) return;
    if (!currentMediaElement) {
      isTeachablePredicting = false;
      btnTeachablePredict.textContent = 'Start Predicting';
      showToast('Load an image or start the camera first.', 'info');
      return;
    }
    try {
      const prediction = await teachableStudio.predict(modelService.mobilenetModel, currentMediaElement);
      if (!prediction.hasTrainedClasses) {
        teachableVerdictText.textContent = prediction.message;
        isTeachablePredicting = false;
        btnTeachablePredict.textContent = 'Start Predicting';
        showToast(prediction.message, 'error');
        return;
      }
      renderTeachablePrediction(prediction);
      const announcement = `${prediction.className} ${prediction.confidencePct}`;
      if (announcement !== lastTeachableAnnouncement) {
        voiceSynthesizer.speak(`Teachable Studio: ${prediction.className}, ${prediction.confidencePct} percent`);
        lastTeachableAnnouncement = announcement;
      }
    } catch (error) {
      isTeachablePredicting = false;
      btnTeachablePredict.textContent = 'Start Predicting';
      showToast(`Prediction error: ${error.message}`, 'error');
      return;
    }
    if (isTeachablePredicting) requestAnimationFrame(predictTeachableFrame);
  };

  btnTeachablePredict?.addEventListener('click', () => {
    isTeachablePredicting = !isTeachablePredicting;
    btnTeachablePredict.textContent = isTeachablePredicting ? 'Stop Predicting' : 'Start Predicting';
    if (isTeachablePredicting) predictTeachableFrame();
  });

  document.querySelectorAll('.teachable-class-name').forEach(input => {
    input.addEventListener('input', () => {
      teachableStudio.setClassName(Number(input.dataset.classId), input.value);
    });
  });

  btnTeachableReset?.addEventListener('click', () => {
    isTeachablePredicting = false;
    btnTeachablePredict.textContent = 'Start Predicting';
    teachableStudio.resetAll();
    [0, 1, 2].forEach(id => {
      document.getElementById(`teachableCount${id}`).textContent = '0 Samples';
    });
    teachableVerdictText.textContent = 'Dataset reset. Ready to capture samples.';
    showToast('Teachable Studio dataset reset.', 'info');
  });

  // Visual Similarity Search Wireup (Page 3)
  btnRunSimilarity?.addEventListener('click', runVisualSimilarity);

  // Multi-File Batch Processor Wireup (Page 4)
  btnTriggerBatchSelect?.addEventListener('click', () => batchFileInput.click());
  const processBatchSelection = (files) => {
    if (!files || files.length === 0) return;
    batchProgressBarWrapper.style.display = 'block';

    batchProcessor.processFiles(
      files,
      ({ current, total, percent, lastResult }) => {
        batchProgressBar.style.width = `${percent}%`;
        batchProgressPctText.textContent = `${percent}% (${current}/${total})`;

        if (current === 1) batchTableBody.innerHTML = '';
        const row = document.createElement('tr');
        const cells = [lastResult.id, lastResult.fileName, lastResult.predictedClass,
          `${lastResult.confidencePct}%`, `${lastResult.latencyMs} ms`, lastResult.modelUsed];
        const indexCell = document.createElement('td');
        indexCell.textContent = `#${cells[0]}`;
        row.appendChild(indexCell);
        const imageCell = document.createElement('td');
        if (lastResult.thumbnailUrl) {
          const thumbnail = document.createElement('img');
          thumbnail.src = lastResult.thumbnailUrl;
          thumbnail.alt = '';
          thumbnail.width = 32;
          thumbnail.height = 32;
          thumbnail.style.objectFit = 'cover';
          imageCell.appendChild(thumbnail);
        }
        row.appendChild(imageCell);
        cells.slice(1).forEach((value, index) => {
          const cell = document.createElement('td');
          cell.textContent = String(value);
          if (index === 1) cell.style.fontWeight = '700';
          row.appendChild(cell);
        });
        batchTableBody.appendChild(row);
      },
      () => {
        batchProgressBarWrapper.style.display = 'none';
      }
    );
  };
  batchFileInput?.addEventListener('change', () => processBatchSelection(batchFileInput.files));
  batchDropTarget?.addEventListener('dragover', event => {
    event.preventDefault();
    batchDropTarget.classList.add('drag-active');
  });
  batchDropTarget?.addEventListener('dragleave', () => batchDropTarget.classList.remove('drag-active'));
  batchDropTarget?.addEventListener('drop', event => {
    event.preventDefault();
    batchDropTarget.classList.remove('drag-active');
    processBatchSelection(event.dataTransfer.files);
  });
  btnExportBatchCSV?.addEventListener('click', () => batchProcessor.exportCSV());
  btnExportBatchJSON?.addEventListener('click', () => batchProcessor.exportJSON());

  // Adversarial Playground Wireup (Page 4)
  const setupAugSlider = (slider, param, labelId, suffix) => {
    slider?.addEventListener('input', (e) => {
      const val = e.target.value;
      document.getElementById(labelId).textContent = `${val}${suffix}`;
      augmentationStudio.setParam(param, val);
      updateAugmentationView();
    });
  };
  setupAugSlider(sliderNoise, 'noise', 'valNoise', '%');
  setupAugSlider(sliderContrast, 'contrast', 'valContrast', '%');
  setupAugSlider(sliderBright, 'brightness', 'valBright', '%');
  setupAugSlider(sliderBlur, 'blur', 'valBlur', ' px');
  setupAugSlider(sliderRot, 'rotation', 'valRot', '°');

  btnResetAugmentation?.addEventListener('click', () => {
    augmentationStudio.resetParams();
    sliderNoise.value = 0; document.getElementById('valNoise').textContent = '0%';
    sliderContrast.value = 100; document.getElementById('valContrast').textContent = '100%';
    sliderBright.value = 100; document.getElementById('valBright').textContent = '100%';
    sliderBlur.value = 0; document.getElementById('valBlur').textContent = '0 px';
    sliderRot.value = 0; document.getElementById('valRot').textContent = '0°';
    updateAugmentationView();
    showToast('Perturbations reset to standard baseline.', 'info');
  });

  // Telemetry & Analytics Dashboard Wireup (Batch & Telemetry workspace)
  btnRefreshAnalytics?.addEventListener('click', refreshAnalyticsView);
  btnClearTelemetryLogs?.addEventListener('click', async () => {
    try {
      await fetch('http://127.0.0.1:8000/api/clear-logs', { method: 'DELETE' });
    } catch (e) {}
    localStorage.removeItem('neurovision_telemetry_logs');
    refreshAnalyticsView();
    showToast('Telemetry database cleared.', 'info');
  });

  // Theme Toggle
  btnThemeToggle?.addEventListener('click', () => {
    const html = document.documentElement;
    const currentTheme = html.getAttribute('data-theme') || 'dark';
    const nextTheme = currentTheme === 'dark' ? 'light' : 'dark';
    html.setAttribute('data-theme', nextTheme);
    localStorage.setItem('neurovision_theme', nextTheme);
  });

  const savedTheme = localStorage.getItem('neurovision_theme');
  if (savedTheme) document.documentElement.setAttribute('data-theme', savedTheme);

  // Auto-evaluate first sample on boot
  setTimeout(() => {
    const firstSample = sampleGalleryContainer.querySelector('.sample-card');
    if (firstSample && !currentMediaElement) {
      firstSample.click();
    }
  }, 500);
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', bootstrap);
} else {
  bootstrap();
}
