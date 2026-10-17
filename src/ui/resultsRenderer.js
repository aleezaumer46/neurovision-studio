/**
 * Results Renderer
 * Handles visualization of CNN predictions, confidence gauge, and probability distribution
 */
import confetti from 'canvas-confetti';
import { showToast } from './toast.js';

export class ResultsRenderer {
  constructor({
    heroCardEl,
    predictionTitleEl,
    confidencePctEl,
    confidenceCircleEl,
    latencyValEl,
    fpsValEl,
    backendValEl,
    probListEl,
    confLevelEl
  }) {
    this.heroCardEl = heroCardEl;
    this.predictionTitleEl = predictionTitleEl;
    this.confidencePctEl = confidencePctEl;
    this.confidenceCircleEl = confidenceCircleEl;
    this.latencyValEl = latencyValEl;
    this.fpsValEl = fpsValEl;
    this.backendValEl = backendValEl;
    this.probListEl = probListEl;
    this.confLevelEl = confLevelEl;

    this.showAllClasses = false;
    this.lastResult = null;
  }

  /**
   * Render inference results
   * @param {Object} result 
   */
  render(result) {
    this.lastResult = result;
    const {
      topClass,
      topConfidence,
      latencyMs,
      fps,
      backend,
      allPredictions
    } = result;

    const confNum = parseFloat(topConfidence);

    // 1. Update Hero Card
    if (this.predictionTitleEl) {
      this.predictionTitleEl.textContent = topClass;
    }

    if (this.confidencePctEl) {
      this.confidencePctEl.textContent = `${confNum}%`;
    }

    if (this.latencyValEl) {
      this.latencyValEl.textContent = `${latencyMs} ms`;
    }

    if (this.fpsValEl) {
      this.fpsValEl.textContent = `${fps} FPS`;
    }

    if (this.backendValEl) {
      this.backendValEl.textContent = backend.toUpperCase();
    }

    // Determine dynamic confidence qualitative status
    let levelText = 'High Confidence';
    let levelColor = '#10b981';

    if (result.isOOD || confNum < 40) {
      levelText = 'Uncertain / Low Confidence (Out-of-Distribution)';
      levelColor = '#f43f5e';
    } else if (confNum < 70) {
      levelText = 'Moderate Confidence';
      levelColor = '#f59e0b';
    } else {
      levelText = 'High Confidence';
      levelColor = '#10b981';
    }

    if (this.confLevelEl) {
      this.confLevelEl.textContent = levelText;
      this.confLevelEl.style.color = levelColor;
    }

    // 2. Animate Circular Progress (Circumference = 2 * PI * 40 ≈ 251.2)
    if (this.confidenceCircleEl) {
      const circumference = 251.2;
      const offset = circumference - (confNum / 100) * circumference;
      this.confidenceCircleEl.style.strokeDashoffset = offset;
    }

    // 3. Render Multi-Class Probability Bars
    this.renderProbabilities(allPredictions);

    // 4. Confetti trigger for high confidence (>85%) on new static predictions
    if (confNum >= 85) {
      this.triggerConfetti();
    }
  }

  /**
   * Render probability bars
   */
  renderProbabilities(predictions) {
    if (!this.probListEl) return;

    const displayCount = this.showAllClasses ? predictions.length : Math.min(5, predictions.length);
    const visiblePredictions = predictions.slice(0, displayCount);

    this.probListEl.innerHTML = visiblePredictions.map(item => {
      const pct = parseFloat(item.confidencePct);
      return `
        <div class="prob-row">
          <div class="prob-header">
            <span class="prob-name">
              <span class="prob-rank">#${item.rank}</span>
              ${item.icon ? `<span style="font-size: 1.1em">${item.icon}</span>` : ''}
              <span>${item.className}</span>
            </span>
            <span class="prob-percent">${pct.toFixed(1)}%</span>
          </div>
          <div class="prob-track">
            <div class="prob-fill" style="width: ${pct}%"></div>
          </div>
        </div>
      `;
    }).join('');
  }

  toggleShowAll() {
    this.showAllClasses = !this.showAllClasses;
    if (this.lastResult) {
      this.renderProbabilities(this.lastResult.allPredictions);
    }
    return this.showAllClasses;
  }

  triggerConfetti() {
    try {
      confetti({
        particleCount: 35,
        spread: 60,
        origin: { y: 0.7 },
        colors: ['#06b6d4', '#3b82f6', '#8b5cf6', '#10b981']
      });
    } catch (e) {
      // silently ignore if not supported
    }
  }

  /**
   * Clear results display
   */
  clear() {
    this.lastResult = null;
    if (this.predictionTitleEl) this.predictionTitleEl.textContent = 'Awaiting Image...';
    if (this.confidencePctEl) this.confidencePctEl.textContent = '0%';
    if (this.confidenceCircleEl) this.confidenceCircleEl.style.strokeDashoffset = '251.2';
    if (this.latencyValEl) this.latencyValEl.textContent = '-- ms';
    if (this.fpsValEl) this.fpsValEl.textContent = '-- FPS';
    if (this.probListEl) this.probListEl.innerHTML = '<div style="color: var(--text-dim); font-size: 0.85rem; text-align: center; padding: 1.5rem;">Upload or select an image to see class probabilities</div>';
  }

  /**
   * Export JSON report
   */
  exportReport() {
    if (!this.lastResult) {
      showToast('No prediction results to export.', 'error');
      return;
    }

    const report = {
      project: 'NeuroVision AI - Client-Side CNN Vision Application',
      preparedFor: 'Aleeza Umer (BSAI Student)',
      timestamp: new Date().toISOString(),
      model: this.lastResult.modelName,
      hardwareBackend: this.lastResult.backend,
      performance: {
        latencyMs: this.lastResult.latencyMs,
        throughputFps: this.lastResult.fps
      },
      topResult: {
        class: this.lastResult.topClass,
        confidencePct: this.lastResult.topConfidence
      },
      fullProbabilityDistribution: this.lastResult.allPredictions
    };

    const dataStr = 'data:text/json;charset=utf-8,' + encodeURIComponent(JSON.stringify(report, null, 2));
    const downloadAnchor = document.createElement('a');
    downloadAnchor.setAttribute('href', dataStr);
    downloadAnchor.setAttribute('download', `CNN_Vision_Report_${Date.now()}.json`);
    document.body.appendChild(downloadAnchor);
    downloadAnchor.click();
    downloadAnchor.remove();

    showToast('JSON Analysis Report downloaded!', 'success');
  }

  /**
   * Copy prediction to clipboard
   */
  async copyResult() {
    if (!this.lastResult) {
      showToast('No prediction to copy.', 'error');
      return;
    }
    const text = `NeuroVision AI CNN Classification:\nPredicted Class: ${this.lastResult.topClass}\nConfidence: ${this.lastResult.topConfidence}%\nInference Latency: ${this.lastResult.latencyMs} ms\nModel: ${this.lastResult.modelName}`;
    try {
      await navigator.clipboard.writeText(text);
      showToast('Classification summary copied to clipboard!', 'success');
    } catch (e) {
      showToast('Clipboard access denied.', 'error');
    }
  }
}
