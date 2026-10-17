/**
 * Module 7: Batch Inference & Dataset Report Exporter
 * Handles asynchronous bulk processing (up to 30 images) and CSV/JSON export
 */
import { modelService } from '../ml/modelService.js';
import { showToast } from './toast.js';

export class BatchProcessor {
  constructor() {
    this.queue = [];
    this.results = [];
    this.isProcessing = false;
  }

  /**
   * Enqueue files for batch evaluation
   */
  async processFiles(files, onProgress, onComplete) {
    if (this.isProcessing) {
      showToast('A batch evaluation is already in progress.', 'error');
      return;
    }

    const imageFiles = Array.from(files).filter(f => f.type.startsWith('image/')).slice(0, 30);
    if (imageFiles.length === 0) {
      showToast('No valid image files selected.', 'error');
      return;
    }

    this.isProcessing = true;
    this.results = [];
    const total = imageFiles.length;

    for (let i = 0; i < total; i++) {
      const file = imageFiles[i];

      try {
        const img = await this.loadImageFromFile(file);
        const result = await modelService.predict(img);

        this.results.push({
          id: i + 1,
          fileName: file.name,
          fileSize: (file.size / 1024).toFixed(1) + ' KB',
          thumbnailUrl: img.src,
          predictedClass: result.topClass,
          confidencePct: result.topConfidence,
          latencyMs: result.latencyMs,
          modelUsed: result.modelName,
          isOOD: result.isOOD || false
        });
      } catch (err) {
        console.warn(`Failed batch processing for ${file.name}:`, err);
        this.results.push({
          id: i + 1,
          fileName: file.name,
          fileSize: (file.size / 1024).toFixed(1) + ' KB',
          thumbnailUrl: '',
          predictedClass: 'Error / Failed',
          confidencePct: '0.0',
          latencyMs: 0,
          modelUsed: 'N/A',
          isOOD: true
        });
      }

      if (onProgress) {
        onProgress({
          current: i + 1,
          total,
          percent: Math.round(((i + 1) / total) * 100),
          lastResult: this.results[this.results.length - 1]
        });
      }

      if (i < total - 1) await new Promise(resolve => requestAnimationFrame(resolve));
    }

    this.isProcessing = false;
    showToast(`Batch processing completed for ${total} images!`, 'success');
    if (onComplete) onComplete(this.results);
  }

  loadImageFromFile(file) {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = (e) => {
        const img = new Image();
        img.onload = () => resolve(img);
        img.onerror = reject;
        img.src = e.target.result;
      };
      reader.onerror = reject;
      reader.readAsDataURL(file);
    });
  }

  exportCSV() {
    if (this.results.length === 0) {
      showToast('No batch results available to export.', 'error');
      return;
    }

    const headers = ['ID', 'File Name', 'File Size', 'Predicted Class', 'Confidence (%)', 'Latency (ms)', 'Model Used', 'Is OOD'];
    const rows = this.results.map(r => [
      r.id,
      `"${r.fileName}"`,
      `"${r.fileSize}"`,
      `"${r.predictedClass.replace(/[^a-zA-Z0-9 /]/g, '').trim()}"`,
      r.confidencePct,
      r.latencyMs,
      `"${r.modelUsed}"`,
      r.isOOD ? 'YES' : 'NO'
    ]);

    const csvContent = 'data:text/csv;charset=utf-8,' + [headers.join(','), ...rows.map(e => e.join(','))].join('\n');
    const encodedUri = encodeURI(csvContent);
    const link = document.createElement('a');
    link.setAttribute('href', encodedUri);
    link.setAttribute('download', `NeuroVision_Batch_Report_${Date.now()}.csv`);
    document.body.appendChild(link);
    link.click();
    link.remove();

    showToast('CSV Dataset Report exported!', 'success');
  }

  exportJSON() {
    if (this.results.length === 0) {
      showToast('No batch results available to export.', 'error');
      return;
    }

    const payload = {
      project: 'NeuroVision Studio Enterprise',
      timestamp: new Date().toISOString(),
      totalImagesProcessed: this.results.length,
      batchResults: this.results
    };

    const dataStr = 'data:text/json;charset=utf-8,' + encodeURIComponent(JSON.stringify(payload, null, 2));
    const link = document.createElement('a');
    link.setAttribute('href', dataStr);
    link.setAttribute('download', `NeuroVision_Batch_Report_${Date.now()}.json`);
    document.body.appendChild(link);
    link.click();
    link.remove();

    showToast('JSON Dataset Report exported!', 'success');
  }
}

export const batchProcessor = new BatchProcessor();
