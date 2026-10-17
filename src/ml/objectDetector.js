/**
 * Module 1: Real-Time Object Detection & Multi-Label Inference
 * Powered by TensorFlow.js COCO-SSD
 */
class ObjectDetector {
  constructor() {
    this.model = null;
    this.isLoaded = false;
    this.isLoading = false;
    this.minConfidence = 0.50;
    this.maxDetections = 20;

    // High-contrast color palette for bounding box overlays
    this.colorPalette = [
      '#06b6d4', '#3b82f6', '#8b5cf6', '#ec4899',
      '#f59e0b', '#10b981', '#14b8a6', '#6366f1',
      '#f43f5e', '#84cc16', '#eab308', '#a855f7'
    ];
  }

  async loadModel() {
    if (this.isLoaded) return;
    if (this.isLoading) return;
    this.isLoading = true;

    try {
      console.log('[COCO-SSD] Loading Object Detection model...');
      const cocoSsd = await import('@tensorflow-models/coco-ssd');
      this.model = await cocoSsd.load({
        base: 'mobilenet_v2'
      });
      this.isLoaded = true;
      console.log('[COCO-SSD] Model loaded successfully!');
    } catch (err) {
      console.error('[COCO-SSD] Model load error:', err);
      throw err;
    } finally {
      this.isLoading = false;
    }
  }

  /**
   * Run object detection on an image, video, or canvas element
   * @param {HTMLImageElement|HTMLVideoElement|HTMLCanvasElement} mediaElement 
   * @returns {Promise<Array>} Array of detections
   */
  async detect(mediaElement) {
    if (!this.isLoaded) {
      await this.loadModel();
    }

    const startTime = performance.now();
    const rawDetections = await this.model.detect(mediaElement, this.maxDetections, this.minConfidence);
    const latencyMs = Math.round(performance.now() - startTime);

    return {
      detections: rawDetections,
      count: rawDetections.length,
      latencyMs,
      fps: Math.round(1000 / Math.max(latencyMs, 1))
    };
  }

  /**
   * Render vector bounding boxes directly over an HTML5 canvas
   */
  renderBoundingBoxes(canvas, detections, displayWidth, displayHeight) {
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    canvas.width = displayWidth;
    canvas.height = displayHeight;
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    if (!detections || detections.length === 0) return;

    detections.forEach((det, idx) => {
      const [x, y, width, height] = det.bbox;
      const scorePct = Math.round(det.score * 100);
      const color = this.colorPalette[idx % this.colorPalette.length];

      // Draw bounding box
      ctx.lineWidth = 3;
      ctx.strokeStyle = color;
      ctx.shadowColor = color;
      ctx.shadowBlur = 8;
      ctx.strokeRect(x, y, width, height);

      // Label background
      ctx.shadowBlur = 0;
      const labelText = `${det.class} ${scorePct}%`;
      ctx.font = '600 13px "Plus Jakarta Sans", sans-serif';
      const textWidth = ctx.measureText(labelText).width;
      const textHeight = 22;

      ctx.fillStyle = color;
      ctx.beginPath();
      ctx.roundRect(x, Math.max(0, y - textHeight - 2), textWidth + 16, textHeight, [4, 4, 0, 0]);
      ctx.fill();

      // Label text
      ctx.fillStyle = '#ffffff';
      ctx.fillText(labelText, x + 8, Math.max(15, y - 7));
    });
  }
}

export const objectDetector = new ObjectDetector();
