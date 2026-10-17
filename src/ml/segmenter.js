/**
 * Module 3: Real-Time Semantic Segmentation & Background Processing
 * Features: Bokeh Blur, Full Background Removal, Chroma-Key Solid Color Replacement
 */

export class SemanticSegmenter {
  constructor() {
    this.mode = 'blur'; // 'blur' | 'remove' | 'chroma'
    this.blurRadius = 12;
    this.chromaColor = '#06b6d4'; // default studio cyan
    this.edgeSoftness = 0.8;
  }

  /**
   * Generates a foreground binary/soft mask from the media element
   * Using adaptive edge-aware color saliency segmentation
   */
  generateForegroundMask(ctx, width, height) {
    const imgData = ctx.getImageData(0, 0, width, height);
    const data = imgData.data;
    const mask = new Float32Array(width * height);

    // Estimate background color from outer perimeter borders
    let bgR = 0, bgG = 0, bgB = 0, borderCount = 0;
    for (let x = 0; x < width; x++) {
      // Top row
      let idx = (x) * 4;
      bgR += data[idx]; bgG += data[idx + 1]; bgB += data[idx + 2];
      // Bottom row
      idx = ((height - 1) * width + x) * 4;
      bgR += data[idx]; bgG += data[idx + 1]; bgB += data[idx + 2];
      borderCount += 2;
    }
    for (let y = 1; y < height - 1; y++) {
      // Left col
      let idx = (y * width) * 4;
      bgR += data[idx]; bgG += data[idx + 1]; bgB += data[idx + 2];
      // Right col
      idx = (y * width + (width - 1)) * 4;
      bgR += data[idx]; bgG += data[idx + 1]; bgB += data[idx + 2];
      borderCount += 2;
    }
    bgR /= borderCount;
    bgG /= borderCount;
    bgB /= borderCount;

    // Center focal prior
    const cx = width / 2;
    const cy = height / 2;
    const maxRadius = Math.sqrt(cx * cx + cy * cy);

    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        const i = y * width + x;
        const pIdx = i * 4;

        const r = data[pIdx];
        const g = data[pIdx + 1];
        const b = data[pIdx + 2];

        // Color distance from perimeter background
        const colorDist = Math.sqrt((r - bgR) ** 2 + (g - bgG) ** 2 + (b - bgB) ** 2);

        // Center spatial prior
        const distFromCenter = Math.sqrt((x - cx) ** 2 + (y - cy) ** 2);
        const spatialWeight = 1.0 - Math.min(1.0, distFromCenter / (maxRadius * 0.95));

        // Combined saliency
        const score = (colorDist / 255.0) * 0.6 + spatialWeight * 0.4;
        mask[i] = score > 0.38 ? 1.0 : (score > 0.28 ? (score - 0.28) / 0.1 : 0.0);
      }
    }

    return mask;
  }

  /**
   * Applies the selected segmentation compositing effect to target canvas
   */
  process(sourceMedia, targetCanvas, mode = null) {
    if (!targetCanvas || !sourceMedia) return;
    const currentMode = mode || this.mode;
    const width = 480;
    const height = Math.round((sourceMedia.videoHeight || sourceMedia.naturalHeight || 360) / (sourceMedia.videoWidth || sourceMedia.naturalWidth || 480) * width);

    targetCanvas.width = width;
    targetCanvas.height = height;
    const ctx = targetCanvas.getContext('2d');

    // 1. Draw original image
    ctx.drawImage(sourceMedia, 0, 0, width, height);

    // 2. Generate mask
    const mask = this.generateForegroundMask(ctx, width, height);

    if (currentMode === 'blur') {
      // Create blurred background buffer
      const blurCanvas = document.createElement('canvas');
      blurCanvas.width = width;
      blurCanvas.height = height;
      const bCtx = blurCanvas.getContext('2d');
      bCtx.filter = `blur(${this.blurRadius}px)`;
      bCtx.drawImage(sourceMedia, 0, 0, width, height);

      const blurImgData = bCtx.getImageData(0, 0, width, height);
      const blurPixels = blurImgData.data;

      const fgImgData = ctx.getImageData(0, 0, width, height);
      const fgPixels = fgImgData.data;

      // Composite foreground over blurred background
      for (let i = 0; i < mask.length; i++) {
        const m = mask[i]; // 1.0 = fg, 0.0 = bg
        const p = i * 4;
        fgPixels[p] = Math.round(fgPixels[p] * m + blurPixels[p] * (1 - m));
        fgPixels[p + 1] = Math.round(fgPixels[p + 1] * m + blurPixels[p + 1] * (1 - m));
        fgPixels[p + 2] = Math.round(fgPixels[p + 2] * m + blurPixels[p + 2] * (1 - m));
      }
      ctx.putImageData(fgImgData, 0, 0);

    } else if (currentMode === 'remove') {
      // Transparent background
      const fgImgData = ctx.getImageData(0, 0, width, height);
      const fgPixels = fgImgData.data;

      for (let i = 0; i < mask.length; i++) {
        const m = mask[i];
        const p = i * 4;
        fgPixels[p + 3] = Math.round(fgPixels[p + 3] * m); // Alpha channel
      }
      ctx.clearRect(0, 0, width, height);
      ctx.putImageData(fgImgData, 0, 0);

    } else if (currentMode === 'chroma') {
      // Solid color replacement
      const hex = this.chromaColor.replace('#', '');
      const cr = parseInt(hex.substring(0, 2), 16) || 6;
      const cg = parseInt(hex.substring(2, 4), 16) || 182;
      const cb = parseInt(hex.substring(4, 6), 16) || 212;

      const fgImgData = ctx.getImageData(0, 0, width, height);
      const fgPixels = fgImgData.data;

      for (let i = 0; i < mask.length; i++) {
        const m = mask[i];
        const p = i * 4;
        fgPixels[p] = Math.round(fgPixels[p] * m + cr * (1 - m));
        fgPixels[p + 1] = Math.round(fgPixels[p + 1] * m + cg * (1 - m));
        fgPixels[p + 2] = Math.round(fgPixels[p + 2] * m + cb * (1 - m));
      }
      ctx.putImageData(fgImgData, 0, 0);
    }
  }
}

export const semanticSegmenter = new SemanticSegmenter();
