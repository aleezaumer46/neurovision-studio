/**
 * Preprocessing & Tensor Utility Engine
 * Handles pixel conversion, resizing, normalization, and visual feature map extraction
 */
import * as tf from '@tensorflow/tfjs';

/**
 * Preprocess image element for CNN inference
 * @param {HTMLImageElement|HTMLVideoElement|HTMLCanvasElement} mediaElement 
 * @param {string} modelType - 'mobilenet' | 'custom_cnn'
 * @returns {tf.Tensor4D} Preprocessed 4D batch tensor [1, 224, 224, 3]
 */
export function preprocessImage(mediaElement, modelType = 'mobilenet') {
  return tf.tidy(() => {
    // 1. Convert DOM image to 3D Tensor [height, width, channels]
    const rawTensor = tf.browser.fromPixels(mediaElement);

    // 2. Resize to canonical CNN resolution [224, 224] with Bilinear Interpolation
    const resized = tf.image.resizeBilinear(rawTensor, [224, 224], true);

    // 3. Cast to float32
    const floatTensor = resized.toFloat();

    // 4. Normalization depending on model training standard
    let normalized;
    if (modelType === 'mobilenet') {
      // MobileNet expects [-1, 1]: (val / 127.5) - 1.0
      const offset = tf.scalar(127.5);
      normalized = floatTensor.sub(offset).div(offset);
    } else {
      // Custom CNN standard expects [0, 1]: val / 255.0
      normalized = floatTensor.div(tf.scalar(255.0));
    }

    // 5. Expand dimension to form batch [1, 224, 224, 3]
    return normalized.expandDims(0);
  });
}

/**
 * Draws the 224x224 preprocessed tensor directly onto a canvas for user inspection
 */
export async function renderPreprocessedCanvas(mediaElement, targetCanvas) {
  if (!targetCanvas) return;
  const rgbPixels = tf.tidy(() => {
    const source = tf.browser.fromPixels(mediaElement, 3);
    const normalized = tf.image.resizeBilinear(source, [224, 224]).toFloat().div(255);
    return Uint8Array.from(normalized.mul(255).dataSync());
  });
  const pixels = new Uint8ClampedArray(224 * 224 * 4);
  for (let sourceIndex = 0, targetIndex = 0; sourceIndex < rgbPixels.length; sourceIndex += 3, targetIndex += 4) {
    pixels[targetIndex] = rgbPixels[sourceIndex];
    pixels[targetIndex + 1] = rgbPixels[sourceIndex + 1];
    pixels[targetIndex + 2] = rgbPixels[sourceIndex + 2];
    pixels[targetIndex + 3] = 255;
  }
  const ctx = targetCanvas.getContext('2d');
  targetCanvas.width = 224;
  targetCanvas.height = 224;
  ctx.putImageData(new ImageData(pixels, 224, 224), 0, 0);
}

/**
 * Generates visual convolutional feature activations from the input image
 * Demonstrates how Conv2D filters extract edges, textures, and gradient patterns
 */
export function generateVisualFeatureMaps(sourceCanvas, targetCanvas, filterType = 'edges') {
  if (!sourceCanvas || !targetCanvas) return;
  const width = 112;
  const height = 112;
  targetCanvas.width = width;
  targetCanvas.height = height;
  const ctx = targetCanvas.getContext('2d');

  // Draw scaled down representation
  ctx.drawImage(sourceCanvas, 0, 0, width, height);
  const imgData = ctx.getImageData(0, 0, width, height);
  const data = imgData.data;

  const luminance = new Float32Array(width * height);
  for (let i = 0, pixel = 0; i < data.length; i += 4, pixel++) {
    luminance[pixel] = 0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2];
  }

  if (filterType === 'edges') {
    for (let y = 1; y < height - 1; y++) {
      for (let x = 1; x < width - 1; x++) {
        const p = y * width + x;
        const gx = -luminance[p - width - 1] + luminance[p - width + 1]
          - 2 * luminance[p - 1] + 2 * luminance[p + 1]
          - luminance[p + width - 1] + luminance[p + width + 1];
        const gy = -luminance[p - width - 1] - 2 * luminance[p - width] - luminance[p - width + 1]
          + luminance[p + width - 1] + 2 * luminance[p + width] + luminance[p + width + 1];
        const edge = Math.min(255, Math.hypot(gx, gy));
        const i = p * 4;
        data[i] = Math.round(edge * 0.15);
        data[i + 1] = Math.round(edge * 0.85);
        data[i + 2] = edge;
      }
    }
  } else {
    const minimum = Math.min(...luminance);
    const maximum = Math.max(...luminance);
    const range = Math.max(1, maximum - minimum);
    for (let pixel = 0; pixel < luminance.length; pixel++) {
      const activation = Math.max(0, (luminance[pixel] - minimum) / range);
      const i = pixel * 4;
      if (filterType === 'activation') {
        data[i] = Math.round(255 * activation);
        data[i + 1] = Math.round(255 * Math.sqrt(activation));
        data[i + 2] = Math.round(255 * (1 - activation));
      } else {
        data[i] = Math.round(255 * activation);
        data[i + 1] = Math.round(255 * (1 - activation));
        data[i + 2] = Math.round(150 * activation);
      }
    }
  }
  ctx.putImageData(imgData, 0, 0);
}
