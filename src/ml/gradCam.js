/**
 * Module 2: Explainable AI (XAI) via Grad-CAM
 * Gradient-Weighted Class Activation Mapping & Jet/Turbo Thermal Heatmap Generator
 */
import * as tf from '@tensorflow/tfjs';

/**
 * Computes Grad-CAM heatmap for an image tensor
 * @param {tf.LayersModel} model - Keras LayersModel
 * @param {tf.Tensor4D} inputTensor - Preprocessed 4D batch tensor [1, 224, 224, 3]
 * @param {number} targetClassIndex - Target class index for backpropagation
 * @param {string} targetLayerName - Final convolutional layer name
 * @returns {Float32Array} 2D Heatmap array [224, 224]
 */
export function computeGradCam(model, inputTensor, targetClassIndex = null, targetLayerName = null) {
  const convLayer = targetLayerName
    ? model.getLayer(targetLayerName)
    : [...model.layers].reverse().find(layer => layer.getClassName().includes('Conv'));
  if (!convLayer) throw new Error('The active LayersModel has no convolutional layer for Grad-CAM.');

  const convLayerIndex = model.layers.indexOf(convLayer);
  const downstreamLayers = model.layers.slice(convLayerIndex + 1);

  return tf.tidy(() => {
    const predictions = model.predict(inputTensor);
    const output = Array.isArray(predictions) ? predictions[0] : predictions;
    const classIndex = targetClassIndex ?? output.argMax(-1).dataSync()[0];
    let activations = inputTensor;
    for (const layer of model.layers.slice(0, convLayerIndex + 1)) {
      if (layer.getClassName() !== 'InputLayer') activations = layer.apply(activations, { training: false });
    }
    const gradients = tf.grad(featureMaps => {
      let scores = featureMaps;
      for (const layer of downstreamLayers) scores = layer.apply(scores, { training: false });
      const scoreTensor = Array.isArray(scores) ? scores[0] : scores;
      return scoreTensor.slice([0, classIndex], [1, 1]).sum();
    })(activations);

    const channelWeights = gradients.mean([1, 2], true);
    const cam = activations.mul(channelWeights).sum(-1).relu();
    const minimum = cam.min();
    const maximum = cam.max();
    const normalized = cam.sub(minimum).div(maximum.sub(minimum).add(1e-8));
    const resized = tf.image.resizeBilinear(normalized.expandDims(-1), [224, 224])
      .squeeze([0, 3]);
    return Float32Array.from(resized.dataSync());
  });
}

/**
 * Converts a normalized 0.0 - 1.0 value to a Jet/Turbo RGB colormap
 */
export function jetColorMap(val) {
  // Clamp
  const v = Math.min(1.0, Math.max(0.0, val));
  let r = 0, g = 0, b = 0;

  if (v < 0.125) {
    r = 0; g = 0; b = 0.5 + 4 * v;
  } else if (v < 0.375) {
    r = 0; g = 4 * (v - 0.125); b = 1;
  } else if (v < 0.625) {
    r = 4 * (v - 0.375); g = 1; b = 1 - 4 * (v - 0.375);
  } else if (v < 0.875) {
    r = 1; g = 1 - 4 * (v - 0.625); b = 0;
  } else {
    r = 1 - 4 * (v - 0.875) * 0.5; g = 0; b = 0;
  }

  return [Math.round(r * 255), Math.round(g * 255), Math.round(b * 255)];
}

/**
 * Draws the Grad-CAM thermal overlay blended onto target canvas
 */
export function renderGradCamOverlay(sourceMedia, targetCanvas, heatmapData, opacity = 0.65) {
  if (!targetCanvas || !heatmapData) return;
  const ctx = targetCanvas.getContext('2d');
  const width = 224;
  const height = 224;
  targetCanvas.width = width;
  targetCanvas.height = height;

  // 1. Draw source image
  ctx.drawImage(sourceMedia, 0, 0, width, height);
  const baseImgData = ctx.getImageData(0, 0, width, height);
  const basePixels = baseImgData.data;

  // 2. Blend Jet Thermal Heatmap
  for (let i = 0; i < heatmapData.length; i++) {
    const val = heatmapData[i];
    const [hr, hg, hb] = jetColorMap(val);
    const pIdx = i * 4;

    basePixels[pIdx] = Math.round(basePixels[pIdx] * (1 - opacity) + hr * opacity);
    basePixels[pIdx + 1] = Math.round(basePixels[pIdx + 1] * (1 - opacity) + hg * opacity);
    basePixels[pIdx + 2] = Math.round(basePixels[pIdx + 2] * (1 - opacity) + hb * opacity);
  }

  ctx.putImageData(baseImgData, 0, 0);
}
