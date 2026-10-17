/**
 * CNN Pipeline Inspector
 * Visualizes the stages of a Convolutional Neural Network pipeline:
 * 1. Resizing & Normalization (224x224)
 * 2. Conv2D Feature Map Filter (Edges)
 * 3. Mid-Level Activation Heatmap
 * 4. Deep Latent Representations & Softmax
 */
import { renderPreprocessedCanvas, generateVisualFeatureMaps } from '../ml/tensorUtils.js';
import * as tf from '@tensorflow/tfjs';
import { preprocessImage } from '../ml/tensorUtils.js';

export class PipelineVisualizer {
  constructor({
    preprocessedCanvas,
    edgeMapCanvas,
    activationMapCanvas,
    deepMapCanvas
  }) {
    this.preprocessedCanvas = preprocessedCanvas;
    this.edgeMapCanvas = edgeMapCanvas;
    this.activationMapCanvas = activationMapCanvas;
    this.deepMapCanvas = deepMapCanvas;
  }

  /**
   * Update all pipeline canvas representations from the source image element
   * @param {HTMLImageElement|HTMLVideoElement|HTMLCanvasElement} sourceMedia 
   */
  update(sourceMedia, model) {
    if (!sourceMedia) return;

    // 1. Draw 224x224 normalized representation
    renderPreprocessedCanvas(sourceMedia, this.preprocessedCanvas);

    // 2. Extract edge filter activation map (Conv2D Layer 1)
    generateVisualFeatureMaps(this.preprocessedCanvas, this.edgeMapCanvas, 'edges');

    if (model) {
      const convolutionLayers = model.layers.filter(layer => layer.getClassName().includes('Conv'));
      const denseLayers = model.layers.filter(layer => layer.getClassName() === 'Dense');
      this.renderLayerActivation(sourceMedia, model, convolutionLayers[Math.min(1, convolutionLayers.length - 1)], this.activationMapCanvas);
      this.renderLayerActivation(sourceMedia, model, denseLayers[Math.max(0, denseLayers.length - 2)], this.deepMapCanvas);
    } else {
      generateVisualFeatureMaps(this.preprocessedCanvas, this.activationMapCanvas, 'activation');
      generateVisualFeatureMaps(this.preprocessedCanvas, this.deepMapCanvas, 'deep');
    }
  }

  renderLayerActivation(sourceMedia, model, layer, canvas) {
    if (!layer || !canvas) return;
    const pixels = tf.tidy(() => {
      const input = preprocessImage(sourceMedia, 'custom_cnn');
      let activations = input;
      const targetLayerIndex = model.layers.indexOf(layer);
      for (const currentLayer of model.layers.slice(0, targetLayerIndex + 1)) {
        if (currentLayer.getClassName() !== 'InputLayer') {
          activations = currentLayer.apply(activations, { training: false });
        }
      }
      const map = activations.rank === 4
        ? activations.relu().mean(-1).expandDims(-1)
        : activations.reshape([1, 1, -1, 1]).relu();
      const resized = tf.image.resizeBilinear(map, [112, 112]).squeeze();
      const min = resized.min();
      const max = resized.max();
      return Float32Array.from(resized.sub(min).div(max.sub(min).add(1e-8)).dataSync());
    });

    canvas.width = 112;
    canvas.height = 112;
    const context = canvas.getContext('2d');
    const image = context.createImageData(112, 112);
    pixels.forEach((value, index) => {
      const offset = index * 4;
      image.data[offset] = Math.round(255 * Math.max(0, Math.min(1, 1.5 * value - 0.25)));
      image.data[offset + 1] = Math.round(255 * Math.max(0, Math.min(1, 1.5 - Math.abs(2 * value - 1))));
      image.data[offset + 2] = Math.round(255 * Math.max(0, Math.min(1, 1.25 - 1.5 * value)));
      image.data[offset + 3] = 255;
    });
    context.putImageData(image, 0, 0);
  }

  clear() {
    [this.preprocessedCanvas, this.edgeMapCanvas, this.activationMapCanvas, this.deepMapCanvas].forEach(canvas => {
      if (!canvas) return;
      const ctx = canvas.getContext('2d');
      ctx.fillStyle = '#070a13';
      ctx.fillRect(0, 0, canvas.width, canvas.height);
    });
  }
}
