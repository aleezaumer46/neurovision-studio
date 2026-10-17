/**
 * Module 4: In-Browser Transfer Learning ("Teachable Studio")
 * Uses KNN Classifier & MobileNetV2 Embedding Truncation
 */
import * as tf from '@tensorflow/tfjs';
import * as knnClassifier from '@tensorflow-models/knn-classifier';

export class TeachableStudio {
  constructor() {
    this.classifier = knnClassifier.create();
    this.customClasses = [
      { id: 0, name: 'Class A', sampleCount: 0, color: '#06b6d4' },
      { id: 1, name: 'Class B', sampleCount: 0, color: '#3b82f6' },
      { id: 2, name: 'Class C', sampleCount: 0, color: '#8b5cf6' }
    ];
    this.isTraining = false;
    this.totalSamples = 0;
  }

  setClassName(classId, name) {
    const targetClass = this.customClasses.find(item => item.id === classId);
    if (targetClass && name.trim()) targetClass.name = name.trim();
  }

  /**
   * Adds an example embedding to a specific class from the active MobileNet model
   * @param {Object} mobilenetModel - Loaded MobileNetV2 model
   * @param {HTMLImageElement|HTMLVideoElement|HTMLCanvasElement} mediaElement 
   * @param {number} classId 
   */
  async addExample(mobilenetModel, mediaElement, classId) {
    if (!mobilenetModel) throw new Error('MobileNet model not loaded.');

    // Extract intermediate 1024-dimensional activation embedding (before softmax)
    const logits = tf.tidy(() => mobilenetModel.infer(mediaElement, true));
    try {
      this.classifier.addExample(logits, classId);
    } finally {
      logits.dispose();
    }

    const targetClass = this.customClasses.find(c => c.id === classId);
    if (targetClass) {
      targetClass.sampleCount++;
      this.totalSamples++;
    }

    return targetClass ? targetClass.sampleCount : 0;
  }

  /**
   * Predicts the custom class for the input media
   */
  async predict(mobilenetModel, mediaElement, k = 3) {
    if (this.classifier.getNumClasses() === 0) {
      return {
        hasTrainedClasses: false,
        message: 'No custom class samples recorded yet. Capture samples first.'
      };
    }

    const logits = tf.tidy(() => mobilenetModel.infer(mediaElement, true));
    let result;
    try {
      result = await this.classifier.predictClass(logits, k);
    } finally {
      logits.dispose();
    }

    const predictedClassId = parseInt(result.label, 10);
    const classInfo = this.customClasses.find(c => c.id === predictedClassId) || { name: `Class ${predictedClassId}` };
    const confPct = Math.round((result.confidences[result.label] || 0) * 100);

    const breakdown = Object.entries(result.confidences).map(([id, conf]) => {
      const cls = this.customClasses.find(c => c.id === parseInt(id, 10)) || { name: `Class ${id}` };
      return {
        id: parseInt(id, 10),
        name: cls.name,
        confidencePct: Math.round(conf * 100)
      };
    });

    return {
      hasTrainedClasses: true,
      classId: predictedClassId,
      className: classInfo.name,
      confidencePct: confPct,
      breakdown
    };
  }

  clearClass(classId) {
    try {
      this.classifier.clearClass(classId);
      const target = this.customClasses.find(c => c.id === classId);
      if (target) {
        this.totalSamples -= target.sampleCount;
        target.sampleCount = 0;
      }
    } catch (e) {
      console.warn('Error clearing class:', e);
    }
  }

  resetAll() {
    this.classifier.clearAllClasses();
    this.customClasses.forEach(c => c.sampleCount = 0);
    this.totalSamples = 0;
  }
}

export const teachableStudio = new TeachableStudio();
