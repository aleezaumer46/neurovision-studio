/**
 * Live Webcam Inference Manager
 * Handles camera streaming and real-time client-side CNN evaluation
 */
import { showToast } from './toast.js';

export class WebcamManager {
  constructor({ videoElement, onFramePredict }) {
    this.videoElement = videoElement;
    this.onFramePredict = onFramePredict;
    this.stream = null;
    this.isRunning = false;
    this.animFrameId = null;
    this.lastPredictTime = 0;
    this.throttleInterval = 120; // run inference every ~120ms to keep UI 60fps responsive
  }

  async startCamera() {
    if (this.isRunning) return;

    try {
      if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        throw new Error('Webcam API is not supported in this browser.');
      }

      this.stream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: { ideal: 640 },
          height: { ideal: 480 },
          facingMode: 'user'
        },
        audio: false
      });

      this.videoElement.srcObject = this.stream;
      await this.videoElement.play();
      this.isRunning = true;
      this.startInferenceLoop();
      showToast('Live Camera Scanner Active', 'success');
      return true;
    } catch (err) {
      console.error('Camera access error:', err);
      showToast(`Cannot access camera: ${err.message}`, 'error');
      return false;
    }
  }

  stopCamera() {
    if (!this.isRunning) return;
    this.isRunning = false;

    if (this.animFrameId) {
      cancelAnimationFrame(this.animFrameId);
      this.animFrameId = null;
    }

    if (this.stream) {
      this.stream.getTracks().forEach(track => track.stop());
      this.stream = null;
    }

    this.videoElement.srcObject = null;
    showToast('Camera Scanner Stopped', 'info');
  }

  startInferenceLoop() {
    const loop = (timestamp) => {
      if (!this.isRunning) return;

      if (timestamp - this.lastPredictTime >= this.throttleInterval) {
        this.lastPredictTime = timestamp;
        if (this.videoElement.readyState >= 2 && this.onFramePredict) {
          this.onFramePredict(this.videoElement);
        }
      }

      this.animFrameId = requestAnimationFrame(loop);
    };

    this.animFrameId = requestAnimationFrame(loop);
  }
}
