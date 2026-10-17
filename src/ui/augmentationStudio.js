/**
 * Module 8: Adversarial Robustness & Image Augmentation Playground
 * Real-time canvas perturbations (Brightness, Contrast, Noise, Blur, Rotation)
 * Stress-tests CNN confidence in response to lighting, distortion, and noise variations
 */

export class AugmentationStudio {
  constructor() {
    this.params = {
      brightness: 100, // 0 - 200%
      contrast: 100,   // 0 - 200%
      saturation: 100, // 0 - 200%
      noise: 0,        // 0 - 50%
      blur: 0,         // 0 - 15px
      rotation: 0      // -180 to 180 deg
    };
  }

  setParam(key, value) {
    if (this.params.hasOwnProperty(key)) {
      this.params[key] = parseFloat(value);
    }
  }

  resetParams() {
    this.params = {
      brightness: 100,
      contrast: 100,
      saturation: 100,
      noise: 0,
      blur: 0,
      rotation: 0
    };
  }

  /**
   * Applies the perturbations to a canvas
   */
  applyAugmentation(sourceMedia, targetCanvas) {
    if (!targetCanvas || !sourceMedia) return;
    const ctx = targetCanvas.getContext('2d');
    const width = 224;
    const height = 224;
    targetCanvas.width = width;
    targetCanvas.height = height;

    ctx.save();
    ctx.clearRect(0, 0, width, height);

    // Apply rotation around center
    if (this.params.rotation !== 0) {
      ctx.translate(width / 2, height / 2);
      ctx.rotate((this.params.rotation * Math.PI) / 180);
      ctx.translate(-width / 2, -height / 2);
    }

    // Apply CSS filters (brightness, contrast, saturation, blur)
    ctx.filter = `brightness(${this.params.brightness}%) contrast(${this.params.contrast}%) saturate(${this.params.saturation}%) blur(${this.params.blur}px)`;
    ctx.drawImage(sourceMedia, 0, 0, width, height);
    ctx.restore();

    // Apply Gaussian / Salt-and-Pepper noise if set
    if (this.params.noise > 0) {
      const imgData = ctx.getImageData(0, 0, width, height);
      const data = imgData.data;
      const noiseAmp = (this.params.noise / 100) * 128;

      for (let i = 0; i < data.length; i += 4) {
        const rand = (Math.random() - 0.5) * noiseAmp * 2;
        data[i] = Math.min(255, Math.max(0, data[i] + rand));
        data[i + 1] = Math.min(255, Math.max(0, data[i + 1] + rand));
        data[i + 2] = Math.min(255, Math.max(0, data[i + 2] + rand));
      }
      ctx.putImageData(imgData, 0, 0);
    }
  }
}

export const augmentationStudio = new AugmentationStudio();
