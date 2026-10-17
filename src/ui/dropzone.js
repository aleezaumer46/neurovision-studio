/**
 * Drag & Drop Image Upload Handler
 * Provides graceful validation, error states, and image decoding
 */
import { showToast } from './toast.js';

export function setupDropzone({
  dropzoneEl,
  fileInputEl,
  onImageSelected,
  onError
}) {
  const VALID_IMAGE_TYPES = ['image/jpeg', 'image/png', 'image/webp', 'image/bmp', 'image/jpg'];
  const MAX_FILE_SIZE_MB = 15;

  function validateFile(file) {
    if (!file) return { valid: false, reason: 'No file selected.' };

    if (!VALID_IMAGE_TYPES.includes(file.type)) {
      return {
        valid: false,
        reason: `Unsupported file format (${file.type || 'unknown'}). Please upload a JPG, PNG, or WEBP image.`
      };
    }

    if (file.size > MAX_FILE_SIZE_MB * 1024 * 1024) {
      return {
        valid: false,
        reason: `Image file exceeds ${MAX_FILE_SIZE_MB}MB limit. Please choose a smaller image.`
      };
    }

    return { valid: true };
  }

  function processFile(file) {
    const validation = validateFile(file);
    if (!validation.valid) {
      showToast(validation.reason, 'error');
      if (onError) onError(validation.reason);
      return;
    }

    const reader = new FileReader();
    reader.onload = (e) => {
      const img = new Image();
      img.onload = () => {
        if (onImageSelected) {
          onImageSelected({
            imgElement: img,
            dataUrl: e.target.result,
            fileName: file.name,
            fileSize: (file.size / 1024).toFixed(1) + ' KB',
            dimensions: `${img.naturalWidth} × ${img.naturalHeight} px`
          });
        }
      };
      img.onerror = () => {
        showToast('Failed to decode image. File might be corrupted.', 'error');
        if (onError) onError('Corrupt image');
      };
      img.src = e.target.result;
    };
    reader.onerror = () => {
      showToast('Error reading image file.', 'error');
    };
    reader.readAsDataURL(file);
  }

  // Drag & Drop events
  ['dragenter', 'dragover'].forEach(eventName => {
    dropzoneEl.addEventListener(eventName, (e) => {
      e.preventDefault();
      e.stopPropagation();
      dropzoneEl.classList.add('dragover');
    });
  });

  ['dragleave', 'drop'].forEach(eventName => {
    dropzoneEl.addEventListener(eventName, (e) => {
      e.preventDefault();
      e.stopPropagation();
      dropzoneEl.classList.remove('dragover');
    });
  });

  dropzoneEl.addEventListener('drop', (e) => {
    const dt = e.dataTransfer;
    const files = dt.files;
    if (files && files.length > 0) {
      processFile(files[0]);
    }
  });

  // Clicking dropzone triggers file picker
  dropzoneEl.addEventListener('click', (e) => {
    // Only trigger if not clicking a button inside dropzone
    if (e.target.closest('button')) return;
    fileInputEl.click();
  });

  fileInputEl.addEventListener('change', () => {
    if (fileInputEl.files && fileInputEl.files.length > 0) {
      processFile(fileInputEl.files[0]);
    }
  });
}
