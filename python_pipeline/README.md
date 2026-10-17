# Python CNN Training & Model Conversion Guide

**Project:** CNN-Based Computer Vision Web Application (Client-Side Inference)  
**Author:** Aleeza Umer (BSAI Student)  
**Architecture:** Deep Convolutional Neural Network (Conv2D, BatchNorm, MaxPool, GAP, Dense, Softmax)

---

## 1. Overview
This pipeline provides the complete Python machine learning workflow to:
1. Design and train a custom Convolutional Neural Network (CNN) in TensorFlow/Keras.
2. Evaluate and save the trained weights.
3. Convert the native model into a browser-executable format (`model.json` + `.bin` weight shards) for **TensorFlow.js**.
4. Deploy and run the model directly inside the client-side web application without any server or backend API.

---

## 2. Directory Structure
```
python_pipeline/
├── train_cnn.py          # Complete training script (Keras CNN + automated TFJS export)
├── export_to_tfjs.py     # Standalone model converter (converts .keras/.h5 to TFJS format)
├── requirements.txt      # Python dependencies
└── README.md             # This comprehensive technical guide
```

---

## 3. Installation & Setup

1. **Create and activate a virtual environment (recommended):**
   ```bash
   python -m venv venv
   # Windows:
   .\venv\Scripts\activate
   # Linux / macOS:
   source venv/bin/activate
   ```

2. **Install required dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## 4. Training the CNN

Run the training pipeline:
```bash
python python_pipeline/train_cnn.py --epochs 5 --output_dir public/models/custom_cnn
```

### What happens during training:
- **Input Dimension:** 224 &times; 224 &times; 3 (RGB).
- **Feature Extraction:** Stride-2 Convolutions, Batch Normalization, and Max Pooling extract hierarchical visual representations (edges &rarr; textures &rarr; parts &rarr; objects).
- **Classification Head:** Global Average Pooling reduces spatial dimensions, followed by a Dropout-regularized Dense layer and 10-class Softmax output.
- **Export:** Automatically generates:
  - `saved_models/custom_cnn.h5` (Native Python model)
  - `public/models/custom_cnn/model.json` (Browser model topology & manifest)
  - `public/models/custom_cnn/group1-shard1of1.bin` (Binary float32 weight shard)

---

## 5. Converting an Existing Keras Model

If you have already trained a model in Python or Google Colab and saved it as `.keras` or `.h5`, you can convert it using either of the two methods below:

### Method A: Using the included Pure-Python Converter (Works on all OS including Windows without C++ build tools)
```bash
python python_pipeline/export_to_tfjs.py --model_path path/to/your_model.h5 --output_dir public/models/custom_cnn
```

### Method B: Using the official `tensorflowjs_converter` CLI (Linux / macOS / Google Colab / WSL)
```bash
pip install tensorflowjs
tensorflowjs_converter --input_format=keras path/to/your_model.h5 public/models/custom_cnn
```

---

## 6. Integrating into the Frontend Web App

1. Place the generated `model.json` and `*.bin` files into the `public/models/custom_cnn/` folder of the web application.
2. In the web application's JavaScript inference engine:
   ```javascript
   import * as tf from '@tensorflow/tfjs';

   // Load the CNN model directly via client-side fetch:
   const model = await tf.loadLayersModel('/models/custom_cnn/model.json');

   // Execute inference entirely in the browser:
   const inputTensor = tf.browser.fromPixels(imageElement)
       .resizeBilinear([224, 224])
       .toFloat()
       .div(255.0)
       .expandDims(0);

   const predictions = model.predict(inputTensor);
   const probabilities = await predictions.data();
   ```
3. The frontend renders the predicted class, top confidence score, and full multi-class probability breakdown in real time with hardware-accelerated WebGL.
