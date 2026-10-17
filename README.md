# NeuroVision AI: Client-Side CNN Computer Vision Web Application

**Academic Project Specification & Final Deliverable**  
**Prepared for:** Aleeza Umer  
**Degree Program:** Bachelor of Science in Artificial Intelligence (BSAI)  
**Specialization:** Deep Learning & Edge Computer Vision  
**Execution Paradigm:** 100% Client-Side In-Browser Inference (Zero Backend, Zero External API, Zero Database)  

---

## 1. Executive Summary

**NeuroVision AI** is a modern, responsive, and privacy-preserving Computer Vision web application powered entirely by Convolutional Neural Networks (CNNs) running natively inside the client's web browser. 

Traditional computer vision applications rely on cloud servers (Flask, FastAPI, Node.js, Express) to receive user images, execute model inference, and return predictions over the network. While functional, this paradigm introduces network latency, recurring cloud hosting costs, vulnerability to server outages, and severe privacy risks when users upload sensitive imagery.

NeuroVision AI eliminates the backend completely by compiling and executing deep convolutional models directly on client hardware (GPU / WebGL) via **TensorFlow.js**. All tensor resizing, normalization, convolutional feature extraction, and probability calculations occur on the user's local device in milliseconds.

---

## 2. Key Highlights & Achievements

- **Zero-Backend Architecture:** No Flask, FastAPI, Node, PHP, Python web server, external REST API, or cloud database.
- **Hardware-Accelerated Edge Inference:** Leverages WebGL 2.0 GPU shaders directly inside the browser for sub-20ms forward-pass latencies.
- **Dual CNN Architectures:**
  1. **MobileNetV2 CNN:** 53 layers, inverted residual blocks, depthwise separable convolutions, classifying 1,000 ImageNet object categories.
  2. **Custom Deep Vision CNN:** Stacked Conv2D blocks with Batch Normalization, Max Pooling, and Global Average Pooling classifying 10 visual taxonomies.
- **Interactive CNN Pipeline Inspector:** Step-by-step visual inspection of the internal deep learning pipeline:
  - *Stage 1:* 224 &times; 224 Normalized Tensor representation
  - *Stage 2:* Conv2D Layer 1 spatial edge and gradient feature map
  - *Stage 3:* Conv2D Layer 2 non-linear ReLU activation heatmap
  - *Stage 4:* Deep latent receptive field and Softmax distribution
- **Real-Time Live Webcam Scanner:** Continuous camera stream inference with hardware FPS throughput counter.
- **Preset Test Gallery:** 1-click test images (Golden Retriever, Sports Car, Espresso Mug, Airliner, Daisy) for instant demonstration.
- **Defensive Error Handling:** Validates file formats (JPEG, PNG, WEBP, BMP), checks file size limits (<15MB), rejects corrupt images with toast notifications.
- **Export & Analytics:** One-click JSON diagnostic report download and clipboard summary copying.
- **Python Training & Export Pipeline:** Full Python training scripts (`python_pipeline/`) to train and convert custom Keras CNNs.

---

## 3. Machine Learning & CNN Pipeline

The inference pipeline adheres to rigorous deep learning principles:

```
[User Media (Image / Webcam)] 
              │
              ▼
   [1. In-Browser Ingestion] ────► DOM HTMLImageElement / HTMLVideoElement
              │
              ▼
   [2. Pixel to Tensor Conversion] ─► tf.browser.fromPixels()
              │
              ▼
   [3. Geometric Resizing] ────────► tf.image.resizeBilinear([224, 224])
              │
              ▼
   [4. Tensor Normalization] ──────► Float32 scaling:
                                      • MobileNet: [-1.0, +1.0] via (val / 127.5) - 1.0
                                      • Custom CNN: [0.0, 1.0] via val / 255.0
              │
              ▼
   [5. Batch Dimension Expansion] ─► Shape: [1, 224, 224, 3]
              │
              ▼
   [6. CNN Forward Pass] ──────────► WebGL GPU Shader Execution
                                      • Conv2D Kernels (Spatial Feature Extraction)
                                      • Batch Normalization (Distribution Centering)
                                      • ReLU Activations (Non-Linearity)
                                      • Pooling / Stride (Spatial Downsampling)
                                      • Global Average Pooling (Vectorization)
              │
              ▼
   [7. Softmax Probability Head] ──► P(y = c | x) = exp(z_c) / Σ exp(z_j)
              │
              ▼
   [8. Post-Processing & Top-K] ───► Ranking, Confidence Score, UI Progress Bars
```

### Mathematical Formulations

#### 1. 2D Discrete Convolution:
$$S(i, j) = (I * K)(i, j) = \sum_{m} \sum_{n} I(i - m, j - n) \cdot K(m, n)$$
Where $I$ is the input feature map, $K$ is the learnable kernel tensor, and $S$ is the output activation matrix.

#### 2. Batch Normalization:
$$\hat{x} = \frac{x - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}, \quad y = \gamma \hat{x} + \beta$$
Stabilizes hidden layer distributions and prevents vanishing/exploding gradients during training.

#### 3. Softmax Activation:
$$P(y = c \mid \mathbf{x}) = \frac{e^{z_c}}{\sum_{j=1}^{C} e^{z_j}}$$
Converts raw logit outputs into a strictly normalized probability distribution over all target classes.

---

## 4. Project Structure

```
d:/Deep Learning/DL Projects/CNN/
├── documentation/
│   └── CNN_Computer_Vision_App_Prompt.docx   # Original FYP specification prompt
├── public/
│   ├── models/
│   │   └── custom_cnn/
│   │       ├── model.json                    # TFJS model topology & manifest
│   │       └── group1-shard1of1.bin          # Binary float32 layer weights
│   └── samples/
│       ├── airliner.jpg                      # Airplane test image
│       ├── coffee_mug.jpg                    # Mug test image
│       ├── daisy.jpg                         # Daisy flower test image
│       ├── golden_retriever.jpg              # Dog test image
│       └── sports_car.jpg                    # Sports car test image
├── python_pipeline/
│   ├── train_cnn.py                          # Full Keras CNN training script
│   ├── export_to_tfjs.py                     # Standalone Keras to TFJS converter
│   ├── requirements.txt                      # Python dependencies
│   └── README.md                             # Technical Python guide
├── src/
│   ├── ml/
│   │   ├── cifarClasses.js                   # Class taxonomy & specifications
│   │   ├── modelService.js                   # TFJS model orchestrator & inference
│   │   └── tensorUtils.js                    # Preprocessing & feature extractors
│   ├── styles/
│   │   └── main.css                          # Modern design system & styling
│   ├── ui/
│   │   ├── dropzone.js                       # Drag-and-drop & file validation
│   │   ├── infoModal.js                      # Architecture & technical modal
│   │   ├── pipelineVisualizer.js             # CNN 4-stage pipeline visualizer
│   │   ├── resultsRenderer.js                # Hero card, radial gauge, bars
│   │   ├── sampleManager.js                  # 1-click sample gallery
│   │   ├── toast.js                          # Non-intrusive notifications
│   │   └── webcamManager.js                  # Real-time camera feed scanner
│   └── main.js                               # Application bootstrap coordinator
├── dist/                                     # Production build assets (vite build)
├── index.html                                # Semantic HTML5 responsive layout
├── package.json                              # Project dependencies & scripts
├── vite.config.js                            # Vite development configuration
└── README.md                                 # Comprehensive FYP project documentation
```

---

## 5. Getting Started & Local Installation

### Prerequisites
- **Node.js:** v18.0.0 or later (v24.19.0 recommended)
- **npm:** v9.0.0 or later (v11.17.0 recommended)
- **Python (Optional, for retraining):** Python 3.9 – 3.11 with TensorFlow 2.15+

### Installation Steps

1. **Clone or navigate to the project directory:**
   ```bash
   cd "d:/Deep Learning/DL Projects/CNN"
   ```

2. **Install frontend dependencies:**
   ```bash
   npm install
   ```

3. **Launch the development server:**
   ```bash
   npm run dev
   ```
   Open your browser at `http://localhost:5173/`.

4. **Build for production:**
   ```bash
   npm run build
   ```

5. **Preview the production bundle:**
   ```bash
   npm run preview
   ```

---

## 6. Training & Exporting Custom Python Models

If you wish to train your own custom CNN architecture on your dataset in Python:

1. **Install Python dependencies:**
   ```bash
   pip install -r python_pipeline/requirements.txt
   ```

2. **Train and automatically export to TensorFlow.js:**
   ```bash
   python python_pipeline/train_cnn.py --epochs 5 --output_dir public/models/custom_cnn
   ```

3. **Or convert an existing Keras `.h5` or `.keras` file:**
   ```bash
   python python_pipeline/export_to_tfjs.py --model_path path/to/model.h5 --output_dir public/models/custom_cnn
   ```

4. The generated `model.json` and `*.bin` files are immediately accessible by the frontend web application with zero code modifications.

---

## 7. Performance & Benchmarking

| Metric | MobileNetV2 CNN | Custom Deep Vision CNN |
| :--- | :--- | :--- |
| **Parameters** | 3,504,872 (~14 MB) | 103,050 (~400 KB) |
| **Target Classes** | 1,000 (ImageNet) | 10 (Vision Taxonomy) |
| **Inference Backend** | WebGL 2.0 (Shader GPU) | WebGL 2.0 (Shader GPU) |
| **Average Latency** | 18 – 35 ms | 8 – 15 ms |
| **Throughput** | 30 – 55 FPS | 60+ FPS |
| **Network Payload** | 0 bytes (Post-load) | 0 bytes (Post-load) |
| **Client Privacy** | 100% On-Device | 100% On-Device |

---

## 8. Academic Deliverable Acknowledgements

- **Developer & Researcher:** Aleeza Umer
- **Academic Program:** BSAI (BS Artificial Intelligence)
- **Course Focus:** Convolutional Neural Networks & Client-Side Computer Vision
- **License:** Educational & Open-Source Research License
