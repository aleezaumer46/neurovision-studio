"""
================================================================================
Convolutional Neural Network (CNN) Training Pipeline
Project: CNN-Based Computer Vision Web Application (Client-Side Inference)
Author: Aleeza Umer (BSAI)
================================================================================
Features:
- Convolutional feature extraction layers (Conv2D, BatchNorm, MaxPooling2D)
- Regularization (Dropout, Batch Normalization)
- Training metrics logging and validation curve plotting
- Automated conversion to TensorFlow.js browser format
"""

import os
import json
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers

def build_cnn_model(input_shape=(224, 224, 3), num_classes=10):
    """
    Constructs a Deep CNN architecture optimized for client-side edge deployment:
    - Low memory footprint (~100k parameters)
    - High feature extraction capability with stacked 3x3 kernels
    - Global Average Pooling to minimize fully-connected weights
    """
    model = models.Sequential([
        layers.Input(shape=input_shape, name="image_input"),

        # Conv Block 1: Low-level edges & textures
        layers.Conv2D(32, (3, 3), strides=2, padding="same", activation="relu", name="conv1"),
        layers.BatchNormalization(name="bn1"),
        layers.MaxPooling2D((2, 2), name="pool1"),

        # Conv Block 2: Mid-level shapes & contours
        layers.Conv2D(64, (3, 3), strides=2, padding="same", activation="relu", name="conv2"),
        layers.BatchNormalization(name="bn2"),
        layers.MaxPooling2D((2, 2), name="pool2"),

        # Conv Block 3: High-level visual semantics
        layers.Conv2D(128, (3, 3), padding="same", activation="relu", name="conv3"),
        layers.BatchNormalization(name="bn3"),

        # Classification Head
        layers.GlobalAveragePooling2D(name="gap"),
        layers.Dense(64, activation="relu", name="fc_latent"),
        layers.Dropout(0.3, name="regularization_dropout"),
        layers.Dense(num_classes, activation="softmax", name="class_probabilities")
    ], name="Aleeza_BSAI_Custom_CNN")

    model.compile(
        optimizer=optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

def train_and_export(epochs=5, batch_size=32, output_dir="public/models/custom_cnn"):
    print("=" * 60)
    print("CNN Training & Browser Deployment Pipeline")
    print("=" * 60)
    
    # 1. Dataset Preparation
    print("[1/4] Preparing dataset (CIFAR-10 resized or synthetic demonstration)...")
    try:
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
        print(f"[*] Loaded CIFAR-10: {len(x_train)} train samples, {len(x_test)} test samples")
        # Subsample for swift demonstration training if needed
        x_train, y_train = x_train[:2000], y_train[:2000]
        x_test, y_test = x_test[:400], y_test[:400]
        # Resize to 224x224 in batches during training or resize directly
        x_train = tf.image.resize(x_train, (224, 224)) / 255.0
        x_test = tf.image.resize(x_test, (224, 224)) / 255.0
    except Exception as e:
        print(f"[!] Offline / fallback mode: Generating synthetic training samples ({e})")
        x_train = np.random.uniform(0, 1, size=(200, 224, 224, 3)).astype(np.float32)
        y_train = np.random.randint(0, 10, size=(200, 1))
        x_test = np.random.uniform(0, 1, size=(50, 224, 224, 3)).astype(np.float32)
        y_test = np.random.randint(0, 10, size=(50, 1))

    # 2. Build Model
    print("[2/4] Initializing CNN Architecture...")
    model = build_cnn_model()
    model.summary()

    # 3. Model Training
    print(f"[3/4] Commencing Training ({epochs} epochs)...")
    history = model.fit(
        x_train, y_train,
        validation_data=(x_test, y_test),
        epochs=epochs,
        batch_size=batch_size,
        verbose=1
    )

    # 4. Save Native Model
    os.makedirs("saved_models", exist_ok=True)
    h5_path = "saved_models/custom_cnn.h5"
    model.save(h5_path)
    print(f"[+] Saved native Keras model to: {h5_path}")

    # 5. Export to Browser Format (TFJS)
    print("[4/4] Exporting model to TensorFlow.js format...")
    from export_to_tfjs import export_keras_to_tfjs
    export_keras_to_tfjs(model, output_dir=output_dir)
    print("\n[COMPLETE] End-to-end training and export completed successfully!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=2, help="Number of training epochs")
    parser.add_argument("--output_dir", type=str, default="public/models/custom_cnn", help="TFJS destination folder")
    args = parser.parse_args()
    
    train_and_export(epochs=args.epochs, output_dir=args.output_dir)
