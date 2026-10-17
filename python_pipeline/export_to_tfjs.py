"""
================================================================================
TensorFlow/Keras to TensorFlow.js Model Exporter
Project: CNN-Based Computer Vision Web Application (Client-Side Inference)
Author: Aleeza Umer (BSAI)
================================================================================
This script converts any trained TensorFlow/Keras CNN model into a browser-compatible
TensorFlow.js format (model.json + binary weight shard), without requiring external
C-compilers or platform-restricted tools.
"""

import os
import json
import argparse
import numpy as np
import tensorflow as tf

def export_keras_to_tfjs(model, output_dir="public/models/custom_cnn", shard_name="group1-shard1of1.bin"):
    """
    Exports a tf.keras Model into TensorFlow.js format:
    1. model.json: JSON manifest containing model topology and weight metadata
    2. group1-shard*.bin: Binary buffer containing concatenated float32 weights
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"[*] Analyzing model: {model.name}")
    print(f"[*] Input shape: {model.input_shape}")
    print(f"[*] Output shape: {model.output_shape}")
    print(f"[*] Total parameters: {model.count_params():,}")

    weight_entries = []
    weight_bytes = bytearray()

    # Extract all layer weights in order
    for layer in model.layers:
        weights = layer.weights
        if not weights:
            continue
        for w in weights:
            tensor_val = w.numpy().astype(np.float32)
            clean_name = w.name.split(':')[0]
            
            weight_entries.append({
                "name": clean_name,
                "shape": list(tensor_val.shape),
                "dtype": "float32"
            })
            # Append binary data in little-endian float32 format
            weight_bytes.extend(tensor_val.tobytes())

    bin_path = os.path.join(output_dir, shard_name)
    with open(bin_path, "wb") as f:
        f.write(weight_bytes)
    print(f"[+] Saved weights binary: {bin_path} ({len(weight_bytes):,} bytes)")

    # Serialize model topology
    model_json_str = model.to_json()
    model_topology = json.loads(model_json_str)

    # Format official TFJS manifest
    tfjs_manifest = {
        "format": "layers-model",
        "generatedBy": f"TensorFlow {tf.__version__}",
        "convertedBy": "TensorFlow.js Universal Exporter",
        "modelTopology": model_topology,
        "weightsManifest": [
            {
                "paths": [shard_name],
                "weights": weight_entries
            }
        ]
    }

    json_path = os.path.join(output_dir, "model.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(tfjs_manifest, f, indent=2)
    print(f"[+] Saved model manifest: {json_path}")
    print("[SUCCESS] Model is ready for browser client-side inference!")
    return json_path, bin_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export Keras CNN model to TensorFlow.js")
    parser.add_argument("--model_path", type=str, default="", help="Path to saved .keras or .h5 file")
    parser.add_argument("--output_dir", type=str, default="public/models/custom_cnn", help="Target output directory")
    args = parser.parse_args()

    if args.model_path and os.path.exists(args.model_path):
        print(f"[*] Loading model from {args.model_path}...")
        model = tf.keras.models.load_model(args.model_path)
    else:
        print("[*] No existing model path provided. Building reference 10-Class Vision CNN...")
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(224, 224, 3), name="input_tensor"),
            
            # Block 1
            tf.keras.layers.Conv2D(16, (3, 3), strides=2, padding="same", activation="relu", name="conv1"),
            tf.keras.layers.BatchNormalization(name="bn1"),
            tf.keras.layers.MaxPooling2D((2, 2), name="pool1"),
            
            # Block 2
            tf.keras.layers.Conv2D(32, (3, 3), strides=2, padding="same", activation="relu", name="conv2"),
            tf.keras.layers.BatchNormalization(name="bn2"),
            tf.keras.layers.MaxPooling2D((2, 2), name="pool2"),
            
            # Block 3
            tf.keras.layers.Conv2D(64, (3, 3), padding="same", activation="relu", name="conv3"),
            tf.keras.layers.BatchNormalization(name="bn3"),
            
            # Classifier Head
            tf.keras.layers.GlobalAveragePooling2D(name="gap"),
            tf.keras.layers.Dense(64, activation="relu", name="dense_fc"),
            tf.keras.layers.Dropout(0.25, name="dropout"),
            tf.keras.layers.Dense(10, activation="softmax", name="softmax_output")
        ], name="Custom_Vision_CNN")
        model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    export_keras_to_tfjs(model, args.output_dir)
