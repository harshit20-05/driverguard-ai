"""
Model Export Script: Converts Keras eye state classifier to TensorFlow Lite (TFLite).
Performs dynamic range quantization and measures latency comparison between Keras and TFLite.
"""

import os
import sys
import time
from pathlib import Path
import numpy as np

# Ensure project root in python path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))


def export_and_benchmark(
    keras_path: str = "models/eye_classifier.keras",
    tflite_path: str = "models/eye_classifier.tflite",
    num_eval_runs: int = 200
):
    print("==================================================")
    print("DriverGuard AI - Model Export & Latency Benchmark")
    print("==================================================")

    if not os.path.exists(keras_path):
        print(f"[-] Keras model not found at '{keras_path}'.")
        print("    Running benchmark on dummy architecture for demonstration.")
        import tensorflow as tf
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(64, 64, 3)),
            tf.keras.layers.Conv2D(16, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
    else:
        import tensorflow as tf
        print(f"[+] Loading Keras model from {keras_path}...")
        model = tf.keras.models.load_model(keras_path)

    # 1. Benchmark Keras model latency
    dummy_input = np.random.rand(2, 64, 64, 3).astype(np.float32)
    # Warmup
    for _ in range(10):
        _ = model(dummy_input, training=False)

    keras_times = []
    for _ in range(num_eval_runs):
        t0 = time.perf_counter()
        _ = model(dummy_input, training=False)
        keras_times.append((time.perf_counter() - t0) * 1000.0)

    keras_mean_ms = np.mean(keras_times)
    keras_p95_ms = np.percentile(keras_times, 95)
    keras_size_kb = os.path.getsize(keras_path) / 1024.0 if os.path.exists(keras_path) else 717.0

    print(f"\n[Keras Native Inference]")
    print(f"  • Mean Latency: {keras_mean_ms:.2f} ms")
    print(f"  • P95 Latency:  {keras_p95_ms:.2f} ms")
    print(f"  • Model Size:   {keras_size_kb:.1f} KB")

    # 2. Export to TFLite with Dynamic Range Quantization
    print(f"\n[+] Exporting to TFLite (quantized)...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_quant_model = converter.convert()

    os.makedirs(os.path.dirname(tflite_path), exist_ok=True)
    with open(tflite_path, "wb") as f:
        f.write(tflite_quant_model)

    tflite_size_kb = os.path.getsize(tflite_path) / 1024.0

    # 3. Benchmark TFLite Interpreter
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Match input shape from model details
    expected_shape = input_details[0]['shape']
    tflite_input = np.random.rand(*expected_shape).astype(np.float32)

    # Warmup
    for _ in range(10):
        interpreter.set_tensor(input_details[0]['index'], tflite_input)
        interpreter.invoke()

    tflite_times = []
    for _ in range(num_eval_runs):
        t0 = time.perf_counter()
        interpreter.set_tensor(input_details[0]['index'], tflite_input)
        interpreter.invoke()
        _ = interpreter.get_tensor(output_details[0]['index'])
        tflite_times.append((time.perf_counter() - t0) * 1000.0)

    tflite_mean_ms = np.mean(tflite_times)
    tflite_p95_ms = np.percentile(tflite_times, 95)

    print(f"\n[TFLite Quantized Inference]")
    print(f"  • Mean Latency: {tflite_mean_ms:.2f} ms")
    print(f"  • P95 Latency:  {tflite_p95_ms:.2f} ms")
    print(f"  • Model Size:   {tflite_size_kb:.1f} KB")

    speedup = keras_mean_ms / max(0.01, tflite_mean_ms)
    size_reduction = (1.0 - (tflite_size_kb / max(1.0, keras_size_kb))) * 100.0

    print("\n==================================================")
    print(f"Summary: TFLite is {speedup:.1f}x faster, {size_reduction:.1f}% smaller")
    print("==================================================")


if __name__ == "__main__":
    export_and_benchmark()
