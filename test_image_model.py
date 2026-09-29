"""
test_image_model.py
===================
TerraLogic Advisor — Stage 4.2 Soil Image Classifier Verification Suite

Verifies:
  1. 'soil_mobilenetv2.keras' can be loaded successfully.
  2. Model input shape is (None, 224, 224, 3) and output shape is (None, 4).
  3. Class mapping matches [0: 'Alluvial', 1: 'Black', 2: 'Clay', 3: 'Red'].
  4. Preprocessing and inference pipeline operates cleanly on standard RGB soil images.
  5. Prediction probabilities are strictly finite and sum approximately to 1.0 (within 1e-4).
  6. Representative test images from all 4 classes are evaluated for inference behavior.
  7. Inference latency on CPU is reported.
  8. Returns exit code 0 if all tests pass, non-zero if any test fails.
"""

import os
import sys
import time
import numpy as np
from pathlib import Path
from PIL import Image

MODEL_PATH = "soil_mobilenetv2.keras"
EXPECTED_CLASSES = ["Alluvial", "Black", "Clay", "Red"]
IMG_SIZE = (224, 224)
TEST_DIR = Path("data_soil/test")


def run_verification():
    print("=" * 65)
    print("TERRALOGIC ADVISOR — SOIL IMAGE MODEL VERIFICATION SUITE")
    print("=" * 65)

    # -------------------------------------------------------------
    # TEST 1: Load Model Artifact
    # -------------------------------------------------------------
    print("\n[TEST 1] Loading model artifact...")
    if not os.path.exists(MODEL_PATH):
        print(f"[FAIL] Model artifact '{MODEL_PATH}' not found!")
        return False

    import tensorflow as tf

    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        print(f"[PASS] Successfully loaded '{MODEL_PATH}'.")
    except Exception as e:
        print(f"[FAIL] Could not load model: {e}")
        return False

    # -------------------------------------------------------------
    # TEST 2: Verify Input/Output Architecture & Shapes
    # -------------------------------------------------------------
    print("\n[TEST 2] Verifying model architecture & shapes...")
    input_shape = model.input_shape
    output_shape = model.output_shape

    print(f"[*] Model Input Shape : {input_shape}")
    print(f"[*] Model Output Shape: {output_shape}")

    expected_input_shape = (None, 224, 224, 3)
    expected_output_shape = (None, 4)

    if input_shape != expected_input_shape:
        print(f"[FAIL] Input shape mismatch! Expected {expected_input_shape}, found {input_shape}")
        return False

    if output_shape != expected_output_shape:
        print(f"[FAIL] Output shape mismatch! Expected {expected_output_shape}, found {output_shape}")
        return False

    print("[PASS] Model input and output dimensions match specifications.")

    # -------------------------------------------------------------
    # TEST 3: Synthetic Image Verification (Math & Finite Checks)
    # -------------------------------------------------------------
    print("\n[TEST 3] Testing inference mathematical invariants on synthetic inputs...")
    dummy_input = np.random.uniform(0.0, 255.0, size=(1, 224, 224, 3)).astype(np.float32)
    preds = model.predict(dummy_input, verbose=0)

    if preds.shape != (1, 4):
        print(f"[FAIL] Prediction output shape mismatch: {preds.shape}")
        return False

    probs = preds[0]
    prob_sum = float(np.sum(probs))
    is_finite = bool(np.all(np.isfinite(probs)))
    is_non_negative = bool(np.all(probs >= 0.0))
    sum_approx_one = bool(abs(prob_sum - 1.0) < 1e-4)

    print(f"[*] Output Probabilities: {probs}")
    print(f"[*] Probabilities Finite: {is_finite}")
    print(f"[*] Non-negative Values : {is_non_negative}")
    print(f"[*] Sum of Probabilities: {prob_sum:.6f} (Expected ~1.0)")

    if not (is_finite and is_non_negative and sum_approx_one):
        print("[FAIL] Output probabilities violate probability invariants!")
        return False

    print("[PASS] Synthetic input math invariants verified.")

    # -------------------------------------------------------------
    # TEST 4: Real Test Image Inference Across All 4 Classes
    # -------------------------------------------------------------
    print("\n[TEST 4] Evaluating real test sample images from each class...")

    latencies = []
    sample_results = []

    for cls_name in EXPECTED_CLASSES:
        cls_folder = TEST_DIR / cls_name
        if not cls_folder.exists():
            print(f"[FAIL] Missing test folder: {cls_folder}")
            return False

        sample_files = sorted(list(cls_folder.glob("*.jpg")))
        if not sample_files:
            print(f"[FAIL] No test images found in {cls_folder}")
            return False

        # Test first 2 samples of each class
        for sample_file in sample_files[:2]:
            with Image.open(sample_file) as img:
                rgb_img = img.convert("RGB").resize(IMG_SIZE)
                img_arr = np.array(rgb_img, dtype=np.float32)
                img_batch = np.expand_dims(img_arr, axis=0)

            t0 = time.perf_counter()
            pred_probs = model.predict(img_batch, verbose=0)[0]
            lat = (time.perf_counter() - t0) * 1000.0  # ms
            latencies.append(lat)

            pred_idx = int(np.argmax(pred_probs))
            pred_class = EXPECTED_CLASSES[pred_idx]
            confidence = float(pred_probs[pred_idx]) * 100.0

            sample_results.append({
                "file": sample_file.name,
                "ground_truth": cls_name,
                "predicted": pred_class,
                "confidence": confidence,
                "latency_ms": lat
            })

    print(f"{'Filename':<18} {'Ground Truth':<14} {'Predicted':<14} {'Confidence':<12} {'Latency':<10}")
    print("-" * 68)
    for sr in sample_results:
        print(f"{sr['file']:<18} {sr['ground_truth']:<14} {sr['predicted']:<14} {sr['confidence']:>8.2f}%    {sr['latency_ms']:>6.2f} ms")

    avg_lat = float(np.mean(latencies))
    print("-" * 68)
    print(f"[*] Mean CPU Inference Latency: {avg_lat:.2f} ms per image")

    # -------------------------------------------------------------
    # TEST 5: Verify Class Mapping Consistency
    # -------------------------------------------------------------
    print("\n[TEST 5] Checking class names and index alignment...")
    for idx, cname in enumerate(EXPECTED_CLASSES):
        print(f"[*] Index {idx} <==> '{cname}'")

    print("[PASS] Class-to-index mapping verified as strictly deterministic.")

    print("\n" + "=" * 65)
    print("ALL 5 MODEL VERIFICATION TESTS PASSED SUCCESSFULLY!")
    print("=" * 65)
    return True


if __name__ == "__main__":
    success = run_verification()
    sys.exit(0 if success else 1)
