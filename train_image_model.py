"""
train_image_model.py
====================
TerraLogic Advisor — Soil Image Classification Training Pipeline
Architecture: MobileNetV2 (Transfer Learning, Pretrained on ImageNet)

Features:
  - 4 Classes: Alluvial, Black, Clay, Red
  - Data Augmentation with Hue-Preserving Color Safeguards
  - Frozen MobileNetV2 Feature Extractor + Custom Classification Head
  - Callbacks: EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
  - Model Export: soil_mobilenetv2.keras
  - Post-training Multi-Class Evaluation (Accuracy, F1-Score, Confusion Matrix)
"""

import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix

# =============================================================================
# CONFIGURATION HYPERPARAMETERS
# =============================================================================
DATA_DIR = "data_soil"
MODEL_OUTPUT_PATH = "soil_mobilenetv2.keras"

IMG_HEIGHT = 224
IMG_WIDTH = 224
IMG_SIZE = (IMG_HEIGHT, IMG_WIDTH)
BATCH_SIZE = 32
EPOCHS = 20
INITIAL_LR = 1e-4
RANDOM_SEED = 42

EXPECTED_CLASSES = ["Alluvial", "Black", "Clay", "Red"]
NUM_CLASSES = len(EXPECTED_CLASSES)

tf.random.set_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# =============================================================================
# DATA LOADING & SPLITTING
# =============================================================================
def load_datasets(data_dir):
    """
    Loads dataset from directory.
    Supports either pre-split structure (train/val/test) or single directory with auto 80/20 split.
    """
    if not os.path.exists(data_dir):
        print(f"[ERROR] Dataset directory '{data_dir}' not found.")
        print("Please create 'data_soil/' with class subdirectories (Alluvial, Black, Clay, Red).")
        sys.exit(1)

    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")
    test_dir = os.path.join(data_dir, "test")

    # Case 1: Pre-partitioned train/val directories exist
    if os.path.isdir(train_dir) and os.path.isdir(val_dir):
        print("[*] Loading from pre-partitioned train/val folders...")
        train_ds = tf.keras.utils.image_dataset_from_directory(
            train_dir,
            image_size=IMG_SIZE,
            batch_size=BATCH_SIZE,
            label_mode="categorical",
            shuffle=True,
            seed=RANDOM_SEED
        )
        val_ds = tf.keras.utils.image_dataset_from_directory(
            val_dir,
            image_size=IMG_SIZE,
            batch_size=BATCH_SIZE,
            label_mode="categorical",
            shuffle=False
        )
        test_ds = None
        if os.path.isdir(test_dir):
            test_ds = tf.keras.utils.image_dataset_from_directory(
                test_dir,
                image_size=IMG_SIZE,
                batch_size=BATCH_SIZE,
                label_mode="categorical",
                shuffle=False
            )
        class_names = train_ds.class_names
    else:
        # Case 2: Unified dataset folder -> automatic 80/20 train/validation split
        print("[*] Loading unified dataset with 80/20 train/validation split...")
        train_ds, val_ds = tf.keras.utils.image_dataset_from_directory(
            data_dir,
            validation_split=0.2,
            subset="both",
            seed=RANDOM_SEED,
            image_size=IMG_SIZE,
            batch_size=BATCH_SIZE,
            label_mode="categorical",
            shuffle=True
        )
        test_ds = val_ds  # Use validation split for final evaluation
        class_names = train_ds.class_names

    print(f"[*] Detected Classes ({len(class_names)}): {class_names}")

    # Optimize pipeline with prefetching
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)
    if test_ds and test_ds is not val_ds:
        test_ds = test_ds.prefetch(buffer_size=AUTOTUNE)

    return train_ds, val_ds, test_ds, class_names


# =============================================================================
# MODEL ARCHITECTURE (MobileNetV2 Transfer Learning)
# =============================================================================
def build_model(num_classes=4):
    """
    Builds MobileNetV2 transfer learning model with data augmentation safeguards.
    """
    # 1. Data Augmentation Layers (Hue-preserving: flips, rotations, mild zoom/brightness)
    data_augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal_and_vertical", seed=RANDOM_SEED),
        layers.RandomRotation(0.08, seed=RANDOM_SEED),    # ~30 deg
        layers.RandomZoom(0.1, seed=RANDOM_SEED),          # 10% zoom
        layers.RandomBrightness(0.1, seed=RANDOM_SEED),    # 10% brightness variation (no hue shift)
    ], name="data_augmentation")

    # 2. Input Layer
    inputs = tf.keras.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3), name="input_image")

    # 3. Apply Augmentation (active during training only)
    x = data_augmentation(inputs)

    # 4. MobileNetV2 Pixel Preprocessing (scales [0, 255] -> [-1.0, 1.0])
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)

    # 5. Base Pretrained Backbone
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(IMG_HEIGHT, IMG_WIDTH, 3),
        include_top=False,
        weights="imagenet"
    )
    base_model.trainable = False  # Freeze base weights for feature extraction

    x = base_model(x, training=False)

    # 6. Classification Head
    x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    x = layers.Dropout(0.3, name="head_dropout")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="soil_classification")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="TerraLogic_Soil_MobileNetV2")
    return model, base_model


# =============================================================================
# TRAINING & EVALUATION PIPELINE
# =============================================================================
def train_and_evaluate():
    print("=" * 65)
    print("TERRALOGIC ADVISOR — SOIL CLASSIFICATION TRAINING (MobileNetV2)")
    print("=" * 65)

    # 1. Load Data
    train_ds, val_ds, test_ds, class_names = load_datasets(DATA_DIR)

    # 2. Build Model
    model, base_model = build_model(num_classes=len(class_names))
    model.summary()

    # 3. Compile Model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=INITIAL_LR),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    # 4. Callbacks Setup
    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        ModelCheckpoint(
            filepath=MODEL_OUTPUT_PATH,
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.2,
            patience=3,
            min_lr=1e-6,
            verbose=1
        )
    ]

    # 5. Train Model
    print("\n" + "=" * 65)
    print(f"[*] Training for up to {EPOCHS} epochs...")
    print("=" * 65)

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks
    )

    print("\n[+] Training complete. Best model saved to:", MODEL_OUTPUT_PATH)

    # 6. Comprehensive Multi-Class Evaluation on Validation/Test Set
    print("\n" + "=" * 65)
    print("FINAL MULTI-CLASS EVALUATION")
    print("=" * 65)

    eval_ds = test_ds if test_ds is not None else val_ds
    y_true = []
    y_pred_probs = []

    for batch_images, batch_labels in eval_ds:
        preds = model.predict(batch_images, verbose=0)
        y_true.extend(np.argmax(batch_labels.numpy(), axis=1))
        y_pred_probs.extend(preds)

    y_true = np.array(y_true)
    y_pred_probs = np.array(y_pred_probs)
    y_pred = np.argmax(y_pred_probs, axis=1)
    confidences = np.max(y_pred_probs, axis=1) * 100

    # Classification Report (Precision, Recall, F1)
    print("\n--- CLASSIFICATION REPORT ---")
    print(classification_report(y_true, y_pred, target_names=class_names, digits=4))

    # Confusion Matrix
    print("--- CONFUSION MATRIX (Rows=Actual, Columns=Predicted) ---")
    cm = confusion_matrix(y_true, y_pred)
    header = "          " + "  ".join([f"{c[:8]:>8}" for c in class_names])
    print(header)
    for idx, row in enumerate(cm):
        row_str = "  ".join([f"{val:>8}" for val in row])
        print(f"{class_names[idx][:8]:>8}  {row_str}")

    # Confidence Threshold Analysis (Threshold = 60%)
    low_conf_mask = confidences < 60.0
    low_conf_count = np.sum(low_conf_mask)
    avg_conf = np.mean(confidences)

    print("\n--- CONFIDENCE ANALYSIS ---")
    print(f"Mean Prediction Confidence : {avg_conf:.2f}%")
    print(f"Low Confidence Samples (<60%): {low_conf_count} / {len(confidences)} ({low_conf_count / len(confidences) * 100:.1f}%)")
    print("=" * 65)


if __name__ == "__main__":
    train_and_evaluate()

