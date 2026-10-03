"""
soil_image_advisory.py
=====================
TerraLogic Advisor — Soil Image Classification Advisory API (Stage 4.3)

SCIENTIFIC BOUNDARY
-------------------
This module identifies the *visual soil type* of a photograph. It does NOT
measure, estimate, or predict nitrogen (N), phosphorus (P), potassium (K),
pH, moisture, temperature, humidity, or rainfall. Those values come from the
numeric soil-test pipeline (see advisory.py) or from user-provided lab data.

Public API
----------
load_soil_image_model()    Cached MobileNetV2 loader.
_preprocess_image()        Input normalisation -> (1, 224, 224, 3) float32.
predict_soil_image()       Inference -> class, confidence, probabilities.
get_soil_type_advisory()   Agronomic advisory for a recognised soil type.
predict_and_advise()       Convenience wrapper combining the two above.
"""

import os
import pathlib

import numpy as np


# -----------------------------
# Module-level paths & constants
# -----------------------------

_PROJECT_ROOT = pathlib.Path(__file__).parent.resolve()
MODEL_PATH = str(_PROJECT_ROOT / "soil_mobilenetv2.keras")

SOIL_CLASSES = ["Alluvial", "Black", "Clay", "Red"]

IMG_SIZE = (224, 224)

DEFAULT_CONFIDENCE_THRESHOLD = 0.6

_CACHED_MODEL = None


_CAUTION = (
    "Do not estimate nitrogen, phosphorus, potassium, or pH from a photograph. "
    "Conduct a laboratory soil test for quantitative nutrient data."
)


SOIL_ADVISORY_DATABASE = {
    "Alluvial": {
        "description": (
            "Alluvial soil is deposited by rivers and is generally fertile, "
            "well-drained, and rich in minerals. It is among the most productive "
            "agricultural soils in India."
        ),
        "characteristics": [
            "Fine to medium texture with good water retention",
            "Generally neutral to slightly alkaline pH",
            "Rich in potash, poor in phosphorus and nitrogen in some regions",
            "Found extensively in Indo-Gangetic plains, river deltas",
        ],
        "suitable_crops": [
            "Rice", "Wheat", "Sugarcane", "Cotton",
            "Maize", "Pulses", "Vegetables", "Oilseeds",
        ],
        "management_tips": [
            "Maintain organic matter with green manure or compost.",
            "Test pH periodically; add lime if acidic.",
            "Practice crop rotation to prevent nutrient depletion.",
            "Use drip or furrow irrigation to avoid waterlogging.",
        ],
        "caution": _CAUTION,
    },
    "Black": {
        "description": (
            "Black soil (Regur) is rich in clay minerals and has high "
            "moisture-retention capacity. It shrinks and cracks in dry "
            "conditions and swells when wet, making tillage timing critical."
        ),
        "characteristics": [
            "High clay content — shrinks and cracks when dry",
            "Excellent water retention; can become waterlogged",
            "Rich in calcium, magnesium, and potash",
            "Typically low in nitrogen and phosphorus",
            "Found in Deccan Plateau, Maharashtra, Madhya Pradesh",
        ],
        "suitable_crops": [
            "Cotton", "Sorghum", "Wheat", "Chickpea",
            "Linseed", "Castor", "Sunflower",
        ],
        "management_tips": [
            "Till only when soil moisture is at a workable level to avoid compaction.",
            "Add phosphorus-rich fertilizer if soil test confirms deficiency.",
            "Use raised beds or ridges to improve drainage.",
            "Avoid heavy irrigation — monitor field capacity carefully.",
        ],
        "caution": _CAUTION,
    },
    "Clay": {
        "description": (
            "Clay soil has very fine particles with high surface area, giving it "
            "strong water and nutrient retention but poor drainage and aeration. "
            "It requires careful management."
        ),
        "characteristics": [
            "Very fine texture; sticky when wet, hard when dry",
            "Slow drainage; prone to waterlogging",
            "High nutrient-holding capacity (CEC)",
            "Poor aeration; may become anaerobic when waterlogged",
        ],
        "suitable_crops": [
            "Rice", "Wheat", "Sugarcane", "Jute",
            "Broccoli", "Brussels sprouts", "Cabbage",
        ],
        "management_tips": [
            "Incorporate organic matter (compost, manure) to improve structure.",
            "Use raised beds and ridges to enhance drainage.",
            "Avoid working soil when wet to prevent compaction.",
            "Add gypsum to improve flocculation in sodic clay soils.",
        ],
        "caution": _CAUTION,
    },
    "Red": {
        "description": (
            "Red soil gets its colour from iron oxide. It is generally "
            "coarse-textured, well-drained, and low in nutrients and organic "
            "matter. It requires enrichment for productive farming."
        ),
        "characteristics": [
            "Reddish colour due to iron oxide content",
            "Sandy to loamy texture; well-drained",
            "Low in nitrogen, phosphorus, and organic matter",
            "Slightly acidic to neutral pH",
            "Found in Tamil Nadu, Karnataka, Andhra Pradesh, Odisha",
        ],
        "suitable_crops": [
            "Groundnut", "Cotton", "Wheat", "Rice",
            "Millets", "Pulses", "Tobacco", "Potato",
        ],
        "management_tips": [
            "Apply organic matter regularly to improve fertility and water retention.",
            "Use phosphorus and nitrogen fertilizers based on soil test results.",
            "Practice mulching to reduce moisture evaporation.",
            "Apply lime if soil test shows pH is below optimal range.",
        ],
        "caution": _CAUTION,
    },
}


# -----------------------------
# Model loading (cached)
# -----------------------------

def load_soil_image_model(model_path=None):
    """
    Load (and cache) the MobileNetV2 soil classifier from disk.

    Parameters
    ----------
    model_path : str or None
        Path to the .keras model file.  Defaults to MODEL_PATH.

    Returns
    -------
    tf.keras.Model
        The loaded Keras model.

    Raises
    ------
    FileNotFoundError
        If the model file does not exist at the resolved path.
    RuntimeError
        If TensorFlow fails to load the model.
    """
    global _CACHED_MODEL

    if _CACHED_MODEL is not None:
        return _CACHED_MODEL

    path = model_path or MODEL_PATH

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Soil image model not found at: {path}\n"
            "Run train_image_model.py to generate soil_mobilenetv2.keras."
        )

    try:
        import tensorflow as tf

        _CACHED_MODEL = tf.keras.models.load_model(path)
    except Exception as exc:
        raise RuntimeError(f"Failed to load soil image model from '{path}': {exc}") from exc

    return _CACHED_MODEL


# -----------------------------
# Image preprocessing
# -----------------------------

def _preprocess_image(image_input):
    """
    Convert any supported input into a normalised (1, 224, 224, 3) float32 array.

    Accepts
    -------
    - str / pathlib.Path : file-system path to an image file
    - PIL.Image.Image    : already-loaded PIL image
    - file-like object  : anything with a .read() method (e.g. BytesIO, UploadedFile)

    Returns
    -------
    np.ndarray  shape (1, 224, 224, 3), dtype float32, values in [0, 255]

    Note: values are deliberately left in the 0-255 range. MobileNetV2's
    preprocess_input is baked into the trained model graph, so the [0, 255]
    convention used at training time must be preserved here.
    """
    from PIL import Image

    if isinstance(image_input, (str, pathlib.Path)):
        img = Image.open(str(image_input)).convert("RGB")

    elif hasattr(image_input, "read"):
        img = Image.open(image_input).convert("RGB")

    elif hasattr(image_input, "convert"):
        img = image_input.convert("RGB")

    else:
        raise TypeError(
            f"Unsupported image_input type: {type(image_input).__name__}. "
            "Pass a file path (str/Path), a PIL.Image, or a file-like object."
        )

    img = img.resize(IMG_SIZE, Image.LANCZOS)
    arr = np.array(img, dtype=np.float32)
    return np.expand_dims(arr, axis=0)


# -----------------------------
# Inference
# -----------------------------

def predict_soil_image(image_input, model_path=None, confidence_threshold=DEFAULT_CONFIDENCE_THRESHOLD):
    """
    Predict the soil type from an image.

    Parameters
    ----------
    image_input : str, pathlib.Path, PIL.Image, or file-like
        The soil image to classify.
    model_path : str or None
        Override path to the .keras model file.
    confidence_threshold : float
        Minimum probability for a prediction to be considered confident.

    Returns
    -------
    dict with keys:
        predicted_class  (str)   : e.g. "Alluvial"
        confidence       (float) : probability of the predicted class, 0–1
        all_probabilities (dict) : {class_name: probability} for all 4 classes
        is_confident     (bool)  : True if confidence >= confidence_threshold
        warning          (str or None) : human-readable note when not confident

    Raises
    ------
    FileNotFoundError  : model file missing
    RuntimeError       : model loading failed
    TypeError          : unsupported image_input type
    ValueError         : model output shape unexpected
    """
    model = load_soil_image_model(model_path)
    arr = _preprocess_image(image_input)

    raw = model.predict(arr, verbose=0)

    if raw.shape != (1, len(SOIL_CLASSES)):
        raise ValueError(
            f"Unexpected model output shape {raw.shape}; "
            f"expected (1, {len(SOIL_CLASSES)})."
        )

    probs = raw[0]
    idx = int(np.argmax(probs))
    predicted_class = SOIL_CLASSES[idx]
    confidence = float(probs[idx])

    all_probabilities = {cls: float(p) for cls, p in zip(SOIL_CLASSES, probs)}

    is_confident = confidence >= confidence_threshold

    warning = None
    if not is_confident:
        warning = (
            f"Low confidence ({confidence:.1%}). The model is uncertain about "
            "this image. Consider using a clearer, well-lit photograph of the soil."
        )

    return {
        "predicted_class": predicted_class,
        "confidence": confidence,
        "all_probabilities": all_probabilities,
        "is_confident": is_confident,
        "warning": warning,
    }


# -----------------------------
# Soil-type advisory lookup
# -----------------------------

def get_soil_type_advisory(soil_type):
    """
    Return the structured advisory dictionary for a given soil type.

    Parameters
    ----------
    soil_type : str
        One of "Alluvial", "Black", "Clay", "Red" (case-insensitive).

    Returns
    -------
    dict  : advisory entry from SOIL_ADVISORY_DATABASE

    Raises
    ------
    ValueError
        If soil_type is not one of the four recognised classes.
    """
    normalised = soil_type.strip().title()

    if normalised not in SOIL_ADVISORY_DATABASE:
        raise ValueError(
            f"Unknown soil type: '{soil_type}'. "
            f"Valid types are: {', '.join(SOIL_ADVISORY_DATABASE.keys())}."
        )

    return SOIL_ADVISORY_DATABASE[normalised]


# -----------------------------
# Combined convenience wrapper
# -----------------------------

def predict_and_advise(image_input, model_path=None, confidence_threshold=DEFAULT_CONFIDENCE_THRESHOLD):
    """
    Predict soil type from an image and return both prediction metadata
    and the soil-type advisory in a single call.

    Parameters
    ----------
    image_input : str, pathlib.Path, PIL.Image, or file-like
        The soil image to classify.
    model_path : str or None
        Override path to the .keras model file.
    confidence_threshold : float
        Threshold for is_confident flag.

    Returns
    -------
    dict with keys:
        prediction  (dict) : output of predict_soil_image()
        advisory    (dict) : output of get_soil_type_advisory()
    """
    prediction = predict_soil_image(image_input, model_path, confidence_threshold)
    advisory = get_soil_type_advisory(prediction["predicted_class"])

    return {
        "prediction": prediction,
        "advisory": advisory,
    }
