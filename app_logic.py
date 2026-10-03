"""
app_logic.py
============
TerraLogic Advisor — Unified Application Logic Layer (Stage 4.4)

Orchestrates the two independent ML pipelines and exposes a single
application-facing entrypoint, run_terralogic_advisory().

PIPELINE SEPARATION
-------------------
    Numeric pipeline : tabular soil/climate measurements (N, P, K, temperature,
                       humidity, pH, rainfall) -> RandomForest crop model
                       + rule-based advisory engine (advisory.py).

    Image pipeline   : RGB soil photograph -> MobileNetV2 soil-type classifier
                       (soil_image_advisory.py).

The two pipelines are never merged into a single prediction. The combined
workflow simply presents both results side by side and cross-references the
recommended crop against crops known to suit the visually identified soil type.

SCIENTIFIC BOUNDARY
-------------------
An RGB photograph identifies *visual soil morphology only*. It does NOT measure
N, P, K, pH, moisture, temperature, humidity, or rainfall. Quantitative nutrient
values require a laboratory soil test.
"""

import os
import sys
import pathlib

import joblib
import numpy as np
import pandas as pd

from advisory import validate_soil_inputs, get_soil_advisory
from soil_image_advisory import predict_soil_image, get_soil_type_advisory


# -----------------------------
# Module-level paths & constants
# -----------------------------

_PROJECT_ROOT = pathlib.Path(__file__).parent.resolve()
_CROP_MODEL_PATH = _PROJECT_ROOT / "crop_model.pkl"

_CACHED_CROP_MODEL = None


NUMERIC_INPUT_KEYS = ("N", "P", "K", "temperature", "humidity", "ph", "rainfall")


SOIL_CROP_AFFINITY = {
    "Alluvial": ["Rice", "Wheat", "Sugarcane", "Cotton", "Maize", "Jute", "Oilseeds", "Vegetables"],
    "Black": ["Cotton", "Sorghum", "Wheat", "Chickpea", "Linseed", "Castor", "Sunflower"],
    "Clay": ["Rice", "Wheat", "Sugarcane", "Jute", "Broccoli", "Cabbage"],
    "Red": ["Groundnut", "Cotton", "Wheat", "Rice", "Millets", "Pulses", "Tobacco", "Potato"],
}


BOUNDARY_NOTE = (
    "The soil image identifies visual soil morphology only. It does not measure "
    "N, P, K, pH, moisture, temperature, or rainfall. Always use laboratory soil "
    "tests for quantitative nutrient data."
)


# -----------------------------
# Crop model loading (cached)
# -----------------------------

def load_crop_model(model_path=None):
    """
    Load (and cache) the RandomForest crop recommendation model.

    Parameters
    ----------
    model_path : str or pathlib.Path or None
        Override path to crop_model.pkl.  Defaults to project root.

    Returns
    -------
    sklearn estimator

    Raises
    ------
    FileNotFoundError  : model file missing
    RuntimeError       : joblib load failed
    """
    global _CACHED_CROP_MODEL

    if _CACHED_CROP_MODEL is not None:
        return _CACHED_CROP_MODEL

    path = pathlib.Path(model_path) if model_path else _CROP_MODEL_PATH

    if not path.exists():
        raise FileNotFoundError(
            f"Crop model not found at: {path}\n"
            "Run train_model.py to generate crop_model.pkl."
        )

    try:
        _CACHED_CROP_MODEL = joblib.load(str(path))
    except Exception as exc:
        raise RuntimeError(f"Failed to load crop model from '{path}': {exc}") from exc

    return _CACHED_CROP_MODEL


# -----------------------------
# Numeric pipeline
# -----------------------------

def analyze_numeric_inputs(
    N,
    P,
    K,
    temperature,
    humidity,
    ph,
    rainfall,
    crop_model_path=None,
):
    """
    Run the full numeric soil/crop pipeline.

    Validates inputs, predicts the recommended crop via the RandomForest,
    and generates a structured soil advisory report.

    Parameters
    ----------
    N, P, K : float  — Nitrogen, Phosphorus, Potassium (kg/ha)
    temperature : float  — °C
    humidity    : float  — % relative humidity
    ph          : float  — soil pH 0–14
    rainfall    : float  — mm
    crop_model_path : str or None  — override crop model path

    Returns
    -------
    dict with keys:
        status              (str)  : "success" or "error"
        error               (str or None)
        recommended_crop    (str or None)
        crop_probabilities  (dict or None) : {crop: probability}
        top_crops           (list) : top-5 crops by probability
        soil_advisory       (dict or None) : full advisory report
    """
    try:
        validate_soil_inputs(N, P, K, temperature, humidity, ph, rainfall)

    except (TypeError, ValueError) as exc:
        return {
            "status": "error",
            "error": str(exc),
            "recommended_crop": None,
            "crop_probabilities": None,
            "top_crops": [],
            "soil_advisory": None,
        }

    try:
        model = load_crop_model(crop_model_path)

        features = pd.DataFrame(
            [[N, P, K, temperature, humidity, ph, rainfall]],
            columns=["N", "P", "K", "temperature", "humidity", "ph", "rainfall"],
        )

        predicted_crop = model.predict(features)[0]
        proba = model.predict_proba(features)[0]
        classes = model.classes_

        crop_probabilities = {
            str(cls): float(p) for cls, p in zip(classes, proba)
        }

        top_crops = sorted(
            crop_probabilities, key=crop_probabilities.get, reverse=True
        )[:5]

    except Exception as exc:
        return {
            "status": "error",
            "error": f"Crop model prediction failed: {exc}",
            "recommended_crop": None,
            "crop_probabilities": None,
            "top_crops": [],
            "soil_advisory": None,
        }

    try:
        soil_advisory = get_soil_advisory(
            N, P, K, temperature, humidity, ph, rainfall,
            crop=str(predicted_crop),
        )
    except Exception:
        soil_advisory = None

    return {
        "status": "success",
        "error": None,
        "recommended_crop": str(predicted_crop),
        "crop_probabilities": crop_probabilities,
        "top_crops": top_crops,
        "soil_advisory": soil_advisory,
    }


# -----------------------------
# Image pipeline
# -----------------------------

def analyze_soil_image(
    image_input,
    image_model_path=None,
    confidence_threshold=0.6,
):
    """
    Run the soil image classification pipeline.

    Parameters
    ----------
    image_input : str, pathlib.Path, PIL.Image, or file-like
    image_model_path : str or None  — override model path
    confidence_threshold : float

    Returns
    -------
    dict with keys:
        status          (str)  : "success" or "error"
        error           (str or None)
        predicted_class (str or None)
        confidence      (float or None)
        is_confident    (bool)
        warning         (str or None)
        all_probabilities (dict or None)
        soil_advisory   (dict or None)
    """
    if image_input is None:
        return {
            "status": "error",
            "error": "No image provided.",
            "predicted_class": None,
            "confidence": None,
            "is_confident": False,
            "warning": None,
            "all_probabilities": None,
            "soil_advisory": None,
        }

    try:
        prediction = predict_soil_image(
            image_input,
            model_path=image_model_path,
            confidence_threshold=confidence_threshold,
        )

    except FileNotFoundError as exc:
        return {
            "status": "error",
            "error": f"Image model not found: {exc}",
            "predicted_class": None,
            "confidence": None,
            "is_confident": False,
            "warning": None,
            "all_probabilities": None,
            "soil_advisory": None,
        }

    except (TypeError, ValueError) as exc:
        return {
            "status": "error",
            "error": f"Invalid image input: {exc}",
            "predicted_class": None,
            "confidence": None,
            "is_confident": False,
            "warning": None,
            "all_probabilities": None,
            "soil_advisory": None,
        }

    except Exception as exc:
        return {
            "status": "error",
            "error": f"Image classification failed: {exc}",
            "predicted_class": None,
            "confidence": None,
            "is_confident": False,
            "warning": None,
            "all_probabilities": None,
            "soil_advisory": None,
        }

    try:
        soil_advisory = get_soil_type_advisory(prediction["predicted_class"])
    except Exception:
        soil_advisory = None

    return {
        "status": "success",
        "error": None,
        "predicted_class": prediction["predicted_class"],
        "confidence": prediction["confidence"],
        "is_confident": prediction["is_confident"],
        "warning": prediction["warning"],
        "all_probabilities": prediction["all_probabilities"],
        "soil_advisory": soil_advisory,
    }


# -----------------------------
# Combined guidance
# -----------------------------

def synthesize_combined_guidance(numeric_result, image_result):
    """
    Produce integrated agronomic guidance when both pipeline results are available.

    This function describes physical/agronomic management tips for the
    identified soil type and notes where the numeric pipeline's crop
    recommendation aligns with that soil type.

    SCIENTIFIC BOUNDARY:
        This function NEVER implies that the image result can measure
        or substitute for numeric soil test values (N, P, K, pH, etc.).

    Parameters
    ----------
    numeric_result : dict  — output of analyze_numeric_inputs()
    image_result   : dict  — output of analyze_soil_image()

    Returns
    -------
    dict with keys:
        soil_type_from_image     (str or None)
        crop_from_numeric        (str or None)
        soil_crop_compatibility  (bool or None)
        affinity_crops           (list)
        guidance_notes           (list of str)
        boundary_note            (str)
    """
    soil_type = None
    if image_result and image_result.get("status") == "success":
        soil_type = image_result.get("predicted_class")

    crop = None
    if numeric_result and numeric_result.get("status") == "success":
        crop = numeric_result.get("recommended_crop")

    affinity_crops = SOIL_CROP_AFFINITY.get(soil_type, []) if soil_type else []

    compatibility = None
    if soil_type and crop:
        # The crop model emits lowercase labels ("rice") while SOIL_CROP_AFFINITY
        # stores Title Case names ("Rice"), so compare case-insensitively.
        compatibility = crop.lower() in [c.lower() for c in affinity_crops]

    guidance_notes = []

    if soil_type:
        adv = image_result.get("soil_advisory") or {}
        tips = adv.get("management_tips", [])
        for tip in tips:
            guidance_notes.append(f"[{soil_type} soil] {tip}")

    if soil_type and crop:
        if compatibility:
            guidance_notes.append(
                f"The numerically recommended crop ({crop}) is well-suited to "
                f"{soil_type} soil."
            )
        else:
            guidance_notes.append(
                f"Note: {crop} is not listed among common crops for {soil_type} "
                "soil. Verify with a local agronomist before planting."
            )

    if not guidance_notes:
        guidance_notes.append(
            "Provide both numeric inputs and a soil image for integrated guidance."
        )

    return {
        "soil_type_from_image": soil_type,
        "crop_from_numeric": crop,
        "soil_crop_compatibility": compatibility,
        "affinity_crops": affinity_crops,
        "guidance_notes": guidance_notes,
        "boundary_note": BOUNDARY_NOTE,
    }


# -----------------------------
# Master entrypoint
# -----------------------------

def run_terralogic_advisory(
    numeric_inputs=None,
    image_input=None,
    crop_model_path=None,
    image_model_path=None,
    confidence_threshold=0.6,
):
    """
    Master entrypoint for TerraLogic Advisor.

    Orchestrates one of three workflows depending on what is provided:
      - numeric-only   : numeric_inputs dict, no image_input
      - image-only     : image_input, no numeric_inputs
      - combined       : both numeric_inputs and image_input

    Parameters
    ----------
    numeric_inputs : dict or None
        Keys: N, P, K, temperature, humidity, ph, rainfall (all float/int).
    image_input : str, pathlib.Path, PIL.Image, file-like, or None
        Soil photograph.
    crop_model_path : str or None
        Override path to crop_model.pkl.
    image_model_path : str or None
        Override path to soil_mobilenetv2.keras.
    confidence_threshold : float
        Minimum confidence for image prediction (default 0.60).

    Returns
    -------
    dict with keys:
        workflow            (str)  : "numeric_only" | "image_only" | "combined" | "error"
        numeric_result      (dict or None)
        image_result        (dict or None)
        combined_guidance   (dict or None) : only present in "combined" workflow
        error               (str or None)  : top-level error if no input provided

    Raises
    ------
    Nothing — all errors are captured and returned in the result dict.
    """
    has_numeric = numeric_inputs is not None
    has_image = image_input is not None

    if not has_numeric and not has_image:
        return {
            "workflow": "error",
            "numeric_result": None,
            "image_result": None,
            "combined_guidance": None,
            "error": (
                "No inputs provided. Supply numeric_inputs (dict) and/or "
                "image_input (file path, PIL Image, or file-like object)."
            ),
        }

    numeric_result = None
    image_result = None
    combined_guidance = None

    if has_numeric:
        try:
            numeric_result = analyze_numeric_inputs(
                numeric_inputs["N"],
                numeric_inputs["P"],
                numeric_inputs["K"],
                numeric_inputs["temperature"],
                numeric_inputs["humidity"],
                numeric_inputs["ph"],
                numeric_inputs["rainfall"],
                crop_model_path=crop_model_path,
            )
        except KeyError as exc:
            numeric_result = {
                "status": "error",
                "error": f"Missing numeric input key: {exc}",
                "recommended_crop": None,
                "crop_probabilities": None,
                "top_crops": [],
                "soil_advisory": None,
            }

    if has_image:
        image_result = analyze_soil_image(
            image_input,
            image_model_path=image_model_path,
            confidence_threshold=confidence_threshold,
        )

    if has_numeric and has_image:
        workflow = "combined"
        combined_guidance = synthesize_combined_guidance(numeric_result, image_result)
    elif has_numeric:
        workflow = "numeric_only"
    else:
        workflow = "image_only"

    return {
        "workflow": workflow,
        "numeric_result": numeric_result,
        "image_result": image_result,
        "combined_guidance": combined_guidance,
        "error": None,
    }
