"""
app.py
======
TerraLogic Advisor — Streamlit Application (Stage 5)

Minimal single-page front end for the two independent TerraLogic pipelines.

  Section 1 : Numeric soil & climate analysis  -> app_logic.analyze_numeric_inputs()
  Section 2 : Soil image identification        -> app_logic.analyze_soil_image()
  Section 3 : Combined guidance                -> app_logic.run_terralogic_advisory()

All prediction and advisory logic lives in app_logic.py, advisory.py and
soil_image_advisory.py. This file only collects input and renders results.

Run:  .venv\\Scripts\\python.exe -m streamlit run app.py
"""

import streamlit as st

import app_logic as al


# -----------------------------
# Page configuration
# -----------------------------

st.set_page_config(
    page_title="TerraLogic Advisor",
    page_icon="🌱",
    layout="centered",
)

BOUNDARY_NOTICE = (
    "Image classification identifies visual soil type/morphology. "
    "It does not directly measure N, P, K, pH, moisture, temperature, or rainfall."
)

CONFIDENT_THRESHOLD = 0.60

NUMERIC_FIELDS = [
    ("N", "Nitrogen (N)", 0.0, 300.0, 90.0),
    ("P", "Phosphorus (P)", 0.0, 300.0, 42.0),
    ("K", "Potassium (K)", 0.0, 300.0, 43.0),
    ("temperature", "Temperature (°C)", 0.0, 60.0, 20.8),
    ("humidity", "Humidity (%)", 0.0, 100.0, 82.0),
    ("ph", "pH", 0.0, 14.0, 6.5),
    ("rainfall", "Rainfall (mm)", 0.0, 500.0, 202.0),
]


# -----------------------------
# Small render helpers
# -----------------------------

def build_summary(result, advisory):
    """
    Build a short plain-language summary from values already computed by
    advisory.py. This only rearranges existing results — it introduces no new
    analysis, thresholds, or scores.
    """
    crop = result.get("recommended_crop") or ""
    crop_name = crop.title() if crop else "No crop"

    sentences = [
        f"{crop_name} is the recommended crop based on the measured soil and "
        f"climate inputs."
    ]

    if advisory:
        score = advisory.get("soil_score")
        quality = advisory.get("soil_quality")
        ph_status = advisory.get("ph_status")

        if score is not None and quality:
            sentences.append(
                f"The soil assessment is {quality} ({score}/100)."
            )
        if ph_status:
            sentences.append(f"The pH status is {ph_status}.")

    if advisory and advisory.get("fertilizer_recommendations"):
        sentences.append(
            "Follow the management recommendation above to address the issues "
            "identified. The nutrient status table above shows each nutrient level."
        )
    else:
        sentences.append(
            "Review the nutrient status table above."
        )

    return " ".join(sentences)


def render_numeric_result(result):
    """Render a successful numeric analysis result."""
    advisory = result.get("soil_advisory")

    crop = result.get("recommended_crop")
    probs = result.get("crop_probabilities") or {}
    confidence = probs.get(crop)

    # Primary result — most important output, shown first and largest.
    col_crop, col_conf = st.columns(2)
    with col_crop:
        st.metric("Recommended crop", crop.title() if crop else "—")
    with col_conf:
        st.metric(
            "Model confidence",
            f"{confidence * 100:.2f}%" if confidence is not None else "—",
        )

    st.markdown("### Assessment Summary")
    st.info(build_summary(result, advisory))

    if not advisory:
        st.warning("The soil advisory report could not be generated.")
        return

    st.markdown("### Soil Assessment")
    st.markdown(
        f"**Soil assessment score:** {advisory['soil_score']}/100 — "
        f"{advisory['soil_quality']}"
    )
    st.markdown(f"**pH status:** {advisory['ph_status']}")

    nutrients = advisory.get("nutrient_status") or {}
    if nutrients:
        st.markdown("### Nutrient Status")
        st.dataframe(
            {
                "Nutrient": list(nutrients.keys()),
                "Status": list(nutrients.values()),
            },
            width="stretch",
            hide_index=True,
        )

    fertilizer = advisory.get("fertilizer_recommendations") or []
    if fertilizer:
        st.markdown("### Fertilizer / Management Recommendations")
        for tip in fertilizer:
            st.markdown(f"- {tip}")

    # Secondary detail — kept available but out of the main reading path.
    top_crops = result.get("top_crops") or []
    if top_crops:
        with st.expander("Top 5 Alternative Crop Predictions"):
            st.dataframe(
                {
                    "Crop": [c.title() for c in top_crops],
                    "Probability": [
                        f"{probs.get(c, 0.0) * 100:.2f}%" for c in top_crops
                    ],
                },
                width="stretch",
                hide_index=True,
            )


def render_image_result(result):
    """Render a successful image classification result."""
    soil_type = result.get("predicted_class")
    confidence = result.get("confidence")

    col_soil, col_conf = st.columns(2)
    with col_soil:
        st.metric("Predicted soil type", soil_type or "—")
    with col_conf:
        st.metric(
            "Model confidence",
            f"{confidence * 100:.2f}%" if confidence is not None else "—",
        )

    # Low-confidence handling — the threshold is a project display setting,
    # not a scientifically validated accuracy guarantee.
    if not result.get("is_confident"):
        st.warning(
            f"Confidence is below the project display threshold of "
            f"{CONFIDENT_THRESHOLD:.0%}. Treat this identification as "
            "**unreliable** and verify the soil type with an expert. "
            "This threshold is a project convention, not a scientifically "
            "validated accuracy guarantee."
        )

    if result.get("warning"):
        st.warning(result["warning"])

    probs = result.get("all_probabilities") or {}
    if probs:
        st.markdown("### Probability Distribution Across Soil Types")
        st.dataframe(
            {
                "Soil type": list(probs.keys()),
                "Probability": [f"{v * 100:.2f}%" for v in probs.values()],
            },
            width="stretch",
            hide_index=True,
        )

    advisory = result.get("soil_advisory") or {}

    if advisory.get("description"):
        st.markdown("### About This Soil Type")
        st.write(advisory["description"])

    if advisory.get("characteristics"):
        st.markdown("### Characteristics")
        for item in advisory["characteristics"]:
            st.markdown(f"- {item}")

    if advisory.get("suitable_crops"):
        st.markdown("### Commonly Grown Crops")
        st.write(", ".join(advisory["suitable_crops"]))

    if advisory.get("management_tips"):
        st.markdown("### Management Tips")
        for tip in advisory["management_tips"]:
            st.markdown(f"- {tip}")

    if advisory.get("caution"):
        st.markdown("### Caution")
        st.caption(advisory["caution"])


def render_combined_guidance(guidance):
    """Render the integrated agronomic guidance."""
    st.markdown("### Analysis Summary")

    crop = guidance.get("crop_from_numeric")
    soil = guidance.get("soil_type_from_image")

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Crop (numeric model)", crop.title() if crop else "—")
    with col2:
        st.metric("Soil type (image model)", soil or "—")

    st.markdown("### Compatibility")
    compatibility = guidance.get("soil_crop_compatibility")
    if compatibility is True:
        st.success("The recommended crop is listed as suitable for this soil type.")
    elif compatibility is False:
        st.warning(
            "The recommended crop is **not** listed among common crops for this "
            "soil type. Verify with a local agronomist before planting."
        )

    affinity = guidance.get("affinity_crops") or []
    if affinity:
        st.markdown("### Crops Commonly Suited to This Soil Type")
        st.write(", ".join(affinity))

    notes = guidance.get("guidance_notes") or []
    if notes:
        st.markdown("### Guidance Notes")
        for note in notes:
            st.markdown(f"- {note}")

    st.markdown("### Scientific Limitation")
    st.info(guidance.get("boundary_note") or BOUNDARY_NOTICE)


# -----------------------------
# Header
# -----------------------------

st.title("TerraLogic Advisor")
st.subheader("AI-assisted soil and crop advisory system")
st.write(
    "TerraLogic Advisor combines two independent machine learning pipelines: a "
    "numeric soil and climate analysis that recommends a crop, and a soil image "
    "classifier that identifies the visual soil type. Submit either or both to "
    "receive guidance."
)
st.info(BOUNDARY_NOTICE)

if "numeric_result" not in st.session_state:
    st.session_state.numeric_result = None
    st.session_state.numeric_inputs = None
    st.session_state.image_result = None


# -----------------------------
# Section 1 — Numeric analysis
# -----------------------------

st.divider()
st.header("1. Soil & Climate Analysis")

st.write(
    "Enter measured soil and climate values. These measurements come from a "
    "soil test and field observations — not from a photograph."
)

with st.form("numeric_form"):
    values = {}
    for key, label, low, high, default in NUMERIC_FIELDS:
        values[key] = st.number_input(
            label,
            min_value=low,
            max_value=high,
            value=default,
            step=1.0,
        )

    submitted = st.form_submit_button("Analyze Soil & Climate", type="primary")

if submitted:
    try:
        result = al.analyze_numeric_inputs(**values)
        st.session_state.numeric_inputs = dict(values)

        if result["status"] == "success":
            st.session_state.numeric_result = result
        else:
            st.session_state.numeric_result = None
            st.error(result.get("error") or "The numeric analysis failed.")

    except FileNotFoundError as exc:
        st.error(f"Model file is missing: {exc}")

    except RuntimeError as exc:
        st.error(f"The crop model could not be loaded: {exc}")

numeric_result = st.session_state.numeric_result
if numeric_result and numeric_result.get("status") == "success":
    with st.expander("Numeric analysis result", expanded=True):
        render_numeric_result(numeric_result)


# -----------------------------
# Section 2 — Image identification
# -----------------------------

st.divider()
st.header("2. Soil Image Identification")

st.write("Upload a clear photograph of bare soil for visual soil-type identification.")

uploaded = st.file_uploader(
    "Upload a soil image",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=False,
)

if uploaded is not None:
    st.image(uploaded, caption="Uploaded soil image", width="stretch")

    try:
        result = al.analyze_soil_image(uploaded)

        if result["status"] == "success":
            st.session_state.image_result = result
            with st.expander("Image identification result", expanded=True):
                render_image_result(result)
        else:
            st.session_state.image_result = None
            st.error(result.get("error") or "The image could not be processed.")

    except FileNotFoundError as exc:
        st.error(f"Model file is missing: {exc}")

    except ValueError as exc:
        st.error(f"Unsupported image: {exc}")
else:
    st.session_state.image_result = None

    if st.session_state.numeric_result:
        st.info(
            "Upload an image to enable the combined guidance section below."
        )


# -----------------------------
# Section 3 — Combined guidance
# -----------------------------

image_result = st.session_state.image_result
numeric_ready = (
    numeric_result is not None and numeric_result.get("status") == "success"
)
image_ready = (
    image_result is not None and image_result.get("status") == "success"
)

if numeric_ready and image_ready:
    st.divider()
    st.header("3. Combined Guidance")

    st.caption(
        "This section combines the numeric soil-test analysis and the visual "
        "soil-type identification. They remain two separate measurements."
    )

    # synthesize_combined_guidance() is the combined-guidance step that
    # run_terralogic_advisory() itself calls for the combined workflow.
    # Calling it directly reuses the two results already computed above
    # instead of re-running both models a second time.
    guidance = al.synthesize_combined_guidance(numeric_result, image_result)

    render_combined_guidance(guidance)


# -----------------------------
# Footer
# -----------------------------

st.divider()
st.caption(
    "TerraLogic Advisor is an educational project prototype. Its recommendations "
    "are model predictions and must be verified with appropriate agricultural "
    "extension services and laboratory soil testing before use."
)