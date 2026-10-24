
import json
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Student Performance Prediction",
    page_icon="🎓",
    layout="centered"
)

@st.cache_resource
def load_model():
    return joblib.load("student_performance_model.pkl")

with open("model_metadata.json", "r", encoding="utf-8") as f:
    metadata = json.load(f)

model = load_model()

st.title("🎓 Student Performance Prediction System")
st.write(
    "Enter student details to estimate the target "
    "value learned from the training dataset."
)

st.caption(
    "This model produces estimates based on historical data. "
    "It does not guarantee actual academic outcomes."
)

features = metadata["features"]
values = {}

with st.form("prediction_form"):
    for column, info in features.items():
        if info["type"] == "numeric":
            minimum = info["min"]
            maximum = info["max"]
            default = info["default"]

            if minimum == maximum:
                values[column] = minimum
                st.write(f"{column}: {minimum}")
            else:
                values[column] = st.number_input(
                    column,
                    min_value=float(minimum),
                    max_value=float(maximum),
                    value=float(min(max(default, minimum), maximum))
                )
        else:
            options = info.get("options", [])
            if options:
                default = info.get("default", options[0])
                index = options.index(default) if default in options else 0
                values[column] = st.selectbox(
                    column, options, index=index
                )
            else:
                values[column] = st.text_input(column, value="")

    submitted = st.form_submit_button("Predict Performance")

if submitted:
    input_df = pd.DataFrame([values])

    try:
        prediction = model.predict(input_df)[0]

        st.subheader("Prediction Result")

        if metadata["task"] == "regression":
            st.success(f"Estimated {metadata['target_column']}: {prediction:.2f}")
        else:
            st.success(f"Predicted {metadata['target_column']}: {prediction}")

        st.write(f"Model used: {metadata['model_name']}")

        if metadata["task"] == "classification" and hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(input_df)[0]
            classes = model.named_steps["model"].classes_

            probability_df = pd.DataFrame({
                "Class": classes,
                "Estimated probability": probabilities
            })

            st.write("Model class probabilities (not guaranteed outcomes):")
            st.dataframe(probability_df, use_container_width=True)

    except Exception as e:
        st.error(f"Prediction failed: {e}")
