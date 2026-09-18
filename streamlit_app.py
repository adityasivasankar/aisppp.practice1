import streamlit as st
import pandas as pd
import pickle

# ============================================================
# GENERALIZED STREAMLIT ML PREDICTION APP
# ============================================================

st.set_page_config(
    page_title="ML Prediction System",
    page_icon="📊",
    layout="centered"
)

MODEL_FILE = "trained_model.pkl"

try:
    with open(MODEL_FILE, "rb") as file:
        package = pickle.load(file)
except FileNotFoundError:
    st.error(
        "trained_model.pkl was not found. Keep it in the same folder as streamlit_app.py."
    )
    st.stop()
except Exception as error:
    st.error(f"Could not load the model file: {error}")
    st.stop()

try:
    model = package["model"]
    target_column = package["target_column"]
    selected_features = package["selected_features"]
    categorical_features = package["categorical_features"]
    numerical_features = package["numerical_features"]
    categorical_values = package["categorical_values"]
    numeric_ranges = package["numeric_ranges"]
except KeyError as error:
    st.error(
        f"The pickle file is missing required information: {error}. "
        "Regenerate trained_model.pkl using the generalized notebook."
    )
    st.stop()

st.title("ML Prediction System")
st.write("Predicting:")
st.subheader(target_column.replace("_", " "))
st.caption(
    "This form is generated automatically from the features selected during model training."
)
st.divider()

input_data = {}

with st.form("prediction_form"):
    for feature in selected_features:
        display_name = feature.replace("_", " ")

        if feature in categorical_features:
            options = categorical_values.get(feature, [])

            if options:
                input_data[feature] = st.selectbox(
                    display_name,
                    options
                )
            else:
                input_data[feature] = st.text_input(display_name)

        else:
            info = numeric_ranges.get(
                feature,
                {"min": 0.0, "max": 100.0, "mean": 0.0}
            )

            minimum = float(info["min"])
            maximum = float(info["max"])
            mean_value = float(info["mean"])

            input_data[feature] = st.number_input(
                display_name,
                min_value=minimum,
                max_value=maximum,
                value=mean_value
            )

            st.caption(
                f"Training range: {minimum:.2f} to {maximum:.2f}"
            )

    submit = st.form_submit_button(
        "Predict",
        use_container_width=True
    )

if submit:
    try:
        input_df = pd.DataFrame([input_data])
        input_df = input_df[selected_features]
        prediction = model.predict(input_df)[0]

        st.success(
            f"Predicted {target_column.replace('_', ' ')}: {prediction:.2f}"
        )

        with st.expander("View entered values"):
            st.dataframe(input_df, use_container_width=True)

    except Exception as error:
        st.error(f"Prediction failed: {error}")

st.divider()
st.caption(
    "Generalized ML Prediction App | "
    "Random Forest Feature Selection + Linear Regression"
)
