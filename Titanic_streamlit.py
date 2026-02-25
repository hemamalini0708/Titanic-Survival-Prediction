import streamlit as st
import pickle
import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
import os

# Page configuration
st.set_page_config(
    page_title="Titanic Survival Predictor",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Title and description
st.title("🚢 Titanic Survival Prediction")
st.markdown("Predict whether a passenger would have survived the Titanic disaster using Naive Bayes ML algorithm")


# Initialize Naive Bayes model
@st.cache_resource
def load_model():
    model = GaussianNB()

    # Try to load pre-trained model from pickle file
    model_path = 'titanic_model.pkl'
    if os.path.exists(model_path):
        try:
            with open(model_path, 'rb') as f:
                loaded_model = pickle.load(f)
                if isinstance(loaded_model, GaussianNB):
                    return loaded_model
        except Exception as e:
            st.warning(f"Error loading pre-trained model: {e}")

    # Train with dummy data if model isn't pre-trained
    try:
        dummy_X = pd.DataFrame({
            'Age': [30, 40, 20, 50, 25, 35, 45, 55, 18, 28],
            'Parch': [0, 1, 0, 2, 1, 0, 1, 2, 0, 1],
            'Fare': [7.25, 50, 30, 100, 15, 25, 75, 120, 5, 45],
            'Sex_female': [0, 1, 1, 0, 1, 1, 0, 1, 0, 1],
            'Sex_male': [1, 0, 0, 1, 0, 0, 1, 0, 1, 0],
            'Embarked_C': [0, 1, 0, 1, 0, 1, 0, 1, 0, 0],
            'Embarked_S': [1, 0, 1, 0, 1, 0, 1, 0, 1, 1]
        })
        dummy_y = [0, 1, 1, 1, 0, 1, 0, 1, 0, 1]  # 0 = Did not survive, 1 = Survived

        model.fit(dummy_X, dummy_y)
        st.info("Model trained with default data")
    except Exception as e:
        st.error(f"Error training model: {e}")

    return model


# Load model
model = load_model()

# Check if model is trained
if not hasattr(model, 'classes_'):
    st.error("Model is not trained. Please provide training data.")
    st.stop()

# Main prediction form
st.markdown("---")
st.subheader("Passenger Information")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input(
        "Age",
        min_value=0,
        max_value=100,
        value=30,
        step=1,
        help="Passenger's age in years"
    )

    fare = st.number_input(
        "Fare (£)",
        min_value=0.0,
        max_value=600.0,
        value=32.0,
        step=0.5,
        help="Ticket fare in pounds"
    )

with col2:
    parch = st.number_input(
        "Parents/Children Aboard",
        min_value=0,
        max_value=10,
        value=0,
        step=1,
        help="Number of parents or children traveling with passenger"
    )

    sex = st.selectbox(
        "Sex",
        options=["Male", "Female"],
        help="Passenger's gender"
    )

embarked = st.selectbox(
    "Port of Embarkation",
    options=["Southampton (S)", "Cherbourg (C)", "Queenstown (Q)"],
    help="Port where passenger boarded"
)

st.markdown("---")

# Create prediction button
col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    if st.button("Predict Survival", use_container_width=True, type="primary"):
        try:
            # Extract port code
            port_code = embarked.split("(")[1].rstrip(")")

            # Create DataFrame with correct feature columns
            input_data = pd.DataFrame({
                'Age': [float(age)],
                'Parch': [int(parch)],
                'Fare': [float(fare)],
                'Sex_female': [1 if sex == "Female" else 0],
                'Sex_male': [1 if sex == "Male" else 0],
                'Embarked_C': [1 if port_code == "C" else 0],
                'Embarked_S': [1 if port_code == "S" else 0]
            })

            # Make prediction
            prediction = model.predict(input_data)[0]

            # Get prediction probability
            prediction_proba = model.predict_proba(input_data)[0]
            confidence = max(prediction_proba) * 100

            # Display result
            st.markdown("---")
            st.subheader("Prediction Result")

            result_col1, result_col2 = st.columns(2)

            with result_col1:
                if prediction == 1:
                    st.success("SURVIVED")
                    st.balloons()
                else:
                    st.error("DID NOT SURVIVE")

            with result_col2:
                st.metric("Model", "Naive Bayes")
                st.metric("Confidence", f"{confidence:.1f}%")

            # Display passenger summary
            st.markdown("---")
            st.subheader("Passenger Summary")

            summary_col1, summary_col2, summary_col3 = st.columns(3)
            with summary_col1:
                st.info(f"**Age:** {int(age)} years")
            with summary_col2:
                st.info(f"**Fare:** £{fare:.2f}")
            with summary_col3:
                st.info(f"**Gender:** {sex}")

            summary_col1, summary_col2 = st.columns(2)
            with summary_col1:
                st.info(f"**Relatives:** {int(parch)}")
            with summary_col2:
                st.info(f"**Port:** {embarked}")

        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #888;'>
    <p>Project Done By Hema Malini | Build by Streamli</p>
    <p>ML Model: Naive Bayes Classifier</p>
    </div>
    """,
    unsafe_allow_html=True
)