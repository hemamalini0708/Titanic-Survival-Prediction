import streamlit as st
import pickle
import pandas as pd
import numpy as np
import os

# Page configuration
st.set_page_config(
    page_title="🚢 Titanic Survival Predictor",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Title and description
st.title("🚢 Titanic Survival Prediction")
st.markdown(
    """
    **Predict whether a Titanic passenger would have survived using a pre-trained Naive Bayes classifier**

    Enter passenger details and click "Predict Survival" to get the prediction.
    """
)


# Load pre-trained model
@st.cache_resource
def load_trained_model():
    """Load the pre-trained Titanic model from Titanic_model.pkl"""

    # Try multiple paths where the model might be located
    possible_paths = [
        'model/Titanic_model.pkl',  # Current directory
        './model/Titanic_model.pkl',  # Current directory with ./
        os.path.expanduser('~model/Titanic_model.pkl'),  # Home directory
    ]

    model_path = None
    for path in possible_paths:
        if os.path.exists(path):
            model_path = path
            break

    if model_path is None:
        st.error(" Model file 'Titanic_model.pkl' not found!")
        st.info("Please place 'Titanic_model.pkl' in the same directory as this script.")
        return None

    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)

        # Verify model is trained
        if hasattr(model, 'classes_'):
            st.sidebar.success(f" Model loaded from: {model_path}")
            return model
        else:
            st.error(" Model is not properly trained")
            return None
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None


# Load model
model = load_trained_model()

if model is None:
    st.stop()

# Model info in sidebar
with st.sidebar:
    st.header("Model Information")
    st.info(
        """
        **Model Type:** Gaussian Naive Bayes

        **Classes:** 
        - 0 = Did Not Survive 
        - 1 = Survived 

        **Features Used:** 7
        - Age
        - Parch (Parents/Children)
        - Fare
        - Sex (Male/Female encoded)
        - Port of Embarkation

        **Status:** Pre-trained and Ready
        """
    )

# Main form section
st.markdown("---")
st.subheader("Passenger Information")
st.markdown("*Enter the details of the passenger you want to analyze*")

# Create two columns for input
col1, col2 = st.columns(2)

with col1:
    age = st.number_input(
        " Age",
        min_value=0.0,
        max_value=100.0,
        value=30.0,
        step=0.5,
        help="Passenger's age in years"
    )

    fare = st.number_input(
        "Fare (£)",
        min_value=0.0,
        max_value=600.0,
        value=50.0,
        step=0.5,
        help="Ticket fare in pounds sterling"
    )

with col2:
    parch = st.number_input(
        "Parents/Children",
        min_value=0,
        max_value=10,
        value=0,
        step=1,
        help="Number of parents or children aboard"
    )

    sex = st.radio(
        "Sex",
        options=["Male", "Female"],
        horizontal=True,
        help="Passenger's gender"
    )

embarked = st.selectbox(
    "Port of Embarkation",
    options=["Southampton (S)", "Cherbourg (C)", "Queenstown (Q)"],
    help="Port where passenger boarded"
)

st.markdown("---")

# Prediction button
col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    predict_button = st.button(
        "Predict Survival",
        use_container_width=True,
        type="primary"
    )

# Make prediction
if predict_button:
    try:
        # Extract port code from selected option
        port_code = embarked.split("(")[1].rstrip(")")

        # Create input DataFrame with exact feature names
        input_data = pd.DataFrame({
            'Age': [float(age)],
            'Parch': [int(parch)],
            'Fare': [float(fare)],
            'Sex_female': [1 if sex == "Female" else 0],
            'Sex_male': [1 if sex == "Male" else 0],
            'Embarked_C': [1 if port_code == "C" else 0],
            'Embarked_S': [1 if port_code == "S" else 0]
        })

        # Make prediction using the pre-trained model
        prediction = model.predict(input_data)[0]

        # Get prediction probability
        prediction_proba = model.predict_proba(input_data)[0]
        confidence = max(prediction_proba) * 100

        # Display results
        st.markdown("---")
        st.subheader("Prediction Result")

        # Main prediction result with balloons
        result_col1, result_col2 = st.columns(2)

        with result_col1:
            if prediction == 1:
                st.success("**SURVIVED**")
                st.balloons()
            else:
                st.error("**DID NOT SURVIVE**")

        with result_col2:
            st.metric("Model", "Naive Bayes")
            st.metric("Confidence", f"{confidence:.1f}%")

        # Detailed prediction probabilities
        st.markdown("---")
        st.subheader(" Prediction Probabilities")

        prob_col1, prob_col2 = st.columns(2)

        with prob_col1:
            st.metric(
                "Did Not Survive",
                f"{prediction_proba[0] * 100:.2f}%"
            )

        with prob_col2:
            st.metric(
                " Survived",
                f"{prediction_proba[1] * 100:.2f}%"
            )

        # Passenger summary
        st.markdown("---")
        st.subheader(" Passenger Summary")

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

        # Feature values table
        st.markdown("---")
        st.subheader("Feature Values Used in Prediction")

        feature_df = pd.DataFrame({
            'Feature': ['Age', 'Parch', 'Fare', 'Sex (Female)', 'Sex (Male)', 'Port (C)', 'Port (S)'],
            'Value': [
                age,
                parch,
                fare,
                input_data['Sex_female'][0],
                input_data['Sex_male'][0],
                input_data['Embarked_C'][0],
                input_data['Embarked_S'][0]
            ]
        })

        st.dataframe(feature_df, use_container_width=True, hide_index=True)

    except Exception as e:
        st.error(f" Error making prediction: {str(e)}")
        st.write("**Please check your input values and try again.**")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #888; font-size: 0.85rem;'>
    <p>Project Done By Hema Malini Build by Streamlit</p>
    </div>
    """,
    unsafe_allow_html=True
)