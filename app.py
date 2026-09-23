
import streamlit as st
import pandas as pd
import joblib
import os


# ============================================
# PAGE CONFIGURATION
# ============================================

st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide"
)


# ============================================
# LOAD TRAINED MODEL
# ============================================

MODEL_PATH = "fraud_detection_model.pkl"

if not os.path.exists(MODEL_PATH):
    st.error(
        "Model file not found. Please run train_model.py first."
    )
    st.stop()

model = joblib.load(MODEL_PATH)


# ============================================
# TITLE
# ============================================

st.title("💳 Credit Card Fraud Detection")
st.write(
    "Machine Learning based system for detecting "
    "fraudulent credit card transactions."
)

st.divider()


# ============================================
# SIDEBAR
# ============================================

st.sidebar.title("About Project")

st.sidebar.info(
    """
    This project uses Machine Learning to classify
    credit card transactions as:

    • Legitimate Transaction
    • Fraudulent Transaction

    Model:
    Logistic Regression

    Dataset:
    Credit Card Fraud Detection Dataset
    """
)


# ============================================
# TRANSACTION INPUT
# ============================================

st.subheader("🔍 Enter Transaction Details")

st.write(
    "Enter the transaction features below and click "
    "**Predict Transaction**."
)


# --------------------------------------------
# Basic transaction information
# --------------------------------------------

col1, col2 = st.columns(2)

with col1:
    time_value = st.number_input(
        "Time",
        value=0.0,
        format="%.4f"
    )

with col2:
    amount_value = st.number_input(
        "Transaction Amount",
        min_value=0.0,
        value=100.0,
        format="%.2f"
    )


st.subheader("Transaction Features")

st.caption(
    "Enter values for V1 to V28. Default value is 0."
)


# ============================================
# V1 - V28 INPUTS
# ============================================

features = {}

columns = st.columns(4)

for i in range(1, 29):

    column_index = (i - 1) % 4

    with columns[column_index]:
        features[f"V{i}"] = st.number_input(
            f"V{i}",
            value=0.0,
            format="%.6f",
            key=f"v{i}"
        )


# ============================================
# PREDICTION BUTTON
# ============================================

st.divider()

predict_button = st.button(
    "🔎 Predict Transaction",
    use_container_width=True
)


# ============================================
# PREDICTION
# ============================================

if predict_button:

    # Create input dictionary
    input_data = {
        "Time": time_value
    }

    # Add V1-V28
    input_data.update(features)

    # Add Amount
    input_data["Amount"] = amount_value

    # Convert to DataFrame
    input_df = pd.DataFrame([input_data])

    # Make prediction
    prediction = model.predict(input_df)[0]

    # Get probability
    probability = model.predict_proba(input_df)[0]

    fraud_probability = probability[1] * 100
    legitimate_probability = probability[0] * 100


    # ========================================
    # RESULT
    # ========================================

    st.divider()

    st.subheader("📊 Prediction Result")

    if prediction == 1:

        st.error(
            "🚨 FRAUDULENT TRANSACTION DETECTED"
        )

        st.write(
            f"Fraud Probability: **{fraud_probability:.2f}%**"
        )

        st.write(
            f"Legitimate Probability: "
            f"**{legitimate_probability:.2f}%**"
        )

    else:

        st.success(
            "✅ TRANSACTION APPEARS LEGITIMATE"
        )

        st.write(
            f"Legitimate Probability: "
            f"**{legitimate_probability:.2f}%**"
        )

        st.write(
            f"Fraud Probability: **{fraud_probability:.2f}%**"
        )


    # ========================================
    # SHOW INPUT DATA
    # ========================================

    with st.expander("View Transaction Data"):

        st.dataframe(
            input_df,
            use_container_width=True
        )


# ============================================
# FOOTER
# ============================================

st.divider()

st.caption(
    "Credit Card Fraud Detection | "
    "Machine Learning + Streamlit"
)
