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

MODEL_PATH = "Fraud Detection Model.pkl"

if not os.path.exists(MODEL_PATH):
    st.error(
        "❌ Model file not found. Please run train_model.py first."
    )
    st.stop()

model = joblib.load(MODEL_PATH)


# ============================================
# HEADER
# ============================================

st.title("💳 Credit Card Fraud Detection")

st.write(
    "A Machine Learning based system for detecting "
    "potentially fraudulent credit card transactions."
)

st.caption(
    "Model: Logistic Regression | "
    "Fraud Recall: 91.84%"
)

st.divider()


# ============================================
# SIDEBAR
# ============================================

st.sidebar.title("📌 About Project")

st.sidebar.info(
    """
This project uses Machine Learning to classify
credit card transactions as:

✅ Legitimate Transaction

🚨 Fraudulent Transaction

Model:
Logistic Regression

Dataset:
Credit Card Fraud Detection Dataset

Dataset Size:
284,807 transactions
"""
)

st.sidebar.markdown("---")

st.sidebar.subheader("📊 Model Performance")

st.sidebar.write("Accuracy: **97.55%**")
st.sidebar.write("Precision: **6.10%**")
st.sidebar.write("Recall: **91.84%**")
st.sidebar.write("F1-Score: **11.44%**")


# ============================================
# TRANSACTION INPUT
# ============================================

st.subheader("🔍 Enter Transaction Details")

st.write(
    "Enter the transaction features below and click "
    "**Predict Transaction**."
)


# ============================================
# BASIC TRANSACTION INFORMATION
# ============================================

col1, col2 = st.columns(2)

with col1:
    time_value = st.number_input(
        "⏱️ Time",
        value=0.0,
        format="%.4f"
    )

with col2:
    amount_value = st.number_input(
        "💰 Transaction Amount",
        min_value=0.0,
        value=100.0,
        format="%.2f"
    )


# ============================================
# TRANSACTION FEATURES
# ============================================

st.subheader("🧮 Transaction Features")

st.caption(
    "Enter values for V1 to V28. Default value is 0."
)


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

    # ------------------------------------------
    # CREATE INPUT DATA
    # ------------------------------------------

    input_data = {
        "Time": time_value
    }

    input_data.update(features)

    input_data["Amount"] = amount_value


    # ------------------------------------------
    # CONVERT INPUT INTO DATAFRAME
    # ------------------------------------------

    input_df = pd.DataFrame([input_data])


    # ------------------------------------------
    # MAKE PREDICTION
    # ------------------------------------------

    prediction = model.predict(input_df)[0]


    # ------------------------------------------
    # GET PREDICTION PROBABILITY
    # ------------------------------------------

    probability = model.predict_proba(input_df)[0]

    legitimate_probability = probability[0] * 100
    fraud_probability = probability[1] * 100


    # ==========================================
    # PREDICTION RESULT
    # ==========================================

    st.divider()

    st.subheader("📊 Prediction Result")


    if prediction == 1:

        st.error(
            "🚨 FRAUDULENT TRANSACTION DETECTED"
        )

    else:

        st.success(
            "✅ TRANSACTION APPEARS LEGITIMATE"
        )


    # ==========================================
    # PROBABILITY METRICS
    # ==========================================

    result_col1, result_col2 = st.columns(2)

    with result_col1:

        st.metric(
            "✅ Legitimate Probability",
            f"{legitimate_probability:.2f}%"
        )

    with result_col2:

        st.metric(
            "🚨 Fraud Probability",
            f"{fraud_probability:.2f}%"
        )


    # ==========================================
    # PROBABILITY CHART
    # ==========================================

    st.subheader("📈 Prediction Probability")

    probability_data = pd.DataFrame(
        {
            "Probability": [
                legitimate_probability,
                fraud_probability
            ]
        },
        index=[
            "Legitimate",
            "Fraud"
        ]
    )

    st.bar_chart(
        probability_data
    )


    # ==========================================
    # INPUT DATA
    # ==========================================

    with st.expander("🔎 View Transaction Input"):

        st.dataframe(
            input_df,
            use_container_width=True
        )


    # ==========================================
    # DISCLAIMER
    # ==========================================

    st.info(
        "⚠️ This project is for educational and "
        "demonstration purposes only. It should not "
        "be used as a real-world financial fraud "
        "detection system."
    )


# ============================================
# FOOTER
# ============================================

st.divider()

st.caption(
    "💳 Credit Card Fraud Detection | "
    "Machine Learning Project"
)