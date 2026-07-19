import streamlit as st
import pickle
import numpy as np

# Page config
st.set_page_config(
    page_title="Loan Approval Predictor",
    page_icon="🏦",
    layout="centered"
)

# Custom CSS for unique design
st.markdown("""
<style>
.main { background-color: #0e1117; }
.stButton>button {
    background-color: #00c853;
    color: white;
    border-radius: 10px;
    width: 100%;
    height: 50px;
    font-size: 18px;
}
</style>
""", unsafe_allow_html=True)

# Title
st.markdown("# 🏦 Loan Approval Prediction System")
st.markdown("### IBM PBEL Virtual Training Project")
st.divider()

# Load model
model = pickle.load(open('Model/loan_model.pkl', 'rb'))

# Two columns layout
col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("👤 Gender", ["Male", "Female"])
    married = st.selectbox("💍 Married", ["Yes", "No"])
    dependents = st.selectbox("👨‍👩‍👧 Dependents", ["0","1","2","3+"])
    education = st.selectbox("🎓 Education", 
                   ["Graduate","Not Graduate"])
    self_employed = st.selectbox("💼 Self Employed", ["Yes","No"])
    property_area = st.selectbox("🏘️ Property Area",
                   ["Urban","Semiurban","Rural"])

with col2:
    applicant_income = st.number_input(
        "💰 Applicant Income", 
        min_value=0, value=5000)
    coapplicant_income = st.number_input(
        "💰 Co-applicant Income", 
        min_value=0, value=0)
    loan_amount = st.number_input(
        "🏷️ Loan Amount (thousands)", 
        min_value=0, value=100)
    loan_term = st.selectbox(
        "📅 Loan Term (months)", 
        [360, 180, 240, 120, 60])
    credit_history = st.selectbox(
        "📊 Credit History", 
        ["Good (1.0)", "Bad (0.0)"])

st.divider()

# Predict button
if st.button("🔍 Check Loan Eligibility"):
    
    # Encode inputs
    gender_val = 1 if gender == "Male" else 0
    married_val = 1 if married == "Yes" else 0
    dependents_val = 3 if dependents == "3+" else int(dependents)
    education_val = 0 if education == "Graduate" else 1
    self_employed_val = 1 if self_employed == "Yes" else 0
    property_val = 2 if property_area == "Urban" else (1 if property_area == "Semiurban" else 0)
    credit_val = 1.0 if "Good" in credit_history else 0.0

    features = np.array([[gender_val, married_val,
                          dependents_val, education_val,
                          self_employed_val, applicant_income,
                          coapplicant_income, loan_amount,
                          loan_term, credit_val, property_val]])

    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0]

    st.divider()
    
    if prediction == 1:
        st.success("# ✅ LOAN APPROVED!")
        st.balloons()
        st.metric("Approval Probability", 
                  f"{probability[1]*100:.1f}%")
    else:
        st.error("# ❌ LOAN REJECTED")
        st.metric("Rejection Probability",
                  f"{probability[0]*100:.1f}%")
    
    # Probability bar
    st.progress(float(probability[1]))
    st.caption(f"Confidence: {max(probability)*100:.1f}%")