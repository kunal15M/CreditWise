import streamlit as st
import pandas as pd
import joblib

# Set page config for premium look
st.set_page_config(
    page_title="CreditWise - Loan Approval Predictor",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main {
        background-color: #0e1117;
        color: #ffffff;
    }
    .stButton>button {
        background: linear-gradient(90deg, #4b6cb7 0%, #182848 100%);
        color: white;
        border-radius: 8px;
        border: none;
        padding: 10px 24px;
        transition: transform 0.2s;
    }
    .stButton>button:hover {
        transform: scale(1.05);
        background: linear-gradient(90deg, #182848 0%, #4b6cb7 100%);
    }
    .title-text {
        font-family: 'Inter', sans-serif;
        color: #4b6cb7;
        font-weight: 800;
        text-align: center;
        padding-bottom: 20px;
    }
    .result-card-approved {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        padding: 20px;
        border-radius: 12px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 15px rgba(56, 239, 125, 0.4);
    }
    .result-card-rejected {
        background: linear-gradient(135deg, #cb2d3e 0%, #ef473a 100%);
        padding: 20px;
        border-radius: 12px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 15px rgba(239, 71, 58, 0.4);
    }
</style>
""", unsafe_allow_html=True)

# Load the trained pipeline
@st.cache_resource
def load_model():
    return joblib.load("creditwise_pipeline.joblib")

model = load_model()

st.markdown("<h1 class='title-text'>🏦 CreditWise: Loan Approval Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #a0aec0;'>Fill in the applicant details below to predict loan approval using our Machine Learning model.</p>", unsafe_allow_html=True)
st.divider()

# Layout
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("👤 Applicant Info")
    applicant_income = st.number_input("Applicant Income ($)", min_value=0, value=5000, step=500)
    coapplicant_income = st.number_input("Coapplicant Income ($)", min_value=0, value=0, step=500)
    age = st.number_input("Age", min_value=18, max_value=100, value=30)
    gender = st.selectbox("Gender", ['Female', 'Male'])
    marital_status = st.selectbox("Marital Status", ['Single', 'Married'])
    dependents = st.number_input("Dependents", min_value=0, max_value=10, value=0)

with col2:
    st.subheader("💼 Employment Details")
    employment_status = st.selectbox("Employment Status", ['Salaried', 'Self-employed', 'Contract', 'Unemployed'])
    employer_category = st.selectbox("Employer Category", ['Private', 'Government', 'MNC', 'Business', 'Unemployed'])
    education_level = st.selectbox("Education Level", ['Graduate', 'Not Graduate'])
    dti_ratio = st.slider("Debt-to-Income (DTI) Ratio", 0.0, 1.0, 0.3)
    credit_score = st.slider("Credit Score", 300, 850, 650)
    
with col3:
    st.subheader("📊 Loan Details")
    loan_amount = st.number_input("Loan Amount ($)", min_value=1000, value=20000, step=1000)
    loan_term = st.selectbox("Loan Term (Months)", [12, 24, 36, 48, 60, 72, 84, 120, 180, 240, 360])
    loan_purpose = st.selectbox("Loan Purpose", ['Personal', 'Car', 'Business', 'Home', 'Education'])
    existing_loans = st.number_input("Existing Loans", min_value=0, max_value=20, value=0)
    savings = st.number_input("Savings ($)", min_value=0, value=5000, step=500)
    collateral_value = st.number_input("Collateral Value ($)", min_value=0, value=10000, step=1000)
    property_area = st.selectbox("Property Area", ['Urban', 'Semiurban', 'Rural'])

st.divider()

if st.button("🔮 Predict Loan Approval", use_container_width=True):
    # Construct input dataframe
    input_data = pd.DataFrame({
        'Applicant_Income': [applicant_income],
        'Coapplicant_Income': [coapplicant_income],
        'Age': [age],
        'Dependents': [dependents],
        'Credit_Score': [credit_score],
        'Existing_Loans': [existing_loans],
        'DTI_Ratio': [dti_ratio],
        'Savings': [savings],
        'Collateral_Value': [collateral_value],
        'Loan_Amount': [loan_amount],
        'Loan_Term': [float(loan_term)],
        'Employment_Status': [employment_status],
        'Marital_Status': [marital_status],
        'Loan_Purpose': [loan_purpose],
        'Property_Area': [property_area],
        'Education_Level': [education_level],
        'Gender': [gender],
        'Employer_Category': [employer_category]
    })
    
    try:
        prediction = model.predict(input_data)[0]
        
        st.write("") # Spacer
        if prediction == "Yes":
            st.markdown(
                """
                <div class='result-card-approved'>
                    <h2>🎉 Loan Approved!</h2>
                    <p>Based on the model, this applicant is likely to be approved for the loan.</p>
                </div>
                """, unsafe_allow_html=True
            )
            st.balloons()
        else:
            st.markdown(
                """
                <div class='result-card-rejected'>
                    <h2>⚠️ Loan Rejected</h2>
                    <p>Based on the model, this applicant is unlikely to be approved.</p>
                </div>
                """, unsafe_allow_html=True
            )
    except Exception as e:
        st.error(f"Error making prediction: {e}")
