import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder 
import joblib
import shap
import matplotlib.pyplot as plt

# Load model and encoders
model = joblib.load("best_model.pkl")
le_gender = joblib.load('le_gender.pkl')
le_job = joblib.load('le_job.pkl')

education_order = {
    'High School': 0,
    "Bachelor's Degree": 1,
    "Master's Degree": 2,
    'PhD': 3
}

st.title("Employee Salary Prediction App")
st.write("Enter the Input Details:")

# Collect inputs
age = st.slider('Age', 21, 50)
gender = st.selectbox('Gender', ['Male', 'Female'])
edu_level = st.selectbox('Education Level', ["High School", "Bachelor's Degree", "Master's Degree", "PhD"])
experience = st.slider('Years of Experience', 0, 30)
job_cate = st.selectbox('Job Category', [
    'Software/Engineering', 'Data/Analytics', 'Product', 'Sales', 'Marketing', 'HR', 
    'Finance', 'Management', 'Support', 'Design', 'Project Management', 'Operations', 
    'Writing', 'Customer Service', 'Admin', 'Other'
])

# Prepare input dataframe
input_df = pd.DataFrame({
    'Age': [age],
    'Gender': [le_gender.transform([gender])[0]],
    'Education Level': [education_order[edu_level]],
    'Years of Experience': [experience],
    'Job Category': [le_job.transform([job_cate])[0]]
})

# Prediction
if st.button("Predict Salary Class"):
    prediction = model.predict(input_df)
    st.success(f"✅ Prediction: Your Monthly Salary is ₹{prediction[0]}")

    # SHAP Explanation
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(input_df)

    base_value = explainer.expected_value[0] if isinstance(explainer.expected_value, list) else explainer.expected_value
    shap_vals = shap_values[0] if isinstance(shap_values, list) else shap_values

    st.subheader("Local Feature Impact (SHAP)")
    st.write("Red - Increases the Predicted Salary ; Blue - Decreases the Predicted Salary")

    plt.figure(figsize=(10, 5))
    shap.waterfall_plot(shap.Explanation(values=shap_vals[0],
                                         base_values=base_value,
                                         data=input_df.iloc[0]))
    st.pyplot(plt.gcf())
    plt.clf()
