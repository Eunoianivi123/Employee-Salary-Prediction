import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder 
import joblib
import shap
import matplotlib.pyplot as plt
# shap.initjs()

model = joblib.load("best_model.pkl")
le_gender = joblib.load('le_gender.pkl')
le_job = joblib.load('le_job.pkl')
# st.set_option('deprecation.showPyplotGlobalUse', False)

education_order ={
    'High School': 0 ,
    "Bachelor's Degree" : 1,
    "Master's Degree" : 2,
    'PhD' : 3
}

st.title("Employee Salary Prediction App")
st.write("Enter the Input Details:")

age = st.slider('Age', 21, 50)
gender = st.selectbox('Gender',['Male','Female'])
edu_level = st.selectbox('Education Level', ["High School","Bachelor's Degree","Master's Degree", "PhD"])
experience = st.slider('Years of Experience', 0,30)
job_cate = st.selectbox('Job Category',['Software/Engineering','Data/Analytics', 'Product' , 'Sales', 'Marketing','HR', 'Finance' , 'Management','Support', 'Design', 'Project Management' , 'Operations', 'Writing' ,'Customer Service' , 'Admin' , 'Other'])

input_df = pd.DataFrame({
    'Age': [age],
    'Gender': [le_gender.transform([gender])[0]],
    'Education Level': [education_order[edu_level]],
    'Years of Experience': [experience],
    'Job Category': [le_job.transform([job_cate])[0]]
    
})

if st.button("Predict Salary Class"):
    prediction = model.predict(input_df)
    st.success(f"✅ Prediction: Your Monthly Salary is ₹{prediction[0]}")

explainer = shap.Explainer(model)
shap_values = explainer(input_df)

st.subheader("Local Feature Impact (SHAP)")
st.write("Red - Increases the Predicted Salary ; Blue - Decreases the Predicted Salary")
fig = shap.plots.waterfall(shap_values[0], show=False)
st.pyplot(bbox_inches='tight')
plt.clf() 
