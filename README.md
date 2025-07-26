# 💼 Employee Salary Prediction App

This is a Streamlit-based web application that predicts the monthly salary class of an individual based on their age, gender, education level, years of experience, and job category. The model is built using machine learning algorithms and provides SHAP-based interpretability to show which features contributed most to the prediction.

### 🔗 Live App

👉 [Click here to try the app](https://employee-salary-prediction-wkuskzo6p7w2zmg8aqwstw.streamlit.app/)

---

## ⚙️ Technologies Used

| Component              | Details                                             |
|------------------------|-----------------------------------------------------|
| Language               | Python                                              |
| Development Platform   | Jupyter Notebook, Streamlit                        |
| Libraries              | pandas, numpy, matplotlib, shap, sklearn, joblib   |
| Deployment             | Streamlit Cloud                                     |

---

## 🧪 Model Building Steps

1. **Data Collection**  
   - Dataset sourced from Kaggle.

2. **Preprocessing**  
   - Handled null values, removed duplicates & outliers, mapped categorical to numeric.

3. **Feature Engineering**  
   - Used `LabelEncoder` for categorical fields.  
   - Standardized features for regression.

4. **Model Selection**  
   - Compared Linear Regression, Polynomial Regression, SVR, Gradient Boosting, and XGBoost using pipeline.  
   - Chose the best model based on validation performance.

5. **Interpretability**  
   - Used SHAP (Shapley Additive Explanations) to show which features most influenced the prediction.

6. **Deployment**  
   - App built using Streamlit and deployed on Streamlit Cloud.

---
