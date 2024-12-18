import pickle
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings('ignore')

# Title 
st.title('Diabetes Prediction')

# Load The Pickle Files
scaler = pickle.load(open('artifacts\scaler.pkl','rb'))
model = pickle.load(open('artifacts\model.pkl','rb'))

    
# Input Fields
Pregnancies = st.number_input("Pregnancies")
Glucose = st.number_input("Glucose")
BloodPressure = st.number_input("BloodPressure")
SkinThickness = st.number_input("SkinThickness")
Insulin = st.number_input("Insulin")
BMI = st.number_input("BMI")
DiabetesPedigreeFunction = st.number_input("DiabetesPedigreeFunction")
Age = st.number_input("Age")

data = {'Pregnancies':Pregnancies,'Glucose':Glucose,'BloodPressure':BloodPressure,'SkinThickness':SkinThickness,
        'Insulin':Insulin,'BMI':BMI,'DiabetesPedigreeFunction':DiabetesPedigreeFunction,'Age':Age}
    
df = pd.DataFrame([data])

scaled = scaler.transform(df)
prediction = model.predict(scaled)

if st.button('Predict'):
    if prediction == 0:
        st.success("The patient has no diabetic.")
    else:
        st.error("The patient has diabetic.")


