# -*- coding: utf-8 -*-
"""
Created on Tue Mar 25 18:21:43 2025

@author: Nandish
"""

import numpy as np
import pandas as pd
import pickle
import streamlit as st
from sklearn.preprocessing import StandardScaler

# ✅ Load the trained model
loaded_model = pickle.load(open('C:/Users/Nandish/OneDrive/Desktop/trained_model.sav', 'rb'))

# ✅ Load dataset for feature names & fit scaler
diabetes_dataset = pd.read_csv('C:/Users/Nandish/OneDrive/Desktop/diabetes.csv')
X = diabetes_dataset.drop(columns='Outcome', axis=1)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # Fit the scaler on training data

def diabetes_prediction(input_data):
    # ✅ Convert input to DataFrame with column names
    input_data_df = pd.DataFrame([input_data], columns=diabetes_dataset.columns[:-1])

    # ✅ Standardize input data using the trained scaler
    input_data_scaled = scaler.transform(input_data_df)

    # ✅ Make prediction
    prediction = loaded_model.predict(input_data_scaled)

    return "The person is Diabetic" if prediction[0] == 1 else "The person is Not Diabetic"

def main():
    # ✅ Title and Subheader
    st.markdown("<h1 style='text-align: center; color: #4CAF50;'>Diabetes Prediction App</h1>", unsafe_allow_html=True)
    st.markdown("<h4 style='text-align: center; color: gray;'>Check your diabetes status by entering health details below</h4>", unsafe_allow_html=True)
    st.write("---")

    # ✅ Input fields organized nicely
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)

        with col1:
            Pregnancies = st.text_input('Number of Pregnancies')
            BloodPressure = st.text_input('Blood Pressure value')
            Insulin = st.text_input('Insulin Level')
            DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function value')

        with col2:
            Glucose = st.text_input('Glucose Level')
            SkinThickness = st.text_input('Skin Thickness value')
            BMI = st.text_input('BMI value')
            Age = st.text_input('Age of the Person')

        submit_button = st.form_submit_button(label='Predict Diabetes')

    diagnosis = ''

    if submit_button:
        try:
            input_data = [float(Pregnancies), float(Glucose), float(BloodPressure), float(SkinThickness),
                          float(Insulin), float(BMI), float(DiabetesPedigreeFunction), float(Age)]
            diagnosis = diabetes_prediction(input_data)
        except ValueError:
            diagnosis = "⚠ Please enter valid numeric values."

    if diagnosis:
        st.success(diagnosis)

if __name__ == '__main__':
    main()
