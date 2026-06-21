# Importing laibrary
import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
import pandas as pd
import pickle

# Loading the model 
model = tf.keras.models.load_model('model.h5')

# Loading preprocessor
with open('preprocessor.pkl','rb') as file:
    preprocessor = pickle.load(file)

# streamlit app
st.title('Customer churn Prediction')

# User input
geography = st.selectbox('Geography', ['France','Germany','Spain'])
gender = st.selectbox('Gender', ['Female','Male'])
age = st.slider('Age', 18, 92)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure', 0, 10)
num_of_products = st.slider('Number of Products', 1, 4)
has_cr_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])

# Prepare the input data
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Geography':[geography],
    'Gender': [gender],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})

input_transformed = preprocessor.transform(input_data)

# Predict churn
prediction = model.predict(input_transformed)
prediction_proba = prediction[0][0]

# Display
st.write(f'Churn Probability: {prediction_proba:.2f}')

if prediction_proba > 0.5:
    st.write('The customer is likely to churn.')
else:
    st.write('Customer is unlikely to churn.')
