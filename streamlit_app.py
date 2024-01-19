

# IMPORT STATEMENTS
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import json


# Install Lottie package

# Load the dataset
df = pd.read_csv('c:\\Users\\Chaymae\\Desktop\\diabetes.csv')

# Replace zeros with NaN in the first six columns
df[df.columns[:6]] = df[df.columns[:6]].replace(0, np.nan)

# Impute missing values with the median based on Outcome
for column in df.columns[:6]:
    df[column][(df['Outcome'] == 0) & (df[column].isnull())] = df[column][(df['Outcome'] == 0)].median()
    df[column][(df['Outcome'] == 1) & (df[column].isnull())] = df[column][(df['Outcome'] == 1)].median()



st.subheader('Training Data Stats')
st.write(df.describe())

# X AND Y DATA
X = df.drop('Outcome', axis=1)
y = df['Outcome']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# PATIENT DATA
def user_report():
    Pregnancies = st.slider("Your Number of Pregnancies", 0,17, 3)
    Glucose = st.slider("Your Glucose", 0,200, 120)
    BloodPressure = st.slider("Your Blood Pressure", 0,122, 70)
    SkinThickness = st.slider("Your Skin thickness", 0,100, 20)
    Insulin = st.slider("Your Insulin", 0,846, 79)
    BMI = st.slider("Your BMI", 0,67, 20 )
    DiabetesPedigreeFunction = st.slider("Your Diabetes Pedigree Function", 0.0,2.4, 0.47)
    Age = st.slider("Your Age", 21,88, 33)

    user_data = pd.DataFrame({
        'Pregnancies': [Pregnancies],
        'Glucose': [Glucose],
        'BloodPressure': [BloodPressure],
        'SkinThickness': [SkinThickness],
        'Insulin': [Insulin],
        'BMI': [BMI],
        'DiabetesPedigreeFunction': [DiabetesPedigreeFunction],
        'Age': [Age]
    })

    return user_data

user_data = user_report()




st.title('Diabetes Prediction')
st.sidebar.header('Patient Data')
st.sidebar.write(user_data)


# MODEL
rf = RandomForestClassifier()
rf.fit(X_train, y_train)









if st.button('Predict'):
    result = rf.predict(user_data)
    updated_res = result.flatten().astype(int)
    if updated_res == 0:
       st.write("You not are diabetic")
    else:
       st.write("You are diabetic")
   

import streamlit as st
from streamlit_lottie import st_lottie
import json

# Example Lottie animation from a local file
animation_file_path = r"C:\Users\Chaymae\Downloads\opener-loading.json"

# Function to load Lottie animation from file
def load_lottie_file(file_path):
    with open(file_path, "r") as file:
        animation_json = json.load(file)
    return st_lottie(animation_json, height=300, key="user")

# Display the Lottie animation
load_lottie_file(animation_file_path)

