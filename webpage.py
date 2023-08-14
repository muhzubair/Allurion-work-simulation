# Importing libraries
import streamlit as st
import pickle
import datetime
import pandas as pd
import numpy as np
import xgboost as xgb
from xgboost import XGBClassifier


# Caling the model we saved using pickle
def load_model():
    with open('saved_steps.pkl', 'rb') as file:
        data = pickle.load(file)
    return data

# loading the model
data = load_model()

# Saving the model in xgboost variable
xgb = data["model"]

# Function to build the web application interface
def show_predict_page():
    # Setting title
    st.title("Track progress of your Allurion Program")
    
    # Providing message to give instructions to users
    st.write("Please enter your information below")
    
    # User variables to input data 
    gender = st.selectbox('Gender', ('Male', 'Female'))
    unit = st.number_input("Unit")
    age = st.number_input("Age")
    height = st.number_input("height")
    weight = st.number_input("Weight")
    bmi = st.number_input("BMI")
    bfat = st.number_input("BodyFat")
    bwater = st.number_input("BodyWater")
    bone = st.number_input("Bone")
    vfat = st.number_input("VisceralFat")
    mmass = st.number_input("MuscleMass")
    bmr = st.number_input("BMR")
    tbwl = st.number_input("TBWL")
    daydiff = st.number_input("Days since the treatement started")
    
    # After the user clicks ok, and enters all information
    ok = st.button("Calculate Success")
    if ok:
        
        # Converting user data into pandas dataframe
        X = np.array([[gender, unit, age, height, weight, bmi, bfat, bwater, bone,  vfat, bmr, mmass, tbwl, daydiff]])
        df = pd.DataFrame(X, columns = ['Gender', 'Unit','Age','Height', 'Weight', 'BMI', 'BodyFat', 'BodyWater', 'Bone', 'VisceralFat', 'BMR', 'MuscleMass', 'TBWL', 'DayDiff'])

        # Label encoding for gender column
        df['Gender'] = df['Gender'].apply(lambda x: 1 if x == 'Female' else 0)
        df = df.astype('float64')

        # Prediction using the model we built 
        prediction = (xgb.predict_proba(df)[:,1] > 0.982)
        if (prediction == 1):
            st.subheader("Congratulations you are on track to be successful in the program")
        else:
            st.subheader("Unfortunately you are not on track to be successful in the program, please contact your allurion personal assistant")      
    

show_predict_page();



