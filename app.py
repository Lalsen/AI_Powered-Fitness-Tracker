import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import joblib
from chatbot import create_chat_interface

# Set page config
st.set_page_config(
    page_title="Fitness Assistant",
    page_icon="💪",
    layout="wide"
)

# Title and description
st.title("💪 Fitness Assistant")
st.markdown("""
This application helps you predict calories burned during exercise and provides AI-powered fitness advice.
""")

# Create tabs for different features
tab1, tab2 = st.tabs(["Calories Predictor", "AI Fitness Assistant"])

# Load and preprocess data
@st.cache_data
def load_data():
    exercise_df = pd.read_csv('exercise.csv')
    calories_df = pd.read_csv('calories.csv')
    
    # Merge the datasets
    df = pd.merge(exercise_df, calories_df, on='User_ID')
    
    # Drop User_ID as it's not needed for prediction
    df = df.drop('User_ID', axis=1)
    
    # Encode gender
    le = LabelEncoder()
    df['Gender'] = le.fit_transform(df['Gender'])
    
    return df

# Train the model
@st.cache_resource
def train_model(df):
    X = df.drop('Calories', axis=1)
    y = df['Calories']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    return model

# Load data and train model
df = load_data()
model = train_model(df)

# Calories Predictor Tab
with tab1:
    st.subheader("Calories Burned Predictor")
    st.markdown("""
    Enter your details below to predict how many calories you'll burn during exercise.
    """)
    
    # Create input form
    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox("Gender", ["male", "female"])
        age = st.number_input("Age", min_value=1, max_value=100, value=30)
        height = st.number_input("Height (cm)", min_value=100, max_value=250, value=170)
        weight = st.number_input("Weight (kg)", min_value=30, max_value=200, value=70)

    with col2:
        duration = st.number_input("Exercise Duration (minutes)", min_value=1, max_value=300, value=30)
        heart_rate = st.number_input("Heart Rate (bpm)", min_value=40, max_value=200, value=100)
        body_temp = st.number_input("Body Temperature (°C)", min_value=35.0, max_value=42.0, value=37.0, step=0.1)

    # Create prediction button
    if st.button("Predict Calories Burned"):
        # Prepare input data
        input_data = pd.DataFrame({
            'Gender': [1 if gender == 'male' else 0],
            'Age': [age],
            'Height': [height],
            'Weight': [weight],
            'Duration': [duration],
            'Heart_Rate': [heart_rate],
            'Body_Temp': [body_temp]
        })
        
        # Make prediction
        prediction = model.predict(input_data)[0]
        
        # Display result
        st.success(f"Estimated calories burned: {prediction:.1f} calories")
        
        # Display feature importance
        st.subheader("Feature Importance")
        feature_importance = pd.DataFrame({
            'Feature': df.drop('Calories', axis=1).columns,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        st.bar_chart(feature_importance.set_index('Feature'))

    # Display data statistics
    st.subheader("Dataset Statistics")
    st.write(df.describe())

# AI Fitness Assistant Tab
with tab2:
    st.subheader("AI Fitness Assistant")
    st.markdown("""
    Chat with our AI assistant to get personalized fitness advice, workout tips, and answers to your health-related questions.
    """)
    create_chat_interface() 