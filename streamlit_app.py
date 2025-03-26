import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import requests

# Set page config
st.set_page_config(
    page_title="Delivery Delay Predictor",
    page_icon="🚚",
    layout="wide"
)

# Title and description
st.title("🚚 Delivery Delay Predictor")
st.markdown("""
This app predicts whether a delivery will be on time or delayed based on various features.
Enter the details below to get a prediction.
""")

# Load the model
try:
    model = joblib.load('random_forest_model.joblib')
except:
    st.error("Error: Model file not found. Please make sure 'random_forest_model.joblib' exists.")
    st.stop()

# Create input fields
st.subheader("Enter Delivery Details")

# Create two columns for better layout
col1, col2 = st.columns(2)

with col1:
    freight_value = st.number_input("Freight Value", min_value=0.0, max_value=1000.0, value=10.0)
    price_per_weight = st.number_input("Price per Weight", min_value=0.0, max_value=1000.0, value=10.0)
    
    # Date input
    purchase_date = st.date_input("Purchase Date")
    purchase_year = purchase_date.year
    purchase_month = purchase_date.month
    purchase_day = purchase_date.day
    purchase_quarter = (purchase_month - 1) // 3 + 1

# Display the entered values
with col2:
    st.markdown("### Entered Values")
    st.write(f"Freight Value: {freight_value}")
    st.write(f"Price per Weight: {price_per_weight}")
    st.write(f"Purchase Date: {purchase_date}")
    st.write(f"Purchase Year: {purchase_year}")
    st.write(f"Purchase Month: {purchase_month}")
    st.write(f"Purchase Day: {purchase_day}")
    st.write(f"Purchase Quarter: {purchase_quarter}")

# Create prediction button
if st.button("Predict Delivery Status"):
    try:
        # Prepare input data
        input_data = pd.DataFrame({
            'freight_value': [float(freight_value)],
            'purchase_year': [int(purchase_year)],
            'purchase_month': [int(purchase_month)],
            'purchase_day': [int(purchase_day)],
            'purchase_quarter': [int(purchase_quarter)],
            'price_per_weight': [float(price_per_weight)]
        })
        
        # Make prediction
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0]
        
        # Display results
        st.subheader("Prediction Results")
        
        # Create a container for the results
        result_container = st.container()
        
        with result_container:
            # Display prediction
            if prediction == 1:
                st.success("✅ On-time Delivery")
            else:
                st.error("⚠️ Delayed Delivery")
            
            # Display probability
            st.metric(
                "Probability of On-time Delivery",
                f"{probability[1]*100:.2f}%"
            )
            
            # Display feature importance
            if hasattr(model, 'feature_importances_'):
                st.subheader("Feature Importance")
                feature_importance = pd.DataFrame({
                    'Feature': input_data.columns,
                    'Importance': model.feature_importances_
                }).sort_values('Importance', ascending=False)
                
                st.bar_chart(feature_importance.set_index('Feature'))
    
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")

# Add some helpful information
st.sidebar.markdown("""
### About the Model
This model predicts delivery delays using the following features:
- Freight Value
- Purchase Date (Year, Month, Day, Quarter)
- Price per Weight

The model was trained on historical delivery data and uses a Random Forest algorithm.
""")

# Add footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>Built with ❤️ using Streamlit</p>
    <p>Model Accuracy: 82.66%</p>
</div>
""", unsafe_allow_html=True) 