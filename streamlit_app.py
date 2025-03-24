import streamlit as st
import requests
import pandas as pd
from datetime import datetime, timedelta
import json

st.set_page_config(page_title="Olist Sales Prediction", page_icon="📊", layout="wide")

st.title("📊 Olist Sales Prediction")
st.markdown("""
This application predicts sales performance based on product and delivery features.
Enter the details below to get a prediction.
""")

# Create two columns for input fields
col1, col2 = st.columns(2)

with col1:
    st.subheader("Product Details")
    price = st.number_input("Product Price (BRL)", min_value=0.0, value=100.0)
    freight_value = st.number_input("Freight Value (BRL)", min_value=0.0, value=20.0)
    product_weight_g = st.number_input("Product Weight (g)", min_value=0.0, value=1000.0)
    product_length_cm = st.number_input("Product Length (cm)", min_value=0.0, value=20.0)
    product_height_cm = st.number_input("Product Height (cm)", min_value=0.0, value=10.0)
    product_width_cm = st.number_input("Product Width (cm)", min_value=0.0, value=10.0)

with col2:
    st.subheader("Delivery Details")
    order_date = st.date_input("Order Purchase Date", datetime.now())
    order_time = st.time_input("Order Purchase Time", datetime.now().time())
    order_purchase_timestamp = datetime.combine(order_date, order_time)
    
    # Calculate delivery dates
    carrier_date = order_purchase_timestamp + timedelta(days=3)
    customer_date = carrier_date + timedelta(days=5)
    
    order_delivered_carrier_date = st.date_input(
        "Order Delivered to Carrier Date",
        carrier_date,
        min_value=order_date
    )
    order_delivered_customer_date = st.date_input(
        "Order Delivered to Customer Date",
        customer_date,
        min_value=order_delivered_carrier_date
    )

# Create prediction button
if st.button("Get Prediction"):
    # Prepare data for API
    data = {
        "price": price,
        "freight_value": freight_value,
        "product_weight_g": product_weight_g,
        "product_length_cm": product_length_cm,
        "product_height_cm": product_height_cm,
        "product_width_cm": product_width_cm,
        "order_purchase_timestamp": order_purchase_timestamp.strftime("%Y-%m-%d %H:%M:%S"),
        "order_delivered_carrier_date": order_delivered_carrier_date.strftime("%Y-%m-%d"),
        "order_delivered_customer_date": order_delivered_customer_date.strftime("%Y-%m-%d")
    }
    
    try:
        # Make API request
        response = requests.post("http://localhost:5000/predict", json=data)
        result = response.json()
        
        if result["status"] == "success":
            st.success("Prediction Successful!")
            
            # Display prediction in a nice format
            col1, col2, col3 = st.columns(3)
            with col2:
                st.metric(
                    label="Predicted Sales Performance",
                    value=f"{result['prediction']:.2f}",
                    delta=None
                )
            
            # Add explanation
            st.markdown("""
            ### Prediction Explanation
            The model considers various factors including:
            - Product characteristics (price, dimensions, weight)
            - Delivery timing and performance
            - Time-based features (seasonality, day of week)
            """)
            
        else:
            st.error(f"Error: {result.get('error', 'Unknown error occurred')}")
            
    except Exception as e:
        st.error(f"Error connecting to the API: {str(e)}")

# Add sidebar with information
with st.sidebar:
    st.header("About")
    st.markdown("""
    This application uses machine learning to predict sales performance
    based on historical data from Olist's marketplace.
    
    ### Features Considered:
    - Product details (price, dimensions, weight)
    - Delivery information
    - Time-based features
    - Derived metrics (price per volume, delivery delay)
    """)
    
    st.header("Model Information")
    st.markdown("""
    The model is trained on historical Olist data and uses:
    - Random Forest algorithm
    - Cross-validation for robust performance
    - Feature engineering for better predictions
    """) 