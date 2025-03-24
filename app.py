from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import os

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load the trained model
model = joblib.load('best_model.joblib')

@app.route('/')
def home():
    return jsonify({
        'status': 'success',
        'message': 'Olist Sales Prediction API is running'
    })

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from request
        data = request.get_json()
        
        # Create DataFrame with input data
        input_data = pd.DataFrame([data])
        
        # Add time-based features
        input_data['order_purchase_timestamp'] = pd.to_datetime(input_data['order_purchase_timestamp'])
        input_data['purchase_year'] = input_data['order_purchase_timestamp'].dt.year
        input_data['purchase_month'] = input_data['order_purchase_timestamp'].dt.month
        input_data['purchase_day'] = input_data['order_purchase_timestamp'].dt.day
        input_data['purchase_weekday'] = input_data['order_purchase_timestamp'].dt.weekday
        input_data['purchase_hour'] = input_data['order_purchase_timestamp'].dt.hour
        input_data['purchase_quarter'] = input_data['order_purchase_timestamp'].dt.quarter
        input_data['is_weekend'] = input_data['purchase_weekday'].isin([5, 6]).astype(int)
        input_data['is_holiday'] = input_data['purchase_month'].isin([12, 1]).astype(int)
        input_data['is_peak_hour'] = input_data['purchase_hour'].isin([9, 10, 11, 12, 13, 14, 15, 16, 17]).astype(int)
        input_data['is_morning'] = input_data['purchase_hour'].isin([6, 7, 8, 9, 10, 11]).astype(int)
        input_data['is_afternoon'] = input_data['purchase_hour'].isin([12, 13, 14, 15, 16, 17]).astype(int)
        input_data['is_evening'] = input_data['purchase_hour'].isin([18, 19, 20, 21, 22, 23]).astype(int)
        
        # Add product features
        input_data['product_volume'] = input_data['product_length_cm'] * input_data['product_height_cm'] * input_data['product_width_cm']
        input_data['product_volume'] = input_data['product_volume'].replace(0, 1)
        input_data['price_per_volume'] = input_data['price'] / input_data['product_volume']
        input_data['price_per_weight'] = input_data['price'] / input_data['product_weight_g'].replace(0, 1)
        input_data['product_density'] = input_data['product_weight_g'] / input_data['product_volume']
        
        # Add delivery features
        input_data['order_delivered_carrier_date'] = pd.to_datetime(input_data['order_delivered_carrier_date'])
        input_data['order_delivered_customer_date'] = pd.to_datetime(input_data['order_delivered_customer_date'])
        input_data['processing_time'] = (input_data['order_delivered_carrier_date'] - input_data['order_purchase_timestamp']).dt.days
        input_data['shipping_time'] = (input_data['order_delivered_customer_date'] - input_data['order_delivered_carrier_date']).dt.days
        input_data['total_delivery_time'] = (input_data['order_delivered_customer_date'] - input_data['order_purchase_timestamp']).dt.days
        
        # Add categorical features with custom bins to handle duplicates
        price_bins = [0, 50, 100, 200, 500, float('inf')]
        price_labels = ['Very Low', 'Low', 'Medium', 'High', 'Very High']
        input_data['price_category'] = pd.cut(input_data['price'], bins=price_bins, labels=price_labels, include_lowest=True)
        
        volume_bins = [0, 1000, 5000, 10000, float('inf')]
        volume_labels = ['Small', 'Medium', 'Large', 'Very Large']
        input_data['product_size_category'] = pd.cut(input_data['product_volume'], bins=volume_bins, labels=volume_labels, include_lowest=True)
        
        # Encode categorical features
        input_data['price_category'] = input_data['price_category'].astype('category').cat.codes
        input_data['product_size_category'] = input_data['product_size_category'].astype('category').cat.codes
        
        # Add interaction features
        input_data['volume_weight_interaction'] = input_data['product_volume'] * input_data['product_weight_g']
        input_data['price_volume_interaction'] = input_data['price'] * input_data['product_volume']
        
        # Select features for prediction
        feature_columns = [
            'price', 'freight_value', 'product_weight_g', 'product_volume',
            'purchase_year', 'purchase_month', 'purchase_day', 'purchase_weekday',
            'purchase_hour', 'purchase_quarter', 'is_weekend', 'is_holiday',
            'is_peak_hour', 'is_morning', 'is_afternoon', 'is_evening',
            'price_per_volume', 'price_per_weight', 'product_density',
            'processing_time', 'shipping_time', 'total_delivery_time',
            'volume_weight_interaction', 'price_volume_interaction',
            'price_category', 'product_size_category'
        ]
        
        # Make prediction
        prediction = model.predict(input_data[feature_columns])
        
        return jsonify({
            'prediction': float(prediction[0]),
            'status': 'success'
        })
        
    except Exception as e:
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 400

if __name__ == '__main__':
    app.run(debug=True, port=5000) 