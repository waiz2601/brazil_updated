from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np
from datetime import datetime

app = Flask(__name__)

# Load the trained model
try:
    model = joblib.load('random_forest_model.joblib')
except Exception as e:
    print(f"Error loading model: {str(e)}")
    model = None

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        'status': 'success',
        'message': 'Delivery Delay Prediction API',
        'endpoints': {
            'predict': '/predict (POST)',
            'health': '/health (GET)'
        },
        'example_request': {
            'freight_value': 10.0,
            'purchase_year': 2023,
            'purchase_month': 7,
            'purchase_day': 15,
            'purchase_quarter': 3,
            'price_per_weight': 10.0
        }
    })

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({
            'error': 'Model not loaded. Please check the server logs.'
        }), 500

    try:
        # Get data from request
        data = request.get_json()
        
        if not data:
            return jsonify({
                'error': 'No data provided. Please send a JSON object with the required fields.'
            }), 400

        # Validate required fields
        required_fields = ['freight_value', 'purchase_year', 'purchase_month', 
                         'purchase_day', 'purchase_quarter', 'price_per_weight']
        missing_fields = [field for field in required_fields if field not in data]
        
        if missing_fields:
            return jsonify({
                'error': f'Missing required fields: {", ".join(missing_fields)}'
            }), 400
        
        # Create a DataFrame with the input data
        input_data = pd.DataFrame({
            'freight_value': [float(data['freight_value'])],
            'purchase_year': [int(data['purchase_year'])],
            'purchase_month': [int(data['purchase_month'])],
            'purchase_day': [int(data['purchase_day'])],
            'purchase_quarter': [int(data['purchase_quarter'])],
            'price_per_weight': [float(data['price_per_weight'])]
        })
        
        # Make prediction
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0]
        
        return jsonify({
            'prediction': int(prediction),
            'probability': float(probability[1]),
            'message': 'On-time delivery' if prediction == 1 else 'Delayed delivery'
        })
    
    except Exception as e:
        return jsonify({
            'error': str(e)
        }), 400

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'timestamp': datetime.now().isoformat()
    })

@app.errorhandler(404)
def not_found(error):
    return jsonify({
        'error': 'Not Found',
        'message': 'The requested URL was not found on the server.'
    }), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        'error': 'Internal Server Error',
        'message': 'An unexpected error has occurred.'
    }), 500

if __name__ == '__main__':
    app.run(debug=True) 