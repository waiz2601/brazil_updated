# Delivery Delay Prediction System

A machine learning system that predicts whether a delivery will be on time or delayed based on various features. The system includes both a Flask API and a Streamlit web interface.

## Features

- **Machine Learning Model**: Random Forest classifier with 82.66% accuracy
- **Flask API**: RESTful API for delivery delay predictions
- **Streamlit Interface**: User-friendly web interface for predictions
- **Feature Engineering**: Handles missing values, outliers, and creates derived features
- **Model Evaluation**: Comprehensive metrics including accuracy, precision, recall, and F1-score

## Project Structure

```
delivery_delay/
├── app.py                 # Flask API
├── streamlit_app.py       # Streamlit web interface
├── delivery_delay.py      # Main model training script
├── requirements.txt       # Project dependencies
├── random_forest_model.joblib  # Trained model
└── README.md             # Project documentation
```

## Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/delivery-delay-prediction.git
cd delivery-delay-prediction
```

2. Create a virtual environment (recommended):

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Running the Flask API

```bash
python app.py
```

The API will be available at `http://localhost:5000`

Available endpoints:

- `GET /`: API documentation
- `GET /health`: Health check
- `POST /predict`: Make predictions

Example API request:

```python
import requests

data = {
    'freight_value': 10.0,
    'purchase_year': 2023,
    'purchase_month': 7,
    'purchase_day': 15,
    'purchase_quarter': 3,
    'price_per_weight': 10.0
}

response = requests.post('http://localhost:5000/predict', json=data)
print(response.json())
```

### Running the Streamlit App

```bash
streamlit run streamlit_app.py
```

The app will be available at `http://localhost:8501`

## Model Features

The model uses the following features:

- Freight Value
- Purchase Date (Year, Month, Day, Quarter)
- Price per Weight

## Model Performance

- Accuracy: 82.66%
- Average Precision: 95.21%
- Cross-validation Accuracy: 74.78% (± 1.02%)

## Data Preprocessing

The system includes comprehensive data preprocessing:

1. Missing Value Handling

   - Numerical features: Median/Mean imputation based on skewness
   - Categorical features: Mode imputation
   - Time-based features: Forward/Backward fill

2. Outlier Handling

   - IQR method for skewed distributions
   - Z-score method for normal distributions

3. Feature Engineering
   - Time-based features
   - Product-related features
   - Derived metrics

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Dataset: Brazilian E-commerce Public Dataset
- Libraries: scikit-learn, pandas, numpy, Flask, Streamlit
