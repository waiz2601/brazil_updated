# Brazilian E-Commerce Sales Prediction

This project implements a machine learning model to predict product prices in the Olist marketplace using historical sales data. The application includes a Flask API backend and a Streamlit frontend for easy interaction.

## Features

- Sales prediction using multiple ML models (XGBoost, Random Forest, Gradient Boosting, AdaBoost)
- Feature engineering for time-based, product, and delivery metrics
- Interactive web interface using Streamlit
- RESTful API using Flask
- Comprehensive model evaluation and visualization

## Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

## Installation

1. Clone the repository:

```bash
git clone https://github.com/waiz2601/Brazillian_E-Commerce_2.git
cd Brazillian_E-Commerce_2
```

2. Install required packages:

```bash
pip install -r requirements.txt
```

## Project Structure

```
Brazillian_E-Commerce_2/
├── app.py                 # Flask API server
├── streamlit_app.py       # Streamlit frontend
├── sales_prediction.py    # Model training script
├── requirements.txt       # Project dependencies
├── best_model.joblib     # Trained model file
└── preprocessed_data2.csv # Preprocessed dataset
```

## Usage

1. Train the model:

```bash
python sales_prediction.py
```

This will:

- Load and preprocess the data
- Train multiple ML models
- Generate performance metrics and visualizations
- Save the best model as 'best_model.joblib'

2. Start the Flask API server:

```bash
python app.py
```

The API will be available at http://localhost:5000

3. Launch the Streamlit interface:

```bash
streamlit run streamlit_app.py
```

The web interface will be available at http://localhost:8501

## API Endpoints

- `GET /`: Health check endpoint
- `POST /predict`: Sales prediction endpoint
  ```json
  {
    "price": float,
    "freight_value": float,
    "product_weight_g": float,
    "product_length_cm": float,
    "product_height_cm": float,
    "product_width_cm": float,
    "order_purchase_timestamp": "YYYY-MM-DD HH:MM:SS",
    "order_delivered_carrier_date": "YYYY-MM-DD HH:MM:SS",
    "order_delivered_customer_date": "YYYY-MM-DD HH:MM:SS"
  }
  ```

## Model Performance

The best model (XGBoost) achieves:

- R² Score: 0.9624 (96.24% accuracy)
- RMSE: 44.11
- MAE: 14.21

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
