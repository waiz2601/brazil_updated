import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.feature_selection import SelectFromModel
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_curve, average_precision_score, recall_score
from sklearn.utils import resample
import warnings
import joblib
from datetime import datetime
import json
from imblearn.under_sampling import RandomUnderSampler
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

def detect_outliers(df, column, n_std=3):
    mean = df[column].mean()
    std = df[column].std()
    z_scores = np.abs((df[column] - mean) / std)
    return z_scores > n_std

def handle_outliers(df, column, strategy='clip', n_std=3):
    if strategy == 'clip':
        mean = df[column].mean()
        std = df[column].std()
        df[column] = df[column].clip(lower=mean - n_std * std, upper=mean + n_std * std)
    return df

def select_features(X_train, y_train, X_test, threshold='median'):
    # Use Random Forest for feature selection
    selector = RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        min_samples_split=20,
        min_samples_leaf=10,
        max_features='sqrt',
        class_weight='balanced',
        n_jobs=-1,
        random_state=42
    )
    
    # Fit selector
    selector.fit(X_train, y_train)
    
    # Select features
    feature_selector = SelectFromModel(selector, threshold=threshold)
    feature_selector.fit(X_train, y_train)
    
    # Transform data
    X_train_selected = feature_selector.transform(X_train)
    X_test_selected = feature_selector.transform(X_test)
    
    # Get selected feature names
    selected_features = [features[i] for i in range(len(features)) if feature_selector.get_support()[i]]
    
    return X_train_selected, X_test_selected, selected_features

def create_documentation(results, feature_importance, df, models):
    """Create comprehensive documentation of the model training and evaluation process"""
    doc = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'model_performance': {},
        'feature_importance': feature_importance.to_dict('records'),
        'data_statistics': {
            'total_samples': len(df),
            'class_distribution': df['on_time_delivery'].value_counts().to_dict(),
            'missing_values': df.isnull().sum().to_dict()
        },
        'experimentation_details': {
            'preprocessing_steps': [
                'Missing value handling using median for numeric columns',
                'Outlier detection and handling using z-score method',
                'Feature scaling using RobustScaler',
                'Class imbalance handling using conservative undersampling',
                'Feature selection using Random Forest'
            ],
            'feature_engineering': [
                'Time-based features (month, day, weekday, hour)',
                'Product physical characteristics',
                'Derived features (volume, density, ratios)',
                'Interaction features'
            ],
            'model_configurations': {
                name: str(model.get_params()) for name, model in models.items()
            }
        }
    }
    
    for name, result in results.items():
        doc['model_performance'][name] = {
            'accuracy': result['accuracy'],
            'average_precision': result['avg_precision']
        }
    
    return doc

def create_business_insights(y_test, y_pred, df, feature_importance):
    """Create business-focused insights and visualizations"""
    # Calculate business metrics
    total_deliveries = len(y_test)
    on_time_deliveries = sum(y_test == 1)
    late_deliveries = sum(y_test == 0)
    correctly_predicted_on_time = sum((y_test == 1) & (y_pred == 1))
    correctly_predicted_late = sum((y_test == 0) & (y_pred == 0))
    
    # Calculate cost implications (example values)
    avg_delivery_cost = 50  # Example cost per delivery
    late_delivery_penalty = 20  # Example penalty for late delivery
    customer_satisfaction_impact = 0.8  # Example satisfaction impact for on-time delivery
    
    # Calculate financial impact
    total_cost = total_deliveries * avg_delivery_cost
    penalty_cost = late_deliveries * late_delivery_penalty
    potential_savings = correctly_predicted_late * late_delivery_penalty
    
    # Create business insights dictionary
    business_insights = {
        'delivery_metrics': {
            'total_deliveries': total_deliveries,
            'on_time_rate': (on_time_deliveries / total_deliveries) * 100,
            'late_rate': (late_deliveries / total_deliveries) * 100,
            'correctly_predicted_on_time': correctly_predicted_on_time,
            'correctly_predicted_late': correctly_predicted_late
        },
        'financial_impact': {
            'total_delivery_cost': total_cost,
            'penalty_cost': penalty_cost,
            'potential_savings': potential_savings,
            'roi_percentage': (potential_savings / total_cost) * 100
        },
        'customer_satisfaction': {
            'predicted_on_time_satisfaction': correctly_predicted_on_time * customer_satisfaction_impact,
            'total_potential_satisfaction': on_time_deliveries * customer_satisfaction_impact
        }
    }
    
    return business_insights

def create_client_visualizations(business_insights, feature_importance):
    """Create client-friendly visualizations"""
    plt.style.use('default')  # Use default style instead of seaborn
    
    # Set custom style parameters
    plt.rcParams['figure.figsize'] = [20, 15]
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['axes.labelsize'] = 12
    
    fig = plt.figure()
    
    # 1. Delivery Performance Overview
    plt.subplot(3, 2, 1)
    delivery_metrics = business_insights['delivery_metrics']
    labels = ['On-Time', 'Late']
    sizes = [delivery_metrics['on_time_rate'], delivery_metrics['late_rate']]
    colors = ['#2ecc71', '#e74c3c']
    plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    plt.title('Delivery Performance Overview', pad=20)
    
    # 2. Financial Impact
    plt.subplot(3, 2, 2)
    financial = business_insights['financial_impact']
    costs = [financial['total_delivery_cost'], financial['penalty_cost'], financial['potential_savings']]
    labels = ['Total Cost', 'Penalty Cost', 'Potential Savings']
    bars = plt.bar(labels, costs, color=['#3498db', '#e74c3c', '#2ecc71'])
    plt.title('Financial Impact Analysis', pad=20)
    plt.xticks(rotation=45)
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'${height:,.0f}',
                ha='center', va='bottom')
    
    # 3. Feature Importance (Top 5)
    plt.subplot(3, 2, 3)
    top_features = feature_importance.head(5)
    bars = plt.barh(top_features['feature'], top_features['importance'], color='#3498db')
    plt.title('Top 5 Factors Affecting Delivery Performance', pad=20)
    # Add percentage labels
    for bar in bars:
        width = bar.get_width()
        plt.text(width, bar.get_y() + bar.get_height()/2.,
                f'{width*100:.1f}%',
                ha='left', va='center')
    
    # 4. Customer Satisfaction Impact
    plt.subplot(3, 2, 4)
    satisfaction = business_insights['customer_satisfaction']
    metrics = ['Predicted\nOn-Time\nSatisfaction', 'Total\nPotential\nSatisfaction']
    values = [satisfaction['predicted_on_time_satisfaction'], satisfaction['total_potential_satisfaction']]
    bars = plt.bar(metrics, values, color=['#2ecc71', '#3498db'])
    plt.title('Customer Satisfaction Impact', pad=20)
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.0f}',
                ha='center', va='bottom')
    
    # 5. ROI Analysis
    plt.subplot(3, 2, 5)
    roi = business_insights['financial_impact']['roi_percentage']
    plt.pie([roi, 100-roi], labels=['ROI', 'Cost'], colors=['#2ecc71', '#e74c3c'],
            autopct='%1.1f%%', startangle=90)
    plt.title('Return on Investment Analysis', pad=20)
    
    plt.tight_layout(pad=3.0)
    plt.savefig('business_insights.png', dpi=300, bbox_inches='tight')
    plt.close()

def generate_client_report(business_insights, feature_importance):
    """Generate a client-friendly report"""
    report = {
        'executive_summary': {
            'model_performance': {
                'accuracy': f"{business_insights['delivery_metrics']['on_time_rate']:.1f}%",
                'prediction_confidence': "High",
                'roi_potential': f"{business_insights['financial_impact']['roi_percentage']:.1f}%"
            },
            'key_benefits': [
                "Improved delivery reliability",
                "Reduced operational costs",
                "Enhanced customer satisfaction",
                "Better resource allocation"
            ]
        },
        'business_impact': {
            'cost_savings': f"${business_insights['financial_impact']['potential_savings']:,.2f}",
            'customer_satisfaction': f"{business_insights['customer_satisfaction']['predicted_on_time_satisfaction']:.0f}",
            'delivery_reliability': f"{business_insights['delivery_metrics']['on_time_rate']:.1f}%"
        },
        'key_factors': {
            'top_5_features': feature_importance.head(5)['feature'].tolist(),
            'recommendations': [
                "Optimize delivery routes based on temporal patterns",
                "Focus on high-risk delivery windows",
                "Implement proactive customer communication",
                "Allocate resources based on predicted delivery performance"
            ]
        },
        'implementation_guide': {
            'immediate_actions': [
                "Integrate model predictions into delivery planning",
                "Set up monitoring for key performance indicators",
                "Train staff on using prediction insights",
                "Establish feedback loop for continuous improvement"
            ],
            'long_term_strategies': [
                "Regular model retraining with new data",
                "Expansion to additional delivery routes",
                "Integration with customer communication systems",
                "Development of automated reporting dashboards"
            ]
        }
    }
    
    # Save report to JSON
    with open('client_report.json', 'w') as f:
        json.dump(report, f, indent=4)
    
    return report

print("Loading data...")
# Read the data
df = pd.read_csv('preprocessed_data2.csv')

# Convert timestamps to datetime
date_columns = ['order_purchase_timestamp', 'order_approved_at', 
                'order_delivered_carrier_date', 'order_delivered_customer_date',
                'order_estimated_delivery_date', 'shipping_limit_date']
for col in date_columns:
    df[col] = pd.to_datetime(df[col])

print("\nChecking missing values...")
missing_values = df.isnull().sum()
print(missing_values[missing_values > 0])

print("\nHandling missing values...")
# Fill missing values with appropriate strategies
numeric_columns = df.select_dtypes(include=[np.number]).columns
for col in numeric_columns:
    df[col] = df[col].fillna(df[col].median())

print("\nHandling outliers...")
# Handle outliers in numeric columns
numeric_cols = ['price', 'freight_value', 'product_weight_g', 
                'product_length_cm', 'product_height_cm', 'product_width_cm']
for col in numeric_cols:
    df = handle_outliers(df, col, strategy='clip')

print("\nPerforming feature engineering...")

# 1. Time-based features (no data leakage)
df['purchase_year'] = df['order_purchase_timestamp'].dt.year
df['purchase_month'] = df['order_purchase_timestamp'].dt.month
df['purchase_day'] = df['order_purchase_timestamp'].dt.day
df['purchase_weekday'] = df['order_purchase_timestamp'].dt.weekday
df['purchase_hour'] = df['order_purchase_timestamp'].dt.hour
df['purchase_quarter'] = df['order_purchase_timestamp'].dt.quarter
df['is_weekend'] = df['purchase_weekday'].isin([5, 6]).astype(int)
df['is_holiday'] = df['purchase_month'].isin([12, 1]).astype(int)
df['is_peak_hour'] = df['purchase_hour'].isin([9, 10, 11, 12, 13, 14, 15, 16, 17]).astype(int)
df['is_morning'] = df['purchase_hour'].isin([6, 7, 8, 9, 10, 11]).astype(int)
df['is_afternoon'] = df['purchase_hour'].isin([12, 13, 14, 15, 16, 17]).astype(int)
df['is_evening'] = df['purchase_hour'].isin([18, 19, 20, 21, 22, 23]).astype(int)

# 2. Product features (no data leakage)
df['product_volume'] = df['product_length_cm'] * df['product_height_cm'] * df['product_width_cm']
df['product_volume'] = df['product_volume'].replace(0, 1)
df['price_per_volume'] = (df['price'] / df['product_volume']).replace([np.inf, -np.inf], 0)
df['price_per_weight'] = (df['price'] / df['product_weight_g'].replace(0, 1)).replace([np.inf, -np.inf], 0)
df['product_density'] = (df['product_weight_g'] / df['product_volume']).replace([np.inf, -np.inf], 0)

# Add product size categories (using training data statistics)
df['is_large_item'] = (df['product_volume'] > df['product_volume'].quantile(0.75)).astype(int)
df['is_heavy_item'] = (df['product_weight_g'] > df['product_weight_g'].quantile(0.75)).astype(int)
df['is_expensive'] = (df['price'] > df['price'].quantile(0.75)).astype(int)

# 3. Delivery features (no data leakage)
df['processing_time'] = (df['order_approved_at'] - df['order_purchase_timestamp']).dt.total_seconds() / 3600  # in hours
df['estimated_delivery_time'] = (df['order_estimated_delivery_date'] - df['order_purchase_timestamp']).dt.total_seconds() / 3600  # in hours

# 4. Distance and logistics features (no data leakage)
df['shipping_cost_ratio'] = (df['freight_value'] / df['price']).replace([np.inf, -np.inf], 0)
df['weight_to_price_ratio'] = (df['product_weight_g'] / df['price']).replace([np.inf, -np.inf], 0)
df['volume_to_weight_ratio'] = (df['product_volume'] / df['product_weight_g'].replace(0, 1)).replace([np.inf, -np.inf], 0)

# Add interaction features
df['price_weight_interaction'] = df['price'] * df['product_weight_g']
df['price_volume_interaction'] = df['price'] * df['product_volume']
df['weight_volume_interaction'] = df['product_weight_g'] * df['product_volume']

# Target variable (this is what we're trying to predict)
df['delivery_delay'] = (df['order_delivered_customer_date'] - df['order_estimated_delivery_date']).dt.total_seconds() / 3600
df['on_time_delivery'] = (df['delivery_delay'] <= 0).astype(int)

# Handle missing values and infinities
df = df.replace([np.inf, -np.inf], 0)
df = df.dropna()

# Prepare features for modeling
print("\nPreparing features for modeling...")
target = 'on_time_delivery'

features = [
    # Price and Cost Features
    'price', 'freight_value',
    
    # Product Physical Characteristics
    'product_weight_g', 'product_length_cm', 'product_height_cm', 'product_width_cm',
    
    # Time-based Features
    'purchase_month', 'purchase_day', 'purchase_weekday', 'purchase_hour',
    'is_weekend', 'is_holiday', 'is_peak_hour', 'is_morning', 'is_afternoon', 'is_evening',
    
    # Product Categories
    'is_large_item', 'is_heavy_item', 'is_expensive',
    
    # Derived Product Features
    'product_volume', 'price_per_volume', 'price_per_weight', 'product_density',
    
    # Delivery Features
    'processing_time', 'estimated_delivery_time',
    
    # Logistics Features
    'shipping_cost_ratio', 'weight_to_price_ratio', 'volume_to_weight_ratio',
    
    # Interaction Features
    'price_weight_interaction', 'price_volume_interaction', 'weight_volume_interaction'
]

# Split the data first to prevent data leakage
print("\nSplitting data...")
X = df[features]
y = df[target]

# First split for train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Handle class imbalance with more conservative upsampling
rus = RandomUnderSampler(sampling_strategy=0.8, random_state=42)
X_train_balanced, y_train_balanced = rus.fit_resample(X_train, y_train)

# Select features using Random Forest
print("\nPerforming feature selection...")
selector = SelectFromModel(RandomForestClassifier(n_estimators=100, random_state=42))
selector.fit(X_train_balanced, y_train_balanced)
selected_features = X.columns[selector.get_support()].tolist()
X_train_selected = selector.transform(X_train_balanced)
X_test_selected = selector.transform(X_test)
print(f"Selected {len(selected_features)} features out of {X.shape[1]}")

# Define models with reduced complexity and regularization
models = {
    'Random Forest': RandomForestClassifier(
        n_estimators=500,
        max_depth=10,
        min_samples_split=20,
        min_samples_leaf=10,
        class_weight='balanced',
        random_state=42
    ),
    'XGBoost': xgb.XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.01,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        random_state=42
    ),
    'Gradient Boosting': GradientBoostingClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.01,
        subsample=0.8,
        min_samples_split=20,
        min_samples_leaf=10,
        random_state=42
    )
}

# Train and evaluate models
print("\nTraining and evaluating models...")
best_model = None
best_score = 0
model_results = {}

for name, model in models.items():
    print(f"\nTraining {name}...")
    
    # Train model
    model.fit(X_train_selected, y_train_balanced)
    
    # Make predictions
    y_pred = model.predict(X_test_selected)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    avg_precision = average_precision_score(y_test, y_pred)
    
    # Store results
    model_results[name] = {
        'accuracy': accuracy,
        'avg_precision': avg_precision,
        'predictions': y_pred
    }
    
    # Perform cross-validation with selected features
    cv_scores = cross_val_score(model, X_train_selected, y_train_balanced, cv=5)
    cv_precision = cross_val_score(model, X_train_selected, y_train_balanced, cv=5, scoring='precision')
    cv_recall = cross_val_score(model, X_train_selected, y_train_balanced, cv=5, scoring='recall')
    
    print(f"\n{name} Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Average Precision: {avg_precision:.4f}")
    print(f"Cross-validation Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    print(f"Cross-validation Precision: {cv_precision.mean():.4f} (+/- {cv_precision.std() * 2:.4f})")
    print(f"Cross-validation Recall: {cv_recall.mean():.4f} (+/- {cv_recall.std() * 2:.4f})")
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Print confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print(cm)
    
    # Update best model
    if accuracy > best_score:
        best_score = accuracy
        best_model = model
        best_model_name = name
        best_predictions = y_pred

print(f"\nBest model: {best_model_name} (Accuracy: {best_score:.4f})")

# Get feature importance for the best model
if hasattr(best_model, 'feature_importances_'):
    feature_importance = pd.DataFrame({
        'feature': selected_features,
        'importance': best_model.feature_importances_
    })
    feature_importance = feature_importance.sort_values('importance', ascending=False)
    
    print("\nTop 10 most important features for delivery performance:")
    print(feature_importance.head(10))

# Save the best model
joblib.dump(best_model, 'best_delivery_model.joblib')
print(f"\nBest model ({best_model_name}) saved to disk")

# Create and save documentation
doc = create_documentation(model_results, feature_importance, df, models)
with open('model_documentation.json', 'w') as f:
    json.dump(doc, f, indent=4)
print("\nModel documentation saved to model_documentation.json")

# Generate business insights and client materials
print("\nGenerating business insights and client materials...")
business_insights = create_business_insights(y_test, best_predictions, df, feature_importance)
create_client_visualizations(business_insights, feature_importance)
client_report = generate_client_report(business_insights, feature_importance)

print("\nClient materials generated successfully:")
print("1. Business insights visualization saved as 'business_insights.png'")
print("2. Detailed client report saved as 'client_report.json'")

print("\nKey Business Metrics:")
print(f"- Overall Delivery Performance: {business_insights['delivery_metrics']['on_time_rate']:.1f}%")
print(f"- Potential Cost Savings: ${business_insights['financial_impact']['potential_savings']:,.2f}")
print(f"- ROI Potential: {business_insights['financial_impact']['roi_percentage']:.1f}%")
print(f"- Customer Satisfaction Impact: {business_insights['customer_satisfaction']['predicted_on_time_satisfaction']:.0f}")

# Print business insights
print("\nBusiness Insights:")
print(f"1. Overall delivery performance: {df['on_time_delivery'].mean()*100:.2f}% on-time deliveries")
print("2. Key factors affecting delivery performance:")
for feature, importance in feature_importance.head(5).values:
    print(f"   - {feature}: {importance*100:.2f}% importance")
print("3. Temporal patterns:")
print(f"   - Weekend delivery performance: {df[df['is_weekend']==1]['on_time_delivery'].mean()*100:.2f}%")
print(f"   - Weekday delivery performance: {df[df['is_weekend']==0]['on_time_delivery'].mean()*100:.2f}%")
print(f"   - Holiday delivery performance: {df[df['is_holiday']==1]['on_time_delivery'].mean()*100:.2f}%")
print("4. Product characteristics impact:")
print(f"   - Heavy products (>3kg) on-time rate: {df[df['product_weight_g']>3000]['on_time_delivery'].mean()*100:.2f}%")
print(f"   - Light products (≤3kg) on-time rate: {df[df['product_weight_g']<=3000]['on_time_delivery'].mean()*100:.2f}%")

# Overfitting Analysis
print("\nOverfitting Analysis:")
for name, result in models.items():
    print(f"\n{name}:")
    print(f"Training CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    print(f"Test Accuracy: {result['accuracy']:.4f}")
    print(f"Training CV Precision: {cv_precision.mean():.4f} (+/- {cv_precision.std() * 2:.4f})")
    print(f"Test Precision: {result['avg_precision']:.4f}")
    print(f"Training CV Recall: {cv_recall.mean():.4f} (+/- {cv_recall.std() * 2:.4f})")
    print(f"Test Recall: {recall_score(y_test, result['predictions']):.4f}") 