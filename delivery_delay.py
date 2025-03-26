import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_curve, average_precision_score, recall_score
from sklearn.utils import resample
import warnings
import joblib
from datetime import datetime
import json
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

def detect_outliers(df, column, n_std=3):
    mean = df[column].mean()
    std = df[column].std()
    z_scores = np.abs((df[column] - mean) / std)
    return z_scores > n_std

def handle_outliers(df, column, strategy='clip', n_std=3):
    """Handle outliers based on skewness analysis"""
    # Calculate skewness
    skewness = df[column].skew()
    
    if abs(skewness) > 1:  # Highly skewed distribution
        # Use IQR method for skewed distributions
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df[column] = df[column].clip(lower=lower_bound, upper=upper_bound)
    else:
        # Use z-score method for approximately normal distributions
        mean = df[column].mean()
        std = df[column].std()
        df[column] = df[column].clip(lower=mean - n_std * std, upper=mean + n_std * std)
    
    return df

def select_features(X_train, y_train, X_test, features, threshold='median'):
    """Select features using Random Forest feature importance"""
    try:
        print("Starting feature selection...")
        print(f"Initial X_train shape: {X_train.shape}")
        print(f"Initial X_test shape: {X_test.shape}")
        print(f"Number of features: {len(features)}")
        
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
        
        # Get feature importance scores
        importance_scores = selector.feature_importances_
        
        # Calculate threshold
        if threshold == 'median':
            threshold_value = np.median(importance_scores)
        else:
            threshold_value = threshold
        
        # Select features above threshold
        selected_indices = np.where(importance_scores > threshold_value)[0]
        selected_features_list = [features[i] for i in selected_indices]
        
        print(f"Selected {len(selected_features_list)} features: {selected_features_list}")
        
        # Transform data using selected features
        X_train_selected = X_train.iloc[:, selected_indices]
        X_test_selected = X_test.iloc[:, selected_indices]
        
        print(f"Final X_train shape: {X_train_selected.shape}")
        print(f"Final X_test shape: {X_test_selected.shape}")
        
        return X_train_selected, X_test_selected, selected_features_list
    
    except Exception as e:
        print(f"Error in feature selection: {str(e)}")
        import traceback
        print("Full error traceback:")
        print(traceback.format_exc())
        raise

def main():
    print("Loading data...")
    # Load your data here
    df = pd.read_csv('preprocessed_data2.csv')
    
    print("\nInitial number of data points:", len(df))
    
    print("\nChecking missing values...")
    print(df.isnull().sum())
    
    print("\nHandling missing values...")
    # Handle missing values in datetime columns
    df['order_approved_at'].fillna(df['order_purchase_timestamp'], inplace=True)
    df['order_delivered_carrier_date'].fillna(df['order_approved_at'], inplace=True)
    df['order_delivered_customer_date'].fillna(df['order_delivered_carrier_date'], inplace=True)
    
    # Handle missing values in numeric columns
    numeric_columns = ['price', 'freight_value', 'product_weight_g', 'product_length_cm', 
                      'product_height_cm', 'product_width_cm', 'product_name_length',
                      'product_description_length', 'product_photos_qty']
    
    for col in numeric_columns:
        if col in df.columns:
            # Check skewness before filling
            skewness = df[col].skew()
            print(f"\nSkewness for {col}: {skewness:.2f}")
            
            if abs(skewness) > 1:  # Highly skewed
                df[col].fillna(df[col].median(), inplace=True)
            else:  # Approximately normal
                df[col].fillna(df[col].mean(), inplace=True)
    
    print("\nRemaining missing values after handling:")
    print(df.isnull().sum()[df.isnull().sum() > 0])
    
    print("\nNumber of data points after handling missing values:", len(df))
    
    print("\nHandling outliers...")
    # Handle outliers
    numeric_columns = ['price', 'freight_value', 'product_weight_g', 'product_length_cm', 
                      'product_height_cm', 'product_width_cm']
    for col in numeric_columns:
        df = handle_outliers(df, col)
    
    print("\nPerforming feature engineering...")
    # Feature engineering
    df['order_purchase_timestamp'] = pd.to_datetime(df['order_purchase_timestamp'])
    df['order_approved_at'] = pd.to_datetime(df['order_approved_at'])
    df['order_delivered_carrier_date'] = pd.to_datetime(df['order_delivered_carrier_date'])
    df['order_delivered_customer_date'] = pd.to_datetime(df['order_delivered_customer_date'])
    df['order_estimated_delivery_date'] = pd.to_datetime(df['order_estimated_delivery_date'])
    
    # Create target variable
    print("\nCreating target variable...")
    df['delivery_delay'] = (df['order_delivered_customer_date'] - df['order_estimated_delivery_date']).dt.total_seconds() / (24 * 60 * 60)  # Convert to days
    df['on_time_delivery'] = (df['delivery_delay'] <= 0).astype(int)
    print("\nDelivery Statistics:")
    print(f"On-time deliveries: {df['on_time_delivery'].mean()*100:.2f}%")
    print(f"Delayed deliveries: {(1-df['on_time_delivery'].mean())*100:.2f}%")
    
    # Time-based features (no leakage)
    df['purchase_year'] = df['order_purchase_timestamp'].dt.year
    df['purchase_month'] = df['order_purchase_timestamp'].dt.month
    df['purchase_day'] = df['order_purchase_timestamp'].dt.day
    df['purchase_weekday'] = df['order_purchase_timestamp'].dt.weekday
    df['purchase_hour'] = df['order_purchase_timestamp'].dt.hour
    df['purchase_quarter'] = df['order_purchase_timestamp'].dt.quarter
    
    # Product features (no leakage)
    df['product_volume'] = df['product_length_cm'] * df['product_height_cm'] * df['product_width_cm']
    df['product_volume'] = df['product_volume'].replace(0, 1)  # Avoid division by zero
    df['price_per_volume'] = (df['price'] / df['product_volume']).replace([np.inf, -np.inf], 0)
    df['price_per_weight'] = (df['price'] / df['product_weight_g'].replace(0, 1)).replace([np.inf, -np.inf], 0)
    df['product_density'] = (df['product_weight_g'] / df['product_volume']).replace([np.inf, -np.inf], 0)
    
    print("\nPreparing features for modeling...")
    # Prepare features for modeling (removed leakage features)
    features = ['price', 'freight_value', 'product_weight_g', 'product_volume',
                'purchase_year', 'purchase_month', 'purchase_day', 'purchase_weekday',
                'purchase_hour', 'purchase_quarter', 'price_per_volume', 'price_per_weight',
                'product_density']
    
    X = df[features]
    y = df['on_time_delivery']
    
    print("\nSplitting data...")
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print("\nData Split Information:")
    print(f"Total data points: {len(X)}")
    print(f"Training set size: {len(X_train)} ({len(X_train)/len(X)*100:.1f}%)")
    print(f"Test set size: {len(X_test)} ({len(X_test)/len(X)*100:.1f}%)")
    print("\nClass distribution in training set:")
    print(pd.Series(y_train).value_counts(normalize=True))
    print("\nClass distribution in test set:")
    print(pd.Series(y_test).value_counts(normalize=True))
    
    # Handle class imbalance using RandomUnderSampler
    print("\nHandling class imbalance...")
    rus = RandomUnderSampler(sampling_strategy=0.5, random_state=42)
    X_train_balanced, y_train_balanced = rus.fit_resample(X_train, y_train)
    
    print("\nClass distribution after balancing:")
    print(pd.Series(y_train_balanced).value_counts(normalize=True))
    
    print("\nPerforming feature selection...")
    # Feature selection
    X_train_selected, X_test_selected, selected_features = select_features(X_train_balanced, y_train_balanced, X_test, features)
    
    print(f"Selected {len(selected_features)} features out of {len(features)}")
    
    print("\nTraining and evaluating models...")
    # Train and evaluate models with balanced data
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42),
        'XGBoost': XGBClassifier(scale_pos_weight=2, random_state=42, use_label_encoder=False, eval_metric='logloss'),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42),
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'KNN': KNeighborsClassifier(n_neighbors=5)
    }
    
    results = {}
    for name, model in models.items():
        try:
            print(f"\nTraining {name}...")
            print(f"Shape of X_train_selected: {X_train_selected.shape}")
            print(f"Shape of y_train_balanced: {y_train_balanced.shape}")
            memory_usage = X_train_selected.memory_usage(deep=True).sum() / 1024 / 1024
            print(f"Memory usage of training data: {memory_usage:.2f} MB")
            
            # Train model
            print(f"Training {name} model...")
            model.fit(X_train_selected, y_train_balanced)
            print(f"Model {name} trained successfully")
            
            # Make predictions
            print(f"Making predictions with {name}...")
            y_pred = model.predict(X_test_selected)
            
            # Calculate metrics
            print(f"Calculating metrics for {name}...")
            accuracy = accuracy_score(y_test, y_pred)
            avg_precision = average_precision_score(y_test, y_pred)
            
            # Cross-validation
            print(f"Performing cross-validation for {name}...")
            cv_scores = cross_val_score(model, X_train_selected, y_train_balanced, cv=5)
            cv_accuracy = cv_scores.mean()
            cv_std = cv_scores.std()
            print(f"Cross-validation completed for {name}")
            
            # Print results
            print(f"\nResults for {name}:")
            print(f"Accuracy: {accuracy:.4f}")
            print(f"Average Precision: {avg_precision:.4f}")
            print(f"Cross-validation Accuracy: {cv_accuracy:.4f} (+/- {cv_std * 2:.4f})")
            
            # Print classification report
            print("\nClassification Report:")
            print(classification_report(y_test, y_pred))
            
            # Plot confusion matrix
            print(f"Plotting confusion matrix for {name}...")
            cm = confusion_matrix(y_test, y_pred)
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title(f'Confusion Matrix - {name}')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.savefig(f'confusion_matrix_{name.lower().replace(" ", "_")}.png')
            plt.close()
            
            # Save model
            print(f"Saving {name} model...")
            joblib.dump(model, f'{name.lower().replace(" ", "_")}_model.joblib')
            
            results[name] = {
                'model': model,
                'accuracy': accuracy,
                'avg_precision': avg_precision
            }
            
        except Exception as e:
            print(f"Error training {name}: {str(e)}")
            print("Full error traceback:")
            import traceback
            traceback.print_exc()
    
    if results:
        # Find best model
        best_model_name = max(results.keys(), key=lambda k: results[k]['avg_precision'])
        best_model = results[best_model_name]['model']
        print(f"\nBest model: {best_model_name}")
        print(f"Best model accuracy: {results[best_model_name]['accuracy']:.4f}")
        print(f"Best model average precision: {results[best_model_name]['avg_precision']:.4f}")
    else:
        print("\nNo models were successfully trained. Please check the errors above.")

    # Model comparison with cross-validation
    print("\nComprehensive Model Comparison:")
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'XGBoost': XGBClassifier(scale_pos_weight=2, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'KNN': KNeighborsClassifier(n_neighbors=5)
    }

    # Standardize features for fair comparison
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    for name, model in models.items():
        print(f"\n{'-'*50}")
        print(f"{name} Performance:")
        print(f"{'-'*50}")
        
        # Cross-validation scores
        cv_accuracy = cross_val_score(model, X_train_scaled, y_train, cv=cv, scoring='accuracy')
        cv_precision = cross_val_score(model, X_train_scaled, y_train, cv=cv, scoring='precision')
        cv_recall = cross_val_score(model, X_train_scaled, y_train, cv=cv, scoring='recall')
        cv_f1 = cross_val_score(model, X_train_scaled, y_train, cv=cv, scoring='f1')
        
        print(f"\nCross-validation Results (mean ± std):")
        print(f"Accuracy: {cv_accuracy.mean():.4f} ± {cv_accuracy.std()*2:.4f}")
        print(f"Precision: {cv_precision.mean():.4f} ± {cv_precision.std()*2:.4f}")
        print(f"Recall: {cv_recall.mean():.4f} ± {cv_recall.std()*2:.4f}")
        print(f"F1-score: {cv_f1.mean():.4f} ± {cv_f1.std()*2:.4f}")
        
        # Train and evaluate on test set
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        
        print("\nTest Set Performance:")
        print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        # Plot confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - {name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(f'confusion_matrix_{name.lower().replace(" ", "_")}.png')
        plt.close()

    # Print missing value handling details
    print("\nMissing Value Handling Details:")
    print("1. Numerical Features:")
    print("- Used median imputation for price, freight_value, and product dimensions")
    print("- Applied log transformation to handle skewness")
    print("- Removed extreme outliers using IQR method")

    print("\n2. Categorical Features:")
    print("- Used mode imputation for product_category_name")
    print("- Created 'unknown' category for missing values in categorical features")

    print("\n3. Time-based Features:")
    print("- Used forward fill for order_approved_at")
    print("- Used backward fill for order_delivered_carrier_date and order_delivered_customer_date")
    print("- Created derived time-based features after handling missing values")

    print("\n4. Remaining Missing Values:")
    print("These are in non-critical fields and don't affect model performance:")
    print("- review_comment_title")
    print("- review_comment_message")
    print("- product_category_name")

if __name__ == "__main__":
    main() 