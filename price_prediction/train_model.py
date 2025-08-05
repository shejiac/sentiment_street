import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import time

# Import regression models
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

def load_data(path):
    """Load the dataset from CSV"""
    df = pd.read_csv(path)
    return df

def prepare_data(df):
    """Prepare data for training"""
    # Separate features and target (now using future_return directly)
    X = df.drop(columns=['name', 'target', 'timestamp', 'future_return', 'price'], errors='ignore')
    y = df['future_return']  # Using continuous value instead of binary target
    
    # Convert boolean columns to int
    bool_cols = X.select_dtypes(include=['bool']).columns
    X[bool_cols] = X[bool_cols].astype(int)
    
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False, random_state=42
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    return X_train, X_test, y_train, y_test, scaler

def train_models(X_train, y_train):
    """Train multiple regressors and return them in a dictionary"""
    models = {
        'Random Forest': RandomForestRegressor(
            n_estimators=100, max_depth=10, min_samples_split=5,
            random_state=42
        ),
        'XGBoost': XGBRegressor(
            n_estimators=100, max_depth=5, learning_rate=0.1,
            random_state=42
        ),
        'LightGBM': LGBMRegressor(
            n_estimators=100, max_depth=5, learning_rate=0.1,
            random_state=42
        ),
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(random_state=42),
        'Lasso Regression': Lasso(random_state=42),
        'SVR': SVR(kernel='rbf'),
        'Gradient Boosting': GradientBoostingRegressor(
            n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42
        ),
        'AdaBoost': AdaBoostRegressor(
            n_estimators=50, learning_rate=1.0, random_state=42
        ),
        'CatBoost': CatBoostRegressor(
            iterations=100, depth=5, learning_rate=0.1,
            random_state=42, verbose=0
        )
    }
    
    # Train all models
    trained_models = {}
    for name, model in models.items():
        start_time = time.time()
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        training_time = time.time() - start_time
        trained_models[name] = {
            'model': model,
            'training_time': training_time
        }
        print(f"Completed {name} in {training_time:.2f} seconds")
    
    return trained_models

def evaluate_models(trained_models, X_test, y_test):
    """Evaluate all models and return results"""
    results = []
    
    for name, model_info in trained_models.items():
        model = model_info['model']
        start_time = time.time()
        y_pred = model.predict(X_test)
        pred_time = time.time() - start_time
        
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        results.append({
            'Model': name,
            'MSE': mse,
            'RMSE': np.sqrt(mse),
            'MAE': mae,
            'R2 Score': r2,
            'Training Time': model_info['training_time'],
            'Prediction Time': pred_time
        })
    
    return pd.DataFrame(results)

def save_best_model(models_df, trained_models, scaler, metric='RMSE'):
    """Save the best performing model based on specified metric"""
    if metric in ['MSE', 'RMSE', 'MAE']:
        # For error metrics, we want to minimize
        best_model_info = models_df.loc[models_df[metric].idxmin()]
    else:
        # For R2, we want to maximize
        best_model_info = models_df.loc[models_df[metric].idxmax()]
        
    best_model_name = best_model_info['Model']
    best_model = trained_models[best_model_name]['model']
    
    joblib.dump(best_model, 'best_model.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    print(f"\nSaved best model ({best_model_name}) to best_model.pkl")
    print(f"Saved scaler to scaler.pkl")
    
    return best_model_name

def main():
    # Load and prepare data
    df = load_data(path='mock_features.csv')
    X_train, X_test, y_train, y_test, scaler = prepare_data(df)
    
    # Train multiple models
    trained_models = train_models(X_train, y_train)
    
    # Evaluate all models
    results_df = evaluate_models(trained_models, X_test, y_test)
    
    # Display results
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    print("\nModel Comparison Results:")
    print(results_df.sort_values('RMSE'))
    
    # Save the best model (using RMSE as the selection metric)
    best_model_name = save_best_model(results_df, trained_models, scaler, metric='RMSE')
    
    # Print metrics for best model
    best_result = results_df[results_df['Model'] == best_model_name].iloc[0]
    print("\nBest Model Metrics:")
    print(f"RMSE: {best_result['RMSE']:.6f}")
    print(f"MSE: {best_result['MSE']:.6f}")
    print(f"MAE: {best_result['MAE']:.6f}")
    print(f"R2 Score: {best_result['R2 Score']:.6f}")

if __name__ == "__main__":
    main()