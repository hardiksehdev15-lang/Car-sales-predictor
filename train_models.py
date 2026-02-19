import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

def train_and_evaluate(input_file_path: str):
    """
    Trains and evaluates Linear Regression and Random Forest models for car price prediction.
    """
    print(f"--- Starting model training and evaluation using {input_file_path} ---")

    try:
        # 1. Load the encoded dataset
        df = pd.read_csv(input_file_path)
    except FileNotFoundError:
        print(f"Error: Encoded features file not found at {input_file_path}. Please run model_prep.py first.")
        return

    # 2. Separate Features (X) and Target (y)
    # The target variable is 'log_price'
    X = df.drop(columns=['log_price'])
    y = df['log_price']

    # 3. Split the data into Training and Testing sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"Training data shape: {X_train.shape}, Testing data shape: {X_test.shape}")

    # --- MODEL 1: LINEAR REGRESSION (BASELINE) ---
    print("\n--- Training Linear Regression Model ---")
    
    # Initialize and train the model
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    
    # Make predictions
    lr_predictions = lr_model.predict(X_test)
    
    # Evaluate the model
    lr_mse = mean_squared_error(y_test, lr_predictions)
    lr_rmse = np.sqrt(lr_mse)
    
    # Interpret the RMSE: Convert log_price RMSE back to an approximate USD error
    lr_approx_error_usd = np.expm1(lr_rmse)

    print(f"Linear Regression RMSE (Log Scale): {lr_rmse:.4f}")
    print(f"Linear Regression Approx. Error (USD): ${lr_approx_error_usd:,.2f}")
    
    # --- MODEL 2: RANDOM FOREST REGRESSOR ---
    print("\n--- Training Random Forest Regressor Model ---")
    
    # Initialize and train the model (using a small number of estimators for speed, you can increase this later)
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1, max_depth=20, min_samples_split=5)
    rf_model.fit(X_train, y_train)
    
    # Make predictions
    rf_predictions = rf_model.predict(X_test)
    
    # Evaluate the model
    rf_mse = mean_squared_error(y_test, rf_predictions)
    rf_rmse = np.sqrt(rf_mse)
    
    # Interpret the RMSE
    rf_approx_error_usd = np.expm1(rf_rmse)

    print(f"Random Forest RMSE (Log Scale): {rf_rmse:.4f}")
    print(f"Random Forest Approx. Error (USD): ${rf_approx_error_usd:,.2f}")

    print("\n--- Model Training and Evaluation Complete ---")

if __name__ == "__main__":
    ENCODED_DATA_FILE = 'used_cars_kaggle_encoded_features.csv'
    train_and_evaluate(ENCODED_DATA_FILE)