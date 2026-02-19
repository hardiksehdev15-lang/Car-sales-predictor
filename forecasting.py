import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

def forecast_prices(input_file_path: str):
    """
    Retrains the best model (Random Forest) and forecasts prices for hypothetical new cars.
    """
    print(f"--- Starting Final Forecasting using {input_file_path} ---")

    try:
        # 1. Load the encoded dataset
        df_encoded = pd.read_csv(input_file_path)
    except FileNotFoundError:
        print(f"Error: Encoded features file not found. Please run model_prep.py first.")
        return

    # Separate Features (X) and Target (y)
    X_train = df_encoded.drop(columns=['log_price'])
    y_train = df_encoded['log_price']

    # 2. Retrain the best model (Random Forest) on the full dataset
    print("Retraining Random Forest Regressor on the full dataset...")
    final_rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1, max_depth=20, min_samples_split=5)
    final_rf_model.fit(X_train, y_train)

    # 3. Define Hypothetical New Car Data (Unseen data for forecasting)
    # These cars must use features present in the original dataset
    hypothetical_cars = [
        # Scenario 1: Nearly New Luxury SUV
        {'model_year': 2024, 'milage': 5000.0, 'brand': 'Land Rover', 'fuel_type': 'Hybrid', 'transmission': '8-Speed Automatic', 'accident': 'None reported', 'clean_title': 'Yes'},
        # Scenario 2: Average Used Sedan
        {'model_year': 2018, 'milage': 75000.0, 'brand': 'Honda', 'fuel_type': 'Gasoline', 'transmission': 'Automatic', 'accident': 'None reported', 'clean_title': 'Yes'},
        # Scenario 3: Old High-Mileage Truck with Accident
        {'model_year': 2010, 'milage': 150000.0, 'brand': 'Ford', 'fuel_type': 'Gasoline', 'transmission': '6-Speed A/T', 'accident': 'At least 1 accident or damage reported', 'clean_title': 'Yes'}
    ]

    # Convert to DataFrame
    df_new = pd.DataFrame(hypothetical_cars)

    # 4. Prepare New Data for Prediction (Encoding)
    
    # Define categorical columns based on the training data
    categorical_cols = ['brand', 'fuel_type', 'transmission', 'accident', 'clean_title']
    
    # One-Hot Encode the new data
    df_new_encoded = pd.get_dummies(df_new, columns=categorical_cols, drop_first=True)

    # Reindex and align columns with the training data (X_train)
    # This is CRITICAL. Missing columns (features not present in the new data) must be added as zeros.
    missing_cols = set(X_train.columns) - set(df_new_encoded.columns)
    for c in missing_cols:
        df_new_encoded[c] = 0
    
    # Ensure the order of columns is the same as the training data
    df_new_encoded = df_new_encoded[X_train.columns]


    # 5. Predict the Log Prices
    log_predictions = final_rf_model.predict(df_new_encoded)

    # 6. Convert Log Prices back to USD
    predicted_prices = np.expm1(log_predictions)

    # 7. Print the Results
    print("\n--- Final Price Forecast ---")
    
    results = df_new.copy()
    results['Predicted_Price_USD'] = predicted_prices.round(2)
    
    # Format the price column for display
    results['Predicted_Price_USD'] = results['Predicted_Price_USD'].apply(lambda x: f"${x:,.2f}")
    
    print(results)
    
    print("\n--- Project Complete: Data Preprocessing, EDA, Modeling, and Forecasting Done! ---")

if __name__ == "__main__":
    ENCODED_DATA_FILE = 'used_cars_kaggle_encoded_features.csv'
    forecast_prices(ENCODED_DATA_FILE)