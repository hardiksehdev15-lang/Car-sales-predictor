import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

def run_full_pipeline(input_file_path: str):
    """
    Executes the entire Used Car Price Prediction pipeline:
    1. Preprocessing and Cleaning
    2. Feature Engineering (Encoding)
    3. Modeling (Random Forest Training)
    4. Visualization of Results
    5. Forecasting Hypothetical Prices
    """
    print("=========================================================")
    print("=== STARTING FULL USED CAR PRICE PREDICTION PIPELINE ====")
    print("=========================================================")

    try:
        df = pd.read_csv(input_file_path)
    except FileNotFoundError:
        print(f"Error: Raw file not found at {input_file_path}.")
        return

    # --- 1. PREPROCESSING AND CLEANING ---
    print("\n--- STEP 1: Data Cleaning and Preprocessing ---")

    # Standardize column names
    df.columns = df.columns.str.strip()

    # Clean 'price' and 'milage'
    df['price'] = df['price'].astype(str).str.replace('$', '', regex=False).str.replace(',', '', regex=False).astype(float)
    df['milage'] = df['milage'].astype(str).str.replace(' mi.', '', regex=False).str.replace(',', '', regex=False).astype(float)

    # Impute missing categorical values with the Mode
    impute_cols = ['fuel_type', 'accident', 'clean_title']
    
    # Handle 'fuel_type' inconsistent values first
    df['fuel_type'] = df['fuel_type'].replace(['–', 'not supported'], np.nan)

    for col in impute_cols:
        mode_val = df[col].mode()[0]
        df[col].fillna(mode_val, inplace=True)
    
    print("Cleaning complete. All missing values handled.")

    # --- 2. FEATURE ENGINEERING AND ENCODING ---
    print("\n--- STEP 2: Feature Engineering and Encoding ---")

    # Log Transformation
    df['log_price'] = np.log1p(df['price'])
    
    # Feature Selection (Excluding high-cardinality/redundant features)
    selected_features = ['model_year', 'milage', 'brand', 'fuel_type', 'transmission', 'accident', 'clean_title']
    df_features = df[selected_features]
    categorical_cols = df_features.select_dtypes(include='object').columns

    # One-Hot Encoding
    df_encoded = pd.get_dummies(df_features, columns=categorical_cols, drop_first=True)
    df_encoded['log_price'] = df['log_price']

    # Final feature set
    X = df_encoded.drop(columns=['log_price'])
    y = df_encoded['log_price']
    
    print(f"Final feature count after encoding: {X.shape[1]}")
    
    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # --- 3. MODELING (RANDOM FOREST) ---
    print("\n--- STEP 3: Model Training (Random Forest Regressor) ---")
    
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1, max_depth=20, min_samples_split=5)
    rf_model.fit(X_train, y_train)
    rf_predictions = rf_model.predict(X_test)
    
    rf_rmse = np.sqrt(mean_squared_error(y_test, rf_predictions))
    rf_approx_error_usd = np.expm1(rf_rmse)

    print(f"Model Trained successfully.")
    print(f"Random Forest RMSE (Log Scale): {rf_rmse:.4f}")
    print(f"Random Forest Approx. Error (USD): ${rf_approx_error_usd:,.2f}")

    # --- 4. VISUALIZATION OF RESULTS ---
    print("\n--- STEP 4: Generating Visualization Plots ---")
    sns.set_style("whitegrid")
    
    # 4.1. Actual vs. Predicted Plot
    plt.figure(figsize=(8, 8))
    sns.scatterplot(x=y_test, y=rf_predictions, alpha=0.6, color='darkblue')
    max_val = max(y_test.max(), rf_predictions.max())
    min_val = min(y_test.min(), rf_predictions.min())
    plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', linewidth=2)
    plt.title('Actual vs. Predicted Prices (Log Scale)', fontsize=16)
    plt.xlabel('Actual Log Price')
    plt.ylabel('Predicted Log Price')
    plt.savefig('actual_vs_predicted_rf.png')
    plt.close()
    print("Plot generated: actual_vs_predicted_rf.png")

    # 4.2. Feature Importance Bar Chart
    plt.figure(figsize=(10, 6))
    feature_importances = pd.Series(rf_model.feature_importances_, index=X_train.columns)
    top_15_features = feature_importances.nlargest(15)
    sns.barplot(x=top_15_features.values, y=top_15_features.index, color='teal')
    plt.title('Top 15 Feature Importances (Random Forest)', fontsize=16)
    plt.xlabel('Importance Score')
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.savefig('feature_importance_rf.png')
    plt.close()
    print("Plot generated: feature_importance_rf.png")
    
    # 4.3. Residuals Plot
    plt.figure(figsize=(8, 6))
    residuals = y_test - rf_predictions
    sns.scatterplot(x=rf_predictions, y=residuals, alpha=0.6, color='darkorange')
    plt.axhline(y=0, color='red', linestyle='--', linewidth=2)
    plt.title('Residuals Plot (Model Diagnostics)', fontsize=16)
    plt.xlabel('Predicted Log Price')
    plt.ylabel('Residuals (Error)')
    plt.savefig('residuals_plot_rf.png')
    plt.close()
    print("Plot generated: residuals_plot_rf.png")

    # --- 5. FORECASTING HYPOTHETICAL PRICES ---
    print("\n--- STEP 5: Final Price Forecasting ---")

    # Retrain model on full dataset for final deployment prediction
    final_rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1, max_depth=20, min_samples_split=5)
    final_rf_model.fit(X, y)
    
    hypothetical_cars = [
        {'model_year': 2024, 'milage': 5000.0, 'brand': 'Land Rover', 'fuel_type': 'Hybrid', 'transmission': '8-Speed Automatic', 'accident': 'None reported', 'clean_title': 'Yes'},
        {'model_year': 2018, 'milage': 75000.0, 'brand': 'Honda', 'fuel_type': 'Gasoline', 'transmission': 'Automatic', 'accident': 'None reported', 'clean_title': 'Yes'},
        {'model_year': 2010, 'milage': 150000.0, 'brand': 'Ford', 'fuel_type': 'Gasoline', 'transmission': '6-Speed A/T', 'accident': 'At least 1 accident or damage reported', 'clean_title': 'Yes'}
    ]
    df_new = pd.DataFrame(hypothetical_cars)
    df_new_encoded = pd.get_dummies(df_new, columns=categorical_cols, drop_first=True)

    # Align columns for prediction
    missing_cols = set(X.columns) - set(df_new_encoded.columns)
    for c in missing_cols:
        df_new_encoded[c] = 0
    df_new_encoded = df_new_encoded[X.columns]

    # Predict and convert back to USD
    log_predictions = final_rf_model.predict(df_new_encoded)
    predicted_prices = np.expm1(log_predictions)

    results = df_new.copy()
    results['Predicted_Price_USD'] = predicted_prices.round(2)
    results['Predicted_Price_USD'] = results['Predicted_Price_USD'].apply(lambda x: f"${x:,.2f}")
    
    print("\n--- Final Predicted Prices ---")
    print(results)
    
    print("\n=========================================================")
    print("=== PIPELINE EXECUTION COMPLETE. CHECK CURRENT DIRECTORY FOR PLOTS. ===")
    print("=========================================================")


if __name__ == "__main__":
    RAW_DATA_FILE = 'used_cars_kaggle.csv'
    run_full_pipeline(RAW_DATA_FILE)