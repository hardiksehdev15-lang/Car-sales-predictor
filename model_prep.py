import pandas as pd
import numpy as np

def prepare_features(input_file_path: str, output_file_path: str):
    """
    Applies feature engineering and one-hot encoding to prepare data for ML models.
    
    Args:
        input_file_path (str): Path to the cleaned CSV file.
        output_file_path (str): Path to save the final encoded feature matrix.
    """
    print(f"--- Starting feature preparation for {input_file_path} ---")

    try:
        # 1. Load the cleaned dataset
        df = pd.read_csv(input_file_path)
    except FileNotFoundError:
        print(f"Error: Cleaned file not found at {input_file_path}. Please run preprocess.py first.")
        return

    # 2. Target Transformation: Create the log-transformed price target variable
    df['log_price'] = np.log1p(df['price'])
    
    # 3. Feature Selection
    # Select key numerical and categorical features.
    # Exclude high-cardinality features (model, engine, ext_col, int_col) to prevent overfitting.
    selected_features = [
        'model_year', 
        'milage', 
        'brand', 
        'fuel_type', 
        'transmission', 
        'accident', 
        'clean_title'
    ]
    
    df_features = df[selected_features]
    
    # Identify categorical columns for encoding
    categorical_cols = df_features.select_dtypes(include='object').columns
    print(f"Categorical features to be encoded: {list(categorical_cols)}")

    # 4. One-Hot Encoding
    # Convert categorical variables into dummy/indicator variables
    df_encoded = pd.get_dummies(df_features, columns=categorical_cols, drop_first=True)
    
    # Add the log_price target variable back to the feature matrix
    df_encoded['log_price'] = df['log_price']

    # Final check of the encoded data shape
    print(f"\nOriginal number of columns: {len(selected_features)}")
    print(f"Final number of columns after One-Hot Encoding: {df_encoded.shape[1]}")
    print(f"Final dataset shape (Rows, Columns): {df_encoded.shape}")

    # 5. Saving the final feature matrix
    df_encoded.to_csv(output_file_path, index=False)
    print(f"--- Feature preparation complete. Encoded data saved to {output_file_path} ---")
    
    return df_encoded

if __name__ == "__main__":
    CLEANED_DATA_FILE = 'used_cars_kaggle_cleaned.csv'
    ENCODED_DATA_FILE = 'used_cars_kaggle_encoded_features.csv'
    
    # Run the feature preparation process
    encoded_df = prepare_features(CLEANED_DATA_FILE, ENCODED_DATA_FILE)

    if encoded_df is not None:
        print("\nHead of the final encoded DataFrame (Features + Target):")
        print(encoded_df.head())