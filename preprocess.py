import pandas as pd
import numpy as np
import warnings

# --- CORRECTED LINE ---
# The path to SettingWithCopyWarning has moved. 
# We now typically import it from pandas.errors.
try:
    from pandas.errors import SettingWithCopyWarning
except ImportError:
    # Fallback for very old pandas versions, though this is less likely to fix the original error
    SettingWithCopyWarning = pd.core.common.SettingWithCopyWarning 

# Suppress the harmless SettingWithCopyWarning 
warnings.filterwarnings('ignore', category=SettingWithCopyWarning)

def clean_data(input_file_path: str, output_file_path: str):
    """
    Performs comprehensive data preprocessing on the used cars dataset.
    """
    print(f"--- Starting data cleaning for {input_file_path} ---")

    try:
        # 1. Load the dataset
        df = pd.read_csv(input_file_path)
    except FileNotFoundError:
        print(f"Error: Input file not found at {input_file_path}")
        return

    # Standardize column names by stripping leading/trailing spaces
    df.columns = df.columns.str.strip()
    
    # --- 2. Clean Numerical Columns (Price and Milage) ---

    # Clean 'price' column (Remove '$', ',', and convert to float)
    print("Cleaning 'price' and 'milage' columns...")
    df['price'] = df['price'].astype(str).str.replace('$', '', regex=False).str.replace(',', '', regex=False).astype(float)

    # Clean 'milage' column (Remove ' mi.', ',', and convert to float)
    df['milage'] = df['milage'].astype(str).str.replace(' mi.', '', regex=False).str.replace(',', '', regex=False).astype(float)

    # --- 3. Handle Missing/Inconsistent Categorical Values (Imputation with Mode) ---

    # a. 'fuel_type' imputation
    # Replace inconsistent/unclear entries ('–' and 'not supported') with NaN
    df['fuel_type'] = df['fuel_type'].replace(['–', 'not supported'], np.nan)
    # Impute remaining NaNs with the mode
    fuel_type_mode = df['fuel_type'].mode()[0]
    df['fuel_type'].fillna(fuel_type_mode, inplace=True)

    # b. 'accident' imputation
    # Impute NaNs with the mode
    accident_mode = df['accident'].mode()[0]
    df['accident'].fillna(accident_mode, inplace=True)

    # c. 'clean_title' imputation
    # Impute NaNs with the mode
    clean_title_mode = df['clean_title'].mode()[0]
    df['clean_title'].fillna(clean_title_mode, inplace=True)

    # --- 4. Final Verification and Saving ---
    
    # Final check for missing values across all columns
    print("\nFinal check for missing values (All counts should be 0):")
    print(df.isnull().sum())
    
    # Save the cleaned DataFrame
    df.to_csv(output_file_path, index=False)
    print(f"\n--- Data cleaning complete. Cleaned data saved to {output_file_path} ---")
    
    return df

if __name__ == "__main__":
    # Define file paths
    RAW_DATA_FILE = 'used_cars_kaggle.csv'
    CLEANED_DATA_FILE = 'used_cars_kaggle_cleaned.csv'
    
    # Run the cleaning process
    cleaned_df = clean_data(RAW_DATA_FILE, CLEANED_DATA_FILE)

    # Display the first 5 rows of the cleaned data
    if cleaned_df is not None:
        print("\nHead of the cleaned DataFrame:")
        print(cleaned_df.head())