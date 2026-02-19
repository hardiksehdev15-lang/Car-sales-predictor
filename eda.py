import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def perform_eda(input_file_path: str):
    """
    Performs Exploratory Data Analysis (EDA) on the cleaned used cars dataset.

    Steps include:
    1. Loading the cleaned dataset.
    2. Log-transforming the 'price' target variable.
    3. Generating key visualizations.
    4. Calculating the correlation matrix.
    """
    print(f"--- Starting EDA for {input_file_path} ---")

    try:
        # 1. Load the cleaned dataset
        df = pd.read_csv(input_file_path)
    except FileNotFoundError:
        print(f"Error: Cleaned file not found at {input_file_path}. Please run preprocess.py first.")
        return

    # 2. Target Transformation: Log-transform the price (Crucial for regression)
    df['log_price'] = np.log1p(df['price'])

    # Configure plot styles
    sns.set_style("whitegrid")

    # --- 3. Key Visualizations ---

    # 3.1 Distribution of Log-Transformed Price (Target Variable)
    plt.figure(figsize=(10, 5))
    sns.histplot(df['log_price'], kde=True, bins=50, color='skyblue')
    plt.title('1. Distribution of Log-Transformed Car Prices (Target)', fontsize=16)
    plt.xlabel('Log(1 + Price)')
    plt.ylabel('Frequency')
    plt.savefig('1_log_price_distribution.png')
    plt.close()
    print("Generated plot: 1_log_price_distribution.png")

    # 3.2 Price vs. Milage (Using Log Price for clearer linear trend)
    plt.figure(figsize=(10, 5))
    sns.scatterplot(x='milage', y='log_price', data=df, alpha=0.6, color='darkred')
    plt.title('2. Log Price vs. Milage', fontsize=16)
    plt.xlabel('Milage (Miles)')
    plt.ylabel('Log(1 + Price)')
    plt.savefig('2_log_price_vs_milage.png')
    plt.close()
    print("Generated plot: 2_log_price_vs_milage.png")

    # 3.3 Price Distribution by Model Year
    # Using raw price here for dollar-value interpretability
    plt.figure(figsize=(12, 6))
    sns.boxplot(x=df['model_year'], y=df['price'], flierprops={'marker': 'o', 'markersize': 2}, color='lightgreen')
    plt.xticks(rotation=45, ha='right')
    plt.title('3. Price Distribution by Model Year (Depreciation)', fontsize=16)
    plt.xlabel('Model Year')
    plt.ylabel('Price (USD)')
    plt.ylim(0, df['price'].quantile(0.99)) # Limiting y-axis to focus on majority of data
    plt.tight_layout()
    plt.savefig('3_price_vs_model_year.png')
    plt.close()
    print("Generated plot: 3_price_vs_model_year.png")

    # --- 4. Correlation Matrix ---
    print("\n4. Correlation Matrix for Numerical Features:")
    corr_matrix = df[['log_price', 'milage', 'model_year']].corr()
    print(corr_matrix)
    
    # Save a heatmap of the correlation matrix
    plt.figure(figsize=(6, 5))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
    plt.title('Correlation Matrix of Key Features', fontsize=14)
    plt.tight_layout()
    plt.savefig('4_correlation_heatmap.png')
    plt.close()
    print("Generated plot: 4_correlation_heatmap.png")
    
    print("\n--- EDA complete. Plots and correlations analyzed. ---")
    
    return df

if __name__ == "__main__":
    CLEANED_DATA_FILE = 'used_cars_kaggle_cleaned.csv'
    
    # Run the EDA process
    df_with_log_price = perform_eda(CLEANED_DATA_FILE)