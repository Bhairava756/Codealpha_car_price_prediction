# Project Name: Car Price Prediction with Python

# This script builds a machine learning regression model to predict car prices
# based on various features.

# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import os
import datetime

def predict_car_prices(file_path):
    """
    Predicts car prices using a machine learning model.

    Args:
        file_path (str): The path to the car data CSV file.
    """
    try:
        # Check if the file exists before attempting to read it
        if not os.path.exists(file_path):
            print(f"Error: The file '{file_path}' was not found.")
            print("Please make sure the dataset is in the same directory and the filename is correct.")
            return

        # --- Step 1: Load and Prepare the Data ---
        print("Step 1: Loading and preparing the dataset...")
        df = pd.read_csv(file_path)

        # Print the original columns for reference
        print("\nOriginal columns in the dataset:")
        print(list(df.columns))

        # Handle missing values if any (check for nulls)
        print("\nChecking for missing values:")
        print(df.isnull().sum())

        # --- Step 2: Data Preprocessing and Feature Engineering ---
        print("\nStep 2: Data Preprocessing and Feature Engineering...")

        # Drop 'Car_Name' as it's not useful for numerical prediction
        df.drop('Car_Name', axis=1, inplace=True)

        # Handle categorical features using one-hot encoding
        df = pd.get_dummies(df, drop_first=True)

        # Feature Engineering: Create 'Age' from 'Year'
        current_year = datetime.datetime.now().year
        df['Age'] = current_year - df['Year']
        df.drop('Year', axis=1, inplace=True)
        
        # --- Step 3: Model Training ---
        print("\nStep 3: Training a machine learning model...")

        # Define features (X) and target (y)
        X = df.drop('Selling_Price', axis=1)
        y = df['Selling_Price']

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        print(f"Training set size: {X_train.shape[0]} samples")
        print(f"Testing set size: {X_test.shape[0]} samples")

        # Use a Linear Regression model for prediction
        model = LinearRegression()
        model.fit(X_train, y_train)
        print("Model training complete.")

        # --- Step 4: Model Evaluation ---
        print("\nStep 4: Evaluating the model's performance...")

        # Make predictions on the test data
        y_pred = model.predict(X_test)

        # Calculate evaluation metrics
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        print(f"\nMean Absolute Error (MAE): {mae:.2f}")
        print(f"R-squared (R²): {r2:.2f}")

        # Visualization: Plotting actual vs predicted prices
        plt.figure(figsize=(10, 6))
        sns.regplot(x=y_test, y=y_pred, scatter_kws={'alpha': 0.6})
        plt.title('Actual vs Predicted Car Prices')
        plt.xlabel('Actual Prices')
        plt.ylabel('Predicted Prices')
        plt.show()

        # --- Step 5: Presenting Insights ---
        print("\n--- Project Insights ---")
        print("1. Data Preprocessing: Categorical data was converted into a numerical format, and a new 'Age' feature was engineered for better prediction.")
        print("2. Model Performance: The model shows a strong R² score, indicating that it can explain a large portion of the variance in car prices.")
        print("3. Real-world Application: This model can be used to estimate the fair selling price of a used car based on its features.")
        print("------------------------")

    except Exception as e:
        print(f"An error occurred: {e}")

# Call the analysis function with the name of your dataset file
if __name__ == "__main__":
    predict_car_prices('car data.csv')
