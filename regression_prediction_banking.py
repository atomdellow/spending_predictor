import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from openpyxl import load_workbook

# Load data from Excel file
def load_data_from_excel(file_name, sheet_name):
    return pd.read_excel(file_name, sheet_name=sheet_name)

def extract_features(df):
    df['Date'] = pd.to_datetime(df['Date'])
    df['Month'] = df['Date'].dt.month
    df['Day'] = df['Date'].dt.day
    return df

def train_and_evaluate_model(X_train, X_test, y_train, y_test):
    # Train the model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("Mean Squared Error: ", mse)
    print("R2 Score: ", r2)

    return model

def save_predicted_data_to_csv(predicted_data, file_name):
    predicted_data.to_csv(file_name, index=False)

def main():
    file_name = "your_excel_file.xlsx"
    sheet_name = "Sheet1"

    # Load data
    data = load_data_from_excel(file_name, sheet_name)

    # Extract features
    data = extract_features(data)

    # Prepare data for training
    X = data[['Month', 'Day']]
    y = data['Spending']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train and evaluate the model
    model = train_and_evaluate_model(X_train, X_test, y_train, y_test)

    # Use the model to predict future spending habits
    future_dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
    future_data = pd.DataFrame(future_dates, columns=['Date'])
    future_data = extract_features(future_data)

    future_X = future_data[['Month', 'Day']]
    future_y = model.predict(future_X)

    future_spending = pd.DataFrame({'Date': future_dates, 'Predicted Spending': future_y})
    print(future_spending)

    # Save predicted data to a new CSV file
    output_file_name = "predicted_spending.csv"
    save_predicted_data_to_csv(future_spending, output_file_name)

if __name__ == "__main__":
    main()