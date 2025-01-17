import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np


def preprocess_data_for_linear_regression(data, country):
    """
    Prepares the dataset for linear regression by filtering, encoding, and scaling features.
    """
    country_data = data[data['Country'] == country].reset_index(drop=True)

    # country_data['Revenue_per_Visitor'] = country_data['Revenue'] / country_data['Visitors']
    country_data['Log_Visitors'] = np.log1p(country_data['Visitors'])
    country_data['Log_Revenue'] = np.log1p(country_data['Revenue'])


    le_category = LabelEncoder()
    country_data['Category_encoded'] = le_category.fit_transform(country_data['Category'])

    return country_data, le_category


def train_and_evaluate_linear_regression(data):
    """
    Trains and evaluates a linear regression model.
    """

    features = data[['Log_Visitors', 'Rating', 'Log_Revenue']]
    target = data['Revenue']

    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

    # train the linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # make predictions
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return model, mse, mae, r2, X_test, y_test, y_pred


def calculate_revenue_hierarchy_linear(test_data, y_pred, data, le_category):
    """
    Calculates the revenue hierarchy based on predictions from the linear regression model.
    """
    test_data['Predicted_Revenue'] = y_pred
    test_data['Category'] = data.iloc[test_data.index]['Category'].values

    avg_predicted_revenue_by_category = (
        test_data.groupby('Category')['Predicted_Revenue'].mean().sort_values(ascending=False)
    )
    return avg_predicted_revenue_by_category


def main_linear_regression(file_path, country):

    data = pd.read_csv(file_path)
    country_data, le_category = preprocess_data_for_linear_regression(data, country)

    model, mse, mae, r2, X_test, y_test, y_pred = train_and_evaluate_linear_regression(country_data)
    print(f"Linear Regression Results for {country}:")
    print(f"Mean Squared Error (MSE): {mse:.2f}")
    print(f"Mean Absolute Error (MAE): {mae:.2f}")
    print(f"R² Score: {r2:.2f}")

    # calculate revenue hierarchy
    test_data = pd.DataFrame(X_test, columns=['Log_Visitors', 'Rating', 'Log_Revenue'])
    revenue_hierarchy = calculate_revenue_hierarchy_linear(test_data, y_pred, country_data, le_category)
    print("\nRevenue Hierarchy:")
    print(revenue_hierarchy)


if __name__ == "__main__":
    file_path = '../../data/tourism_dataset.csv'
    countries = ['India', 'USA', 'Brazil', 'France', 'Egypt', 'China', 'Australia']
    for country in countries:
        print(f"\n--- Results for {country} ---")
        main_linear_regression(file_path, country)
