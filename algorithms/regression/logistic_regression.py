import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score
import numpy as np


def preprocess_data_for_logistic_regression(data, country):
    """
    Prepares the dataset for logistic regression by filtering, encoding, and scaling features.
    """
    country_data = data[data['Country'] == country].reset_index(drop=True)

    country_data['Revenue_per_Visitor'] = country_data['Revenue'] / country_data['Visitors']
    country_data['Log_Visitors'] = np.log1p(country_data['Visitors'])

    le_category = LabelEncoder()
    le_accommodation = LabelEncoder()
    country_data['Category_encoded'] = le_category.fit_transform(country_data['Category'])
    country_data['Accommodation_encoded'] = le_accommodation.fit_transform(country_data['Accommodation_Available'])

    # select relevant features
    features = country_data[['Log_Visitors', 'Rating', 'Accommodation_encoded', 'Revenue_per_Visitor']]
    target = country_data['Category_encoded']

    # standardize numerical features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    return features_scaled, target, le_category


def train_and_evaluate_logistic_regression(X, y):
    """
    Trains and evaluates a logistic regression model with default parameters.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # train the logistic regression model
    model = LogisticRegression(random_state=42, max_iter=200)
    model.fit(X_train, y_train)

    # make predictions
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    return model, accuracy, X_test, y_test, y_pred


def calculate_revenue_hierarchy(test_data, y_pred, y_test, le_category, country_data):
    """
    Calculates the revenue hierarchy based on predictions from the logistic regression model.
    """
    test_data['Category_pred_label'] = le_category.inverse_transform(y_pred)
    test_data['Actual_label'] = le_category.inverse_transform(y_test)

    test_data['Revenue'] = country_data.iloc[test_data.index]['Revenue'].values

    avg_revenue_by_category = (
        test_data.groupby('Category_pred_label')['Revenue'].mean().sort_values(ascending=False)
    )
    return avg_revenue_by_category


def main(file_path, countries):
    """
    Main function to execute the logistic regression pipeline for multiple countries.
    """
    data = pd.read_csv(file_path)

    for country in countries:
        print(f"\n--- Results for {country} ---")
        X, y, le_category = preprocess_data_for_logistic_regression(data, country)

        model, accuracy, X_test, y_test, y_pred = train_and_evaluate_logistic_regression(X, y)

        print(f"Accuracy for {country}: {accuracy * 100:.2f}%")

        # calculate revenue hierarchy
        test_data = pd.DataFrame(X_test, columns=['Log_Visitors', 'Rating', 'Accommodation_encoded', 'Revenue_per_Visitor'])
        revenue_hierarchy = calculate_revenue_hierarchy(test_data, y_pred, y_test, le_category, data)
        print("\nRevenue Hierarchy:")
        print(revenue_hierarchy)


if __name__ == "__main__":
    file_path = '../../data/tourism_dataset.csv'
    countries = ['India', 'USA', 'Brazil', 'France', 'Egypt', 'China', 'Australia']
    main(file_path, countries)
