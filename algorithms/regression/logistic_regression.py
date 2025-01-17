import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
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

    country_data['Category_encoded'] = le_category.fit_transform(country_data['Category'])

    return country_data, le_category


def train_and_evaluate_logistic_regression(data):
    """
    Trains and evaluates a logistic regression model with default parameters.
    """

    features = data[['Log_Visitors', 'Rating', 'Revenue_per_Visitor']]
    target = data['Category_encoded']

    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42, stratify=target)

    # train the logistic regression model
    model = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
    model.fit(X_train, y_train)

    # make predictions
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    return model, accuracy


def calculate_revenue_hierarchy_logistic(test_data, data, le_category):
    """
    Calculates the revenue hierarchy based on predictions from the logistic regression model.
    """
    test_data['Category_pred_label'] = le_category.inverse_transform(test_data['Category_pred'])
    test_data['Revenue'] = data.iloc[test_data.index]['Revenue'].values

    avg_revenue_by_category = (
        test_data.groupby('Category_pred_label')['Revenue'].mean().sort_values(ascending=False)
    )
    return avg_revenue_by_category


def main(file_path, country):

    data = pd.read_csv(file_path)
    country_data, le_category = preprocess_data_for_logistic_regression(data, country)

    model, accuracy = train_and_evaluate_logistic_regression(country_data)
    print(f"Logistic Regression Results for {country}:")
    print(f"Accuracy: {accuracy * 100:.2f}%")

    features = country_data[['Log_Visitors', 'Rating', 'Revenue_per_Visitor']]
    test_data = features.copy()
    test_data['Category_pred'] = model.predict(features)

    revenue_hierarchy = calculate_revenue_hierarchy_logistic(test_data, country_data, le_category)
    print("\nRevenue Hierarchy:")
    print(revenue_hierarchy)


if __name__ == "__main__":
    file_path = '../../data/tourism_dataset.csv'
    countries = ['India', 'USA', 'Brazil', 'France', 'Egypt', 'China', 'Australia']
    for country in countries:
        print(f"\n--- Results for {country} ---")
        main(file_path, country)
