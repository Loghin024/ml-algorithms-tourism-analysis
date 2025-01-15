import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import numpy as np


def preprocess_data_for_id3(data, country):
    """
    Preprocesses the data for ID3 algorithm.
    """
    country_data = data[data['Country'] == country].reset_index(drop=True)

    country_data['Revenue_per_Visitor'] = country_data['Revenue'] / country_data['Visitors']
    country_data['Log_Visitors'] = np.log1p(country_data['Visitors'])
    country_data['Log_Revenue'] = np.log1p(country_data['Revenue'])

    le_category = LabelEncoder()
    le_accommodation = LabelEncoder()
    country_data['Category_encoded'] = le_category.fit_transform(country_data['Category'])
    country_data['Accommodation_encoded'] = le_accommodation.fit_transform(country_data['Accommodation_Available'])

    return country_data, le_category


def train_and_evaluate_id3(data):
    """
    Trains and evaluates the ID3 (Decision Tree) algorithm.
    """

    features = data[['Log_Visitors', 'Rating', 'Accommodation_encoded', 'Revenue_per_Visitor']]
    target = data['Category_encoded']

    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42, stratify=target)

    # initialize and train the ID3 model
    model = DecisionTreeClassifier(criterion='entropy', random_state=42)
    model.fit(X_train, y_train)

    # make predictions
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    # Create a DataFrame with predictions and actual values
    test_data = X_test.copy()
    test_data['Category_pred'] = y_pred
    test_data['Actual'] = y_test.reset_index(drop=True)

    return model, test_data, accuracy, report


def calculate_revenue_hierarchy_id3(test_data, data, le_category):
    """
    Calculates the revenue hierarchy based on ID3 predictions.
    """

    test_data['Category_pred_label'] = le_category.inverse_transform(test_data['Category_pred'])
    test_data['Revenue'] = data.iloc[test_data.index]['Revenue'].values

    # calculate average revenue by predicted category
    avg_revenue_by_category = (
        test_data.groupby('Category_pred_label')['Revenue'].mean().sort_values(ascending=False)
    )
    return avg_revenue_by_category


def main(file_path, country):
    data = pd.read_csv(file_path)
    country_data, le_category = preprocess_data_for_id3(data, country)

    model, test_data, accuracy, report = train_and_evaluate_id3(country_data)
    print(f"Accuracy for {country}: {accuracy * 100:.2f}%")
    print("Classification Report:\n", report)

    revenue_hierarchy = calculate_revenue_hierarchy_id3(test_data, country_data, le_category)
    print("\nRevenue Hierarchy:")
    print(revenue_hierarchy)


if __name__ == "__main__":
    file_path = '../data/tourism_dataset.csv'

    countries = ['India', 'USA', 'Brazil', 'France', 'Egypt', 'China', 'Australia']
    for country in countries:
        main(file_path, country)
