import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import numpy as np


def preprocess_data_for_id3(data, country, q_visitors, q_revenue):
    """
    Preprocesses the data for ID3 algorithm with specified discretization intervals.
    """

    country_data = data[data['Country'] == country].reset_index(drop=True)

    country_data['Revenue_per_Visitor'] = country_data['Revenue'] / country_data['Visitors']
    country_data['Log_Visitors'] = np.log1p(country_data['Visitors'])

    # discretize variables with specified quantiles
    country_data['Visitors_Bins'] = pd.qcut(country_data['Log_Visitors'], q=q_visitors, labels=False)
    country_data['Revenue_per_Visitor_Bins'] = pd.qcut(country_data['Revenue_per_Visitor'], q=q_revenue, labels=False)

    le_category = LabelEncoder()
    le_accommodation = LabelEncoder()
    country_data['Category_encoded'] = le_category.fit_transform(country_data['Category'])
    country_data['Accommodation_encoded'] = le_accommodation.fit_transform(country_data['Accommodation_Available'])

    return country_data, le_category


def train_and_evaluate_id3(data):
    """
    Trains and evaluates the ID3 (Decision Tree) algorithm.
    """
    features = data[['Visitors_Bins', 'Rating', 'Accommodation_encoded', 'Revenue_per_Visitor_Bins']]
    target = data['Category_encoded']

    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42, stratify=target)

    # train the ID3 model
    model = DecisionTreeClassifier(criterion='entropy', random_state=42)
    model.fit(X_train, y_train)

    # make predictions
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    test_data = X_test.copy()
    test_data['Category_pred'] = y_pred
    test_data['Actual'] = y_test.reset_index(drop=True)

    return model, test_data, accuracy


def optimize_discretization(data, country, max_q=10):
    """
    Optimizes the number of quantiles (q) for Visitors_Bins and Revenue_Bins.
    """
    best_accuracy = 0
    best_params = None

    # iterate over all combinations of q for Visitors_Bins and Revenue_Bins
    for q_visitors in range(2, max_q + 1):
        for q_revenue in range(2, max_q + 1):
            processed_data, _ = preprocess_data_for_id3(data, country, q_visitors, q_revenue)

            _, _, accuracy = train_and_evaluate_id3(processed_data)

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_params = (q_visitors, q_revenue)

    return best_params


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


def main(file_path):
    data = pd.read_csv(file_path)
    countries = ['India', 'USA', 'Brazil', 'France', 'Egypt', 'China', 'Australia']

    for country in countries:
        print(f"\n--- Results for {country} ---")

        best_params = optimize_discretization(data, country, max_q=10)

        print(f"Best q for Visitors_Bins: {best_params[0]}, Revenue_Bins: {best_params[1]}")

        processed_data, le_category = preprocess_data_for_id3(data, country, best_params[0], best_params[1])
        model, test_data, accuracy = train_and_evaluate_id3(processed_data)

        print(f"Accuracy for {country}: {accuracy * 100:.2f}%")

        revenue_hierarchy = calculate_revenue_hierarchy_id3(test_data, processed_data, le_category)
        print("\nRevenue Hierarchy:")
        print(revenue_hierarchy)


if __name__ == "__main__":
    file_path = '../data/tourism_dataset.csv'
    main(file_path)
