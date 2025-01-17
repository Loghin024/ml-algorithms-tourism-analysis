import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from custom_adaboost import AdaBoost
import numpy as np


def preprocess_data_for_adaboost(data, country):
    """
    Preprocesses the data for AdaBoost algorithm by filtering, encoding, and scaling features.
    """
    country_data = data[data['Country'] == country].reset_index(drop=True)

    country_data['Revenue_per_Visitor'] = country_data['Revenue'] / country_data['Visitors']
    country_data['Log_Visitors'] = np.log1p(country_data['Visitors'])

    le_category = LabelEncoder()
    country_data['Category_encoded'] = le_category.fit_transform(country_data['Category'])

    return country_data, le_category


def evaluate_adaboost_sklearn(data):
    """
    Trains and evaluates the sklearn AdaBoostClassifier.
    """
    features = data[['Log_Visitors', 'Rating', 'Revenue_per_Visitor']]
    target = data['Category_encoded']

    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42, stratify=target)

    base_estimator = DecisionTreeClassifier(max_depth=1, random_state=42)
    model = AdaBoostClassifier(estimator=base_estimator, n_estimators=50, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    return model, X_test, y_test, y_pred, accuracy


def evaluate_adaboost_custom(data):
    """
    Trains and evaluates the custom AdaBoost implementation.
    """
    features = data[['Log_Visitors', 'Rating', 'Revenue_per_Visitor']].values
    target = data['Category_encoded'].values

    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42, stratify=target)

    adaboost = AdaBoost(base_classifier=DecisionTreeClassifier(max_depth=1, random_state=42), n_estimators=50)
    adaboost.fit(X_train, y_train)

    y_pred = adaboost.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    return adaboost, X_test, y_test, y_pred, accuracy


def calculate_revenue_hierarchy(test_data, y_pred, le_category, data):
    """
    Calculates the revenue hierarchy based on predictions.
    """
    test_data['Category_pred'] = y_pred
    test_data['Category_pred_label'] = le_category.inverse_transform(test_data['Category_pred'])
    test_data['Revenue'] = data.iloc[test_data.index]['Revenue'].values

    avg_revenue_by_category = (
        test_data.groupby('Category_pred_label')['Revenue']
        .mean()
        .sort_values(ascending=False)
    )
    return avg_revenue_by_category


def main(file_path):
    data = pd.read_csv(file_path)
    countries = ['India', 'USA', 'Brazil', 'France', 'Egypt', 'China', 'Australia']

    for country in countries:
        print(f"\n=== Results for {country} ===")

        country_data, le_category = preprocess_data_for_adaboost(data, country)

        # evaluate sklearn AdaBoost
        sklearn_model, X_test, y_test, sklearn_y_pred, sklearn_accuracy = evaluate_adaboost_sklearn(country_data)
        print(f"Sklearn AdaBoost Accuracy for {country}: {sklearn_accuracy * 100:.2f}%")
        sklearn_revenue_hierarchy = calculate_revenue_hierarchy(X_test.copy(), sklearn_y_pred, le_category, country_data)
        print("Sklearn Revenue Hierarchy:")
        print(sklearn_revenue_hierarchy)

        # evaluate custom AdaBoost
        custom_model, custom_X_test, custom_y_test, custom_y_pred, custom_accuracy = evaluate_adaboost_custom(country_data)
        print(f"Custom AdaBoost Accuracy for {country}: {custom_accuracy * 100:.2f}%")
        custom_revenue_hierarchy = calculate_revenue_hierarchy(pd.DataFrame(custom_X_test, columns=['Log_Visitors', 'Rating', 'Revenue_per_Visitor']),
                                                               custom_y_pred, le_category, country_data)
        print("Custom Revenue Hierarchy:")
        print(custom_revenue_hierarchy)


if __name__ == "__main__":
    file_path = '../../data/tourism_dataset.csv'
    main(file_path)
