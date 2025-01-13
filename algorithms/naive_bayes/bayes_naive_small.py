import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score


def preprocess_data(data, country):
    """
    Filters data for a specific country and performs label encoding for categorical features.
    """

    country_data = data[data['Country'] == country].reset_index(drop=True)

    # encode categorical columns
    le_category = LabelEncoder()
    le_accommodation = LabelEncoder()
    country_data['Category_encoded'] = le_category.fit_transform(country_data['Category'])
    country_data['Accommodation_encoded'] = le_accommodation.fit_transform(country_data['Accommodation_Available'])

    return country_data, le_category


def train_and_evaluate_naive_bayes(data):
    """
    Trains a Naive Bayes model and evaluates its accuracy on the test set.
    """

    features = data[['Visitors', 'Rating', 'Accommodation_encoded']]
    target = data['Category_encoded']

    # split into training and testing sets (80%-20%)
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

    # train a Naive Bayes model
    model = GaussianNB()
    model.fit(X_train, y_train)

    # predict on test data
    y_pred = model.predict(X_test)

    # add predictions to the test data
    test_data = X_test.copy()
    test_data['Category_pred'] = y_pred
    test_data['Actual'] = y_test.reset_index(drop=True)

    accuracy = accuracy_score(y_test, y_pred)

    return test_data, accuracy


def calculate_revenue_hierarchy(test_data, data, le_category):
    """
    Calculates the revenue hierarchy based on predictions.
    """
    test_data['Category_pred_label'] = le_category.inverse_transform(test_data['Category_pred'])
    test_data['Revenue'] = data.iloc[test_data.index]['Revenue'].values

    # group by predicted category and calculate average revenue
    avg_revenue_by_category = (
        test_data.groupby('Category_pred_label')['Revenue'].mean().sort_values(ascending=False)
    )
    return avg_revenue_by_category


def main(file_path, country):
    data = pd.read_csv(file_path)
    country_data, le_category = preprocess_data(data, country)

    test_data, accuracy = train_and_evaluate_naive_bayes(country_data)
    print(f"Model Accuracy for {country}: {accuracy * 100:.2f}%")

    revenue_hierarchy = calculate_revenue_hierarchy(test_data, country_data, le_category)
    # print("\nRevenue Hierarchy:")
    # print(revenue_hierarchy)


if __name__ == "__main__":
    file_path = '../../data/tourism_dataset.csv'
    # country = 'USA'
    countries = ['India', 'USA', 'Brazil', 'France', 'Egypt', 'China', 'Australia']
    for country in countries:
        main(file_path, country)
