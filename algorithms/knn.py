import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def preprocess_data(data):
    """
    preprocess the dataset by encoding categorical variables and scaling numerical features.
    """
    # encode categorical variables
    le_category = LabelEncoder()
    le_accommodation = LabelEncoder()
    data['Category_encoded'] = le_category.fit_transform(data['Category'])
    data['Accommodation_encoded'] = le_accommodation.fit_transform(data['Accommodation_Available'])

    # select relevant features and target
    features = data[['Visitors', 'Rating', 'Accommodation_encoded']]
    target = data['Revenue']

    # normalize the features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    return features_scaled, target, le_category


def tune_knn(data, country):
    """
    Tunes the kNN regressor for a specific country by optimizing k, weights, and distance metrics.
    """

    country_data = data[data['Country'] == country].reset_index(drop=True)
    features, revenue, le_category = preprocess_data(country_data)

    X_train, X_test, y_train, y_test = train_test_split(
        features, revenue, test_size=0.2, random_state=42
    )

    # define hyperparameter grid
    param_grid = {
        'n_neighbors': range(1, 21),
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan', 'minkowski']
    }

    # initialize and run GridSearchCV
    knn = KNeighborsRegressor()
    grid_search = GridSearchCV(knn, param_grid, cv=5, scoring='r2', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    # retrieve the best model and parameters
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_

    # evaluate the best model on the test set
    y_pred = best_model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Generate ranking by predicted revenue
    test_data = pd.DataFrame({
        'Category': country_data.iloc[y_test.index]['Category'].values,
        'Predicted_Revenue': y_pred
    })
    ranking = test_data.groupby('Category')['Predicted_Revenue'].mean().sort_values(ascending=False)
    print(ranking)
    return best_params, mae, mse, r2


def main():
    """
    main function to load data, tune kNN, and display results.
    """

    file_path = '../data/tourism_dataset.csv'
    tourism_data = pd.read_csv(file_path)

    countries = ['India', 'USA', 'Brazil', 'France', 'Egypt', 'China', 'Australia']

    # tune kNN for the specified country
    for country_to_tune in countries:
        best_params, mae, mse, r2 = tune_knn(tourism_data, country_to_tune)

        # Display results
        print(f"Results for country: {country_to_tune}")
        print(f"Best Parameters: {best_params}")
        print(f"Mean Absolute Error (MAE): {mae:.2f}")
        print(f"Mean Squared Error (MSE): {mse:.2f}")
        print(f"R² Score: {r2:.2f}")


if __name__ == "__main__":
    main()
