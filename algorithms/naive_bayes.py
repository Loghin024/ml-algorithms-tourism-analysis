import pandas as pd
from math import prod


class NaiveBayesClassifier:
    def __init__(self, data: pd.DataFrame, target: str):
        """
        Initializes the Naive Bayes Classifier.

        :param data: A pandas DataFrame containing the dataset.
        :param target: The name of the target column.
        """
        self.data = data
        self.target = target

    def calculate_conditional_probabilities(self, feature, value, target_class):
        """
        Calculate conditional probabilities P(feature=value | target_class).

        :param feature: The feature column name.
        :param value: The value of the feature.
        :param target_class: The target class.
        :return: Conditional probability.
        """
        subset = self.data[self.data[self.target] == target_class]
        return (subset[feature] == value).sum() / len(subset)

    def naive_bayes_predict(self, instance):
        """
        Predict the target class using Naive Bayes.

        :param instance: A pandas Series representing an instance.
        :return: Predicted target class.
        """
        classes = self.data[self.target].unique()
        probs = {}

        for target_class in classes:
            # prior probability of the class
            class_prob = len(self.data[self.data[self.target] == target_class]) / len(self.data)

            # product of conditional probabilities for all features
            conditional_probs = [
                self.calculate_conditional_probabilities(feature, instance[feature], target_class)
                for feature in instance.index if feature != self.target
            ]

            # final probability for the class
            probs[target_class] = class_prob * prod(conditional_probs)

        # return the class with the highest probability
        return max(probs, key=probs.get)

    def calculate_training_error(self):
        """
        Calculate the training error of the classifier.

        :return: Training error as a fraction.
        """
        incorrect_predictions = sum(
            1 for _, row in self.data.iterrows() if self.naive_bayes_predict(row) != row[self.target]
        )
        return incorrect_predictions / len(self.data)

    def calculate_cvloo_error(self):
        """
        Calculate the cross-validation leave-one-out (CVLOO) error.

        :return: CVLOO error as a fraction.
        """
        incorrect_predictions = 0
        for i in range(len(self.data)):
            train_df = self.data.drop(index=i)
            test_instance = self.data.iloc[i]
            # cgit reate a temporary classifier for the subset
            temp_classifier = NaiveBayesClassifier(train_df, self.target)
            prediction = temp_classifier.naive_bayes_predict(test_instance)
            if prediction != test_instance[self.target]:
                incorrect_predictions += 1
        return incorrect_predictions / len(self.data)


def main():
    # Example Usage:
    data = [
        {"Outlook": "Sunny", "Temperature": "Hot", "Humidity": "High", "Wind": "Weak", "EnjoyTennis": "No"},
        {"Outlook": "Sunny", "Temperature": "Hot", "Humidity": "High", "Wind": "Strong", "EnjoyTennis": "No"},
        {"Outlook": "Overcast", "Temperature": "Hot", "Humidity": "High", "Wind": "Weak", "EnjoyTennis": "Yes"},
        {"Outlook": "Rain", "Temperature": "Mild", "Humidity": "High", "Wind": "Weak", "EnjoyTennis": "Yes"},
        {"Outlook": "Rain", "Temperature": "Cool", "Humidity": "Normal", "Wind": "Weak", "EnjoyTennis": "Yes"},
        {"Outlook": "Rain", "Temperature": "Cool", "Humidity": "Normal", "Wind": "Strong", "EnjoyTennis": "No"},
        {"Outlook": "Overcast", "Temperature": "Cool", "Humidity": "Normal", "Wind": "Strong", "EnjoyTennis": "Yes"},
        {"Outlook": "Sunny", "Temperature": "Mild", "Humidity": "High", "Wind": "Weak", "EnjoyTennis": "No"},
        {"Outlook": "Sunny", "Temperature": "Cool", "Humidity": "Normal", "Wind": "Weak", "EnjoyTennis": "Yes"},
        {"Outlook": "Rain", "Temperature": "Mild", "Humidity": "Normal", "Wind": "Weak", "EnjoyTennis": "Yes"},
        {"Outlook": "Sunny", "Temperature": "Mild", "Humidity": "Normal", "Wind": "Strong", "EnjoyTennis": "Yes"},
        {"Outlook": "Overcast", "Temperature": "Mild", "Humidity": "High", "Wind": "Strong", "EnjoyTennis": "Yes"},
        {"Outlook": "Overcast", "Temperature": "Hot", "Humidity": "Normal", "Wind": "Weak", "EnjoyTennis": "Yes"},
        {"Outlook": "Rain", "Temperature": "Mild", "Humidity": "High", "Wind": "Strong", "EnjoyTennis": "No"}
    ]

    df = pd.DataFrame(data)
    classifier = NaiveBayesClassifier(df, target="EnjoyTennis")

    # Compute errors
    training_error = classifier.calculate_training_error()
    cvloo_error = classifier.calculate_cvloo_error()

    print(f"Training Error: {training_error:.2f}")
    print(f"CVLOO Error: {cvloo_error:.2f}")


if __name__ == "__main__":
    main()
