import numpy as np
from sklearn.base import clone


class AdaBoost:
    def __init__(self, base_classifier, n_estimators):
        """
        Initialize the AdaBoost model.
        """
        self.classes_ = None
        self.base_classifier = base_classifier
        self.n_estimators = n_estimators
        self.models = []
        self.alphas = []

    def fit(self, X, y):
        """
        Train the AdaBoost model using the One-vs-All strategy.
        A classifier is constructed for each class, where it is treated as positive and all others are treated as negative.
        """
        self.classes_ = np.unique(y)
        n_samples = X.shape[0]

        for c in self.classes_:
            # create a binary target for the current class
            y_binary = np.where(y == c, 1, -1)

            # initialize uniform weights
            weights = np.ones(n_samples) / n_samples
            models_c = []
            alphas_c = []

            for _ in range(self.n_estimators):
                # train weak classifier
                classifier = clone(self.base_classifier)
                classifier.fit(X, y_binary, sample_weight=weights)
                y_pred = classifier.predict(X)

                # Calculate weighted error
                err = np.sum(weights * (y_pred != y_binary)) / np.sum(weights)

                # the error is too high
                if err > 0.5:
                    break

                # calculate alpha
                alpha = 0.5 * np.log((1 - err) / max(err, 1e-10))

                # update weights
                weights *= np.exp(-alpha * y_binary * y_pred)
                weights /= np.sum(weights)  # Normalize weights

                # store model and alpha
                models_c.append(classifier)
                alphas_c.append(alpha)

            self.models.append(models_c)
            self.alphas.append(alphas_c)

    def predict(self, X):
        """
        Predict class labels for X.
        """
        n_samples = X.shape[0]
        scores = np.zeros((n_samples, len(self.classes_)))

        # calculate scores for each class
        for idx, c in enumerate(self.classes_):
            for alpha, model in zip(self.alphas[idx], self.models[idx]):
                scores[:, idx] += alpha * model.predict(X)

        return self.classes_[np.argmax(scores, axis=1)]
