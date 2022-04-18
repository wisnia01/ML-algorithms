import numpy as np
from collections import Counter

def euclidian_distance(x1, x2):
    return np.sqrt(np.sum((x1-x2)**2))

class KNN:

    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y


    def predict(self, X):
        predicted_labels = [self._predict(x) for x in X]
        return np.array(predicted_labels)

    def _predict(self, x):
        # compute distances
        distances = [euclidian_distance(x, x_train) for x_train in self.X_train]
        # get k nearest sample
        k_indicies = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indicies]
        # majority vote, most common class label
        most_commom = Counter(k_nearest_labels).most_common(1)
        return most_commom[0][0]


