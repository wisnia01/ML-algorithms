import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt

bc = datasets.load_breast_cancer()
X, y = bc.data, bc.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)


def accuracy(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy


from logistic_regression import LogisticRegression

regressor = LogisticRegression(lr=0.0001, n_iters=1000)  # 0.001
regressor.fit(X_train, y_train)
predicted = regressor.predict(X_test)

print(accuracy(y_test, predicted))
