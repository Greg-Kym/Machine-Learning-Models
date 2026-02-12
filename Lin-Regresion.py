import numpy as np


class Linear_Regression():

    def __init__(self, learning_rate, no_of_iterations):
        self.learning_rate = learning_rate
        self.no_of_iterations = no_of_iterations

    def fit(self, X, Y):

        self.rows, self.features = X.shape

        self.w = np.zeros(self.features)
        self.b = 0
        self.X = X
        self.Y = Y

        for i in range(self.no_of_iterations):
            self.update_weights()

    def update_weights(self):
        Y_pred = self.predict(self.X)

        error = Y_pred - self.Y

        dw = (2 / self.rows) * np.dot(self.X.T, error)
        db = (2 / self.rows) * np.sum(error)

        self.w -= self.learning_rate * dw
        self.b -= self.learning_rate * db

    def my(self):
        return

    def my(self):
        return

    def predict(self, X):
        return np.dot(X, self.w) + self.b
