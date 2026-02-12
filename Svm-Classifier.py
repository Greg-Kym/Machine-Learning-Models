import numpy as np


class Svm_Classifier():

    def __init__(self, learninig_rate, no_of_iterations, lambda_parameter):
        self.learninig_rate = learninig_rate
        self.no_of_iterations = no_of_iterations
        self.lambda_parameter = lambda_parameter

    def fit(self, X, Y):

        self.rows, self.features = X.shape

        self.w = np.zeros(self.features)
        self.b = 0
        self.X = X
        self.Y = Y

        for i in range(self.no_of_iterations):
            self.upadate_weights()

    def upadate_weights(self):

        Y_label = np.where(self.Y <= 0, -1, 1)

        for index, x_i in enumerate(self.X):

            condition = Y_label[index] * (np.dot(x_i, self.w) - self.b) >= 1

            if (condition):
                dw = 2 * self.lambda_parameter * self.w
                db = 0

            else:
                dw = 2 * self.lambda_parameter * \
                    self.w - (np.dot(Y_label[index], x_i))
                db = Y_label[index]

            self.w -= self.learninig_rate * dw
            self.b -= self.learninig_rate * db

    def predict(self, X):
        output = np.dot(X, self.w) - self.b

        new_output = np.sign(output)

        y_pred = np.where(new_output <= -1, 0, 1)

        return y_pred
