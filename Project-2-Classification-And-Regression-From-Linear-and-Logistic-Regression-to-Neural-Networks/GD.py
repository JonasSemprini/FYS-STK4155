from functions import *


class GradientDescent:
    def __init__(self, X, y, optimizer=None, gradient_function=None, lmbd=0.001):
        self.lmbd = lmbd
        self.optimizer = optimizer
        self.gradient_function = gradient_function

        self.beta = np.random.randn(X.shape[1]).reshape(-1, 1) * 0.01
        self.X = X
        self.y = y

    def train(self, epochs=1000):
        for i in range(epochs):
            gradient = self.gradient_function(self.beta, self.X, self.y, self.lmbd)

            update = self.optimizer.update(gradient)

            self.beta -= update

        # return np.dot(self.X, self.beta)

    def predict(self, X):
        return np.dot(X, self.beta)


class StochasticGradientDescent:
    def __init__(self, X, y, optimizer=None, gradient_function=None, lmbd=0.001):
        self.lmbd = lmbd
        self.optimizer = optimizer
        self.gradient_function = gradient_function

        self.beta = np.random.randn(X.shape[1]).reshape(-1, 1)
        self.X = X
        self.y = y

    def train(self, epochs=1000, batch_size=10):
        n = self.X.shape[0]
        n_batches = n // batch_size
        for i in range(epochs):
            for j in range(n_batches):
                k = batch_size * np.random.randint(n_batches)

                X_batch = self.X[k : k + batch_size]
                y_batch = self.y[k : k + batch_size]

                gradient = self.gradient_function(
                    self.beta, X_batch, y_batch, self.lmbd
                )

                update = self.optimizer.update(gradient)

                self.beta -= update

        # return np.dot(self.X, self.beta)

    def predict(self, X):
        return np.dot(X, self.beta)


class LogReg:
    def __init__(self, X, y, optimizer=None):
        self.optimizer = optimizer

        self.beta = np.random.randn(X.shape[1]).reshape(-1, 1)
        self.X = X
        self.y = y

    def train(self, epochs=1000, batch_size=10):
        n = self.X.shape[0]
        n_batches = n // batch_size

        for i in range(epochs):
            for j in range(n_batches):
                k = batch_size * np.random.randint(n_batches)

                X_batch = self.X[k : k + batch_size]
                y_batch = self.y[k : k + batch_size]
                gradient = -X_batch.T @ (y_batch - sigmoid(X_batch @ self.beta))

                update = self.optimizer.update(gradient)

                self.beta -= update

            self.optimizer.reset()

    def predict(self, X):
        return np.where(X @ self.beta >= 0.5, 1, 0)
