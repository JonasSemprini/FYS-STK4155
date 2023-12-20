# Importing libraries
import numpy as np
import matplotlib.pyplot as plt
import autograd.numpy as np
import seaborn as sns

from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.datasets import load_breast_cancer
from autograd import grad
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from mpl_toolkits.mplot3d import Axes3D, axes3d
from imageio import imread
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from matplotlib import cm
from scipy.special import softmax
from matplotlib import colors as mcolors
from matplotlib.ticker import PercentFormatter
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset


"""Set plotting parameters"""
plt.rcParams.update(
    {"text.usetex": True, "font.family": "serif", "font.serif": ["ComputerModern"]}
)
plt.rcParams["figure.figsize"] = (8, 6)
plt.rcParams.update({"font.size": 20})
plt.rc(
    "axes",
    facecolor="whitesmoke",
    edgecolor="black",
    axisbelow=True,
    grid=True,
    lw=1.5,
)  # whitesmoke floralwhite
plt.rc("grid", color="w", linestyle="solid", lw=2.0)
plt.rc("lines", linewidth=2)
plt.rc("legend", edgecolor="black", facecolor="w", fancybox=True)

colors = [
    "steelblue",
    "firebrick",
    "seagreen",
    "darkcyan",
    "mediumvioletred",
    "darkslategray",
]

"""Functions for generating data"""


def f_poly(x, params):
    degree = len(params)

    y = params[0]

    for i in range(1, degree):
        y += params[i] * x**i

    return y


def generate_polynomial_data(xmin=-1, xmax=1, low=-1, high=1, n=100, degree=2):
    x = np.random.uniform(low=xmin, high=xmax, size=(n, 1))
    params = np.random.uniform(low=low, high=high, size=(degree,))
    y = f_poly(x, params)

    return x, y


def franke_function(x, y, noise=True):
    if noise:
        random_noise = np.random.normal(0.0, 0.1, x.shape)  # 0.05
    else:
        random_noise = 0

    term1 = 0.75 * np.exp(-(0.25 * (9 * x - 2) ** 2) - 0.25 * ((9 * y - 2) ** 2))
    term2 = 0.75 * np.exp(-((9 * x + 1) ** 2) / 49.0 - 0.1 * (9 * y + 1))
    term3 = 0.5 * np.exp(-((9 * x - 7) ** 2) / 4.0 - 0.25 * ((9 * y - 3) ** 2))
    term4 = -0.2 * np.exp(-((9 * x - 4) ** 2) - (9 * y - 7) ** 2)
    return term1 + term2 + term3 + term4 + random_noise


def generate_franke_data(noise=True, step_size=0.05, reshape=False):
    # Arrange x and y
    x = np.arange(0, 1, step_size)
    y = np.arange(0, 1, step_size)

    # Create meshgrid of x and y
    X, Y = np.meshgrid(x, y)

    # Calculate the values for Franke function
    z = franke_function(X, Y, noise=noise).flatten()

    # Flatten x and y for plotting
    x = X.flatten()
    y = Y.flatten()

    if reshape:
        x = x.reshape(-1, 1)
        y = y.reshape(-1, 1)
        z = z.reshape(-1, 1)

    return x, y, z


def generate_cancer_data(scale=True):
    cancer = load_breast_cancer()
    X = cancer.data
    y = cancer.target.reshape(-1, 1)

    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # if scale:
    #     scaler = MinMaxScaler()
    #     scaler.fit(X_train)
    #     X_train = scaler.transform(X_train)
    #     X_test = scaler.transform(X_test)

    return X, y  # X_train, X_test, y_train, y_test


"""Helper functions"""


def design_matrix_1d(x, degree, intercept=True):
    n = len(x)

    if intercept:
        X = np.ones((n, degree))

        for i in range(1, degree):
            X[:, i] = (x**i).ravel()

    else:
        X = np.zeros((n, degree - 1))

        for i in range(1, degree):
            X[:, i - 1] = (x**i).ravel()

    return X


def design_matrix_2d(x, y, degree, intercept=True):
    if len(x.shape) > 1:
        x = np.ravel(x)
        y = np.ravel(y)

    N = len(x)
    l = int((degree + 1) * (degree + 2) / 2)

    if intercept:
        X = np.ones((N, l))

        for i in range(1, degree + 1):
            q = int((i) * (i + 1) / 2)
            for k in range(i + 1):
                X[:, q + k] = (x ** (i - k)) * (y**k)
    else:
        X = np.ones((N, l - 1))

        for i in range(1, degree + 1):
            q = int((i) * (i + 1) / 2) - 1
            for k in range(i + 1):
                X[:, q + k] = (x ** (i - k)) * (y**k)

    return X


"""Plotting functions"""


def plot_surf(x, y, z, title="", filename=""):
    x = x.reshape(-1)
    y = y.reshape(-1)
    z = z.reshape(-1)

    # Creating figure
    fig = plt.figure(figsize=(14, 9))

    ax = plt.axes(projection="3d")
    ax.set_title(title)

    # Creating plot
    surf = ax.plot_trisurf(x, y, z, linewidth=0.2, antialiased=True, cmap=cm.coolwarm)
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    ax.view_init(30, -50)

    # show plot
    plt.savefig(filename)
    plt.show()


"""Regression functions"""


def mse(y, ypred):
    return mean_squared_error(y, ypred)


def mse_derivative(y, ypred, X):
    return 2 * (ypred - y) / np.size(ypred)


"""Classification functions"""


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def cross_entropy_loss(y, y_pred):
    loss = 0

    for i in range(len(y_pred)):
        loss = loss + (-1 * y[i] * np.log(y_pred[i]))

    return loss


def cross_entropy_loss_derivative(y, ypred, X):
    gradient = np.mean(ypred - y, axis=1).reshape(-1, 1)

    return gradient


def accuracy(y, ypred):
    """Accuracy function for classification"""
    return np.sum(y == ypred) / len(ypred)


"""Cost functions"""


def CostOLS(beta, X, y, lmbd):
    n = len(y)
    return (1.0 / n) * np.sum((X @ beta - y) ** 2)


def DerivativeCostOLS(beta, X, y, lmbd):
    n = len(y)
    return (2.0 / n) * X.T @ (X @ beta - y)


def CostRidge(beta, X, y, lmbd):
    n = len(y)
    return (1.0 / n) * np.sum((X @ beta - y) ** 2) + (lmbd) * (beta.T @ beta)


def DerivativeCostRidge(beta, X, y, lmbd):
    n = len(y)
    return (2.0 / n) * X.T @ (X @ beta - y) + 2.0 * lmbd * beta


"""Optimizer classes"""


class Normal:
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate

    def update(self, gradient):
        update = self.learning_rate * gradient

        return update

    def reset(self):
        pass


class Momentum:
    def __init__(self, momentum=0.8, learning_rate=0.01):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.change = 0.0

    def update(self, gradient):
        self.change = self.learning_rate * gradient + self.momentum * self.change
        update = self.change

        return update

    def reset(self):
        self.change = 0.0


class RMSProp:
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate
        self.epsilon = 1e-6
        self.rho = 0.90
        self.r = 0.0

    def update(self, gradient):
        self.r = self.rho * self.r + (1 - self.rho) * (gradient**2)
        update = self.learning_rate / (np.sqrt(self.epsilon + (self.r))) * gradient

        return update

    def reset(self):
        self.r = 0.0


class Adagrad:
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate
        self.epsilon = 1e-7
        self.r = 0.0

    def update(self, gradient):
        self.r += gradient**2
        update = self.learning_rate / (self.epsilon + np.sqrt(self.r)) * gradient

        return update

    def reset(self):
        self.r = 0.0


class Adam:
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate
        self.rho1 = 0.9
        self.rho2 = 0.999
        self.epsilon = 1e-8
        self.s = 0.0
        self.r = 0.0
        self.t = 0

    def update(self, gradient):
        self.t += 1
        self.s = self.rho1 * self.s + (1 - self.rho1) * gradient
        self.r = self.rho2 * self.r + (1 - self.rho2) * (gradient**2)
        s_hat = self.s / (1 - self.rho1**self.t)
        r_hat = self.r / (1 - self.rho2**self.t)
        update = s_hat * self.learning_rate / (np.sqrt(r_hat) + self.epsilon)

        return update

    def reset(self):
        self.s = 0.0
        self.r = 0.0
        self.t = 0


"""Activation functions"""


from functions import *


class ActivationFunction:
    def __init__(self, activation_function, activation_function_derivative):
        self.activation_function = activation_function
        self.activation_function_derivative = activation_function_derivative

        self.input = None
        self.output = None

    def forward(self, input):
        self.input = input

        self.output = self.activation_function(self.input)

        return self.output

    def backward(self, output_gradient):
        return np.multiply(
            output_gradient, self.activation_function_derivative(self.input)
        )


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


class Sigmoid(ActivationFunction):
    def __init__(self):
        super().__init__(self.sigmoid, self.sigmoid_derivative)

    @staticmethod
    def sigmoid(x):
        return sigmoid(x)

    @staticmethod
    def sigmoid_derivative(x):
        return sigmoid(x) * (1 - sigmoid(x))

    # @staticmethod
    # def sigmoid_derivative(x):
    #     return np.exp(x) / (1 + np.exp(-x)) ** 2


class Relu(ActivationFunction):
    def __init__(self):
        super().__init__(self.relu, self.relu_derivative)

    @staticmethod
    def relu(x):
        return np.maximum(0.0, x)

    @staticmethod
    def relu_derivative(x):
        return np.where(x < 0.0, 0.0, 1.0)


class Hyperbolic(ActivationFunction):
    def __init__(self):
        super().__init__(self.tanh, self.tanh_derivative)

    @staticmethod
    def tanh(x):
        return np.tanh(x)

    @staticmethod
    def tanh_derivative(x):
        return 1 - np.tanh(x) ** 2


class Linear(ActivationFunction):
    def __init__(self):
        super().__init__(self.linear, self.linear_derivative)

    @staticmethod
    def linear(x):
        return x

    @staticmethod
    def linear_derivative(x):
        return np.ones_like(x)


class LeakyReLU(ActivationFunction):
    def __init__(self):
        super().__init__(self.leakyrelu, self.leakyrelu_derivative)

    @staticmethod
    def leakyrelu(x, alpha=0.01):
        return np.maximum(alpha * x, x)

    @staticmethod
    def leakyrelu_derivative(x, alpha=0.01):
        return np.where(x < 0, alpha, 1)


class ELU(ActivationFunction):
    def __init__(self):
        super().__init__(self.elu, self.elu_derivative)

    @staticmethod
    def elu(x, alpha=1.0):
        return np.where(x <= 0, alpha * (np.exp(x) - 1), x)

    @staticmethod
    def elu_derivative(x, alpha=1.0):
        return np.where(x < 0, alpha * np.exp(x), 1)


class Softmax(ActivationFunction):
    def __init__(self):
        super().__init__(self.softmax, self.softmax_derivative)

    @staticmethod
    def softmax(x):
        # return np.exp(x) / np.sum(np.exp(x), axis=0, keepdims=True)
        # e_x = np.exp(x - np.max(x))
        # return e_x / e_x.sum()
        # return np.exp(x - np.max(x)) / np.exp(x - np.max(x)).sum(axis=1, keepdims=True)
        return softmax(x)

    @staticmethod
    def softmax_derivative(x):
        # return np.ones_like(x)
        e = np.exp(x - np.max(x))
        s = np.sum(e, axis=1, keepdims=True)
        return e / s
        # return sigmoid(x) * (1 - sigmoid(x))
