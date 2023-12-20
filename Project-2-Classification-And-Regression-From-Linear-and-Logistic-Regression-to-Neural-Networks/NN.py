from functions import *


class Layer:
    def __init__(
        self,
        input_size,
        output_size,
        optimizer_weights=Normal(learning_rate=0.01),
        optimizer_biases=Normal(learning_rate=0.01),
        regularization=1.0,
        gradient_clipping=False,
    ):
        self.gradient_clipping = gradient_clipping
        self.regularization = regularization
        self.optimizer_weights = optimizer_weights
        self.optimizer_biases = optimizer_biases
        self.weights = np.random.randn(input_size, output_size)
        self.biases = np.zeros((1, output_size)) + 0.01
        # The bias weights are often initialized to zero, but a small
        # value like 0.01 ensures all neurons have some output which can be backpropagated in the first training cycle.

    def forward(self, input):
        self.input = input
        self.output = np.dot(self.input, self.weights) + self.biases

        return self.output

    def backward(self, gradient):
        weights_gradient = np.dot(self.input.T, gradient)
        biases_gradient = np.mean(gradient, axis=0, keepdims=True)

        update_weights = self.optimizer_weights.update(
            weights_gradient * self.regularization
        )
        update_biases = self.optimizer_biases.update(biases_gradient)

        self.weights -= update_weights
        self.biases -= update_biases

        # self.weights -= learning_rate * weights_gradient
        # self.biases -= learning_rate * biases_gradient

        # np.clip(self.weights, 1e-6, 1e6, out=self.weights)

        input_gradient = np.dot(gradient, self.weights.T)

        if self.gradient_clipping:
            input_gradient = np.clip(input_gradient, 1e-16, 1e16)

        return input_gradient


def add_layer_activation(nn, activation_function):
    if activation_function == "sigmoid":
        nn.add_activation(Sigmoid())

    elif activation_function == "relu":
        nn.add_activation(Relu())

    elif activation_function == "leakyrelu":
        nn.add_activation(LeakyReLU())

    elif activation_function == "hyperbolic":
        nn.add_activation(Hyperbolic())

    elif activation_function == "linear":
        nn.add_activation(Linear())

    elif activation_function == "softmax":
        nn.add_activation(Softmax())

    elif activation_function == "elu":
        nn.add_activation(ELU())


def add_optimizer(optimizer, learning_rate):
    if not optimizer:
        opt = Normal(learning_rate=learning_rate)

    elif optimizer == "momentum":
        opt = Momentum(learning_rate=learning_rate)

    elif optimizer == "adagrad":
        opt = Adagrad(learning_rate=learning_rate)

    elif optimizer == "rmsprop":
        opt = RMSProp(learning_rate=learning_rate)

    elif optimizer == "adam":
        opt = Adam(learning_rate=learning_rate)

    return opt


def build_neural_network(
    input_size,
    output_size,
    n_hidden_layer,
    n_hidden_nodes,
    cost_function,
    cost_function_derivative,
    activation_function="hyperbolic",
    optimizer=None,
    last_activation="linear",
    learning_rate=0.01,
    regularization=1.0,
    gradient_clipping=False,
):
    if n_hidden_layer == 1:
        n_hidden_nodes = input_size

    nn = NeuralNetwork(
        cost_function,
        cost_function_derivative,
    )

    if n_hidden_layer > 1:
        nn.add_layer(
            Layer(
                input_size,
                n_hidden_nodes,
                optimizer_weights=add_optimizer(optimizer, learning_rate),
                optimizer_biases=add_optimizer(optimizer, learning_rate),
                regularization=regularization,
                gradient_clipping=gradient_clipping,
            )
        )
        add_layer_activation(nn, activation_function)

    if n_hidden_layer > 2:
        for i in range(n_hidden_layer - 2):
            nn.add_layer(
                Layer(
                    n_hidden_nodes,
                    n_hidden_nodes,
                    optimizer_weights=add_optimizer(optimizer, learning_rate),
                    optimizer_biases=add_optimizer(optimizer, learning_rate),
                    regularization=regularization,
                    gradient_clipping=gradient_clipping,
                )
            )
            add_layer_activation(nn, activation_function)

    nn.add_layer(
        Layer(
            n_hidden_nodes,
            output_size,
            optimizer_weights=add_optimizer(optimizer, learning_rate),
            optimizer_biases=add_optimizer(optimizer, learning_rate),
            regularization=regularization,
            gradient_clipping=gradient_clipping,
        )
    )

    add_layer_activation(nn, last_activation)

    return nn


class NeuralNetwork:
    def __init__(
        self,
        cost_function,
        cost_function_derivative,
    ):
        self.network = []
        self.activations = []
        self.cost_function = cost_function
        self.cost_function_derivative = cost_function_derivative

    def add_layer(self, layer):
        self.network.append(layer)

    def add_activation(self, activation):
        self.activations.append(activation)

    def train(
        self,
        X,
        y,
        epochs,
        method="gd",
        minibatch_size=10,
        return_all_costs=False,
    ):
        if return_all_costs:
            self.cost_list = []
        else:
            self.cost_list = None

        if method == "gd":
            for i in range(epochs):
                ypred = self.predict(X)

                gradient = self.cost_function_derivative(y, ypred, X)
                self.backpropagation(gradient)

        elif method == "sgd":
            n_minibatches = int(len(X) / minibatch_size)

            for i in range(epochs):
                for batch in range(n_minibatches):
                    k = minibatch_size * np.random.randint(n_minibatches)

                    X_batch = X[k : k + minibatch_size]
                    y_batch = y[k : k + minibatch_size]

                    ypred_batch = self.predict(X_batch)

                    if np.isnan(ypred_batch).any():
                        print("nan")
                        raise ValueError("The values are nan, e.i. overflow!")

                    if self.cost_list:
                        cost = self.cost_function(y_batch, ypred_batch, X_batch)
                        self.cost_list.append(cost)

                    gradient = self.cost_function_derivative(
                        y_batch, ypred_batch, X_batch
                    )

                    self.backpropagation(gradient)

                for layer in self.network:
                    layer.optimizer_weights.reset()
                    layer.optimizer_biases.reset()

    def backpropagation(self, gradient):
        for layer, activation_function in zip(
            reversed(self.network), reversed(self.activations)
        ):
            gradient = activation_function.backward(gradient)
            gradient = layer.backward(gradient)

    def predict(self, input):
        # self.best_cost = np.min(self.cost_list)
        output = input

        for layer, activation_function in zip(self.network, self.activations):
            output = layer.forward(output)
            output = activation_function.forward(output)

        return output

    def classify(self, input):
        output = input

        for layer, activation_function in zip(self.network, self.activations):
            output = layer.forward(output)
            output = activation_function.forward(output)

        return np.where(output >= 0.5, 1, 0)

    def get_cost_list(self):
        return self.cost_list
