from pinn import *

import seaborn as sns


def gridsearch_l1(X):
    """
    Gridsearch for learning rate and regularization
    """
    learning_rates = [float(i) for i in range(-4, 0)]
    regularizations = [float(i) for i in range(-4, 1)]

    grid = np.zeros((len(learning_rates), len(regularizations)))

    for i, learning_rate in enumerate(learning_rates):
        for j, regularization in enumerate(regularizations):
            activation_function = nn.Tanh()

            model = PINN(40, 3, activation_function)
            optimizer = optim.Adam(
                model.parameters(),
                lr=10**learning_rate,
                weight_decay=10**regularization,
            )

            loss = train(X, optimizer, model, epochs=1000)

            grid[i, j] = loss

    return grid


def gridsearch_l2(X):
    """
    Gridsearch for number of nodes and layers
    """
    nodes = [i for i in range(10, 101, 10)]
    layers = [i for i in range(1, 6)]

    grid = np.zeros((len(nodes), len(layers)))

    for i, node in enumerate(nodes):
        for j, layer in enumerate(layers):
            activation_function = nn.Tanh()

            model = PINN(node, layer, activation_function)
            optimizer = optim.Adam(
                model.parameters(),
                lr=10**-2,
                weight_decay=10**-4,
            )

            loss = train(X, optimizer, model, epochs=1000)

            grid[i, j] = loss

    return grid


def gridsearch_l3(X):
    """
    Gridsearch for activation function and epochs
    """
    epochs = [100, 500, 1000, 2000, 5000]
    activation_functions = [
        nn.Tanh(),
        nn.Sigmoid(),
        nn.ReLU(),
        nn.LeakyReLU(),
        nn.ELU(),
    ]
    activation_functions_names = ["Tanh", "Sigmoid", "ReLU", "LeakyReLU", "ELU"]

    grid = np.zeros((len(epochs), len(activation_functions)))

    for i, epoch in enumerate(epochs):
        for j, activation_function in enumerate(activation_functions):
            model = PINN(80, 4, activation_function)
            optimizer = optim.Adam(
                model.parameters(),
                lr=10**-2,
                weight_decay=10**-4,
            )

            loss = train(X, optimizer, model, epochs=epoch)

            grid[i, j] = loss

    return grid


def gridsearch_l4(X):
    """
    Gridsearch for epochs
    """
    epochs = np.arange(2000, 5001, 500)

    grid = np.zeros((len(epochs)))

    for i, epoch in enumerate(epochs):
        model = PINN(80, 4, nn.Tanh())
        optimizer = optim.Adam(
            model.parameters(),
            lr=10**-2,
            weight_decay=10**-4,
        )

        loss = train(X, optimizer, model, epochs=epoch)

        grid[i] = loss

    return grid


def plot_heatmap(
    grid, xticklabels, yticklabels, title, xlabel, ylabel, output_filename=None
):
    """
    Plot heatmap
    """
    im = sns.heatmap(
        grid,
        xticklabels=xticklabels,
        yticklabels=yticklabels,
        cmap="YlGn_r",
        alpha=1.0,
        annot=True,
        linewidth=0.5,
        fmt=".1e",
        annot_kws={"fontsize": 9},
        vmin=0.0,
        vmax=1.0,
        cbar_kws={"label": "Loss"},
    )

    # im.set_title(title)
    im.set_xlabel(xlabel)
    im.set_ylabel(ylabel)
    plt.tight_layout()

    plt.savefig(f"{output_filename}")
    plt.clf()
    # plt.show()


if __name__ == "__main__":
    N = 40
    X, x_mesh, t_mesh = generate_data(N=N)

    # grid_1 = gridsearch_l1(X)
    # np.save("data/grid_1.npy", grid_1)

    grid_1 = np.load("data/grid_1.npy")

    learning_rates = [float(i) for i in range(-4, 0)]
    regularizations = [float(i) for i in range(-4, 1)]

    plot_heatmap(
        grid_1,
        regularizations,
        learning_rates,
        "Gridsearch",
        "Weight decay ($log_{10}$)",
        "Learning Rate ($log_{10}$)",
        "figures/gridsearch_1.eps",
    )

    # grid_2 = gridsearch_l2(X)
    # np.save("data/grid_2.npy", grid_2)

    grid_2 = np.load("data/grid_2.npy")

    nodes = [i for i in range(10, 101, 10)]
    layers = [i for i in range(1, 6)]

    plot_heatmap(
        grid_2,
        layers,
        nodes,
        "Gridsearch",
        "Layers",
        "Nodes",
        "figures/gridsearch_2.eps",
    )

    # grid_3 = gridsearch_l3(X)
    # np.save("data/grid_3.npy", grid_3)

    grid_3 = np.load("data/grid_3.npy")

    epochs = [100, 500, 1000, 2000, 5000]
    activation_functions_names = ["Tanh", "Sigmoid", "ReLU", "LeakyReLU", "ELU"]

    plot_heatmap(
        grid_3,
        activation_functions_names,
        epochs,
        "Gridsearch",
        "Activation Function",
        "Epochs",
        "figures/gridsearch_3.eps",
    )

    # grid_4 = gridsearch_l4(X)
    # np.save("data/grid_4.npy", grid_4)

    # grid_4 = np.load("data/grid_4.npy")

    # epochs = np.arange(2000, 5001, 500)

    # plot_heatmap(
    #     grid_4,
    #     epochs,
    #     epochs,
    #     "Gridsearch",
    #     "Epochs",
    #     "Epochs",
    #     "figures/gridsearch_4.eps",
    # )
