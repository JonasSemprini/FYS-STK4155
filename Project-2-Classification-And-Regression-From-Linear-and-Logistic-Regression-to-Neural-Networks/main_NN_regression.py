from functions import *
from NN import *


def gridsearch_lr_lmbd_opt(x_train, x_test, y_train, y_test):
    """Grid search for learning rate, lambda and optimizer"""
    learning_rates = [float(i) for i in range(-6, 0)]
    lambdas = [float(i) for i in range(-6, 1)]
    optimizers = [None, "momentum", "adagrad", "rmsprop", "adam"]

    grid = np.zeros((len(optimizers), len(learning_rates), len(lambdas)))
    iterations = len(optimizers) * len(learning_rates) * len(lambdas)

    for i, optimizer in enumerate(optimizers):
        for j, learning_rate in enumerate(learning_rates):
            for k, lmbd in enumerate(lambdas):
                nn = build_neural_network(
                    input_size=x_train.shape[1],
                    output_size=y_train.shape[1],
                    n_hidden_layer=2,
                    n_hidden_nodes=20,
                    cost_function=mse,
                    cost_function_derivative=mse_derivative,
                    activation_function="elu",
                    last_activation="linear",
                    optimizer=optimizer,
                    regularization=10**lmbd,
                )

                nn.train(
                    x_train,
                    y_train,
                    epochs=1000,
                    method="gd",
                )

                y_pred = nn.predict(x_test)

                mse_test = mse(y_test, y_pred)

                grid[i, j, k] = mse_test

                print(
                    f"Progression: {100 * (i * len(learning_rates) * len(lambdas) + j * len(lambdas) + k) / iterations:.2f} %"
                )

    # np.save("grid.npy", grid)
    # grid = np.load("grid.npy")

    # for i, optimizer in enumerate(optimizers):
    #     data = grid[i]

    #     im = sns.heatmap(
    #         data,
    #         xticklabels=lambdas,
    #         yticklabels=learning_rates,
    #         cmap="RdYlGn_r",
    #         alpha=1.0,
    #         annot=True,
    #         linewidth=0.5,
    #         fmt=".1e",
    #         annot_kws={"fontsize": 9},
    #     )

    #     im.set_title(f"MSE for {optimizer}")
    #     im.set_xlabel("lmbd $\lambda$")
    #     im.set_ylabel("Learning Rate")
    #     plt.tight_layout()

    #     # plt.savefig(f"figures/{output_filename}")
    #     plt.show()
    #     plt.clf()

    return grid


def gridsearch_lr_lmbd_opt_sgd(x_train, x_test, y_train, y_test):
    """Grid search for learning rate, lambda and optimizer"""
    learning_rates = [float(i) for i in range(-6, 0)]
    lambdas = [float(i) for i in range(-6, 1)]
    optimizers = [None, "momentum", "adagrad", "rmsprop", "adam"]

    grid = np.zeros((len(optimizers), len(learning_rates), len(lambdas)))
    iterations = len(optimizers) * len(learning_rates) * len(lambdas)

    for i, optimizer in enumerate(optimizers):
        for j, learning_rate in enumerate(learning_rates):
            for k, lmbd in enumerate(lambdas):
                nn = build_neural_network(
                    input_size=x_train.shape[1],
                    output_size=y_train.shape[1],
                    n_hidden_layer=2,
                    n_hidden_nodes=20,
                    cost_function=mse,
                    cost_function_derivative=mse_derivative,
                    activation_function="elu",
                    last_activation="linear",
                    optimizer=optimizer,
                    regularization=10**lmbd,
                )

                nn.train(
                    x_train,
                    y_train,
                    epochs=100,
                    method="sgd",
                    minibatch_size=20,
                )

                y_pred = nn.predict(x_test)

                mse_test = mse(y_test, y_pred)

                grid[i, j, k] = mse_test

                print(
                    f"Progression: {100 * (i * len(learning_rates) * len(lambdas) + j * len(lambdas) + k) / iterations:.2f} %"
                )

    return grid

    # for i, optimizer in enumerate(optimizers):
    #     data = grid[i]

    #     im = sns.heatmap(
    #         data,
    #         xticklabels=lambdas,
    #         yticklabels=learning_rates,
    #         cmap="RdYlGn_r",
    #         alpha=1.0,
    #         annot=True,
    #         linewidth=0.5,
    #         fmt=".1e",
    #         annot_kws={"fontsize": 9},
    #     )

    #     im.set_title(f"MSE for {optimizer}")
    #     im.set_xlabel("lmbd $\lambda$")
    #     im.set_ylabel("Learning Rate")
    #     plt.tight_layout()

    #     # plt.savefig(f"figures/{output_filename}")
    #     plt.show()
    #     plt.clf()


def gridsearch_epoch_batchsize(x_train, x_test, y_train, y_test):
    epochs = [10, 50, 100, 1000, 3000, 5000]
    batchsizes = [10, 20, 30, 50, 75, 100]

    grid = np.zeros((len(epochs), len(batchsizes)))
    iterations = len(epochs) * len(batchsizes)

    for i, epoch in enumerate(epochs):
        for j, batchsize in enumerate(batchsizes):
            nn = build_neural_network(
                input_size=x_train.shape[1],
                output_size=y_train.shape[1],
                n_hidden_layer=2,
                n_hidden_nodes=20,
                cost_function=mse,
                cost_function_derivative=mse_derivative,
                activation_function="elu",
                last_activation="linear",
                optimizer="adam",
                regularization=1e-2,
                learning_rate=1e-1,
                gradient_clipping=False,
            )

            nn.train(
                x_train,
                y_train,
                epochs=epoch,
                method="sgd",
                minibatch_size=batchsize,
            )

            y_pred = nn.predict(x_test)

            mse_test = mse(y_test, y_pred)

            grid[i, j] = mse_test

            print(f"Progression: {100 * (i * len(batchsizes) + j) / iterations:.2f} %")

    # im = sns.heatmap(
    #     grid,
    #     xticklabels=batchsizes,
    #     yticklabels=epochs,
    #     cmap="RdYlGn_r",
    #     alpha=1.0,
    #     annot=True,
    #     linewidth=0.5,
    #     fmt=".1e",
    #     annot_kws={"fontsize": 9},
    # )

    # im.set_title(f"MSE")
    # im.set_xlabel("Batchsize")
    # im.set_ylabel("Epoch")
    # plt.tight_layout()

    # print(np.min(grid))

    # # plt.savefig(f"figures/{output_filename}")
    # plt.show()
    # plt.clf()

    return grid


def gridsearch_epoch_activation_function(x_train, x_test, y_train, y_test):
    activation_functions = ["sigmoid", "hyperbolic", "relu", "leakyrelu", "elu"]

    grid = np.zeros((len(activation_functions), 1))
    iterations = len(activation_functions)

    for i, activation_function in enumerate(activation_functions):
        nn = build_neural_network(
            input_size=x_train.shape[1],
            output_size=y_train.shape[1],
            n_hidden_layer=2,
            n_hidden_nodes=20,
            cost_function=mse,
            cost_function_derivative=mse_derivative,
            activation_function=activation_function,
            last_activation="linear",
            optimizer="adam",
            regularization=1e-2,
            learning_rate=1e-1,
        )

        nn.train(
            x_train,
            y_train,
            epochs=3000,
            method="gd",
            minibatch_size=10,
        )

        y_pred = nn.predict(x_test)

        mse_test = mse(y_test, y_pred)

        grid[i, 0] = mse_test

        print(f"Progression: {100 * i / iterations:.2f} %")

    return grid


def gridsearch_layer_nodes(x_train, x_test, y_train, y_test):
    n_hidden_layers = [1, 2, 3, 4, 5]
    n_hidden_nodes = [5, 10, 20, 30, 50, 100]

    grid = np.zeros((len(n_hidden_layers), len(n_hidden_nodes)))
    iterations = len(n_hidden_layers) * len(n_hidden_nodes)

    for i, n_hidden_layer in enumerate(n_hidden_layers):
        for j, n_hidden_node in enumerate(n_hidden_nodes):
            nn = build_neural_network(
                input_size=x_train.shape[1],
                output_size=y_train.shape[1],
                n_hidden_layer=n_hidden_layer,
                n_hidden_nodes=n_hidden_node,
                cost_function=mse,
                cost_function_derivative=mse_derivative,
                activation_function="sigmoid",
                last_activation="linear",
                optimizer="adam",
                regularization=1e-2,
                learning_rate=1e-2,
                gradient_clipping=False,
            )

            nn.train(
                x_train,
                y_train,
                epochs=1000,
                method="gd",
                # minibatch_size=10,
            )

            y_pred = nn.predict(x_test)

            mse_test = mse(y_test, y_pred)

            grid[i, j] = mse_test

            print(
                f"Progression: {100 * (i * len(n_hidden_layers) + j) / iterations:.2f} %"
            )

    return grid


def plot_heatmap(
    grid,
    xticklabels,
    yticklabels,
    title,
    xlabel,
    ylabel,
    output_filename=None,
    full_grid=np.array([False]),
):
    plt.figure(figsize=(6, 4))

    if full_grid.any() != False:
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
            vmin=np.min(full_grid),
            vmax=0.1,
        )
    else:
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
            vmin=np.min(grid),
            vmax=0.1,
        )
    #     im.collections[0].set_clim(0.1, np.min(full_grid))
    # else:
    #     im.collections[0].set_clim(0.1, np.min(grid))

    im.set_title(title)
    im.set_xlabel(xlabel)
    im.set_ylabel(ylabel)
    plt.tight_layout()

    plt.savefig(f"figures/{output_filename}")
    # plt.show()
    plt.clf()


if __name__ == "__main__":
    np.random.seed(2023)

    learning_rates = [float(i) for i in range(-6, 0)]
    lambdas = [float(i) for i in range(-6, 1)]
    optimizers = [None, "momentum", "adagrad", "rmsprop", "adam"]
    epochs = [10, 50, 100, 1000, 3000, 5000]
    batchsizes = [10, 20, 30, 50, 75, 100]
    activation_functions = ["sigmoid", "hyperbolic", "relu", "leakyrelu", "elu"]
    n_hidden_layers = [1, 2, 3, 4, 5]
    n_hidden_nodes = [5, 10, 20, 30, 50, 100]

    x, y, z = generate_franke_data(noise=True, step_size=0.05, reshape=True)

    """Franke data 2 features"""
    X = design_matrix_2d(x, y, 1, intercept=False)

    (
        x_train,
        x_test,
        y_train,
        y_test,
        z_train,
        z_test,
        X_train,
        X_test,
    ) = train_test_split(x, y, z, X, test_size=0.2, random_state=2023)

    # grid_franke_2_lr_lmbd_opt_gd = gridsearch_lr_lmbd_opt(
    #     X_train, X_test, z_train, z_test
    # )  # lr = 1e-2, lmbd = 1e-2, optimizer = adam (9.7e-3)
    # grid_franke_2_lr_lmbd_opt_sgd = gridsearch_lr_lmbd_opt_sgd(
    #     X_train, X_test, z_train, z_test
    # )  # lr = 1e-1, lmbd = 1e-2, optimizer = adam (1.0e-2)

    # grid_franke_2_epoch_batchsize = gridsearch_epoch_batchsize(
    #     X_train, X_test, z_train, z_test
    # )  # epoch = 3000, batchsize = 10 (1.3e-2)

    # grid_franke_2_epoch_activation = gridsearch_epoch_activation_function(
    #     X_train, X_test, z_train, z_test
    # )
    # elu: 1.4e-2 relu: 1.8e-2 leakyrelu: 2.0e-2 hyperbolic: 8.6e-2 sigmoid: 1.6e-2

    # grid_franke_2_layer_nodes = gridsearch_layer_nodes(X_train, X_test, z_train, z_test)

    # np.save("data/grid_franke_2_lr_lmbd_opt_gd.npy", grid_franke_2_lr_lmbd_opt_gd)
    # np.save("data/grid_franke_2_lr_lmbd_opt_sgd.npy", grid_franke_2_lr_lmbd_opt_sgd)
    # np.save("data/grid_franke_2_epoch_batchsize.npy", grid_franke_2_epoch_batchsize)
    # np.save("data/grid_franke_2_epoch_activation.npy", grid_franke_2_epoch_activation)
    # np.save("data/grid_franke_2_layer_nodes.npy", grid_franke_2_layer_nodes)

    grid_franke_2_lr_lmbd_opt_gd = np.load("data/grid_franke_2_lr_lmbd_opt_gd.npy")
    grid_franke_2_lr_lmbd_opt_sgd = np.load("data/grid_franke_2_lr_lmbd_opt_sgd.npy")
    grid_franke_2_epoch_batchsize = np.load("data/grid_franke_2_epoch_batchsize.npy")
    grid_franke_2_epoch_activation = np.load("data/grid_franke_2_epoch_activation.npy")
    grid_franke_2_layer_nodes = np.load("data/grid_franke_2_layer_nodes.npy")

    # print(grid_franke_2_lr_lmbd_opt_gd)
    # print(grid_franke_2_lr_lmbd_opt_sgd)
    # print(grid_franke_2_epoch_batchsize)
    # print(grid_franke_2_epoch_activation)
    # print(grid_franke_2_layer_nodes)

    optimizers = ["no optimizer", "momentum", "adagrad", "rmsprop", "adam"]
    # for i, grid in enumerate(grid_franke_2_lr_lmbd_opt_gd):
    #     plot_heatmap(
    #         grid,
    #         lambdas,
    #         learning_rates,
    #         f"MSE for {optimizers[i]}",
    #         "Regularization",
    #         "Learning Rate",
    #         f"grid_franke_2_lr_lmbd_{optimizers[i]}_gd.eps",
    #         full_grid=grid_franke_2_lr_lmbd_opt_gd,
    #     )

    # for i, grid in enumerate(grid_franke_2_lr_lmbd_opt_sgd):
    #     plot_heatmap(
    #         grid,
    #         lambdas,
    #         learning_rates,
    #         f"MSE for {optimizers[i]}",
    #         "Regularization",
    #         "Learning Rate",
    #         f"grid_franke_2_lr_lmbd_{optimizers[i]}_sgd.eps",
    #         full_grid=grid_franke_2_lr_lmbd_opt_sgd,
    #     )

    # plot_heatmap(
    #     grid_franke_2_epoch_batchsize,
    #     batchsizes,
    #     epochs,
    #     "MSE",
    #     "Batchsize",
    #     "Epoch",
    #     "grid_franke_2_epoch_batchsize.eps",
    # )
    # plot_heatmap(
    #     grid_franke_2_epoch_activation,
    #     ["None"],
    #     activation_functions,
    #     "MSE",
    #     "None",
    #     "Activation Function",
    #     "grid_franke_2_epoch_activation.eps",
    # )

    # plot_heatmap(
    #     grid_franke_2_layer_nodes,
    #     n_hidden_nodes,
    #     n_hidden_layers,
    #     "MSE",
    #     "Hidden Nodes",
    #     "Hidden Layers",
    #     "grid_franke_2_layer_nodes.eps",
    # )

    # GD: reg = 1.0, lr = 1e-3, opt = adam, act = elu, epoch = 1000
    # SGD: reg = 1e-2, lr = 1e-2, opt = adam, act = elu/sigmoid, epoch = 3000, batchsize = 10, layers = 4, nodes = 10

    # nn = build_neural_network(
    #     input_size=X_train.shape[1],
    #     output_size=z_train.shape[1],
    #     n_hidden_layer=4,
    #     n_hidden_nodes=10,
    #     cost_function=mse,
    #     cost_function_derivative=mse_derivative,
    #     activation_function="sigmoid",
    #     last_activation="linear",
    #     optimizer="adam",
    #     regularization=1e-2,
    #     learning_rate=1e-2,
    # )

    # nn.train(
    #     X_train,
    #     z_train,
    #     epochs=3000,
    #     method="sgd",
    #     minibatch_size=10,
    # )

    # z_pred = nn.predict(X_test)

    # print(f"MSE: {mse(z_test, z_pred)}")

    """Franke data 36 features"""
    X = design_matrix_2d(x, y, 7, intercept=True)
    print(X.shape)

    (
        x_train,
        x_test,
        y_train,
        y_test,
        z_train,
        z_test,
        X_train,
        X_test,
    ) = train_test_split(x, y, z, X, test_size=0.2, random_state=2023)

    # grid_franke_14_lr_lmbd_opt_gd = gridsearch_lr_lmbd_opt(
    #     X_train, X_test, z_train, z_test
    # )  # lr = 1e-2, lmbd = 1e-2, optimizer = adam (9.7e-3)
    # grid_franke_14_lr_lmbd_opt_sgd = gridsearch_lr_lmbd_opt_sgd(
    #     X_train, X_test, z_train, z_test
    # )  # lr = 1e-1, lmbd = 1e-2, optimizer = adam (1.0e-2)

    # grid_franke_14_epoch_batchsize = gridsearch_epoch_batchsize(
    #     X_train, X_test, z_train, z_test
    # )  # epoch = 3000, batchsize = 10 (1.3e-2)

    # grid_franke_14_epoch_activation = gridsearch_epoch_activation_function(
    #     X_train, X_test, z_train, z_test
    # )
    # # elu: 1.4e-2 relu: 1.8e-2 leakyrelu: 2.0e-2 hyperbolic: 8.6e-2 sigmoid: 1.6e-2

    grid_franke_14_layer_nodes = gridsearch_layer_nodes(
        X_train, X_test, z_train, z_test
    )

    # np.save("data/grid_franke_14_lr_lmbd_opt_gd.npy", grid_franke_14_lr_lmbd_opt_gd)
    # np.save("data/grid_franke_14_lr_lmbd_opt_sgd.npy", grid_franke_14_lr_lmbd_opt_sgd)
    # np.save("data/grid_franke_14_epoch_batchsize.npy", grid_franke_14_epoch_batchsize)
    # np.save("data/grid_franke_14_epoch_activation.npy", grid_franke_14_epoch_activation)

    # np.save("data/grid_franke_14_layer_nodes.npy", grid_franke_14_layer_nodes)

    # grid_franke_14_lr_lmbd_opt_gd = np.load("data/grid_franke_14_lr_lmbd_opt_gd.npy")
    # grid_franke_14_lr_lmbd_opt_sgd = np.load("data/grid_franke_14_lr_lmbd_opt_sgd.npy")
    # grid_franke_14_epoch_batchsize = np.load("data/grid_franke_14_epoch_batchsize.npy")
    # grid_franke_14_epoch_activation = np.load(
    #     "data/grid_franke_14_epoch_activation.npy"
    # )
    # grid_franke_14_layer_nodes = np.load("data/grid_franke_14_layer_nodes.npy")

    # for i, grid in enumerate(grid_franke_14_lr_lmbd_opt_gd):
    #     plot_heatmap(
    #         grid,
    #         lambdas,
    #         learning_rates,
    #         f"MSE for {optimizers[i]}",
    #         "Regularization",
    #         "Learning Rate",
    #         f"grid_franke_14_lr_lmbd_{optimizers[i]}_gd.eps",
    #         full_grid=grid_franke_14_lr_lmbd_opt_gd,
    #     )

    # for i, grid in enumerate(grid_franke_14_lr_lmbd_opt_sgd):
    #     plot_heatmap(
    #         grid,
    #         lambdas,
    #         learning_rates,
    #         f"MSE for {optimizers[i]}",
    #         "Regularization",
    #         "Learning Rate",
    #         f"grid_franke_14_lr_lmbd_{optimizers[i]}_sgd.eps",
    #         full_grid=grid_franke_14_lr_lmbd_opt_sgd,
    #     )

    # plot_heatmap(
    #     grid_franke_14_epoch_batchsize,
    #     batchsizes,
    #     epochs,
    #     "MSE",
    #     "Batchsize",
    #     "Epoch",
    #     "grid_franke_14_epoch_batchsize.eps",
    # )
    # plot_heatmap(
    #     grid_franke_14_epoch_activation,
    #     [""],
    #     activation_functions,
    #     "MSE",
    #     "None",
    #     "Activation Function",
    #     "grid_franke_14_epoch_activation.eps",
    # )
    plot_heatmap(
        grid_franke_14_layer_nodes,
        n_hidden_nodes,
        n_hidden_layers,
        "MSE",
        "Hidden Nodes",
        "Hidden Layers",
        "grid_franke_14_layer_nodes.eps",
    )

    # GD franke 14: reg = 1.0, lr = 1e-2, opt = adam, act = elu, epoch = 1000
    # SGD franke 2: reg = 1.0, lr = 1e-3, opt = adam, act = elu/sigmoid, epoch = 100, batchsize = 10, layers = 2, nodes = 30

    """Best nn for franke data 14"""

    # r2_test_list_sgd = []
    # r2_train_list_sgd = []
    # r2_test_list_gd = []
    # r2_train_list_gd = []
    # epochs_plot = []
    # iterations = []

    # n_minibatches = int(len(X_train) / 20)
    # epochs = [int(10000 / (32 * i)) for i in range(1, 18)]

    # for epoch in epochs:
    #     nn = build_neural_network(
    #         input_size=X_train.shape[1],
    #         output_size=z_train.shape[1],
    #         n_hidden_layer=2,
    #         n_hidden_nodes=20,
    #         cost_function=mse,
    #         cost_function_derivative=mse_derivative,
    #         activation_function="relu",
    #         last_activation="linear",
    #         optimizer="adam",
    #         regularization=1.0,
    #         learning_rate=1e-2,
    #     )

    #     nn.train(
    #         X_train,
    #         z_train,
    #         epochs=epoch,
    #         method="sgd",
    #         minibatch_size=10,
    #     )

    #     z_tilde = nn.predict(X_train)
    #     z_pred = nn.predict(X_test)

    #     r2_train = r2_score(z_train, z_tilde)
    #     r2_test = r2_score(z_test, z_pred)

    #     r2_test_list_sgd.append(r2_train)
    #     r2_train_list_sgd.append(r2_test)
    #     iterations.append(n_minibatches * epoch)

    # epochs = [
    #     100,
    #     200,
    #     500,
    #     750,
    #     1000,
    #     1250,
    #     1500,
    #     1750,
    #     2000,
    #     3000,
    #     4000,
    #     5000,
    # ]

    # min_mse = 1e10
    # for epoch in epochs:
    #     epochs_plot.append(epoch)
    #     nn = build_neural_network(
    #         input_size=X_train.shape[1],
    #         output_size=z_train.shape[1],
    #         n_hidden_layer=2,
    #         n_hidden_nodes=20,
    #         cost_function=mse,
    #         cost_function_derivative=mse_derivative,
    #         activation_function="relu",
    #         last_activation="linear",
    #         optimizer="adam",
    #         regularization=1.0,
    #         learning_rate=1e-2,
    #     )

    #     nn.train(
    #         X_train,
    #         z_train,
    #         epochs=epoch,
    #         method="gd",
    #     )

    #     z_tilde = nn.predict(X_train)
    #     z_pred = nn.predict(X_test)

    #     r2_train = r2_score(z_train, z_tilde)
    #     r2_test = r2_score(z_test, z_pred)

    #     r2_test_list_gd.append(r2_test)
    #     r2_train_list_gd.append(r2_train)

    # if mse(z_test, z_pred) < min_mse:
    #     min_mse = mse(z_test, z_pred)
    #     best_epoch = epoch

    # if mse_test < 1.5e-2:
    #     print(f"Epoch: {epoch}, MSE: {mse_test}")
    #     break

    # plt.plot(
    #     epochs_plot, r2_test_list_gd, c=colors[0], linestyle="--", label="R2 test, GD"
    # )
    # plt.plot(epochs_plot, r2_train_list_gd, c=colors[1], label="R2 train, GD")
    # plt.plot(
    #     iterations, r2_test_list_sgd, c=colors[2], linestyle="--", label="R2 test, SGD"
    # )
    # plt.plot(
    #     iterations,
    #     r2_train_list_sgd,
    #     c=colors[3],
    #     label="R2 train, SGD",
    # )

    # plt.legend()

    # plt.title("Comparing FFNN methods")
    # plt.xlabel("Iterations")
    # plt.ylabel("R2 score")
    # plt.ylim(0.0, 1.0)
    # plt.savefig("figures/nn_franke_14_r2_gd_sgd_comparison.eps")
    # plt.clf()
    # # plt.show()

    # print(f"Best epoch: {best_epoch}, MSE: {min_mse}")

    """Best nn for franke data 14"""
    # epochs = [50, 100, 200, 400, 600, 1000, 1250, 1500, 1750, 2000]

    # epochs_plot = []
    # mse_test_list = []
    # mse_train_list = []
    # for epoch in epochs:
    #     nn = build_neural_network(
    #         input_size=X_train.shape[1],
    #         output_size=z_train.shape[1],
    #         n_hidden_layer=2,
    #         n_hidden_nodes=30,
    #         cost_function=mse,
    #         cost_function_derivative=mse_derivative,
    #         activation_function="elu",
    #         last_activation="linear",
    #         optimizer="adam",
    #         regularization=1.0,
    #         learning_rate=1e-3,
    #     )

    #     nn.train(
    #         X_train,
    #         z_train,
    #         epochs=epoch,
    #         method="sgd",
    #         minibatch_size=10,
    #     )

    #     z_tilde = nn.predict(X_train)
    #     z_pred = nn.predict(X_test)

    #     mse_train = mse(z_train, z_tilde)
    #     mse_test = mse(z_test, z_pred)

    #     mse_test_list.append(mse_test)
    #     mse_train_list.append(mse_train)
    #     epochs_plot.append(epoch)

    #     # if mse_test < 1.5e-2:
    #     #     print(f"Epoch: {epoch}, MSE: {mse_test}")
    #     #     break

    # fig, ax = plt.subplots(figsize=(10, 6))

    # ax.plot(epochs_plot, mse_train_list, c=colors[0], label="MSE train, batchsize: 10")
    # ax.plot(
    #     epochs_plot,
    #     mse_test_list,
    #     c=colors[1],
    #     linestyle="--",
    #     label="MSE test, batchsize: 10",
    # )

    # # axins1 = zoomed_inset_axes(ax, zoom=2.4, loc="center right")
    # # axins1.plot(epochs_plot, mse_train_list, c=colors[0])
    # # axins1.plot(epochs_plot, mse_test_list, c=colors[1], linestyle="--")

    # epochs_plot = []
    # mse_test_list = []
    # mse_train_list = []
    # for epoch in epochs:
    #     nn = build_neural_network(
    #         input_size=X_train.shape[1],
    #         output_size=z_train.shape[1],
    #         n_hidden_layer=2,
    #         n_hidden_nodes=30,
    #         cost_function=mse,
    #         cost_function_derivative=mse_derivative,
    #         activation_function="elu",
    #         last_activation="linear",
    #         optimizer="adam",
    #         regularization=1.0,
    #         learning_rate=1e-3,
    #     )

    #     nn.train(
    #         X_train,
    #         z_train,
    #         epochs=epoch,
    #         method="sgd",
    #         minibatch_size=100,
    #     )

    #     z_tilde = nn.predict(X_train)
    #     z_pred = nn.predict(X_test)

    #     mse_train = mse(z_train, z_tilde)
    #     mse_test = mse(z_test, z_pred)

    #     mse_test_list.append(mse_test)
    #     mse_train_list.append(mse_train)
    #     epochs_plot.append(epoch)

    #     # if mse_test < 1.5e-2:
    #     #     print(f"Epoch: {epoch}, MSE: {mse_test}")
    #     #     break

    # ax.plot(epochs_plot, mse_train_list, c=colors[2], label="MSE train, batchsize: 100")
    # ax.plot(
    #     epochs_plot,
    #     mse_test_list,
    #     c=colors[3],
    #     linestyle="--",
    #     label="MSE test, batchsize: 100",
    # )

    # axins1.plot(epochs_plot, mse_train_list, c=colors[2])
    # axins1.plot(epochs_plot, mse_test_list, c=colors[3], linestyle="--")

    # SPECIFY THE LIMITS
    # x1, x2, y1, y2 = 900, 1001, 0.01, 0.015
    # axins1.set_xlim(x1, x2)
    # axins1.set_ylim(y1, y2)

    # plt.xticks(visible=False)
    # plt.yticks(visible=False)

    # ax.legend(loc="upper center")
    # # mark_inset(ax, axins1, loc1=4, loc2=3, fc="none", ec="0.5")
    # ax.set_xlabel("Epochs")
    # ax.set_ylabel("MSE")

    # plt.savefig("figures/nn_franke_2_10_100_batches.eps")
    # plt.clf()

    # plt.show()

    # print(np.min(mse_test_list))
