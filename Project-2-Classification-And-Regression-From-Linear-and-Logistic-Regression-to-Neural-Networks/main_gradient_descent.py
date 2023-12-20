from functions import *
from GD import *
from NN import *


def f(x):
    return 1 + x + 3 * x**2 - 4 * x**3


def gridsearch_lr_lmbd_opt(x_train, x_test, y_train, y_test):
    """Grid search for learning rate, lambda and optimizer"""
    learning_rates = [float(i) for i in range(-6, 0)]

    lambdas = [float(i) for i in range(-6, 1)]
    print(learning_rates)
    optimizers = [None, "momentum", "adagrad", "rmsprop", "adam"]

    grid = np.zeros((len(optimizers), len(learning_rates), len(lambdas)))

    print(grid.shape)

    for i, optimizer in enumerate(optimizers):
        for j, learning_rate in enumerate(learning_rates):
            for k, lmbd in enumerate(lambdas):
                optimizer_object = add_optimizer(optimizer, 10**learning_rate)

                model = GradientDescent(
                    x_train,
                    y_train,
                    optimizer_object,
                    DerivativeCostRidge,
                    lmbd=10**lmbd,
                )
                model.train(epochs=1000)
                y_pred = model.predict(x_test)

                mse_test = mse(y_test, y_pred)

                grid[i, j, k] = mse_test

    for i, optimizer in enumerate(optimizers):
        data = grid[i]

        im = sns.heatmap(
            data,
            xticklabels=lambdas,
            yticklabels=learning_rates,
            cmap="YlGn_r",
            alpha=1.0,
            annot=True,
            linewidth=0.5,
            fmt=".1e",
            annot_kws={"fontsize": 9},
            vmin=np.min(grid),
            vmax=1.0,
        )

        if optimizer == None:
            optimizer = "no optimizer"

        im.set_title(f"MSE for {optimizer}")
        im.set_xlabel("lmbd $\lambda$")
        im.set_ylabel("Learning Rate")
        # plt.tight_layout()

        plt.savefig(f"figures/gridsearch_lr_lmbd_opt_GD_franke_{optimizer}.eps")
        plt.show()
        plt.clf()


def gridsearch_lr_lmbd_opt_sgd(x_train, x_test, y_train, y_test):
    """Grid search for learning rate, lambda and optimizer"""
    learning_rates = [float(i) for i in range(-6, -1)]
    lambdas = [float(i) for i in range(-6, 1)]
    optimizers = [None, "momentum", "adagrad", "rmsprop", "adam"]

    grid = np.zeros((len(optimizers), len(learning_rates), len(lambdas)))

    for i, optimizer in enumerate(optimizers):
        for j, learning_rate in enumerate(learning_rates):
            for k, lmbd in enumerate(lambdas):
                optimizer_object = add_optimizer(optimizer, 10**learning_rate)

                model = StochasticGradientDescent(
                    x_train,
                    y_train,
                    optimizer_object,
                    DerivativeCostRidge,
                    lmbd=10**lmbd,
                )
                model.train(epochs=250, batch_size=20)
                y_pred = model.predict(x_test)

                mse_test = mse(y_test, y_pred)

                grid[i, j, k] = mse_test

    for i, optimizer in enumerate(optimizers):
        data = grid[i]

        im = sns.heatmap(
            data,
            xticklabels=lambdas,
            yticklabels=learning_rates,
            cmap="YlGn_r",
            alpha=1.0,
            annot=True,
            linewidth=0.5,
            fmt=".1e",
            annot_kws={"fontsize": 9},
            vmin=np.min(grid),
            vmax=np.max(grid),
        )

        if optimizer == None:
            optimizer = "no optimizer"

        im.set_title(f"MSE for {optimizer}")
        im.set_xlabel("lmbd $\lambda$")
        im.set_ylabel("Learning Rate")
        # plt.tight_layout()

        plt.savefig(f"figures/gridsearch_lr_lmbd_opt_SGD_franke_{optimizer}.eps")
        plt.show()
        plt.clf()


def gridsearch_epoch_batchsize(x_train, x_test, y_train, y_test):
    epochs = [10, 50, 100, 1000, 3000, 5000]
    batchsizes = [10, 20, 30, 50, 75, 100]

    grid = np.zeros((len(epochs), len(batchsizes)))

    for i, epoch in enumerate(epochs):
        for j, batchsize in enumerate(batchsizes):
            model = StochasticGradientDescent(
                x_train,
                y_train,
                Adam(learning_rate=0.01),
                DerivativeCostRidge,
                lmbd=10 ** (-5),
            )
            model.train(epochs=epoch, batch_size=batchsize)
            y_pred = model.predict(x_test)

            mse_test = mse(y_test, y_pred)
            # mse_test = r2_score(y_test, y_pred)

            grid[i, j] = mse_test

    im = sns.heatmap(
        grid,
        xticklabels=batchsizes,
        yticklabels=epochs,
        cmap="YlGn_r",
        alpha=1.0,
        annot=True,
        linewidth=0.5,
        fmt=".1e",
        annot_kws={"fontsize": 9},
    )

    im.set_title(f"MSE")
    im.set_xlabel("Batchsize")
    im.set_ylabel("Epoch")
    plt.tight_layout()

    # plt.savefig(f"figures/{output_filename}")
    plt.savefig(f"figures/gridsearch_epoch_batchsize_SGD_franke.eps")
    plt.show()
    plt.clf()

    print(np.min(grid))


if __name__ == "__main__":
    np.random.seed(2023)

    # """Generate Polynomial Data"""
    # x = np.random.normal(-1, 1, 1000).reshape(-1, 1)
    # y = f(x)

    # X = design_matrix_1d(x, 3, intercept=True)

    # x_train, x_test, y_train, y_test, X_train, X_test = train_test_split(
    #     x, y, X, test_size=0.2, random_state=2023
    # )

    # r2_list = []
    # r2_train_list = []
    # epochs_list = []

    # for epoch in np.arange(10, 2000, 50):
    #     model = GradientDescent(
    #         X_train,
    #         y_train,
    #         Adam(learning_rate=0.1),
    #         DerivativeCostRidge,
    #         lmbd=10 ** (-4),
    #     )
    #     model.train(epochs=epoch)
    #     y_pred = model.predict(X_test)
    #     y_tilde = model.predict(X_train)

    #     r2_test = r2_score(y_test, y_pred)
    #     r2_train = r2_score(y_train, y_tilde)

    #     epochs_list.append(epoch)
    #     r2_list.append(r2_test)
    #     r2_train_list.append(r2_train)

    #     print(f"MSE: {mse(y_test, y_pred)}")

    # # xs, ypredicts = zip(*sorted(zip(x_test, y_pred)))

    # # plt.scatter(x_test, y_test, label="True", c=colors[0])
    # # plt.plot(xs, ypredicts, label="Predicted", c=colors[1])
    # # plt.legend()
    # # plt.show()
    # for i in range(1, len(r2_list)):
    #     if abs(r2_list[i] - r2_list[i - 1]) < 1e-4:
    #         point = i
    #         break

    #     point = i
    # plt.axvline(
    #     x=epochs_list[point], color=colors[4], linestyle="--", label="Convergence"
    # )

    # plt.plot(epochs_list, r2_list, linestyle="--", color=colors[0], label="Test")
    # plt.plot(epochs_list, r2_train_list, color=colors[1], label="Train")
    # plt.legend()
    # plt.title("R2 score polynomial data")
    # plt.xlabel("Epochs")
    # plt.ylabel("R2 score")
    # # plt.show()
    # plt.savefig("figures/epochs_r2_gd_poly.eps")
    # plt.clf()

    # gridsearch_lr_lmbd_opt(X_train, X_test, y_train, y_test)
    # gridsearch_lr_lmbd_opt_sgd(X_train, X_test, y_train, y_test)
    # gridsearch_epoch_batchsize(X_train, X_test, y_train, y_test)

    x, y, z = generate_franke_data(noise=True, step_size=0.05, reshape=True)

    X = design_matrix_2d(x, y, 7, intercept=True)

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

    # # gridsearch_lr_lmbd_opt(X_train, X_test, z_train, z_test)
    # # gridsearch_lr_lmbd_opt_sgd(X_train, X_test, z_train, z_test)
    # # gridsearch_epoch_batchsize(X_train, X_test, z_train, z_test)

    # r2_list = []
    # r2_train_list = []
    # epochs_list = []

    # for epoch in np.arange(10, 10011, 500):
    #     model = GradientDescent(
    #         X_train,
    #         z_train,
    #         Adam(learning_rate=0.01),
    #         DerivativeCostOLS,
    #         lmbd=10 ** (-4),
    #     )
    #     model.train(epochs=epoch)  # , batch_size=30)
    #     y_pred = model.predict(X_test)
    #     y_tilde = model.predict(X_train)

    #     r2_test = r2_score(z_test, y_pred)
    #     r2_train = r2_score(z_train, y_tilde)

    #     epochs_list.append(epoch)
    #     r2_list.append(r2_test)
    #     r2_train_list.append(r2_train)

    #     print(f"MSE: {mse(z_test, y_pred)}")

    # # xs, ypredicts = zip(*sorted(zip(x_test, y_pred)))

    # # plt.scatter(x_test, y_test, label="True", c=colors[0])
    # # plt.plot(xs, ypredicts, label="Predicted", c=colors[1])
    # # plt.legend()
    # # plt.show()
    # # for i in range(1, len(r2_list)):
    # #     if abs(r2_list[i] - r2_list[i - 1]) < 1e-4:
    # #         point = i
    # #         break

    # #     point = i
    # # plt.axvline(
    # #     x=epochs_list[point], color=colors[4], linestyle="--", label="Convergence"
    # # )

    # plt.plot(epochs_list, r2_list, linestyle="--", color=colors[0], label="Test")
    # plt.plot(epochs_list, r2_train_list, color=colors[1], label="Train")
    # plt.legend()
    # plt.title("R2 score Franke data")
    # plt.xlabel("Epochs")
    # plt.ylabel("R2 score")

    # plt.ylim(0.0, 1.0)
    # plt.show()
    # plt.savefig("figures/epochs_r2_gd_franke_gd.eps")
    # plt.clf()

    # mse_list_test_ridge = []
    # mse_list_train_ridge = []
    # mse_list_train_ridge_8 = []
    # mse_list_test_ridge_8 = []
    # mse_list_test_ols = []
    # mse_list_train_ols = []
    # epochs_list = []

    # for epoch in np.arange(10, 20010, 500):
    #     model = GradientDescent(
    #         X_train,
    #         z_train,
    #         Adam(learning_rate=0.01),
    #         DerivativeCostRidge,
    #         lmbd=10 ** (-4),
    #     )
    #     model.train(epochs=epoch)  # , batch_size=30)
    #     y_pred = model.predict(X_test)
    #     y_tilde = model.predict(X_train)

    #     mse_list_train_ridge.append(mse(z_train, y_tilde))
    #     mse_list_test_ridge.append(mse(z_test, y_pred))
    #     epochs_list.append(epoch)

    #     model = GradientDescent(
    #         X_train,
    #         z_train,
    #         Adam(learning_rate=0.01),
    #         DerivativeCostRidge,
    #         lmbd=10 ** (-8),
    #     )
    #     model.train(epochs=epoch)  # , batch_size=30)
    #     y_pred = model.predict(X_test)
    #     y_tilde = model.predict(X_train)

    #     mse_list_train_ridge_8.append(mse(z_train, y_tilde))
    #     mse_list_test_ridge_8.append(mse(z_test, y_pred))

    #     model = GradientDescent(
    #         X_train,
    #         z_train,
    #         Adam(learning_rate=0.1),
    #         DerivativeCostOLS,
    #     )
    #     model.train(epochs=epoch)  # , batch_size=50)
    #     y_pred = model.predict(X_test)
    #     y_tilde = model.predict(X_train)

    #     mse_list_train_ols.append(mse(z_train, y_tilde))
    #     mse_list_test_ols.append(mse(z_test, y_pred))
    #     print(f"MSE: {mse(z_test, y_pred)}")

    # plt.plot(
    #     epochs_list,
    #     mse_list_train_ridge,
    #     color=colors[1],
    #     label="Train Ridge (lmbd=1e-4)",
    # )
    # plt.plot(
    #     epochs_list,
    #     mse_list_test_ridge,
    #     linestyle="--",
    #     color=colors[0],
    #     label="Test Ridge (lmbd=1e-4)",
    # )

    # plt.plot(
    #     epochs_list,
    #     mse_list_train_ridge_8,
    #     color=colors[5],
    #     label="Train Ridge (lmbd=1e-8)",
    # )
    # plt.plot(
    #     epochs_list,
    #     mse_list_test_ridge_8,
    #     color=colors[4],
    #     label="Test Ridge (lmbd=1e-8)",
    # )

    # plt.plot(epochs_list, mse_list_train_ols, color=colors[3], label="Train OLS")
    # plt.plot(
    #     epochs_list,
    #     mse_list_test_ols,
    #     linestyle="--",
    #     color=colors[2],
    #     label="Test OLS",
    # )

    # plt.legend()
    # plt.title("MSE score Franke data")
    # plt.xlabel("Epochs")
    # plt.ylabel("MSE")

    # plt.savefig("figures/epochs_mse_gd_olsvridge_franke.eps")
    # plt.show()

    model = GradientDescent(
        X_train,
        z_train,
        Adam(learning_rate=0.01),
        DerivativeCostOLS,
    )
    model.train(epochs=10000)
    y_pred = model.predict(X_test)

    print(f"MSE GD: {mse(z_test, y_pred)}")

    plot_surf(
        x_test,
        y_test,
        y_pred,
        title="Gradient Descent",
        filename="figures/gd_franke.eps",
    )

    nn = build_neural_network(
        input_size=X_train.shape[1],
        output_size=z_train.shape[1],
        n_hidden_layer=3,
        n_hidden_nodes=100,
        cost_function=mse,
        cost_function_derivative=mse_derivative,
        activation_function="sigmoid",
        last_activation="linear",
        optimizer="adam",
        regularization=1e-2,
        learning_rate=1e-2,
    )

    nn.train(
        X_train,
        z_train,
        epochs=1000,
        method="gd",
    )

    z_pred = nn.predict(X_test)

    print(f"MSE NN: {mse(z_test, z_pred)}")

    plot_surf(
        x_test,
        y_test,
        z_pred,
        title="Neural Network",
        filename="figures/nn_franke.eps",
    )

    plot_surf(
        x_test,
        y_test,
        z_test,
        title="Franke Function",
        filename="figures/franke.eps",
    )
