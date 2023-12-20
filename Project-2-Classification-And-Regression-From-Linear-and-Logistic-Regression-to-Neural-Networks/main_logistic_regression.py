from functions import *
from GD import *
from NN import *


def gridsearch_learningrate_logreg(x, y):
    learning_rates = [float(i) for i in range(-6, 0)]
    epochs = [10, 100, 500, 1000, 5000]

    grid = np.zeros((len(learning_rates), len(epochs)))

    for i, learning_rate in enumerate(learning_rates):
        for j, epoch in enumerate(epochs):
            model = LogReg(x, y, optimizer=Adam(learning_rate=10**learning_rate))
            model.train(epochs=epoch)
            y_pred = model.predict(x)

            accuracy_score = accuracy(y, y_pred)
            grid[i, j] = accuracy_score

    return grid


def gridsearch_learningrate_nn(x, y):
    learning_rates = [float(i) for i in range(-6, 0)]
    epochs = [10, 100, 500, 1000, 5000]

    grid = np.zeros((len(learning_rates), len(epochs)))

    for i, learning_rate in enumerate(learning_rates):
        for j, epoch in enumerate(epochs):
            nn = build_neural_network(
                input_size=x.shape[1],
                output_size=y.shape[1],
                n_hidden_layer=1,
                n_hidden_nodes=30,
                cost_function=cross_entropy_loss,
                cost_function_derivative=cross_entropy_loss_derivative,
                activation_function="sigmoid",
                optimizer="adam",
                learning_rate=10**learning_rate,
                last_activation="sigmoid",
                regularization=1.0,
            )
            nn.train(x, y, epochs=epoch, method="sgd", minibatch_size=20)
            y_pred = nn.classify(x)

            accuracy_score = accuracy(y, y_pred)
            grid[i, j] = accuracy_score

    return grid


def histogram_nn(X, yOR, iterations=100):
    accuracy_score_list = []
    input_size = X.shape[1]
    output_size = yOR.shape[1]

    for i in range(iterations):
        np.random.seed(i)

        nn = build_neural_network(
            input_size,
            output_size,
            n_hidden_layer=2,
            n_hidden_nodes=50,
            cost_function=cross_entropy_loss,
            cost_function_derivative=cross_entropy_loss_derivative,
            activation_function="sigmoid",
            optimizer="adam",
            last_activation="sigmoid",
            learning_rate=0.1,
            regularization=1.0,
        )

        nn.train(
            X,
            yOR,
            epochs=5000,
            method="sgd",
            minibatch_size=10,
        )
        ypred = nn.classify(X)

        accuracy_score = accuracy(yOR, ypred)
        accuracy_score_list.append(accuracy_score)

        print(f"{100*(i+1)/(iterations):.1f}%")

    bins = [0, 0.25, 0.5, 0.75, 1.0]
    plt.hist(
        accuracy_score_list,
        bins=5,
        lw=1,
        ec=colors[4],
        fc=colors[1],
        alpha=1.0,
        align="mid",
    )
    # plt.show()
    # ticks = [(patch._x0 + patch._x1) / 2 for patch in patches]
    ticklabels = ["0\%", "25%", "50%", "75%", "100%"]
    plt.xticks(ticks=bins, labels=ticklabels)
    plt.xlabel("Accuracy")
    plt.ylabel("Frequency")
    plt.title(f"Histogram of accuracy scores for {iterations} iterations")

    # plt.savefig("figures/cancer_nn_histogram.eps")
    # plt.clf()
    plt.show()


def plot_heatmap(
    grid, xticklabels, yticklabels, title, xlabel, ylabel, output_filename=None
):
    im = sns.heatmap(
        grid,
        xticklabels=xticklabels,
        yticklabels=yticklabels,
        cmap="YlGn",
        alpha=1.0,
        annot=True,
        linewidth=0.5,
        fmt=".2f",
        annot_kws={"fontsize": 9},
        vmin=0.0,
        vmax=1.0,
    )

    im.set_title(title)
    im.set_xlabel(xlabel)
    im.set_ylabel(ylabel)
    plt.tight_layout()

    plt.savefig(f"{output_filename}")
    # plt.show()
    plt.clf()


if __name__ == "__main__":
    np.random.seed(2023)

    learning_rates = [float(i) for i in range(-6, 0)]
    epochs = [10, 100, 500, 1000, 5000]

    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    yOR = np.array([[0], [1], [1], [1]])
    yXOR = np.array([[0], [1], [1], [0]])
    yAND = np.array([[0], [0], [0], [1]])

    # grid_learningrates_epochs_OR = gridsearch_learningrate_logreg(X, yOR)
    # grid_learningrates_epochs_XOR = gridsearch_learningrate_logreg(X, yXOR)
    # grid_learningrates_epochs_AND = gridsearch_learningrate_logreg(X, yAND)

    # np.save("grid_learningrates_epochs_OR.npy", grid_learningrates_epochs_OR)
    # np.save("grid_learningrates_epochs_XOR.npy", grid_learningrates_epochs_XOR)
    # np.save("grid_learningrates_epochs_AND.npy", grid_learningrates_epochs_AND)

    # grid_learningrates_epochs_OR = np.load("grid_learningrates_epochs_OR.npy")
    # grid_learningrates_epochs_XOR = np.load("grid_learningrates_epochs_XOR.npy")
    # grid_learningrates_epochs_AND = np.load("grid_learningrates_epochs_AND.npy")

    # print("Logitsic Regression")
    # print("OR")
    # print(grid_learningrates_epochs_OR)
    # print("XOR")
    # print(grid_learningrates_epochs_XOR)
    # print("AND")
    # print(grid_learningrates_epochs_AND)

    # plot_heatmap(
    #     grid_learningrates_epochs_OR,
    #     epochs,
    #     learning_rates,
    #     "OR",
    #     "Epochs",
    #     "Learning Rate",
    #     "figures/OR_logreg.eps",
    # )

    # plot_heatmap(
    #     grid_learningrates_epochs_XOR,
    #     epochs,
    #     learning_rates,
    #     "XOR",
    #     "Epochs",
    #     "Learning Rate",
    #     "figures/XOR_logreg.eps",
    # )

    # plot_heatmap(
    #     grid_learningrates_epochs_AND,
    #     epochs,
    #     learning_rates,
    #     "AND",
    #     "Epochs",
    #     "Learning Rate",
    #     "figures/AND_logreg.eps",
    # )

    # grid_learningrates_epochs_OR_nn = gridsearch_learningrate_nn(X, yOR)
    # grid_learningrates_epochs_XOR_nn = gridsearch_learningrate_nn(X, yXOR)
    # grid_learningrates_epochs_AND_nn = gridsearch_learningrate_nn(X, yAND)

    # np.save("grid_learningrates_epochs_OR_nn.npy", grid_learningrates_epochs_OR_nn)
    # np.save("grid_learningrates_epochs_XOR_nn.npy", grid_learningrates_epochs_XOR_nn)
    # np.save("grid_learningrates_epochs_AND_nn.npy", grid_learningrates_epochs_AND_nn)

    # grid_learningrates_epochs_OR_nn = np.load("grid_learningrates_epochs_OR_nn.npy")
    # grid_learningrates_epochs_XOR_nn = np.load("grid_learningrates_epochs_XOR_nn.npy")
    # grid_learningrates_epochs_AND_nn = np.load("grid_learningrates_epochs_AND_nn.npy")

    # print("Feed Forward Neural Network")
    # print("OR")
    # print(grid_learningrates_epochs_OR_nn)
    # print("XOR")
    # print(grid_learningrates_epochs_XOR_nn)
    # print("AND")
    # print(grid_learningrates_epochs_AND_nn)

    # plot_heatmap(
    #     grid_learningrates_epochs_OR_nn,
    #     epochs,
    #     learning_rates,
    #     "OR",
    #     "Epochs",
    #     "Learning Rate",
    #     "figures/OR_nn.eps",
    # )

    # plot_heatmap(
    #     grid_learningrates_epochs_XOR_nn,
    #     epochs,
    #     learning_rates,
    #     "XOR",
    #     "Epochs",
    #     "Learning Rate",
    #     "figures/XOR_nn.eps",
    # )

    # plot_heatmap(
    #     grid_learningrates_epochs_AND_nn,
    #     epochs,
    #     learning_rates,
    #     "AND",
    #     "Epochs",
    #     "Learning Rate",
    #     "figures/AND_nn.eps",
    # )

    # Best learning rate for OR: 0.1, epochs: 5000
    # histogram_nn(X, yOR, iterations=1000)
