from pinn import *

if __name__ == "__main__":
    np.random.seed(2023)
    torch.manual_seed(2023)

    N = 40
    (
        X_train,
        X_test,
        x_mesh_train,
        x_mesh_test,
        t_mesh_train,
        t_mesh_test,
    ) = generate_data(N=N, split=True)

    model = PINN(80, 4, nn.Tanh())
    optimizer = optim.Adam(
        model.parameters(),
        lr=10**-2,
        weight_decay=10**-4,
    )
    epochs = 2000

    loss = train(X_train, optimizer, model, epochs=epochs)

    u_pred_train = model.trial(X_train[:, 0:1], X_train[:, 1:2])
    u_pred_train = u_pred_train.detach().cpu().numpy()
    u_pred_train = u_pred_train.reshape((32, 40))

    u_exact_train = analytical_solution(x_mesh_train, t_mesh_train)
    difference_train = abs(u_exact_train - u_pred_train)

    u_pred = model.trial(X_test[:, 0:1], X_test[:, 1:2])
    u_pred = u_pred.detach().cpu().numpy()
    u_pred = u_pred.reshape((8, 40))

    u_exact_test = analytical_solution(x_mesh_test, t_mesh_test)
    difference_test = abs(u_exact_test - u_pred)

    print(f"Training loss: {loss}")
    print(f"MSE train: {np.mean(difference_train**2)}")
    print(f"MSE test: {np.mean(difference_test**2)}")
