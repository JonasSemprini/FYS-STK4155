from pinn import *

if __name__ == "__main__":
    np.random.seed(2023)
    torch.manual_seed(2023)

    N = 40
    X, x_mesh, t_mesh = generate_data(N=N)

    print(X[:, 0:1].shape)

    model = PINN(80, 4, nn.Tanh())
    optimizer = optim.Adam(
        model.parameters(),
        lr=10**-2,
        weight_decay=10**-4,
    )
    epochs = 5000

    loss = train(X, optimizer, model, epochs=epochs)

    u_pred = model.trial(X[:, 0:1], X[:, 1:2])
    u_pred = u_pred.detach().cpu().numpy()
    u_pred = u_pred.reshape((N, N))

    u_exact = analytical_solution(x_mesh, t_mesh)

    difference = abs(u_exact - u_pred)

    print(f"LOSS: {loss}")
    print(f"MSE: {np.mean(difference**2)}")

    plot3D_Matrix(x_mesh, t_mesh, u_pred, save=True)

    # N = 20
    # X, x_mesh, t_mesh = generate_data(N=N, T_max=0.5)

    # model = PINN(80, 4, nn.Tanh())
    # optimizer = optim.Adam(
    #     model.parameters(),
    #     lr=10**-2,
    #     weight_decay=10**-4,
    # )
    # epochs = 500

    # loss = train(X, optimizer, model, epochs=epochs)

    # N = 40

    # X, x_mesh, t_mesh = generate_data(N=N, T_max=1.0)

    # u_pred = model.trial(X[:, 0:1], X[:, 1:2])
    # u_pred = u_pred.detach().cpu().numpy()
    # u_pred = u_pred.reshape((N, N))

    # plot3D_Matrix(x_mesh, t_mesh, u_pred)
