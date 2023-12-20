import torch
import torch.autograd as autograd
from torch import Tensor
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from sklearn.model_selection import train_test_split
import numpy as np


# Set default dtype to float32
torch.set_default_dtype(torch.float)

# Setting random seeds
torch.manual_seed(2023)
np.random.seed(2023)

# Device configuration
device = "mps" if torch.backends.mps.is_available() else "cpu"

"""Set plotting parameters"""
plt.rcParams.update(
    {"text.usetex": True, "font.family": "serif", "font.serif": ["ComputerModern"]}
)
plt.rcParams.update({"font.size": 20})
# plt.rc(
#     "axes",
#     facecolor="whitesmoke",
#     edgecolor="black",
#     axisbelow=True,
#     grid=True,
#     lw=1.5,
# )  # whitesmoke floralwhite
# plt.rc("grid", color="w", linestyle="solid", lw=2.0)
# plt.rc("lines", linewidth=2)
# plt.rc("legend", edgecolor="black", facecolor="w", fancybox=True)

colors = [
    "steelblue",
    "firebrick",
    "seagreen",
    "darkcyan",
    "mediumvioletred",
    "darkslategray",
]


def plot3D_Matrix(x, t, u, save=False):
    """
    Plot contour and 3D surface
    """
    X, T = x, t
    U_xt = u
    fig = plt.figure(figsize=(10, 7))

    plt.contourf(T, X, U_xt, 20, cmap="coolwarm")
    plt.colorbar(label="$u(x,t)$")
    plt.xlabel("$t$", labelpad=10)
    plt.ylabel("$x$", labelpad=10)
    if save:
        plt.savefig("figures/pred_contour.eps")
        plt.clf()
    else:
        plt.show()

    fig = plt.figure(figsize=(10, 7))
    ax = plt.axes(projection="3d")
    ax.plot_surface(T, X, U_xt, cmap="coolwarm")
    ax.set_xlabel("$t$", labelpad=10)
    ax.set_ylabel("$x$", labelpad=10)
    ax.set_zlabel("$u(x,t)$", labelpad=25)
    ax.zaxis.set_rotate_label(False)
    if save:
        plt.savefig("figures/pred_3d.eps")
        plt.clf()
    else:
        plt.show()


class PINN(nn.Module):
    """
    Physics-Informed Neural Network (PINN) class
    """

    def __init__(self, num_hidden_nodes, num_layers, activation_function):
        super(PINN, self).__init__()

        self.optimizer = None

        layers = [
            nn.Linear(2, num_hidden_nodes),
            activation_function,
        ]  # Input ---> 1st layer

        for _ in range(num_layers - 2):
            layers += [
                nn.Linear(num_hidden_nodes, num_hidden_nodes),
                activation_function,
            ]

        layers += [
            nn.Linear(num_hidden_nodes, 1)
        ]  # Last Layer ----> Output (no activation)

        self.model = nn.Sequential(*layers)
        self.model.to(device)

        for layer in self.model:
            if isinstance(layer, nn.Linear):
                torch.nn.init.xavier_normal_(layer.weight, gain=1.0)
                torch.nn.init.zeros_(layer.bias)

    def forward(self, X):
        """
        Forward pass
        """
        if torch.is_tensor(X) != True:
            X = torch.from_numpy(X)

        return self.model(X)

    def trial(self, x, t):
        """
        Trial Solution
        """
        g = (1 - t) * torch.sin(torch.pi * x) + x * (1 - x) * t * self.model(
            torch.hstack((x, t))
        )
        return g

    def Cost(self, X):
        """
        Cost function
        """
        x, t = X[:, 0:1], X[:, 1:2]

        x.requires_grad_(True)
        t.requires_grad_(True)

        g = self.trial(x, t)

        u_dt = torch.autograd.grad(
            g, t, grad_outputs=torch.ones_like(g).to(device), create_graph=True
        )[0]
        u_dx = torch.autograd.grad(
            g, x, grad_outputs=torch.ones_like(g).to(device), create_graph=True
        )[0]
        u_dxx = torch.autograd.grad(
            u_dx, x, grad_outputs=torch.ones_like(u_dx).to(device), create_graph=True
        )[0]

        cost = u_dt - u_dxx

        return cost

    def closure(self):
        """
        Closure method for optimizers
        """
        self.optimizer.zero_grad()
        loss = torch.mean(torch.square(self.Cost(X)))  # MSE
        loss.backward()

        return loss


def train(X, optimizer, model, epochs=100):
    """
    Training loop
    """
    model.optimizer = optimizer

    for i in range(epochs):
        optimizer.zero_grad()
        loss = torch.mean(torch.square(model.Cost(X)))  # MSE
        loss.backward()

        optimizer.step()

        if i % 100 == 0:
            print(f"loss = {loss.item()}, epoch={i}")

    return loss.item()


def analytical_solution(x, t):
    """
    Analytical solution
    """
    return np.sin(np.pi * x) * np.exp(-np.pi**2 * t)


def generate_data(N=40, L=1.0, T_max=1.0, split=False):
    """
    Generate data
    """
    x = np.linspace(0, L, N)
    t = np.linspace(0, T_max, N)

    x_mesh, t_mesh = np.meshgrid(x, t)

    if split:
        x_mesh_train, x_mesh_test, t_mesh_train, t_mesh_test = train_test_split(
            x_mesh, t_mesh, test_size=0.2, random_state=1
        )

        X_train_stack = np.hstack(
            (x_mesh_train.reshape(-1, 1), t_mesh_train.reshape(-1, 1))
        )

        X_train = torch.tensor(X_train_stack, dtype=torch.float32, device=device)

        X_test_stack = np.hstack(
            (x_mesh_test.reshape(-1, 1), t_mesh_test.reshape(-1, 1))
        )

        X_test = torch.tensor(X_test_stack, dtype=torch.float32, device=device)

        return X_train, X_test, x_mesh_train, x_mesh_test, t_mesh_train, t_mesh_test

    else:
        x_ = x_mesh.reshape(-1, 1)
        t_ = t_mesh.reshape(-1, 1)

        X_stack = np.hstack((x_, t_))

        X = torch.tensor(X_stack, dtype=torch.float32, device=device)

        return X, x_mesh, t_mesh


if __name__ == "__main__":
    N = 40
    epochs = 1000
    X, x_mesh, t_mesh = generate_data(N=N)

    activation_function = nn.Tanh()

    model = PINN(40, 3, activation_function)
    optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-3)

    train(X, optimizer, model, epochs=100)

    # X_plot = torch.tensor(np.hstack((x_.reshape(-1, 1), t_.reshape(-1, 1))), dtype=torch.float32)

    u_pred = model.trial(X[:, 0:1], X[:, 1:2])

    u_pred = u_pred.detach().cpu().numpy()  # Convert PyTorch tensor to NumPy array

    # Reshape for 2D plotting
    u_pred = u_pred.reshape((N, N))

    fig = plt.figure(figsize=(10, 7))
    ax = plt.axes(projection="3d")
    ax.plot_surface(
        t_mesh, x_mesh, u_pred, rstride=1, cstride=1, cmap="viridis", edgecolor="none"
    )
    ax.set_title("Approx")
    plt.show()
    print(np.shape(u_pred))

    u_exact = analytical_solution(x_mesh, t_mesh)

    difference = abs(u_exact - u_pred)

    plot3D_Matrix(x_mesh, t_mesh, difference)
