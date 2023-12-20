import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from pinn import *

"""Set plotting parameters"""
plt.rcParams.update(
    {"text.usetex": True, "font.family": "serif", "font.serif": ["ComputerModern"]}
)
plt.rcParams["figure.figsize"] = (8, 6)
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


def forward_euler_diffusion(L, T_max, dx):
    """
    Forward Euler for diffusion equation
    """
    dt = 0.50 * dx**2

    x = np.arange(0, L + dx, dx)  # x array from 0 to L with dx space in between
    t = np.arange(0, T_max + dt, dt)  # t array from 0 to T_max with dt space in between

    Nx = len(x) - 1
    Nt = len(t) - 1

    u = np.zeros((Nt + 1, Nx + 1))
    u[0, :] = np.sin(np.pi * x)

    a = dt / dx**2
    mid = Nx // 2

    mid_T = [u[0, mid]]

    for n in range(0, Nt):
        for i in range(1, Nx):
            # u[i] = a * u_prev[i - 1] + (1 - 2 * a) * u_prev[i] + a * u_prev[i + 1]
            u[n + 1, i] = a * u[n, i - 1] + (1 - 2 * a) * u[n, i] + a * u[n, i + 1]

        u[n + 1, 0] = 0
        u[n + 1, Nx] = 0

        mid_T.append(u[n + 1, mid])

    return u, x, t


def analytical_solution(x, t):
    """
    Analytical solution to diffusion equation
    """
    return np.sin(np.pi * x) * np.exp(-np.pi**2 * t)


def plot_initial(L):
    """
    Plot initial condition
    """
    x = np.linspace(0, L, 1000)
    u = np.sin(np.pi * x)

    plt.plot(x, u, label="Initial condition")


def mse(u_true, u_pred):
    return np.mean((u_true - u_pred) ** 2)


if __name__ == "__main__":
    # Parameters
    L = 1.0  # Length of the rod
    T_max = 1.0  # Maximum time list
    dx_list = [0.1, 0.01]  # Spatial step size

    for i, dx in enumerate(dx_list):
        u_euler, x, t = forward_euler_diffusion(L, T_max=T_max, dx=dx)

        t1 = len(t) // 100
        t2 = (len(t) // 10) * 8

        plt.plot(x, u_euler[0, :], label=f"$t_0$ = {t[0]:.2f}$s$", color=colors[0])
        plt.plot(x, u_euler[t1, :], label=f"$t_1$ = {t[t1]:.2f}$s$", color=colors[1])
        for i in range(15):
            plt.plot(
                x,
                u_euler[t1 * ((2 * i) + 2), :],
                color="gray",
            )
        plt.plot(x, u_euler[t2, :], label=f"$t_2$ = {t[t2]:.2f}$s$", color=colors[2])

        plt.ylim(-0.1, 1.0)
        plt.xlabel("$x$")
        plt.ylabel("$u(x,t)$")
        plt.legend(loc="upper right")
        plt.savefig(f"figures/euler_dx_{dx}.eps", format="eps")
        plt.clf()

    dx = 0.01
    u_euler, x, t = forward_euler_diffusion(L, T_max=T_max, dx=dx)

    x_mesh, t_mesh = np.meshgrid(x, t)

    fig = plt.figure(figsize=(10, 7))
    plt.contourf(t_mesh, x_mesh, u_euler, 20, cmap="coolwarm")
    plt.colorbar(label="$u(x,t)$", rotation=90)
    plt.xlabel("$t$", labelpad=10)
    plt.ylabel("$x$", labelpad=10)
    plt.savefig("figures/euler_contour.eps", format="eps")
    plt.clf()
    # plt.show()

    # fig = plt.figure(figsize=(10, 7))
    ax = plt.axes(projection="3d")
    ax.plot_surface(t_mesh, x_mesh, u_euler, cmap="coolwarm")
    ax.set_xlabel("$t$", labelpad=10)
    ax.set_ylabel("$x$", labelpad=10)
    ax.zaxis.set_rotate_label(False)
    ax.set_zlabel("$u(x,t)$", labelpad=25)
    plt.savefig("figures/euler_3d.eps", format="eps")
    plt.clf()
    # plt.show()

    x = x.reshape(-1, 1)
    t = t.reshape(-1, 1)

    u_exact = (np.exp(-np.pi**2 * t)) @ (np.sin(np.pi * x)).T

    print(f"MSE: {mse(u_exact, u_euler)}")

    # u_preds = []

    # for n in range(len(T_max_list)):
    #     T_max = T_max_list[n]

    #     fig, ax = plt.subplots()
    #     axins = zoomed_inset_axes(ax, zoom=3, loc=2)

    #     for i in range(len(dx_list)):
    #         dx = dx_list[i]
    #         u, x = forward_euler_diffusion(L, T_max, dx)
    #         ax.plot(x, u, label=f"$\Delta$x = {dx}", color=colors[i])

    #         axins.plot(x, u, color=colors[i])

    #         u_preds.append(u)

    #     x = np.linspace(0, L, 1000)
    #     u_true = analytical_solution(x, T_max)
    #     ax.plot(x, u_true, label="Analytical solution", color=colors[2], linestyle="--")
    #     axins.plot(x, u_true, color=colors[2], linestyle="--")

    #     if n == 0:
    #         x1, x2, y1, y2 = 0.45, 0.55, 0.82, 0.98
    #         mark_inset(ax, axins, loc1=1, loc2=4, ec="0.5")
    #     elif n == 1:
    #         x1, x2, y1, y2 = 0.45, 0.55, -0.08, 0.08
    #         mark_inset(ax, axins, loc1=1, loc2=3, ec="0.5")

    #     axins.set_xlim(x1, x2)
    #     axins.set_ylim(y1, y2)

    #     plt.xticks(visible=False)
    #     plt.yticks(visible=False)

    #     ax.legend()
    #     ax.set_ylim(-0.1, 1.0)
    #     ax.set_title(f"Solution after {T_max} seconds")
    #     axins.grid()
    #     ax.set_xlabel("x")
    #     ax.set_ylabel("u(x,t)")

    #     plt.savefig(
    #         f"figures/euler_time_{T_max_list[n]}.eps",
    #         format="eps",
    #         dpi=1000,
    #         # bbox_inches="tight",
    #     )
    #     plt.clf()
