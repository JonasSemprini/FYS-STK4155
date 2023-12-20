import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import linear_model
from sklearn.utils import resample
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from mpl_toolkits.mplot3d import Axes3D
from imageio import imread
from numpy.random import normal, uniform
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from matplotlib import cm


from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning


def FrankeFunction(x, y, noise=True):
    if noise: 
        random_noise = np.random.normal(0, 0.1, x.shape)
    else: 
        random_noise = 0
    
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4 + random_noise

def generate_data(noise=True, step_size=0.05):
    # Arrange x and y
    x = np.arange(0, 1, step_size)
    y = np.arange(0, 1, step_size)

    # Create meshgrid of x and y
    X, Y = np.meshgrid(x, y)
    
    # Calculate the values for Franke function
    z = FrankeFunction(X, Y, noise=noise).flatten()
    
    # Flatten x and y for plotting
    x = X.flatten()
    y = Y.flatten()

    return x, y, z

def generate_dataRealRand(data, N, scale=True):
    # Get the dimensions of the original data
    rows, cols = np.shape(data)
    K = N//2
    
    # Randomly sample K rows and N columns from data
    sampled_rows = np.random.choice(rows, N, replace=False)
    sampled_cols = np.random.choice(cols, K, replace=False)
    
    # Create a new matrix filled with zeros
    z = np.zeros((N, K))
    
    # Fill z with sampled data from the original matrix
    for i, row_idx in enumerate(sampled_rows):
        for j, col_idx in enumerate(sampled_cols):
            z[i, j] = data[row_idx, col_idx]

    # Arrange x and y
    x = np.linspace(0, 1, N)
    y = np.linspace(0, 1, K)   
    
    # Create meshgrid of x and y
    X, Y = np.meshgrid(x, y)

    if scale:
        # Apply MinMax scaling
        scaler = MinMaxScaler()
        z = scaler.fit_transform(z)
    
    # Flatten x, y, and z for plotting
    x = X.flatten()
    y = Y.flatten()
    z = z.flatten()

    return x, y, z

def generate_dataReal(data, N=80, all=False):
    # Downsample the data
    if all:
        z = data
    else:
        z = data[::N, ::N]

    # Calculate the downsampled dimensions
    downsampled_rows, downsampled_cols = np.shape(z)

    # Create meshgrid of x and y based on downsampled dimensions
    x = np.linspace(0, 1, downsampled_cols)
    y = np.linspace(0, 1, downsampled_rows)

    X, Y = np.meshgrid(x, y)

    # Apply MinMax scaling
    scaler = MinMaxScaler()
    z = scaler.fit_transform(z)

    # Flatten x, y, and z for plotting
    x = X.flatten()
    y = Y.flatten()
    z = z.flatten()

    return x, y, z



def plot_surf(x, y, z, save=False, out_name="out.png"):
    # Create a 3D plot
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    # Plot the surface
    surf = ax.plot_trisurf(x, y, z, cmap=cm.coolwarm, linewidth=0, antialiased=False)

    # Customize the z axis.
    ax.set_zlim(-0.10, 1.40)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)

    if save:
        plt.savefig(f"plots/" + out_name)
    else: 
        plt.show()
