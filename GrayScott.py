import matplotlib as matplotlib
# %matplotlib notebook
import numpy as np
import matplotlib.pyplot as pl

import sys
np.set_printoptions(threshold=1000)
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor


A = np.ones((3,3))
A[1,1] = 0
A

right_neighbor = np.roll(A, # the matrix to permute
                         (0,-1), # we want the right neighbor, so we shift the whole matrix -1 in the x-direction)
                         (0,1), # apply this in directions (y,x)
                        )
right_neighbor


def discrete_laplacian(M):
    """Get the discrete Laplacian of matrix M"""
    L = -4 * M
    L += np.roll(M, (0, -1), (0, 1))  # right neighbor
    L += np.roll(M, (0, +1), (0, 1))  # left neighbor
    L += np.roll(M, (-1, 0), (0, 1))  # top neighbor
    L += np.roll(M, (+1, 0), (0, 1))  # bottom neighbor

    return L

discrete_laplacian(A)


def gray_scott_update(A, B, DA, DB, f, k, delta_t):
    """
    Updates a concentration configuration according to a Gray-Scott model
    with diffusion coefficients DA and DB, as well as feed rate f and
    kill rate k.
    """

    # Let's get the discrete Laplacians first
    LA = discrete_laplacian(A)
    LB = discrete_laplacian(B)

    # Now apply the update formula
    diff_A = (DA * LA - A * B ** 2 + f * (1 - A)) * delta_t
    diff_B = (DB * LB + A * B ** 2 - (k + f) * B) * delta_t

    A += diff_A
    B += diff_B

    return A, B


def get_initial_configuration(N, random_influence=0.2):
    """
    Initialize a concentration configuration. N is the side length
    of the (N x N)-sized grid.
    `random_influence` describes how much noise is added.
    """

    # We start with a configuration where on every grid cell
    # there's a lot of chemical A, so the concentration is high
    A = (1 - random_influence) * np.ones((N, N)) + random_influence * np.random.random((N, N))

    # Let's assume there's only a bit of B everywhere
    B = random_influence * np.random.random((N, N))

    # Now let's add a disturbance in the center
    N2 = N // 2
    radius = r = int(N / 10.0)

    A[N2 - r:N2 + r, N2 - r:N2 + r] = 0.50
    B[N2 - r:N2 + r, N2 - r:N2 + r] = 0.25

    # A and B are NxN matrices of concentrations of A and B respectively
    return A, B

def draw(A,B):
    """draw the concentrations"""
    # We get two subplots here. One for the Concentration of A and one for B
    fig, ax = pl.subplots(1,2,figsize=(5.65,4))
    ax[0].imshow(A, cmap='Greys')
    ax[1].imshow(B, cmap='Greys')
    ax[0].set_title('A')
    ax[1].set_title('B')
    ax[0].axis('off')
    ax[1].axis('off')


def saveGreyScaleArray(A, B):
    np.savez("GrayScottEquilibria.npz", A)
    np.savez("GrayScottEquilibria.npz", B)


def paramTests(minDiffusion, maxDiffusion, feed, kill, stepSize):
    # set necessary paramters and initalize the concentration matrices
    print("test")
    delta_t = 1.0
    A, B = get_initial_configuration(200, .5)
    DA = maxDiffusion
    DB = minDiffusion
    print("test")
    for i in range(int((maxDiffusion - minDiffusion) // stepSize)):

        for t in range(N_simulation_steps):
            # update system until equilibria is reached
            A, B = gray_scott_update(A, B, DA, DB, feed, kill, delta_t)

        saveGreyScaleArray(A, B)
        print("Saved array")
        # increment DA down and DB up
        DA -= 0.001
        DB += 0.001

# PARAMETERS

# update in time
delta_t = 1.0

# Diffusion coefficients
DA = 0.16
DB = 0.08

# define feed/kill rates
f = 0.060
k = 0.062

# grid size
N = 200

# simulation steps
N_simulation_steps = 10000

A, B = get_initial_configuration(200, .5)

for t in range(N_simulation_steps):
    A, B = gray_scott_update(A, B, DA, DB, f, k, delta_t)
print("test")
draw(A, B)

saveGreyScaleArray(A, B)
# use code below to then load the file of arrays
loaded_arrays = np.load("GrayScottEquilibria.npz", allow_pickle=True)
loaded_arrays.files
loaded_arrays['arr_0']

# DA, DB, f, k = 0.14, 0.06, 0.035, 0.065 # bacteria
#DA, DB, f, k = 0.05, 0.05, 0.02, 0.03  # bacteria
#A, B = get_initial_configuration(200)

#for t in range(N_simulation_steps):
  #  A, B = gray_scott_update(A, B, DA, DB, f, k, delta_t)

#draw(A, B)