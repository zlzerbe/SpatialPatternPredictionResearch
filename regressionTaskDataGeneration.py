'''This file is current for 11/27/24. The use of this file is meant to be for testing paramter sets for different patterns.
'''

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





def discrete_laplacian(M):
    """Get the discrete Laplacian of matrix M"""
    L = -4 * M
    L += np.roll(M, (0, -1), (0, 1))  # right neighbor
    L += np.roll(M, (0, +1), (0, 1))  # left neighbor
    L += np.roll(M, (-1, 0), (0, 1))  # top neighbor
    L += np.roll(M, (+1, 0), (0, 1))  # bottom neighbor

    return L



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


'''Initializes two NxN grids of concentration values of reactants A and B. 
Splotches of 100% A and B are added to their opposite grids for more noise. This tends
to help the systems better develop a pattern.
'''
def initialize_system(N, n_splotches):
    A = 4 * np.random.random((N, N))
    B = np.random.random((N, N))
    summed = A + B
    A = A / summed
    B = B / summed
    for i in range(n_splotches):
        i = np.random.randint(11, N - 11)
        j = np.random.randint(11, N - 11)
        # index = np.random.randint(11, N - 11)
        A[i - 10: i + 10, j - 10: j] = 0.2
        i = np.random.randint(11, N - 11)
        j = np.random.randint(11, N - 11)
        B[i - 10: i + 10, j - 10: j] = 0.8

    tensorA = torch.from_numpy(A)
    tensorB = torch.from_numpy(B)
    return tensorA, tensorB


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


# The following function will simulate the gray-scott reaction and draw the resulting grid
def simulate_feed_variance(N_simulation_steps, N, splotches, delta_t, DA, DB, kill, feedLower, feedUpper, increment, serial):
    quality_count = 0
    iterations = int((feedUpper - feedLower) / increment)
    A, B = initialize_system(N, splotches)
    for j in range(iterations):

        for t in range(N_simulation_steps):
            A, B = gray_scott_update(A, B, DA, DB, feedLower, kill, delta_t)

        print("A:", A, "B:", B)
        print(torch.std(A), torch.std(B))
        if (torch.std(A) > 0.00001) and (torch.std(B) > 0.00001):
            quality_count += 1
            reactionParameters = [DA, DB, feedLower, kill]
            np.savez( str(serial) + "Equilibria" + str(j) + ".npz", EquilibriaA=A,
                     EquilibriaB=B, parameters=reactionParameters)

        # increment feed up slightly
        feedLower += increment
        # re-initialize
        A, B = initialize_system(N, splotches)
    print("Quality simulations:" + str(quality_count) + "out of:" + str(iterations) + "total")



#GENERATION:

# PARAMETERS – for some example parameter sets, see the git repository

# time step
delta_t = 1.0

# Diffusion coefficients
DA = 0.078
DB = 0.013

# feed/kill rates
feedLower = 0.011
feedUpper = 0.04
k = 0.059


simulate_feed_variance(10000, 200, 10,delta_t,DA,DB,k, feedLower, feedUpper, 0.001, 9)




