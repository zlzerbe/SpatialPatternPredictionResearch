import matplotlib as matplotlib
# %matplotlib notebook
import numpy as np
import matplotlib.pyplot as pl

import sys
np.set_printoptions(threshold=1000)
import torch


#Generates the discrete laplacian tensor of a tensor M
def discrete_laplacian(M):
    """Get the discrete Laplacian of matrix M"""
    L = -4 * M
    L += torch.roll(M, (0, -1), (0, 1))  # right neighbor
    L += torch.roll(M, (0, +1), (0, 1))  # left neighbor
    L += torch.roll(M, (-1, 0), (0, 1))  # top neighbor
    L += torch.roll(M, (+1, 0), (0, 1))  # bottom neighbor

    return L

#Updates the finite difference equation after time delta_t
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


#Plots the two concentrations
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
    pl.show()


#Hash mapping tuples of feed and kill rates to specific patterns
#Note: not all diffusions coefficients work for the f,k tuples in the map
patternHash = {

    (0.050, 0.063): "PatternKappa",
    (0.042, 0.059): "PatternDelta",
    (0.022, 0.059): "PatternEpsilon",
    (0.046, 0.0594): "PatternIota",
    (0.060, 0.062): "PatternGamma",
    (0.046, 0.0594): "PatternRho",
    (0.094, 0.063): "PatternEta",
    (0.009299, 0.030): "ripple",
    (0.0115, 0.033): "PatternAlpha",
    (0.049, 0.0597): "PatternBeta"


}




'''The below initialization method was determined
 to produce solid results for a wide range of parameter sets and patterns '''
def initialize_system(N, splotches):
    A = 4 * np.random.random((N, N))
    B = np.random.random((N, N))
    summed = A + B
    A = A / summed
    B = B / summed
    for i in range(splotches):
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


'''Generates .npz files containing the given equilibria tensors corresponding 
    to a specific pattern class. The npz files contain tensors for matrices A and B, as well as the 
    set of parameters that produced them. The method filters out system equilibria that have not 
    converged to a classified pattern or diverged to essentially 1's and 0's
    
    The number of finite difference update steps is hard coded to 10000'''
def equilibriaGenerate( diffusionA, diffusionB, feed, kill, iterations, patterns,serial, N = 200):

    delta_t = 1.0
    # Initializes tensors A and B
    A, B = initialize_system(N, 10)
    # set the diffusion coefficients
    DB = diffusionB
    DA = diffusionA
    for i in range(iterations):
        for t in range(10000):
            #update system until equilibria is reached
            A, B = gray_scott_update(A, B, DA, DB, feed, kill, delta_t)

        #Save the equilibria to file if they contain a meaningful pattern
        if (torch.std(A) > 0.00001) and (torch.std(B) > 0.00001):
            reactionParameters = [diffusionA, diffusionB, feed, kill]
            np.savez(str(serial) + patterns.get((feed, kill)) + "Equilibria" + str(i) + ".npz", EquilibriaA=A,
                     EquilibriaB=B, parameters=reactionParameters)
        #reinitialize system for next generation
        A, B = initialize_system(N, 10)


#Example data generation below

'''Number of simulations of a particular set of parameters (note: equilibria of the same parameters
will look similar but in fact will be slightly different)'''
n_instances = 75
#PatternDelta
#equilibriaGenerate(0.1, 0.04, 0.042, 0.059, n_instances, patternHash, 1)

#PatternKappa
#equilibriaGenerate(0.09, 0.03, 0.050, 0.063, n_instances, patternHash, 1)

#PatternEpsilon
#equilibriaGenerate(0.078, 0.013, 0.022, 0.059, n_instances, patternHash, 1)

#PatternIota
#equilibriaGenerate(0.1, 0.05, 0.046, 0.0594, n_instances, patternHash, 1)

#PatternGamma
#equilibriaGenerate(0.16, 0.08, 0.060, 0.062, n_instances, patternHash, 1)

#PatternRho
#equilibriaGenerate(0.053, 0.038, 0.046, 0.0594, nreps, patternHash)

#PatternEta
#equilibriaGenerate(0.0047, 0.0001, 0.094, 0.063 , nreps, patternHash)

#Patternripple
#equilibriaGenerate(0.087, 0.020, 0.009299, 0.030, n_instances, patternHash, 1)

#PatternAlpha
#equilibriaGenerate(0.0061, 0.005, 0.0115, 0.033, n_instances, patternHash, 2)

#PatternBeta
#equilibriaGenerate(0.01, 0.008, 0.049, 0.0597, nreps, patternHash)

