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

#Thinking ahead to the advection part of the model, we need to generate a NxNxmaxElev grid, where we randomly generate
#some topological noise



#Initializes two numpy arrays representing the concentration of A and B
# Then converts the two arrays to tensors to return
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
    #N2 = N // 2
    radius = r = int(N / 10.0)

    #A[N2 - r:N2 + r, N2 - r:N2 + r] = 0.50
    #B[N2 - r:N2 + r, N2 - r:N2 + r] = 0.25

    A = np.random.random((N,N))

    # A and B are NxN matrices of concentrations of A and B respectively
    tensorA = torch.from_numpy(A)
    tensorB = torch.from_numpy(B)
    return tensorA, tensorB

def getInitialConfig(N):
    A = 4 * np.random.random((N,N))
    B = np.random.random((N,N))
    summed = A + B
    tensorA = torch.from_numpy(A / summed)
    tensorB = torch.from_numpy(B / summed)
    return tensorA, tensorB




tensorA0, tensorB0 = get_initial_configuration(200, random_influence=0.4)
tensorA1, tensorB1 = get_initial_configuration(200, random_influence=0.4)

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




# Returns a uniformly distributed float in between the lower and upper bounds inputted
#Used to generate random noise for the initial conditions
def randomNoise(lower,upper):
    rng = np.random.default_rng()
    noise = rng.uniform(0,1)
    while( not((noise < upper) and (noise > lower))):
        noise = rng.uniform(0,1)
    return noise


#Hash mapping tuples of feed and kill rates to specific patterns
#Note: not all diffusions coefficients work for the f,k tuples in the map

'''(0.046, 0.0594): "PatternRho",
    (0.094, 0.063): "PatternEta",
    (0.087, 0.020): "PatternTauSigma",
    (0.0115, 0.033): "PatternAlpha",
    (0.049, 0.0597): "PatternBeta"'''

patternHash = {

    (0.050, 0.063): "PatternKappa",
    (0.042, 0.059): "PatternDelta",
    (0.022, 0.059): "PatternEpsilon",
    (0.046, 0.0594): "PatternIota",
    (0.060, 0.062): "PatternGamma"


}

# Generates a nxnx2xk tensor of groups of two, square tensors of concentrations of A and B given a certain
# set of parameters. k refers to how many of these groups of two we have.
# k is given by int( (maxDiffusion - minDiffusion) // stepSize)
# **NOTE: Not all sets of parameter produce valid equilibrium states.
N_simulation_steps = 10000

'''def equilibriaGenerate(diffusionA, diffusionB, feed, kill, iterations, patterns):
    # set necessary parameters and initialize the concentration matrices
    delta_t = 1.0
    n = 200
    #Initializes tensors A and B
    #A, B = getInitialConfig(n)
    A,B = get_initial_configuration(n)
    #set the diffusion coefficients
    DB = diffusionB
    DA = diffusionA
    #Correct dimensions for the big tensors storing all the equilibria tensors
    #savedTensors = torch.zeros([n, n, 2, numberOfArraysToSave])

    for i in range(iterations):
        print(i)
        # use standard deviation across A and B, if less than 10 ^ -12
        #while torch.std(A) < (10^-12) and torch.std(B) < (10^-12):
        for t in range(N_simulation_steps):
            # update system until equilibria is reached
            A, B = gray_scott_update(A, B, DA, DB, feed, kill, delta_t)

        # Save the pair of tensors to file number according to its loop iteration
        reactionParameters = [diffusionA, diffusionB, feed, kill]
        np.savez(patterns.get((feed, kill)) + "Equilibria" + str(i) + ".npz", EquilibriaA=A,
                         EquilibriaB=B, parameters=reactionParameters)
        A, B = get_initial_configuration(n)
    print("done")'''

'''After some tinkering, the below initialization method was determined
 to produce solid results for a wide range of parameter sets and patterns '''
def official_initial_config(N, splotches):
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


N_simulation_steps = 10000

'''Generates .npz files containing the given equilibria tensors correspopmnding 
    to a specific pattern class. The npz files contain tensors for matrices A and B, as well as the 
    set of parameters that produced them. The method filters out system equilibria that have not 
    converged to a classified pattern or diverged to essentially 1's and 0's'''
def equilibriaGenerate( diffusionA, diffusionB, feed, kill, iterations, patterns, N = 200):
    # set necessary parameters and initialize the concentration matrices
    delta_t = 1.0

    # Initializes tensors A and B
    # A, B = getInitialConfig(n)
    A, B = official_initial_config(N, 10)
    # set the diffusion coefficients
    DB = diffusionB
    DA = diffusionA
    # Correct dimensions for the big tensors storing all the equilibria tensors
    # savedTensors = torch.zeros([n, n, 2, numberOfArraysToSave])

    for i in range(iterations):
        print(i)
        for t in range(N_simulation_steps):
            # update system until equilibria is reached
            A, B = gray_scott_update(A, B, DA, DB, feed, kill, delta_t)

        # if the eq are not good:
        '''if (torch.std(A) < (10 ^ -20) or torch.std(B) < (10 ^ -20)):
            iterations += 1
        print(iterations)'''
        # Save the pair of tensors to file number according to its loop iteration
        reactionParameters = [diffusionA, diffusionB, feed, kill]
        np.savez(patterns.get((feed, kill)) + "Equilibria" + str(i) + ".npz", EquilibriaA=A,
                 EquilibriaB=B, parameters=reactionParameters)
        A, B = official_initial_config(N, 10)


#Basic data generation below

#Number of simulations of a particular set of parameters (note: equilibria of the same parameters
#will look similar but in fact will be slightly different)
n_instances = 75
#PatternDelta
#equilibriaGenerate(0.1, 0.04, 0.042, 0.059, n_instances, patternHash)

#PatternKappa
#equilibriaGenerate(0.09, 0.03, 0.050, 0.063, n_instances, patternHash)

#PatternEpsilon
#equilibriaGenerate(0.078, 0.013, 0.022, 0.059, n_instances, patternHash)

#PatternIota
#equilibriaGenerate(0.1, 0.05, 0.046, 0.0594, n_instances, patternHash)

#PatternGamma
equilibriaGenerate(0.16, 0.08, 0.060, 0.062, n_instances, patternHash)

#PatternRho
#equilibriaGenerate(0.053, 0.038, 0.046, 0.0594, nreps, patternHash)

#PatternEta
#equilibriaGenerate(0.0047, 0.0001, 0.094, 0.063 , nreps, patternHash)

#PatternTauSigma
#equilibriaGenerate(0.004299, 0.02, 0.087, 0.020, nreps, patternHash)

#PatternAlpha
#equilibriaGenerate(0.0061, 0.005, 0.0115, 0.033, nreps, patternHash)

#PatternBeta
#equilibriaGenerate(0.01, 0.008, 0.049, 0.0597, nreps, patternHash)
