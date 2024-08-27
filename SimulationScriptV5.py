#% matplotlib notebook
import numpy as np
import matplotlib.pyplot as pl
import matplotlib as mpl
import sys

np.set_printoptions(threshold=1000)

from numpy import load
import torch

torch.set_printoptions(16)


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


def getInitialConfig2(N, splotches):
    A = 4 * np.random.random((N, N))
    B = np.random.random((N, N))
    summed = A + B
    A = A / summed
    B = B / summed
    for i in range(splotches):
        i = np.random.randint(11, N - 11)
        j = np.random.randint(11, N - 11)
        # index = np.random.randint(11, N - 11)
        A[i - 10: i + 10, j - 10: j] = 0
        i = np.random.randint(11, N - 11)
        j = np.random.randint(11, N - 11)
        B[i - 10: i + 10, j - 10: j] = 1

    tensorA = torch.from_numpy(A)
    tensorB = torch.from_numpy(B)
    return tensorA, tensorB


# Will return a size x N x N tensor of initialized gray-scott systems
def initializeSystemTensors(N, size, splotches=40):
    A = 4 * np.random.random((size, N, N))
    B = np.random.random((size, N, N))
    summed = A + B
    A = A / summed
    B = B / summed
    # add opposite binary plotches to the intiial system
    for i in range(splotches):
        i = np.random.randint(11, N - 11)
        j = np.random.randint(11, N - 11)
        # index = np.random.randint(11, N - 11)
        A[:, i - 10: i + 10, j - 10: j] = 0
        i = np.random.randint(11, N - 11)
        j = np.random.randint(11, N - 11)
        B[:, i - 10: i + 10, j - 10: j] = 1

    tensorA = torch.from_numpy(A)
    tensorB = torch.from_numpy(B)

    tensorA = torch.reshape(tensorA, (size, 1, N, N))
    tensorB = torch.reshape(tensorB, (size, 1, N, N))

    return tensorA.float(), tensorB.float()


def draw(A, B):
    """draw the concentrations"""
    # We get two subplots here. One for the Concentration of A and one for B
    fig, ax = pl.subplots(1, 2, figsize=(5.65, 4))
    ax[0].imshow(A, cmap='Greys')
    ax[1].imshow(B, cmap='Greys')
    ax[0].set_title('A')
    ax[1].set_title('B')
    ax[0].axis('off')
    ax[1].axis('off')


def getInitialConfig2(N, splotches):
    A = 4 * np.random.random((N, N))
    B = np.random.random((N, N))
    summed = A + B
    A = A / summed
    B = B / summed
    for i in range(splotches):
        i = np.random.randint(11, N - 11)
        j = np.random.randint(11, N - 11)
        # index = np.random.randint(11, N - 11)
        A[i - 10: i + 10, j - 10: j] = 0
        i = np.random.randint(11, N - 11)
        j = np.random.randint(11, N - 11)
        B[i - 10: i + 10, j - 10: j] = 1

    tensorA = torch.from_numpy(A)
    tensorB = torch.from_numpy(B)
    return tensorA, tensorB


def paramTests(diffusionA, diffusionB, feed, kill, increment, feedUB, feedLB):
    # set necessary paramters and initalize the concentration matrices
    passableParameters = []
    delta_t = 1.0
    N_simulation_steps = 10000
    qualityMatrices = 0
    # np.random.randint(splotchesLB, splotchesUB)
    A, B = getInitialConfig2(200, 40)

    print("Samples:")
    samples = int((feedUB - feedLB) / increment)
    print(str(samples))

    for i in range(samples):
        print(str(i))

        for t in range(N_simulation_steps):
            # update system until equilibria is reached
            A, B = gray_scott_update(A, B, diffusionA, diffusionB, feedLB, kill, delta_t)

        if (torch.std(A) > 0.00001 and torch.std(B) > 0.00001):
            qualityMatrices += 1
            passableParameters.append([diffusionA, diffusionB, feedLB, kill])
            print(A)
            print(B)
            print(torch.std(A))
            print(torch.std(B))
            # draw(A,B)
        feedLB += increment
        # kill += increment
        A, B = getInitialConfig2(200, 40)

    for i in range(samples):
        print(str(i))

        for t in range(N_simulation_steps):
            # update system until equilibria is reached
            A, B = gray_scott_update(A, B, diffusionA, diffusionB, feedUB, kill, delta_t)

        if (torch.std(A) > 0.00001 and torch.std(B) > 0.00001):
            qualityMatrices += 1
            passableParameters.append([diffusionA, diffusionB, feedLB, kill])
            # draw(A,B)
            print(A)
            print(B)
            print(torch.std(A))
            print(torch.std(B))
        # increment DA down and DB up
        feedUB -= increment
        # kill -= increment
        A, B = getInitialConfig2(200, 40)

    print(str(qualityMatrices))
    print("Percent usable:")
    print(str((qualityMatrices / (2 * samples)) * 100) + "%")
    return passableParameters

# paramList = paramTests(0.09, 0.03, 0.09, 0.04, 0.001, .97, .03)

##Feed Variance

# DEFINING CIRCULAR PADDING AND THEN A LAPLACIAN
class CircularPad2d(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        if type(dim) is int:
            self.dims = [dim for i in range(4)]
        elif type(dim) is list:
            self.dims = dim

    def forward(self, x):
        return torch.nn.functional.pad(x, pad=self.dims, mode='circular')


# manually setting input channels to see if it will help
grid_lap_layer = torch.nn.Conv2d(1, 1, kernel_size=3)
grid_lap_layer.weight = torch.nn.Parameter(
    # a special module
    torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]]).float().reshape((1, 1, 3, 3)), requires_grad=False)
grid_lap_layer.bias = torch.nn.Parameter(torch.zeros_like(grid_lap_layer.bias), requires_grad=False)

torch_discrete_laplacian = torch.nn.Sequential(
    CircularPad2d(1),
    grid_lap_layer
)


# A BATCHWISE gray-scott update in Torch.
def gray_scott_tensor_update(tA, tB, tDA, tDB, f, k, delta_t):
    LA = torch_discrete_laplacian(tA)
    LB = torch_discrete_laplacian(tB)
    f = f.reshape(-1, 1, 1, 1)
    k = k.reshape(-1, 1, 1, 1)
    tDA = tDA.reshape(-1, 1, 1, 1)
    tDB = tDB.reshape(-1, 1, 1, 1)

    # Now apply the update formula
    # print(tA.shape, tB.shape, tDA.shape,tDB.shape, f.shape, k.shape)

    diff_A = (tDA * LA - tA * torch.pow(tB, 2) + f * (1 - tA)) * delta_t
    diff_B = (tDB * LB + tA * torch.pow(tB, 2) - (k + f) * tB) * delta_t

    return tA + diff_A, tB + diff_B


'''
def parameterRangeTensor(N, diffusionA, diffusionB, feed, kill, samples, intervalRadius, splotches):
    delta_t = 1.0
    N_simulation_steps = 10000

    A = 4 * np.random.random((samples, N, N))
    B = np.random.random((samples, N, N))
    summed = A + B
    A = A / summed
    B = B / summed
    for i in range(splotches):
        i = np.random.randint(11, N - 11)
        j = np.random.randint(11, N - 11)
        # index = np.random.randint(11, N - 11)
        A[:,i - 10: i + 10, j - 10: j] = 0
        i = np.random.randint(11, N - 11)
        j = np.random.randint(11, N - 11)
        B[:,i - 10: i + 10, j - 10: j] = 1


    tensorA = torch.from_numpy(A)
    tensorB = torch.from_numpy(B)
    torch.reshape(tensorA, (1, 100, N, N))
    torch.reshape(tensorB, (1, 100, N, N))

    #tensor of parameters ranging from the lower bound of desired 'feed range' to the upper bound
    paramTensor = torch.linspace(feed - intervalRadius, feed + intervalRadius, samples)
    print(tensorA.shape)
    print(tensorB.shape)
    print(paramTensor.shape)



    return tensorA, tensorB, paramTensor'''


def patternRangeTesting(DA, DB, tFeed, kill, tA, tB):
    # necessary for tensor operations
    tDA = DA * torch.ones(tFeed.shape)
    tDB = DB * torch.ones(tFeed.shape)
    tKill = kill * torch.ones(tFeed.shape)
    # set necessary paramters and initalize the concentration matrices
    passableParameters = []
    delta_t = 1.0
    N_simulation_steps = 10000
    qualityMatrices = 0
    # np.random.randint(splotchesLB, splotchesUB)
    A, B = getInitialConfig2(200, 40)

    samples = tFeed.shape[0]

    for t in range(N_simulation_steps):
        print(t)
        # update system until equilibria is reached
        tA, tB = gray_scott_tensor_update(tA, tB, tDA, tDB, tFeed, tKill, delta_t)
    return tA, tB

    '''
        if (torch.std(A) > 0.000001 and torch.std(B) > 0.000001):
            qualityMatrices += 1
            passableParameters.append([diffusionA,diffusionB,feed - intervalRadius, kill])
            print(A)
            print(B)
            print(torch.std(A))
            print(torch.std(B))
            #draw(A,B)
        ## Feed variance
        feed += increment
        #kill += increment
        A, B = getInitialConfig2(200, 40)'''

def parameterRangeTensor(N, diffusionA, diffusionB, feed, kill, samples, intervalRadius, splotches):
        delta_t = 1.0
        N_simulation_steps = 10000

        A = 4 * np.random.random((samples, N, N))
        B = np.random.random((samples, N, N))
        summed = A + B
        A = A / summed
        B = B / summed
        # add opposite binary plotches to the intiial system
        for i in range(splotches):
            i = np.random.randint(11, N - 11)
            j = np.random.randint(11, N - 11)
            # index = np.random.randint(11, N - 11)
            A[:, i - 10: i + 10, j - 10: j] = 0
            i = np.random.randint(11, N - 11)
            j = np.random.randint(11, N - 11)
            B[:, i - 10: i + 10, j - 10: j] = 1

        tensorA = torch.from_numpy(A)
        tensorB = torch.from_numpy(B)
        # stack the A and B matrices
        # eqTensors = torch.stack((tensorA, tensorB))
        # eqTensors = torch.reshape(eqTensors, (samples, 2, N, N))

        tensorA = torch.reshape(tensorA, (samples, 1, N, N))
        tensorB = torch.reshape(tensorB, (samples, 1, N, N))

        # tensor of parameters ranging from the lower bound of desired 'feed range' to the upper bound
        paramTensor = torch.linspace(feed - intervalRadius, feed + intervalRadius, samples)
        # print(eqTensors.shape)
        print(paramTensor.shape[0])

        return tensorA.float(), tensorB.float(), paramTensor


def feedRangeTensor(feed, intervalRadius, samples):
    # tensor of parameters ranging from the lower bound of desired 'feed range' to the upper bound
    paramTensor = torch.linspace(feed - intervalRadius, feed + intervalRadius, samples)

    return paramTensor


def equilibriaRangeGenerate(diffusionA, diffusionB, feed, kill, iterations, label, intervalRadius=0.03, N=200,
                            splotches=45):
    # set necessary parameters and initialize the concentration matrices
    delta_t = 1.0
    N_simulation_steps = 10000
    tA, tB, params = parameterRangeTensor(200, diffusionA, diffusionB, feed, kill, iterations, intervalRadius,
                                          splotches)

    tDA = diffusionA * torch.ones(params.shape)
    tDB = diffusionB * torch.ones(params.shape)
    tKill = kill * torch.ones(params.shape)

    for t in range(N_simulation_steps):
        if t % 100 == 0:
            print(t)
        # update system until equilibria is reached
        tA, tB = gray_scott_tensor_update(tA, tB, tDA, tDB, params, tKill, delta_t)

    stdA = torch.std(tA, (2, 3))
    stdB = torch.std(tB, (2, 3))
    # if it passes, save to file
    for i in range(stdA.shape[0]):
        if stdA[i] > 0.0001 and stdB[i] > 0.0001:
            reactionParameters = [tDA[i], tDB[i], params[i], tKill[i]]
            # Save the pair of tensors to file number according to its loop iteration
            # probably need a better way of naming the files
            np.savez(label + "Equilibria" + str(i) + ".npz", EquilibriaA=tA[i],
                     EquilibriaB=tB[i], parameters=reactionParameters)

    print("Done")







