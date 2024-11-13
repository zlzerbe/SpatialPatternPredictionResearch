import os
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
import matplotlib.pyplot as plt

#Manage computing source
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")


patternHash = {

    (0.058, 0.063): "PatternKappa",
    (0.042, 0.059): "PatternDelta",
    (0.022, 0.059): "PatternEpsilon",
    (0.046, 0.0594): "PatternRho",
    (0.094, 0.063): "PatternEta",
    (0.087, 0.020): "PatternTauSigma",
    (0.0115, 0.033): "PatternAlpha",
    (0.049, 0.0597): "PatternBeta"
}

# Generates a nxnx2xk tensor of groups of two, square tensors of concentrations of A and B given a certain
# set of parameters. k refers to how many of these groups of two we have.
# k is given by int( (maxDiffusion - minDiffusion) // stepSize)
# **NOTE: Not all sets of parameter produce valid equilibrium states.
N_simulation_steps = 10000

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


N_simulation_steps = 10000


def equilibriaGenerate( diffusionA, diffusionB, feed, kill, iterations, patterns, N = 200):
    # set necessary parameters and initialize the concentration matrices
    delta_t = 1.0

    # Initializes tensors A and B
    # A, B = getInitialConfig(n)
    A, B = getInitialConfig2(N,40)
    # set the diffusion coefficients
    DB = diffusionB
    DA = diffusionA
    # Correct dimensions for the big tensors storing all the equilibria tensors
    # savedTensors = torch.zeros([n, n, 2, numberOfArraysToSave])

    for i in range(iterations):
        for t in range(N_simulation_steps):
            # update system until equilibria is reached
            A, B = gray_scott_update(A, B, DA, DB, feed, kill, delta_t)

        # if the eq are not good:
        if (torch.std(A) < 10 ^ -20 or torch.std(B) < 10 ^ -20):
            iterations += 1
        print(iterations)
        # Save the pair of tensors to file number according to its loop iteration
        reactionParameters = [diffusionA, diffusionB, feed, kill]
        np.savez(patterns.get((feed, kill)) + "Equilibria" + str(i) + ".npz", EquilibriaA=A,
                 EquilibriaB=B, parameters=reactionParameters)
        A, B = getInitialConfig2(N,40)


# model class

class CNN(nn.Module):

    def __init__(self, output_dim=2):
        super(CNN, self).__init__()
        # 1st convolution
        conv1 = nn.Conv2d(2, 8, (3, 3), padding=1)
        rel1 = nn.ReLU()
        pool1 = nn.MaxPool2d((3, 3), 2, padding=1)
        # 100,100
        # 2nd convolution
        conv2 = nn.Conv2d(8, 16, (3, 3), padding=1)
        rel2 = nn.ReLU()
        pool2 = nn.MaxPool2d((3, 3), 2, padding=1)
        # 50x50
        # 3rd convolution
        conv3 = nn.Conv2d(16, 32, (3, 3), padding=1)
        rel3 = nn.ReLU()
        pool3 = nn.MaxPool2d((3, 3), 2, padding=1)
        # 50x50
        # 4th convolution
        conv4 = nn.Conv2d(32, 64, (3, 3), padding=1)
        rel4 = nn.ReLU()
        pool4 = nn.MaxPool2d((3, 3), 2, padding=1)

        # 5th convolution
        conv5 = nn.Conv2d(64, 64, (3, 3), padding=1)
        rel5 = nn.ReLU()
        pool5 = nn.MaxPool2d((3, 3), 2, padding=1)

        # 6th convolution
        conv6 = nn.Conv2d(64, 64, (3, 3), padding=1)
        rel6 = nn.ReLU()
        pool6 = nn.MaxPool2d((3, 3), 2, padding=1)

        # make a module lists of the different types of layers
        self.conv_layers = torch.nn.ModuleList([conv1, conv2, conv3, conv4, conv5, conv6])
        self.pool_layers = torch.nn.ModuleList([pool1, pool2, pool3, pool4, pool5, pool6])
        self.rels = torch.nn.ModuleList([rel1, rel2, rel3, rel4, rel5, rel6])

        # 25x25
        # Flatten the 25x25 dimesion into a 1x625x32
        self.flat = nn.Flatten(1, 3)
        # Linear transformations to extract one value
        self.lin1 = nn.Linear(1024, 512)
        self.act1 = nn.ELU()
        self.lin2 = nn.Linear(512, 64)
        self.act2 = nn.ELU()
        self.lin3 = nn.Linear(64, output_dim)
        # final layer for this one, apply sigmoid to each entry to guarantee a positive value
        # sampled parameters need to be normally distributed
        #
        self.final = torch.nn.Sigmoid()

    def forward(self, x):
        for conv, pool, act in zip(self.conv_layers, self.pool_layers, self.rels):
            x = conv(x)
            # print(x.shape)
            x = act(x)
            # print(x.shape)
            x = pool(x)
            # print(x.shape)

        x = self.flat(x)
        # print(x.shape)

        x = self.lin1(x)
        x = self.act1(x)

        x = self.lin2(x)
        x = self.act2(x)

        x = self.lin3(x)
        x = self.final(x)

        # x = x.max(-1)[0]
        # print(x.shape)

        # x = x.max(-1)[0]
        # print(x.shape)
        return x

        # Softmax- turns arbitrary numbers into probabilities

        # Argmax– the index of the largest logit

        # max largest logit


# Generates the discrete laplacian tensor of a tensor M
def discrete_laplacian(M):
    """Get the discrete Laplacian of matrix M"""
    L = -4 * M
    L += torch.roll(M, (0, -1), (0, 1))  # right neighbor
    L += torch.roll(M, (0, +1), (0, 1))  # left neighbor
    L += torch.roll(M, (-1, 0), (0, 1))  # top neighbor
    L += torch.roll(M, (+1, 0), (0, 1))  # bottom neighbor

    return L


# Updates the finite difference equation after time delta_t
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


# returns tensors of initialized matrices A and B
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
    # N2 = N // 2
    radius = r = int(N / 10.0)

    # A[N2 - r:N2 + r, N2 - r:N2 + r] = 0.50
    B  # [N2 - r:N2 + r, N2 - r:N2 + r] = 0.25

    A = np.random.random((N, N))

    # A and B are NxN matrices of concentrations of A and B respectively
    tensorA = torch.from_numpy(A)
    tensorB = torch.from_numpy(B)
    return tensorA, tensorB


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


grid_lap_layer = torch.nn.Conv2d(1, 1, kernel_size=3)

grid_lap_layer.weight = torch.nn.Parameter(
    torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]]).float().reshape((1, 1, 3, 3)), requires_grad=False)

grid_lap_layer.bias = torch.nn.Parameter(torch.zeros_like(grid_lap_layer.bias), requires_grad=False)

torch_discrete_laplacian = torch.nn.Sequential(
    CircularPad2d(1),
    grid_lap_layer
).to(device)


# A BATCHWISE gray-scott update in Torch.
def gray_scott_tensor_update(tA, tB, tDA, tDB, f, k, delta_t):
    # Laplacian is a Circular padding of the tensors followed by a convolution
    LA = torch_discrete_laplacian(tA)
    LB = torch_discrete_laplacian(tB)

    # print(tDA.shape, LA.shape)

    # Now apply the update formula
    tDA = tDA.reshape(-1, 1, 1, 1)
    tDB = tDB.reshape(-1, 1, 1, 1)
    f = f.reshape(-1, 1, 1, 1)
    k = k.reshape(-1, 1, 1, 1)

    diff_A = (tDA * LA - tA * torch.pow(tB, 2) + f * (1 - tA)) * delta_t
    diff_B = (tDB * LB + tA * torch.pow(tB, 2) - (k + f) * tB) * delta_t

    return tA + diff_A, tB + diff_B


# batch is Bx2xNxN
# pred is Bx4
# bY is truth parameters
# Question, where are we getting the equilibria

def newLoss(bX, pred, bY, batchSize):
    delta_t = 1.0
    tensorA, tensorB = bX.split(1, 1)

    tensorA = tensorA.reshape((bX.shape[0], 1, 200, 200))
    tensorB = tensorB.reshape((bX.shape[0], 1, 200, 200))

    # print(tensorA.dtype)

    truthA, truthB = gray_scott_tensor_update(tensorA, tensorB, bY[:, 0], bY[:, 1], bY[:, 2], bY[:, 3], delta_t)

    predA, predB = gray_scott_tensor_update(tensorA, tensorB, pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3], delta_t)

    return torch.nn.functional.mse_loss(predA, truthA) + torch.nn.functional.mse_loss(predB, truthB)

# MSE between pred and true labels
# Build loss
# take the parameters predicted by model, plug into a PDE update step, the difference between the eq
# generated by the truth vector and that generated by the pred vector is what we want to minimize
#

#
# Computing cluster password – tigers214834 user – zzerbe

os.path.join


def loadEquilibriumFiles(directory):
    # listDirectory = os.fsencode(directory)
    loaded_tensors = []
    param_labels = []
    # label_mapping = {"Delta":0, "Kappa":1, "Rho":2, "Epsilon":3, "Eta":4, "TauSigma":5,"Alpha":6,"Beta":7}
    for file in os.listdir(directory):
        # print(file)
        if file.endswith(".npz"):

            try:
                arr = np.load(os.path.join(directory, file), allow_pickle=True)
                # print(arr["EquilibriaA"])
                # print(arr["EquilibriaB"])
                # print(arr["parameters"])
                loaded_tensors.append(torch.tensor(
                    np.stack([torch.from_numpy(arr["EquilibriaA"]), torch.from_numpy(arr["EquilibriaB"])])
                ))
                param_labels.append(torch.from_numpy(arr["parameters"]))

                # print(len(loaded_tensors), len(param_labels))
            except KeyError:
                print("Key does not exist")
            except FileNotFoundError:
                print("file not found")

    return torch.squeeze(torch.stack(loaded_tensors)), torch.stack(param_labels)


EquilibriaTensors, TruthLabels = loadEquilibriumFiles("/home/zzerbe")

print(TruthLabels)

# poss_labels = int(TruthLabels.max()) + 1
# print(poss_labels)
# making the model

def train_loop(dataloader, model, optimizer):
    size = len(dataloader.dataset)
    # Set the model to training mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.train()
    current = 0
    for batch in loader:
        bX, bY = batch
        # print(bX.shape, bY.shape)
        bX = bX.to(device)
        bY = bY.to(device)
        pred = model(bX)
        # print(bX[:,0,:,:].shape)
        # bX.reshape()
        loss = newLoss(bX, pred, bY, 32)
        # now bX is a tensor of size (B, 2, K, K)
        # and bY is a tensor of size (B, )

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        current += len(bX)
        # after every hundred batches
        if current % 100 == 0:
            loss = loss.item()
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


# Test/Validation Loop


def test_loop(dataloader, model):
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss = 0
    correct = 0

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for bX, bY in dataloader:
            bX = bX.to(device)
            bY = bY.to(device)
            pred = model(bX)

            # print(pred.argmax(1), bY)
            test_loss += newLoss(bX, pred, bY, 32).item()
            # correct += (pred.argmax(1) == bY).type(torch.float).sum().item()

    test_loss /= num_batches
    # correct /= size
    print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

#instantiate a CNN called model
model = CNN(output_dim= 4).to(device)
#print(poss_labels, model)

#criterion = nn.NLLLoss()
#criterion = newLoss()



#adam optimizer

learning_rate = 1e-3
optimizer = torch.optim.Adam(model.parameters(), lr= learning_rate)


#Training loop

#NOTE that in this version I have hard coded our new loss function into the train and
#test loops



batch_size = 64

data = torch.utils.data.TensorDataset(EquilibriaTensors.float(), TruthLabels.float())
loader = torch.utils.data.DataLoader(data, shuffle=True, batch_size=32)



epochs = 200
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(loader, model, optimizer)
    test_loop(loader, model)
print("Done!")