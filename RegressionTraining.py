import os
import torch
from torch import nn
from torch.utils.data import DataLoader
import numpy as np

#---------------------
# Constants
#---------------------

current_path = os.getcwd()
OUTPUT_DIMENSION = 4
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")
os.path.join

#---------------------
# Helper functions
#---------------------

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

'''A BATCHWISE gray-scott update in Torch. Updates an entire batch of initialized systems one step using 
    a finite difference method
'''
def gray_scott_tensor_update(tA, tB, tDA, tDB, f, k, delta_t):
    # Laplacian is a Circular padding of the tensors followed by a convolution
    LA = torch_discrete_laplacian(tA)
    LB = torch_discrete_laplacian(tB)
    tDA = tDA.reshape(-1, 1, 1, 1)
    tDB = tDB.reshape(-1, 1, 1, 1)
    f = f.reshape(-1, 1, 1, 1)
    k = k.reshape(-1, 1, 1, 1)
    # Now apply the update formula
    diff_A = (tDA * LA - tA * torch.pow(tB, 2) + f * (1 - tA)) * delta_t
    diff_B = (tDB * LB + tA * torch.pow(tB, 2) - (k + f) * tB) * delta_t

    return tA + diff_A, tB + diff_B


# ** Custom Loss Function **
# take the parameters predicted by model, plug into a finite difference update step. The difference between the eq
# generated by the truth vector and that generated by the pred vector is what we want to minimize
# MSE between pred and true labels
def newLoss(bX, pred, bY):
    delta_t = 1.0
    tensorA, tensorB = bX.split(1, 1)

    tensorA = tensorA.reshape((bX.shape[0], 1, 200, 200))
    tensorB = tensorB.reshape((bX.shape[0], 1, 200, 200))

    truthA, truthB = gray_scott_tensor_update(tensorA, tensorB, bY[:, 0], bY[:, 1], bY[:, 2], bY[:, 3], delta_t)

    predA, predB = gray_scott_tensor_update(tensorA, tensorB, pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3], delta_t)

    return torch.nn.functional.mse_loss(predA, truthA) + torch.nn.functional.mse_loss(predB, truthB)


# Function to load Equilibria files from given directory and preprocess them for training
def loadEquilibriumFilesForRegressionTask(directory):
    loaded_tensors = []
    param_labels = []
    for file in os.listdir(directory):
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


#---------------------
#MODEL DEFINITION
#---------------------

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


#---------------------
# Training and Testing loop definitions
#---------------------

#training loop
def train_loop(dataloader, model, optimizer):
    size = len(dataloader.dataset)
    model.train()
    current = 0
    for batch in loader:
        bX, bY = batch
        bX = bX.to(device)
        bY = bY.to(device)
        pred = model(bX)
        # print(bX[:,0,:,:].shape)
        # bX.reshape()
        loss = newLoss(bX, pred, bY, 32)


        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        current += len(bX)
        # after every hundred batches
        if current % 100 == 0:
            loss = loss.item()
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


#Validation loop
def test_loop(dataloader, model):
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss = 0
    with torch.no_grad():
        for bX, bY in dataloader:
            bX = bX.to(device)
            bY = bY.to(device)
            pred = model(bX)
            test_loss += newLoss(bX, pred, bY, 32).item()
    test_loss /= num_batches
    print(f"Test Error: \n  Avg loss: {test_loss:>8f} \n")

#-----------------------
# Train and Test Script
#-----------------------

model = CNN(output_dim = OUTPUT_DIMENSION).to(device)
EquilibriaTensors, TruthLabels = loadEquilibriumFilesForRegressionTask(current_path)

# ** Hyperparameters **
learning_rate = 1e-4
batchSize = 64
epochs = 200


optimizer = torch.optim.Adam(model.parameters(), lr= learning_rate)
data = torch.utils.data.TensorDataset(EquilibriaTensors.float(), TruthLabels.float())
loader = torch.utils.data.DataLoader(data, shuffle=True, batch_size=batchSize)


for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(loader, model, optimizer)
    test_loop(loader, model)
print("Done!")

