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




class CNN(nn.Module):

    def __init__(self, output_dim=2):
        super(CNN, self).__init__()
        #1st convolution
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

        #make a module lists of the different types of layers
        self.conv_layers = torch.nn.ModuleList([conv1, conv2, conv3, conv4, conv5, conv6])
        self.pool_layers = torch.nn.ModuleList([pool1, pool2, pool3, pool4, pool5, pool6])
        self.rels = torch.nn.ModuleList([rel1, rel2, rel3, rel4, rel5, rel6])

        # Flatten the 25x25 dimesion into a 1x625x32
        self.flat = nn.Flatten(1, 3)
        # Linear transformations to extract one value
        self.lin1 = nn.Linear(1024, 512)
        self.act1 = nn.ELU()
        self.lin2 = nn.Linear(512, 64)
        self.act2 = nn.ELU()
        self.lin3 = nn.Linear(64, output_dim)
        self.final = nn.Softmax(1)


    def forward(self, x):
        for conv, pool, act in zip(self.conv_layers, self.pool_layers, self.rels):
            x = conv(x)
            #print(x.shape)
            x = act(x)
            #print(x.shape)
            x = pool(x)
            #print(x.shape)

        x = self.flat(x)
        #print(x.shape)

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



'''Method to load training data (equilbria tensors and truth labels) from their directory.
    '''
def loadEquilibriumFiles(directory):
    loaded_tensors = []
    tensor_labels = []
    label_mapping = {"Delta":0, "Kappa":1, "Iota":2, "Epsilon":3, "Gamma":4, "ripple":5,"Alpha":6,"Beta":7, "TauSigma":8 }
    for file in os.listdir(directory):
        for key in label_mapping.keys():
            if key in file:
                try:
                    arr = np.load(file, allow_pickle=True)
                    loaded_tensors.append(torch.tensor(
                    np.stack([arr["EquilibriaA"], arr["EquilibriaB"]])
                    ))
#                   Map file name to truth label
                    for poss_lab in label_mapping.keys():
                        if poss_lab in file:
                            tensor_labels.append(label_mapping[poss_lab])
                except KeyError:
                    print("Key does not exist")
                except FileNotFoundError:
                    print("file not found")

    return torch.stack(loaded_tensors).float(), torch.LongTensor(tensor_labels)



EquilibriaTensors, TruthLabels = loadEquilibriumFiles("/Users/zachzerbe/PycharmProjects/SpatialPatternPredictionResearch")

#Training loop
def train_loop(dataloader, model, lossFunction, optimizer):
    size = len(dataloader.dataset)
    model.train()
    current = 0
    for batch in loader:
        bX, bY = batch
        bX = bX.to(device)
        bY = bY.to(device)
        pred = model(bX)
        loss = lossFunction(pred, bY)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        current += len(bX)
        if current % 100 == 0:
            loss = loss.item()
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

#Test/Validation Loop
def test_loop(dataloader, model, lossFunction):
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
            test_loss += lossFunction(pred, bY).item()
            correct += (pred.argmax(1) == bY).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")



#output_dim should be set to the number of patterns being differentiated
model = CNN(output_dim= 7).to(device)

# ** Hyperparameters **

criterion = nn.NLLLoss()

learning_rate = 1e-4
optimizer = torch.optim.Adam(model.parameters(), lr= learning_rate)

batchSize = 64
epochs = 200

data = torch.utils.data.TensorDataset(EquilibriaTensors.float(), TruthLabels.long())
loader = torch.utils.data.DataLoader(data, shuffle=True, batch_size=batchSize)



for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(loader, model, criterion, optimizer)
    test_loop(loader, model, criterion)
print("Done!")






