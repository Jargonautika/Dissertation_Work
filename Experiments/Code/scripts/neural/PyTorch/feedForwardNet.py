# https://github.com/Niranjankumar-c/DeepLearning-PadhAI/tree/master/DeepLearning_Materials

import math
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

# This is maybe a failed class, I can't get it to multiply with the data I have
class myNeuralNetwork(nn.Module):

    def __init__(self, X_train, Y_train, ckpt_dest):

        super().__init__()
        torch.manual_seed(42)
        self.ckpt_dest = ckpt_dest

        # Initialize the weights and biases using Xavier Initialization
        self.weights1 = nn.Parameter(torch.randn(X_train.shape[0], X_train.shape[-1]) / math.sqrt(X_train.shape[-1]))
        self.bias1 = nn.Parameter(torch.zeros(X_train.shape[-1]))
        self.weights2 = nn.Parameter(torch.randn(2, 4) / math.sqrt(2))
        self.bias2 = nn.Parameter(torch.zeros(4))

        #set the parameters for training the model
        self.learning_rate = 1e-6
        self.epochs = 10000
        self.opt = optim.SGD(self.parameters(), lr=self.learning_rate, momentum=0.9) #define optimizer
        self.X_train = X_train.float()
        self.Y_train = Y_train.long()
        self.loss_arr = list()
        self.acc_arr = list()

    def train(self):

        for epoch in range(self.epochs):
            y_hat = self.forward(self.X_train)
            loss = self.loss_fn(y_hat, self.Y_train)
            self.loss_arr.append(loss.item())
            self.acc_arr.append(self.accuracy(y_hat, self.Y_train))
            loss.backward()
            opt.step()          #updating each parameter.
            opt.zero_grad()     #resets the gradient to 0

            # https://pytorch.org/tutorials/recipes/recipes/saving_and_loading_a_general_checkpoint.html
            if epoch % 100 == 0:
                torch.save({'epoch': epoch,
                            'model_state_dict': self.state_dict(),
                            'optimizer_state_dict': self.opt.state_dict(),
                            'loss': self.loss
                    }, os.path.join(self.ckpt_dest, "feedForwardNN_{}".format(epoch))
                    )

    # Characterize the forward pass
    def forward(self, x):

        A1 = torch.matmul(x, self.weights1) + self.bias1
        H1 = A1.sigmoid()
        A2 = torch.matmul(H1, self.weights2) + self.bias2
        H2 = A2.exp() / A2.exp().sum(-1).unsqueeze(-1)

        return H2

    # Calculate accuracy of the model
    def accuracy(self, y_hat, y):

        pred = torch.argmax(y_hat, dim = 1)

        return (pred == y).float().mean()

    # Loss function
    def loss(self, y_hat, y):

        return F.cross_entropy(y_hat, y)


class Feedforward(nn.Module):

    def __init__(self, input_size, ckpt_dest):

        super(Feedforward, self).__init__()
    
        self.input_size = input_size
        self.ckpt_dest = ckpt_dest
        self.criterion = nn.BCELoss()
        self.loss_arr = list()
        self.acc_arr = list()

        # Fully-connected layers, no skip layers
        self.fc1 = nn.Linear(self.input_size, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 2)
        # self.fc3 = nn.Linear(128, 1)
        # self.fc3 = nn.Linear(128, 256)
        # self.fc4 = nn.Linear(256, 512)
        # self.fc5 = nn.Linear(512, 1024)
        # self.fc6 = nn.Linear(1024, 512)
        # self.fc7 = nn.Linear(512, 256)
        # self.fc8 = nn.Linear(256, 128)
        # self.fc9 = nn.Linear(128, 64)
        # self.fc10 = nn.Linear(64, 1)
        self.softmax = nn.Softmax()

        # For optimization
        self.optimizer = optim.SGD(self.parameters(), lr = 0.01)

    def forward(self, x):

        # Alternate activation and hidden layers
        a1 = self.fc1(x)
        h1 = self.relu(a1)
        a2 = self.fc2(h1)
        # h2 = self.relu(a2)
        # a3 = self.fc3(h2)
        # h3 = self.relu(a3)
        # a4 = self.fc4(h3)
        # h4 = self.relu(a4)
        # a5 = self.fc5(h4)
        # h5 = self.relu(a5)
        # a6 = self.fc6(h5)
        # h6 = self.relu(a6)
        # a7 = self.fc7(h6)
        # h7 = self.relu(a7)
        # a8 = self.fc8(h7)
        # h8 = self.relu(a8)
        # a9 = self.fc9(h8)
        # h9 = self.relu(a9)
        # a10 = self.fc10(h9)
        # h10 = self.relu(a10)
        output = self.softmax(a2, dim = 2)

        return output

    def accuracy(self, y_hat, y):
    
        pred = torch.argmax(y_hat, dim = 1)
        
        return (pred == y).float().mean()
