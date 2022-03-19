import torch
from torch import nn


class Network(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Inputs to hidden layer linear transformation
        self.hidden = nn.Linear(784, 256)
        # Output layer, 10 units - one for each digit
        self.output = nn.Linear(256, 10)
        
        # Define sigmoid activation and softmax output 
        self.softmax = nn.Softmax()
        
    def forward(self, x, n_classes):
        # Pass the input tensor through each of our operations
        x = self.hidden(x)
        x = nn.functional.relu(x) 
        x = self.output(x)
        x = self.softmax(x , n_classes)
        
        return x


 