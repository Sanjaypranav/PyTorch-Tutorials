import torch
from torch import nn
import torch.nn.functional as functional #Activation function

"""From images/mnist.png you can get a basic idea of 
    Convolutional Neural Networks 
    CNN are Deep learning model 
    here they used for MNIST Dataset
    points to ponder 
    >3 C1: Convolutional layer t scans the input image for features it learned during training. It outputs a map of where it saw each of its learned features in the image. This “activation map” is downsampled in layer S2.
    >3 Layer C3 is another convolutional layer, this time scanning C1’s activation map for combinations of features. It also puts out an activation map describing the spatial locations of these feature combinations, which is downsampled in layer S4.
    >3 Finally, the fully-connected layers at the end, F5, F6, and OUTPUT, are a classifier that takes the final activation map, and classifies it into one of ten bins representing the 10 digits"""


# our model inherits nn.Module for layers 
class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 3)
        self.conv2 = nn.Conv2d(6, 16, 3)
        self.fc1 = nn.Linear(16 * 6 * 6, 120)  # 6*6 from image dimension
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self , X):
        X = functional.max_pool2d(functional.relu(self.conv1(X)), kernel_size = (2, 2))
        X = functional.max_pool2d(functional.relu(self.conv2(X)), kernel_size = 2)
        X = X.view(-1, self.num_flat_features(X))
        X = functional.relu(self.fc1(X))
        X = functional.relu(self.fc2(X))
        X = self.fc3(X)
        return X
        
    def num_flat_features(self, X):
        size=X.size()[1:]  # all dimensions except the batch dimension
        num_features=1
        for s in size:
            num_features *= s
        return num_features


model = Model();
print(model)                         # what does the object tell us about itself?

input = torch.rand(1, 1, 32, 32)   # stand-in for a 32x32 black & white image
print('\nImage batch shape:')
print(input.shape)

output = model(input)                # we don't call forward() directly
print('\nRaw output:')
print(output)
print(output.shape)
