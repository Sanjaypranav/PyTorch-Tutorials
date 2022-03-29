import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)
labels_map = {
    0: "T-Shirt",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle Boot",
}

train_dataloader = DataLoader(training_data, batch_size=60000, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=10000, shuffle=True)

train_features, train_labels = next(iter(train_dataloader))
test_features, test_labels = next(iter(test_dataloader))

print('Training data size =',train_features.size(), len(train_labels))
print('Testing data size',test_features.size(),len(train_labels))

