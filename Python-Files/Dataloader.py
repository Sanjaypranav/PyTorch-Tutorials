import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5,0.5),(0.5,0.5,0.5))])

# the above code is for transforming our in coming image to tensor and Normalizing it 
#lets load CIFAR-10 dataset (32 x 32 )  10 classes 6 animals 4 vehicles
#data will be loaded to new dir which will be named for root directory

trainset = torchvision.datasets.CIFAR10(root='./Load_Data', train = True, download = True, transform = transform)

# after download complete, you can give it to the DataLoader

train = torch.utils.data.DataLoader(trainset, batch_size= 6, shuffle=True, num_workers = 2)

classes = ('plane','car','bird','cat','dog','deer','frog','horse','ship','truck')
len(classes)

def imshow(img):
    img = img / 2 + 0.5 #inverse_transform(img) 
    mpimg = img.numpy()
    plt.imshow(np.transpose(mpimg, (1, 2, 0)))

dataiter = iter(train)
images , labels = dataiter.next()

imshow(torchvision.utils.make_grid(images))

print(' '.join('%5s' % classes[labels[j]] for j in range(4)))