import torch 
import numpy as np
# ways to create tensors from torch 
#1 zeros 

zeros  = np.zeros((3,3))
zeros_tensor  = torch.zeros(3,3)

print("Numpy zeros  \n",zeros)
print("Tensor zeros \n",zeros_tensor)

#2 ones

ones  = np.ones((3,3))
ones_tensor  = torch.ones(3,3)

print("Numpy ones  \n",ones)
print("Tensor ones \n",ones_tensor)

#3 random numbers
r1 = np.random.rand(2,2)
r2 = torch.rand(2, 2)
print('A random numpy array:\n',r1)
print('A random Tensor :\n',r2)

r = (torch.rand(2, 2) - 0.5) * 2# values between -1 and 1
print('A random matrix, r:')
print(r)

# Common mathematical operations are supported:
print('\nAbsolute value of r:')
print(torch.abs(r))

# ...as are trigonometric functions:
print('\nInverse sine of r:')
print(torch.asin(r))

# ...and linear algebra operations like determinant and singular value decomposition
print('\nDeterminant of r:')
print(torch.det(r))
print('\nSingular value decomposition of r:')
print(torch.svd(r))

# ...and statistical and aggregate operations:
print('\nAverage and standard deviation of r:')
print(torch.std_mean(r))
print('\nMaximum value of r:')
print(torch.max(r))


#4 working with concurrent update here b updates cuncurrently
a = np.zeros((2,2))
print("Numpy \n",a)
b = torch.from_numpy(a)
print("b from numpy \n",b)

a += 1
print("a = ",a)
print("b = ",b)