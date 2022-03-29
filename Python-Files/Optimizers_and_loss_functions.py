import torch
from torch import optim
#1 SGD 
#torch.optim.SGD(weights, lr= 0.01)
# optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

#2 Adam
# optimizer = optim.Adam([var1, var2], lr=0.0001)

"""the above are most used you can find many optimizers
in PyTorch Doc :https://pytorch.org/docs/stable/optim.html
 
 Adadelta

Implements Adadelta algorithm.

Adagrad

Implements Adagrad algorithm.

Adam

Implements Adam algorithm.

AdamW

Implements AdamW algorithm.

SparseAdam

Implements lazy version of Adam algorithm suitable for sparse tensors.

Adamax

Implements Adamax algorithm (a variant of Adam based on infinity norm).

ASGD

Implements Averaged Stochastic Gradient Descent.

LBFGS

Implements L-BFGS algorithm, heavily inspired by minFunc.

NAdam

Implements NAdam algorithm.

RAdam

Implements RAdam algorithm.

RMSprop

Implements RMSprop algorithm.

Rprop

Implements the resilient backpropagation algorithm.

SGD

Implements stochastic gradient descent (optionally with momentum)."""


#for loss functions 
from torch import nn
#1 MSE

mse = nn.MSELoss()

#2 MAE

mae = nn.L1Loss()

#3 Log likelihood loss 
"""https://pytorch.org/docs/stable/generated/torch.nn.NLLLoss.html#torch.nn.NLLLoss"""

ll_loss = nn.NLLLoss()

#4 Cross - Entropy Loss
"""https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html#torch.nn.CrossEntropyLoss"""

cross_entrophy = nn.CrossEntropyLoss()

#5 Hinge Loss

hinge_loss = nn.HingeEmbeddingLoss()

"""
    further more loss funtions visit : https://neptune.ai/blog/pytorch-loss-functions
"""