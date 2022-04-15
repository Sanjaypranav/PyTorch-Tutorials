from pyexpat import model
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

dataset = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data',names=['sepal_length', 'sepal_width','petal_length', 'petal_width', 'species'])
dataset['species'] = pd.Categorical(dataset['species']).codes

dataset = dataset.sample(frac=1, random_state=1234)

train_input = dataset.values[:120, :4]
train_target = dataset.values[:120, 4]

test_input = dataset.values[120:, :4]
test_target = dataset.values[120:, 4]

hidden_units = 5
model = torch.nn.Sequential(torch.nn.Linear(4, hidden_units),
torch.nn.ReLU(),
torch.nn.Linear(hidden_units, 3))

Loss = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1,momentum=0.9)

epochs = 50
for epoch in range(epochs):
    inputs = torch.autograd.Variable(torch.Tensor(train_input).float())
    targets =torch.autograd.Variable(torch.Tensor(train_target).long())
    optimizer.zero_grad()
    out = model(inputs)
    loss = Loss(out, targets)
    loss.backward()
    optimizer.step()
    if epoch == 0 or (epoch + 1) % 10 == 0:
        print('Epoch %d Loss: %.4f' % (epoch + 1, loss.item()))


inputs = torch.autograd.Variable(torch.Tensor(test_input).float())
targets = torch.autograd.Variable(torch.Tensor(test_target).long())
optimizer.zero_grad()
out = model(inputs)
_, predicted = torch.max(out.data, 1)
error_count = test_target.size - np.count_nonzero((targets==predicted).numpy())
print('Errors: %d; Accuracy: %d%%' % (error_count, 100 *torch.sum(targets == predicted) / test_target.size))