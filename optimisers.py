import numpy as np
import torch

class Optimisers:
    
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.weights = np.ones((X.shape[1],1))

        
    def SGD(self, l_rate =  0.1):
        self.weights -= l_rate * np.dot(np.dot(self.X, self.weights) - self.y , self.weights) 
        return self.weights

    def SGD_momentum(self,):  #Vt = Vt-1 + lr * gradient  #W_new = W_old + Vt
        sgd = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        return 0

    def rmsprop():
        sgd = torch.optim.RMSprop()
        return 0

    def adagrad():
        sgd = torch.optim.Adagrad()
        return 0

    def adadelta():
        sgd = torch.optim.Adadelta()
        return 0

    def adam():
        sgd = torch.optim.Adam()
        return 0

    def Nadam():
        sgd = torch.optim.NAdam()
        return 0


if __name__ == "__main__":

    X = np.array([[1 , 2],[3 , 4]])
    y = np.array([6 , 14])
    print(X.shape,y.shape)
    optimize = Optimisers(X, y)
    print(optimize.SGD())
