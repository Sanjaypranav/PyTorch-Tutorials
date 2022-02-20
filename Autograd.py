import torch
"""
    Autograd:
    To see more about Autograd i wish you to learn more about gradient descent
    gradient w.r.t the input tensor is computed step by step from loss to top level
    in reverse 
    to see more follow this lik 
    youtube : https://www.youtube.com/watch?v=IC0_FRiX-sw
    at time stamp 8:00 to 9:50 """


x = torch.randn(3, requires_grad = True)
print("X = ",x) # you can able to see the values printed along with requires_grad = True


y = x + 2
print("y = ",y) # you can see the values printed along with gradient function in backward

y = y.mean()

y.backward() #calling backward function will to get gradients from

print("y after mean",y)

# okay :) how to get rid of gradient
#3 simple methods 
#1) x.detach()
#2) x.requires_grad_(False)
#3) 
with torch.no_grad():
    z = x + 2
    print("no grad",z)