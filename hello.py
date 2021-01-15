import numpy as np
print("What is your name?")
name = input()
print("Hello {}! This is a random number: {}".format(name, np.random.randint(10)))
print("And this is a PyTorch tensor:")
import torch
x = torch.randn(10)
if torch.cuda.is_available():
    device = torch.device("cuda")
    y = torch.ones_like(x, device = device)
    x = x.to(device)
print(x)
print(y)