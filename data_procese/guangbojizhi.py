import torch
import torch.nn as nn

x = torch.tensor([1,1,1])
y = torch.tensor([1,1,1])
z = torch.tensor([2,2,2])
print(torch.cat([x*z,y*z],dim=0))
print(torch.cat([x,y],dim=0)*z)