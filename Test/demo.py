import torch

a = torch.randn(size=(2, 3, 4))
b = torch.randn(size=(2, 3, 4))
print(a)
print(a.permute(2, 1, 0))
print(a.is_leaf)