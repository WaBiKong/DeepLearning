# 2.1.1 入门
import torch

x = torch.arange(12)

print(x)
print(x.shape) # shape访问张量形状
print(x.numel()) # numel()访问元素总量

X = x.reshape(3, 4) # reshape()改变张量形状
print(X)
XX = x.reshape(2, -1) # 用 -1 来让reshape()自动推断维度
print(XX)

a = torch.zeros((2, 3, 4)) # zero()元素全0
b = torch.ones((2, 3, 4)) # ones()元素全1
print(a)
print(b)



x = torch.tensor([[3, 2, 1], [6, 5, 4], [9, 8, 7]])
print(x)
print(x.shape)

