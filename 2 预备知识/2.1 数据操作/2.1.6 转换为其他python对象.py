# 2.1.6 转换为其他python对象
import torch

Y = torch.tensor([[2.0, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])

A = Y.numpy()  # tensor转python中的numpy()数组
print(type(A))  # 打印A的类型
print(A)
B = torch.tensor(A)  # python中的numpy()数组转tensor
print(type(B))
print(B)

a = torch.tensor([3.5])
print(a, a.item(), float(a), int(a))