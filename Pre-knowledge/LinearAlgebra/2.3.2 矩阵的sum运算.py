# 矩阵的sum运算
import torch

A = torch.arange(20, dtype=torch.float32).reshape(5, 4)
print('A:\n', A)
print('A.shape:', A.shape)
print('A.sum():', A.sum())
print('A.sum(axis=0):', A.sum(axis=0))  # 沿第0维求和，并压缩此维度
print('A.sum(axis=1):', A.sum(axis=1))  # 沿第1维求和，并压缩此维度
# keepdims=True 保持维度不变进行求和，维度大小变为1
print('A.sum(axis=1, keepdims=True:\n', A.sum(axis=1, keepdim=True))
print('A.sum(axis=[0, 1]:\n', A.sum(axis=[0, 1]))  # 计算所有维度的总和
print('A.mean():', A.mean())  # 求均值
print('A.sum() / A.numel():', A.sum() / A.numel())  # numel()访问元素总数量
