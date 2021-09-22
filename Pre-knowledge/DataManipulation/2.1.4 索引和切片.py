# 2.1.4 索引和切片
import torch

X = torch.arange(12, dtype=torch.float32).reshape(3, 4)

print('X:\n', X)
print('X[-1]:\n', X[-1])  # 用[-1]选择最后一个元素
print('X[1:3]:\n', X[1:3])  # 用[1:3]选择第2个和第3个元素，左开右闭

print('X:\n', X)
X[1, 2] = 9  # 写入元素
print('X:\n', X)
X[0:2, :] = 12  # 写入元素。
print('X:\n', X)
