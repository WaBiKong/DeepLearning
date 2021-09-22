# 矩阵向量乘积
import torch

# 向量-向量相乘（点积）
x = torch.arange(4, dtype=torch.float32)
y = torch.arange(4, dtype=torch.float32)
print('x:', x)
print('y:', y)
print('torch.dot(x, y):', torch.dot(x, y))  # torch.dot()计算向量乘向量的点积
print('matmul:\n', torch.matmul(x, y))

# 矩阵-向量相乘（向量积）
A = torch.arange(20, dtype=torch.float32).reshape(5, 4)
print('A:\n', A)
print('x:', x)
print('torch.mv(A, x):', torch.mv(A, x))  # torch.mv()计算矩阵乘向量
print('matmul:', torch.matmul(A, x))

# 矩阵-矩阵相乘（向量积）
B = torch.ones(4, 3)  # 4*3维 全1 矩阵
print('B:\n', B)
print('torch.mm(A, B):\n', torch.mm(A, B))  # torch.mm()计算矩阵乘矩阵
print('matmul:\n', torch.matmul(A, B))

