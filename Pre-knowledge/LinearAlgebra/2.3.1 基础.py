# 线性代数
import torch

# 标量与变量
x = torch.tensor([3.0])
y = torch.tensor([2.0])
print(x + y)
print(x * y)
print(x / y)
print(x**y)

# 向量
x = torch.arange(4)
print('x:\n', x)
print('x[3]:', x[3])  # 通过张量的索引来访问任一元素
print('张量的形状:', x.shape)
print('张量的长度:', len(x))  # len()访问第一维的长度
z = torch.arange(24).reshape(2, 3, 4)
print('三维张量长度:', len(z))

# 矩阵
A = torch.arange(20).reshape(5, 4)
print('A:\n', A)
print('A.shape:', A.shape)
print('A.shape[-1]:', A.shape[-1])
print('A.T:\n', A.T)  # 矩阵的转置

# 矩阵的计算
A = torch.arange(20, dtype=torch.float32).reshape(5, 4)
B = A.clone()  # 通过分配新内存，将A的一个副本分配给B
print('A:\n', A)
print('B:\n', B)
print('A + B:\n', A + B)  # 矩阵相加
print('A * B:\n', A * B)  # 矩阵相乘

a = 2
X = torch.arange(24).reshape(2, 3, 4)
print("X:\n", X)
print('a + X:\n', a + X)
print('a * X:\n', a * X)
print((a * X).shape)
