# 范数，矩阵访问
import torch

# L2范数：欧几里得距离
# L1范数：向量元素的绝对值之和
u = torch.tensor([3.0, -4.0])
print('向量的L2范数:', torch.norm(u))  # 向量的L2范数
print('向量的L1范数:', torch.abs(u).sum())  # 向量的L1范数

# 类似于向量的L2范数，矩阵的弗罗贝尼乌斯范数(Frobenius norm)是矩阵元素平方和的平方根
# 弗罗贝尼乌斯范数满足向量范数的所有性质，它就像是矩阵形向量的 L2 范数
v = torch.ones(4, 9)
print('v:\n', v)
print('矩阵的L2范数:', torch.norm(v))  # 矩阵的L2范数
