# 2.1.5 节省内存
import torch

X = torch.arange(3, 15).reshape((3, 4))
Y = torch.arange(12).reshape((3, 4))

before = id(Y)
Y = Y + X
print(id(Y) == before)

# 使用 Y += X 或 Y[:] = Y + X 来减少内存开销
before = id(Y)
Y += X
print(id(Y) == before)

before = id(Y)
Y[:] = Y + X
print(id(Y) == before)