# 2.1.2 运算
import torch

x = torch.tensor([1.0, 2, 4, 8])
y = torch.tensor([2, 2, 2, 2], dtype=torch.float32)
# 使用dtype=torch.float32，将y的类型强制转换

print(x + y)
print(x - y)
print(x * y)
print(x / y)
print(x ** y)
print(torch.exp(x))  # e ** x


X = torch.arange(12, dtype=torch.float32).reshape(3, 4)
Y = torch.tensor([[2.0, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])

# 连结（concatenate），将张量端到端堆叠以形成更大的张量
# dim=0 表示在第0维合并
print('cat操作 dim=0\n', torch.cat((X, Y), dim=0), '\n', torch.cat((X, Y), dim=0).shape)
print('cat操作 dim=1\n', torch.cat((X, Y), dim=1))

# 通过逻辑运算符构建二元张量
print( X == Y)
print( X < Y)

print(X.sum())