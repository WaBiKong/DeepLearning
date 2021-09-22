# 2.1.3 广播机制
import torch

a = torch.arange(3).reshape((3, 1))
b = torch.arange(2).reshape((1, 2))

print(a, '\n', b)

# 0 0         0 1
# 1 1    +    0 1
# 2 2         0 1

print(a + b)