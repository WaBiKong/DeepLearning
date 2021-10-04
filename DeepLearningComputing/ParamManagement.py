import torch
from torch import nn

# 访问参数
net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 1))
X = torch.rand(size=(2, 4))
print(net(X))

# 把net当作一个列表，net[2]表示nn.Linear(8, 1)这一层
print(net[2].state_dict())

# 访问目标参数
print(type(net[2].bias))
print(net[2].bias)
print(net[2].bias.data)

print(net[2].weight.grad is None)

# 一次性访问所有参数
print(*[(name, param.shape) for name, param in net[0].named_parameters()])
print(*[(name, param.shape) for name, param in net.named_parameters()])

print(net.state_dict()['2.bias'].data)


# 从嵌套快手机参数
def block1():
    return nn.Sequential(nn.Linear(4, 8), nn.ReLU(),
                         nn.Linear(8, 4), nn.ReLU())


def block2():
    net = nn.Sequential()
    for i in range(4):
        # 在这里嵌套
        net.add_module(f'block{i}', block1())
    return net


rgent = nn.Sequential(block2(), nn.Linear(4, 1))
print(rgent(X))
print(rgent)
print(rgent[0][1][0].bias.data)


# 参数初始化
def init_normal(m):
    if type(m) == nn.Linear:
        # 将所有权重参数初始化为均值为0、标准差为0.01的高斯随机变量
        # 将偏置参数设置为0
        nn.init.normal_(m.weight, mean=0, std=0.01)
        nn.init.zeros_(m.bias)


net.apply(init_normal)  # 对net里的所有函数操作
print(net[0].weight.data[0], net[0].bias.data[0])


