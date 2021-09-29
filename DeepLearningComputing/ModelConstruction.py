# 层和块:块可以描述单个层、由多个层组成的组件或整个模型本身
# nn.Module 类的使用

import torch
from torch import nn
from torch.nn import functional as F

net = nn.Sequential(nn.Linear(20, 256), nn.ReLU(), nn.Linear(256, 10))

X = torch.rand(2, 20)
# 我们一直在通过net(X)调用我们的模型来获得模型的输出
# 这实际上是net.__call__(X)的简写
print(net(X))


# 自定义块
# 实现将严重依赖父类，只需要提供我们自己的构造函数（Python中的__init__函数）和正向传播函数
class MLP(nn.Module):
    # 用模型参数声明层。这里，我们声明两个全连接的层
    def __init__(self):
        # 调用'MLP'的父类'Block'的构造函数来执行并要的初始化。
        # 这样，在类实例化时也可以指定其他函数参数，例如模型参数'params'
        super().__init__()
        self.hidden = nn.Linear(20, 256)  # 隐藏层
        self.out = nn.Linear(256, 10)  # 输出层

    # 定义模型的正向传播，即如和根据输入'X'返回所需的模型输出
    def forward(self, X):
        # 这里使用在nn.functional模块中定义的Relu的函数版本
        return self.out(F.relu(self.hidden(X)))


net = MLP()
print(net(X))


# 顺序块 Sequential
class MySequential(nn.Module):
    def __init__(self, *args):
        super().__init__()
        for block in args:
            # 这里'block'是'Module'子类的一个实例，
            # 我们把它保存在'Module'类的成员变量'_Module'中。
            # 'block'的类型是OrderedDict。
            self._modules[block] = block

    def forward(self, X):
        # OrederedDict保证了按照成员添加的顺序遍历它们
        for block in self._modules.values():
            X = block(X)
        return X


net = MySequential(nn.Linear(20, 256), nn.ReLU(), nn.Linear(256, 10))
print(net(X))


# 在正向传播函数中执行代码
class FixedHiddenMLP(nn.Module):
    def __init__(self):
        super().__init__()
        # 不计算梯度的随机权重参数。因此其在训练期间保持不变。
        self.rand_weight = torch.rand((20, 20), requires_grad=False)
        self.linear = nn.Linear(20, 20)

    def forward(self, X):
        X = self.linear(X)
        # 使用创建的常量参数以及'relu'和'dot'函数
        X = F.relu(torch.mm(X, self.rand_weight) + 1)
        # 复合全连接层，这相当于两个全连接层共享参数
        X = self.linear(X)
        # 控制流
        while X.abs().sum() > 1:
            X /= 2
        return X.sum()


net = FixedHiddenMLP()
print(net(X))


class NestMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(20, 64), nn.ReLU(),
                                 nn.Linear(64, 32), nn.ReLU())
        self.linear = nn.Linear(32, 16)

    def forward(self, X):
        return self.linear(self.net(X))


chimera = nn.Sequential(NestMLP(), nn.Linear(16, 20), FixedHiddenMLP())
print(chimera(X))