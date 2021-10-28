import torch
from torch import nn

torch.device('cpu'), torch.cuda.device('cuda')  # cuda和cuda:0 等价

# print(torch.cuda.device_count())  # 查询GPU数量


def try_gpu(i=0):
    """如果存在，则返回gpu(i)，否则返回cpu()。"""
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')


def try_all_gpus():
    """返回所有可用GPU，如果没有GPU，则返回cpu"""
    devices = [torch.device(f'cuda:{i}')
               for i in range(torch.cuda.device_count())]
    return devices if devices else [torch.device('cpu')]


# print(try_gpu())
# print(try_gpu(10))
# print(try_all_gpus())


# # 张量与gpu
# x = torch.tensor([1, 2, 3])
# print('x.device:', x.device)
# y = torch.tensor([1, 2, 3], device=try_gpu())
# print('y.device:', y.device)
# X = torch.ones(2, 3, device=try_gpu())
# print(X)
# # 计算 x + y
# z = x.cuda(0)
# print('x:', x)
# print('z:', z)
# print('y + z:', y + z)
#
#
# # 神经网络与GPU
# net = nn.Sequential(nn.Linear(3, 1))
# net = net.to(device=try_gpu())
# print(net(X))
# print(net[0].weight.data.device)