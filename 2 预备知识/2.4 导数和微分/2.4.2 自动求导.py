import torch

# 自动梯度计算
x = torch.arange(4.0, requires_grad=True)  # requires_grad=True表示变量x拥有梯度
print('x:', x)
print('x.grad:', x.grad)  # 用x.grad来存放梯度
y = 2 * torch.dot(x, x)
print('y:', y)
y.backward() # 反向传播求梯度
print('x.grad:', x.grad)

x.grad.zero_()  # 重设为0
y = x.sum()
print('y:', y)
y.backward()
print('x.grad:', x.grad)

x.grad.zero_()
y = x * x
print('y:', y)
y.sum().backward()
print('x.grad:', x.grad)

# python控制流梯度计算
def f(a):
    b = a * 2
    print(b.norm())
    while b.norm() < 1000:
        b = b * 2
    if b.sum() > 0:
        c = b
    else:
        c = 100 * b
    return c

a = torch.tensor(2.0)
a.requires_grad_(True)
print('a:', a)
d = f(a)
print('d:', d)
d.backward()
print('a.grad:', a.grad)

