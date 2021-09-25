import torch
import random

# y = Xw + b + c
# 构造数据集
def synthetic_data(w, b, num_examples):
    X = torch.normal(0, 1, (num_examples, len(w)))
    # matmul可以进行张量乘法, 输入可以是高维.
    y = torch.matmul(X, w) + b
    noise = torch.normal(0, 0.01, y.shape)
    y += noise
    return X, y.reshape((-1, 1))

true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = synthetic_data(true_w, true_b, 1000)

# d2l.set_figsize()
# d2l.plt.scatter(features[:, 1].detach().numpy(), labels.detach().numpy(), 1)
# d2l.plt.show()

# 读取数据集
def data_iter(batch_size, features, labels):
    num_examples = len(features)
    # list() 函数是对象迭代器，可以把range()返回的可迭代对象转为一个列表，返回的变量类型为列表
    indices = list(range(num_examples))
    # 这些样本是随机读取的，没有特定的顺序
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        batch_indices = torch.tensor(
            indices[i: min(i + batch_size, num_examples)]
        )
        yield features[batch_indices], labels[batch_indices]

batch_size = 10

# # 查看数据集
# for X, y in data_iter(batch_size, features, labels):
#     print(X, '\n', y)

# 初始化模型参数
w = torch.normal(0, 0.01, size=(2, 1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)

# 定义模型
def linreg(X, w, b):
    """线性回归模型"""
    return torch.matmul(X, w) + b

# 定义损失函数
def squared_loss(y_hat, y):
    """均方损失"""
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2

# 定义优化算法(参数更新）
def sgd(params, lr, batch_size):
    """小批量随机梯度下降"""
    with torch.no_grad():
        for param in params:
            # 计算梯度
            # 损失函数中未求均值，所以这里除以batch_size
            param -= lr * param.grad / batch_size
            # pytorch不会自动设为0
            # 下次计算时就不会和上一次相关
            param.grad.zero_()

# 训练函数
lr = 0.028  # 学习率
num_epochs = 20  # 训练次数
net = linreg
loss = squared_loss

for epoch in range(num_epochs):
    # 每个batch_size拿来更新一下，一共1000/batch_size次
    for X, y in data_iter(batch_size, features, labels):
        l = loss(net(X, w, b), y)
        l.sum().backward()
        sgd([w, b], lr, batch_size)
    # with torch.no_grad() 主要是用于停止autograd模块的工作，
    # 以起到加速和节省显存的作用，具体行为就是停止gradient计算，从而节省了GPU算力和显存，但是并不会影响dropout和batchnorm层的行为。
    with torch.no_grad():
        train_l = loss(net(features, w, b), labels)
        print(f'epoch{epoch + 1}, loss{float(train_l.mean()):f}')

