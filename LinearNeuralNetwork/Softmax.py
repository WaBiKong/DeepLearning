import torch
import torchvision
from torch.utils import data
from torchvision import transforms

import kun


# def get_dataloader_workers():
#     return 0
#
# def load_data_fashion_mnist(batch_size, resize=None):
#     """下载Fashion-MNIST数据集，然后将其加载到内存中。"""
#     trans = [transforms.ToTensor()]
#     if resize:
#         trans.insert(0, transforms.Resize(resize))
#     trans = transforms.Compose(trans)
#     mnist_train = torchvision.datasets.FashionMNIST(
#         root="../data", train=True, transform=trans, download=True
#     )
#     mnist_test = torchvision.datasets.FashionMNIST(
#         root="../data", train=False, transform=trans, download=True
#     )
#     return (data.DataLoader(mnist_train, batch_size, shuffle=True,
#                             num_workers=get_dataloader_workers()),
#             data.DataLoader(mnist_test, batch_size, shuffle=False,
#                             num_workers=get_dataloader_workers()))


# # 定义softmax操作
# # 对每一行（一个图片）而言，概率总和为 1
# # 处理完后(X_exp / partition).shape == X.shape
# def softmax(X):
#     X_exp = torch.exp(X)
#     partition = X_exp.sum(1, keepdim=True)
#     return X_exp / partition  # 运用了广播机制


# 定义模型
def net(X):
    return kun.softmax(torch.matmul(X.reshape((-1, W.shape[0])), W) + b)


# 定义损失函数
# 交叉熵损失函数
# y_hat[range(len(y_hat)), y]：
# y为每个样例所属的类别，shape为(n)
# 这个表示把每个真实类别所对应的预测概率取出
def cross_entropy(y_hat, y):
    return - torch.log(y_hat[range(len(y_hat)), y])


# # 定义准确率计算函数
# def accuracy(y_hat, y):
#     """计算预测正确的数量。"""
#     if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
#         # 使用argmax获得每行中最大元素的索引来获得预测类别
#         y_hat = y_hat.argmax(axis=1)
#     cmp = y_hat.type(y.dtype) == y  # cmp.shape == y.shape == size(n)
#     return float(cmp.type(y.dtype).sum())  # sum计算cmp中元素为1的总和，即正确数量
#
#
# def evaluate_accuracy(net, data_iter):
#     """计算在指定数据集上模型的精度"""
#     if isinstance(net, torch.nn.Module):
#         net.eval()  # 将模型设置为评估模型
#     metric = kun.Accumulator(2)  # 正确预测数、预测总数
#     for X, y in data_iter:
#         metric.add(accuracy(net(X), y), y.numel())
#     return metric[0] / metric[1]


# # 定义一个函数来训练一个迭代周期
# def train_epoch_ch3(net, train_iter, loss, updater):
#     """训练模型的一个迭代周期"""
#     # 将模式设置为训练模式
#     if isinstance(net, torch.nn.Module):
#         net.train()
#     # 训练损失总和、训练准确度总和、样本数
#     metric = kun.Accumulator(3)
#     for X, y in train_iter:
#         # 计算梯度并更新参数
#         y_hat = net(X)
#         l = loss(y_hat, y)
#         if isinstance(updater, torch.optim.Optimizer):
#             # 使用PyTorch内置的优化器和损失函数
#             # updater是更新模型参数的常用函数，它接受批量大小作为参数
#             # updater可以是封装的d2l.sgd函数，也可以是框架的内置优化函数
#             updater.zero_grad()
#             l.backward()
#             updater.step()
#             metric.add(float(1) * len(y), kun.accuracy(y_hat, y),
#                        y.size().numel())
#         else:
#             # 使用定制的优化器和损失函数
#             l.sum().backward()
#             updater(X.shape[0])
#             metric.add(float(l.sum()), kun.accuracy(y_hat, y), y.numel())
#     # 返回训练损失和训练准确率
#     return metric[0] / metric[2], metric[1] / metric[2]


# 定义训练函数
def train_ch3(net, train_iter, test_iter, loss, num_epochs, updater):  # @save
    """训练模型"""
    animator = kun.Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0.3, 0.9],
                            legend=['train loss', 'train acc', 'test acc'])
    for epoch in range(num_epochs):
        train_metrics = kun.train_epoch_ch3(net, train_iter, loss, updater)
        test_acc = kun.evaluate_accuracy(net, test_iter)
        print('start')
        animator.add(epoch + 1, train_metrics + (test_acc,))
        print('end')
    train_loss, train_acc = train_metrics
    assert train_loss < 0.5, train_loss
    assert train_acc <= 1 and train_acc > 0.7, train_acc
    assert test_acc <= 1 and test_acc > 0.7, test_acc


# 使用小批量随机梯度下降法更新参数
def updater(batch_szie):
    return kun.sgd([W, b], lr, batch_size)


# 预测
def predict_ch3(net, test_iter, n=6):
    """预测标签。"""
    for X, y in test_iter:
        break
    trues = kun.get_fashion_mnist_labels(y)
    preds = kun.get_fashion_mnist_labels(net(X).argmax(axis=1))
    titles = [true + '\n' + pred for true, pred in zip(trues, preds)]

    kun.show_images(
        X[0:n].reshape((n, 28, 28)), 1, n, titles=titles[0:n]
    )
    kun.plt.show()


# 获取批量大小为256的数据集（Fashion-MNIST数据集）
batch_size = 256
train_iter, test_iter = kun.load_data_fashion_mnist(batch_size, 28)

# 初始化模型参数
num_inputs = 784
num_outputs = 10

W = torch.normal(0, 0.1, size=(num_inputs, num_outputs), requires_grad=True)
b = torch.zeros(num_outputs, requires_grad=True)

lr = 0.1
num_epochs = 10
train_ch3(net, train_iter, test_iter, cross_entropy, num_epochs, updater)

predict_ch3(net, test_iter)
