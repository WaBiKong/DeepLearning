import torch
from IPython import display
from matplotlib import pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from torchvision import transforms
import torchvision


# 设置 matplotlib 的轴
def set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend):
    """设置 matplotlib 的轴。"""
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    axes.set_xscale(xscale)
    axes.set_yscale(yscale)
    axes.set_xlim(xlim)
    axes.set_ylim(ylim)
    if legend:
        axes.legend(legend)
    axes.grid()


# 动画中绘制数据的实用程序类
class Animator:
    """在动画中绘制数据。"""

    def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None,
                 ylim=None, xscale='linear', yscale='linear',
                 fmts=('-', 'm--', 'g-.', 'r:'), nrows=1, ncols=1,
                 figsize=(9, 7)):
        # 增量地绘制多条线
        if legend is None:
            legend = []
        # display.set_matplotlib_formats('svg')
        self.fig, self.axes = plt.subplots(nrows, ncols, figsize=figsize)
        if nrows * ncols == 1:
            self.axes = [self.axes, ]
        # 使用lambda函数捕获参数
        self.config_axes = lambda: set_axes(
            self.axes[0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
        self.X, self.Y, self.fmts = None, None, fmts
        plt.show(block=False)
        plt.pause(0.1)

    def add(self, x, y):
        # 向图表中添加多个数据点
        n = len(y)
        if not hasattr(y, "__len__"):
            y = [y]
            n = len(y)
        if not hasattr(x, "__len__"):
            x = [x] * n
        if not self.X:
            self.X = [[] for _ in range(n)]
        if not self.Y:
            self.Y = [[] for _ in range(n)]
        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
        self.axes[0].cla()
        for x, y, fmt in zip(self.X, self.Y, self.fmts):
            self.axes[0].plot(x, y, fmt)
            self.config_axes()
        plt.show(block=False)
        plt.pause(0.1)
        display.clear_output(wait=True)


# 定义绘制图像列表函数
def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5):
    """绘制图像列表。"""
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.flatten()
    for i, (ax, img) in enumerate(zip(axes, imgs)):
        if torch.is_tensor(img):
            # Tensor Image
            ax.imshow(img.numpy())
        else:
            # PIL Image
            ax.imshow(img)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        if titles:
            ax.set_title(titles[i])
    return axes


# Accumulator是一个实用程序类，用于对多个变量进行累加
class Accumulator:
    """在`n`个变量上累加。"""

    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

# 创造数据 y = Xw + b + noise。
def synthetic_data(w, b, num_examples):
    X = torch.normal(0, 1, (num_examples, len(w)))
    # matmul可以进行张量乘法, 输入可以是高维.
    y = torch.matmul(X, w) + b
    noise = torch.normal(0, 0.01, y.shape)
    y += noise
    return X, y.reshape((-1, 1))

# 设置不使用多线程（若不为0，则使用多线程）
def get_dataloader_workers():
    return 0


# 下载Fashion-MNIST数据集，然后将其加载到内存中
def load_data_fashion_mnist(batch_size, resize=None):
    """下载Fashion-MNIST数据集，然后将其加载到内存中。"""
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)
    mnist_train = torchvision.datasets.FashionMNIST(
        root="../data", train=True, transform=trans, download=True
    )
    mnist_test = torchvision.datasets.FashionMNIST(
        root="../data", train=False, transform=trans, download=True
    )
    return (DataLoader(mnist_train, batch_size, shuffle=True,
                       num_workers=get_dataloader_workers()),
            DataLoader(mnist_test, batch_size, shuffle=False,
                       num_workers=get_dataloader_workers()))


# 定义优化算法(参数更新）:小批量随机梯度下降
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

# 定义线性神经网络模型
def linreg(X, w, b):
    """线性回归模型"""
    return torch.matmul(X, w) + b

# 定义平方损失函数
def squared_loss(y_hat, y):
    """均方损失"""
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2

# 定义softmax操作
# 对每一行（一个图片）而言，概率总和为 1
# 处理完后(X_exp / partition).shape == X.shape
def softmax(X):
    X_exp = torch.exp(X)
    partition = X_exp.sum(1, keepdim=True)
    return X_exp / partition  # 运用了广播机制


# 定义Fashion-MNIST数据集的文本标签返回函数
def get_fashion_mnist_labels(labels):
    """返回 Fashion-MNIST 数据集的文本标签。"""
    text_labels = [
        't-shirt', 'trouser', 'pullover', 'dress', 'coat', 'sandal', 'shirt',
        'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]


# 定义准确率计算函数
def accuracy(y_hat, y):
    """计算预测正确的数量。"""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        # 使用argmax获得每行中最大元素的索引来获得预测类别
        y_hat = y_hat.argmax(axis=1)
    cmp = astype(y_hat, y.dtype) == y  # cmp.shape == y.shape == size(n)
    return float(reduce_sum(astype(cmp, y.dtype)))  # sum计算cmp中元素为1的总和，即正确数量


# 定义精度计算函数
def evaluate_accuracy(net, data_iter):
    """计算在指定数据集上模型的精度"""
    if isinstance(net, torch.nn.Module):
        net.eval()  # 将模型设置为评估模型
    metric = Accumulator(2)  # 正确预测数、预测总数
    for X, y in data_iter:
        metric.add(accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]

# 评估给定数据集上模型的损失
def evaluate_loss(net, data_iter, loss):
    """评估给定数据集上模型的损失。"""
    metric = Accumulator(2)  # Sum of losses, no. of examples
    for X, y in data_iter:
        out = net(X)
        y = torch.reshape(y, out.shape)
        l = loss(out, y)
        metric.add(reduce_sum(l), size(l))
    return metric[0] / metric[1]

# 定义一个函数来训练一个迭代周期
def train_epoch_ch3(net, train_iter, loss, updater):
    """训练模型的一个迭代周期"""
    # 将模式设置为训练模式
    if isinstance(net, torch.nn.Module):
        net.train()
    # 训练损失总和、训练准确度总和、样本数
    metric = Accumulator(3)
    for X, y in train_iter:
        # 计算梯度并更新参数
        y_hat = net(X)
        l = loss(y_hat, y)
        if isinstance(updater, torch.optim.Optimizer):
            # 使用PyTorch内置的优化器和损失函数
            # updater是更新模型参数的常用函数，它接受批量大小作为参数
            # updater可以是封装的d2l.sgd函数，也可以是框架的内置优化函数
            updater.zero_grad()
            l.backward()
            updater.step()
            metric.add(float(l) * len(y), accuracy(y_hat, y), y.numel())
        else:
            # 使用定制的优化器和损失函数
            l.sum().backward()
            updater(X.shape[0])
            metric.add(float(l.sum()), accuracy(y_hat, y), y.numel())
    # 返回训练损失和训练准确率
    return metric[0] / metric[2], metric[1] / metric[2]

# 定义训练函数
def train_ch3(net, train_iter, test_iter, loss, num_epochs, updater):
    """Train a model (defined in Chapter 3)."""
    animator = Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0.3, 0.9],
                        legend=['train loss', 'train acc', 'test acc'])
    for epoch in range(num_epochs):
        train_metrics = train_epoch_ch3(net, train_iter, loss, updater)
        test_acc = evaluate_accuracy(net, test_iter)
        animator.add(epoch + 1, train_metrics + (test_acc,))
    train_loss, train_acc = train_metrics
    assert train_loss < 0.5, train_loss
    assert train_acc <= 1 and train_acc > 0.7, train_acc
    assert test_acc <= 1 and test_acc > 0.7, test_acc

# pytorch数据迭代器
def load_array(data_arrays, batch_size, is_train=True):
    """构建一个pytorch数据迭代器"""
    dataset = TensorDataset(*data_arrays)
    return DataLoader(dataset, batch_size, shuffle=is_train)


reduce_sum = lambda x, *args, **kwargs: x.sum(*args, **kwargs)
size = lambda x, *args, **kwargs: x.numel(*args, **kwargs)
astype = lambda x, *args, **kwargs: x.type(*args, **kwargs)