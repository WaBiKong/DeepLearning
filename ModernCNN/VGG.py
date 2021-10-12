import torch
from torch import nn
from d2l import torch as d2l
import kun


def vgg_block(num_convs, in_channels, out_channels):
    layers = []
    for _ in range(num_convs):
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
        layers.append(nn.ReLU())
        in_channels = out_channels
    layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
    return nn.Sequential(*layers)


conv_arch = ((1, 64), (1, 128), (2, 256), (2, 512), (2, 512)) # 五个卷积快，卷积层数为 1 + 1 + 2 + 2 + 2 = 8


def vgg(conv_arch):
    conv_blks = []
    in_channels = 1;  # 因为我们用的Fashion-MNIST（灰度图），所以第一次的输入通道为 1
    # 卷积层部分，五个卷积快，八层
    for (num_convs, out_channels) in conv_arch:
        conv_blks.append(vgg_block(num_convs, in_channels, out_channels))
        in_channels = out_channels
    return nn.Sequential(*conv_blks, nn.Flatten(),
                         # 全连接层部分，三个全连接层
                         nn.Linear(in_channels * 7 * 7, 4096), nn.ReLU(),
                         nn.Dropout(0.5),
                         nn.Linear(4096, 4096), nn.ReLU(),
                         nn.Dropout(0.5),
                         nn.Linear(4096, 10))  # 因为我们使用的是Fashion-MNIST，只有十类，所以最后的输出通道为10，实际网络设计为1000


net = vgg(conv_arch)
X = torch.randn(size=(1, 1, 224, 224))
for blk in net:
    X = blk(X)
    print(blk.__class__.__name__, 'output shape:\t', X.shape)

batch_size = 128
train_iter, test_iter = kun.load_data_fashion_mnist(batch_size, resize=224)


def evaluate_accuracy_gpu(net, data_iter, device=None):
    """使用GPU计算模型在数据集上的精度"""
    if isinstance(net, torch.nn.Module):
        net.eval()  # 设置为评估模型
        if not device:
            device = next(iter(net.parameters())).device
    # 正确预测的数量，总预测的数量
    metric = kun.Accumulator(2)
    for X, y in data_iter:
        if isinstance(X, list):
            # BERT微调
            X = [x.to(device) for x in X]
        else:
            X = X.to(device)
        y = y.to(device)
        metric.add(kun.accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]


def try_gpu(i=0):
    """如果存在，则返回gpu(i)，否则返回cpu()。"""
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')


def train(net, train_iter, test_iter, num_epochs, lr, device):
    """用GPU训练模型"""
    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)
    net.apply(init_weights)
    print('training on', device)
    net.to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    loss = nn.CrossEntropyLoss()
    animator = kun.Animator(xlabel='epoch', xlim=[1, num_epochs],
                            legend=['train loss', 'train acc', 'test acc'])
    timer = d2l.Timer()
    num_batches = len(train_iter)
    for epoch in range(num_epochs):
        metric = kun.Accumulator(3)
        net.train() # 在训练模型时会在前面加上，在测试模型时在前面使用：model.eval()
        for i, (X, y) in enumerate(train_iter):
            timer.start()
            optimizer.zero_grad()
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            l.backward()
            optimizer.step()
            with torch.no_grad():
                metric.add(l * X.shape[0], kun.accuracy(y_hat, y), X.shape[0])
            timer.stop()
            train_l = metric[0] / metric[2]
            train_acc = metric[1] / metric[2]
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches, (train_l, train_acc, None))
        test_acc = evaluate_accuracy_gpu(net, test_iter)
        animator.add(epoch + 1, (None, None, test_acc))
    print(f'loss{train_l:.3f}, tarin acc{train_acc:.3f},test acc{test_acc:.3f}')
    print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec on {str(device)}')


ratio = 4
# conv_arch = ((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))
# 将输出通道数减小，以减小自己实现时的时间，实际使用时不可
# small_conv_arch = ((1, 16), (1, 32), (2, 64), (2, 103), (2, 103))
small_conv_arch = [(pair[0], pair[1] // ratio) for pair in conv_arch]
# 由于VGG-11比AlexNet计算量更大，因此我们构建了一个通道数较少的网络，足够用于训练Fashion-MNIST数据集。
net = vgg(small_conv_arch)

lr, num_epochs = 0.05, 10
train(net, train_iter, test_iter, num_epochs, lr, try_gpu())