import math
import numpy as np
from torch import nn
import kun
import torch


max_degree = 20  # 多项式的最大阶数
n_train, n_test = 100, 100  # 训练和测试数据集大小
true_w = np.zeros(max_degree)  # 分配大量空间
true_w[0:4] = np.array([5, 1.2, -3.4, 5.6])

features = np.random.normal(size=(n_train + n_test, 1))  # 随机生成特征
np.random.shuffle(features)  # 将特征数据打乱
poly_features = np.power(features, np.arange(max_degree).reshape(1, -1))
for i in range(max_degree):
    poly_features[:, i] /= math.gamma(i + 1)
labels = np.dot(poly_features, true_w)
labels += np.random.normal(scale=0.1, size=labels.shape)  # 加上噪音

# NumPy ndarry 转换为 tensor
true_w, features, poly_features, labels = [torch.tensor(x, dtype=
         torch.float32) for x in [true_w, features, poly_features, labels]]

def evaluate_loss(net, data_iter, loss):
    """评估给定数据集上模型的损失。"""
    metric = kun.Accumulator(2)
    for X, y in data_iter:
        out = net(X)
        y = y.reshape(out.shape)
        l = loss(out, y)
        metric.add(l.sum(), l.numel())
    return metric[0] / metric[1]

# 训练函数
def train(train_features, test_features, train_labels, test_labels, num_epochs=400):
    loss = nn.MSELoss()
    input_shape = train_features.shape[-1]
    # 不设置偏置，因为我们已经在多项式特征中实现了它
    net = nn.Sequential(nn.Linear(input_shape, 1, bias=False))
    batch_szie = min(10, train_labels.shape[0])
    train_iter = kun.load_array((train_features, train_labels.reshape(-1, 1)),
                                batch_szie)
    test_iter = kun.load_array((test_features, test_labels.reshape(-1, 1)),
                               batch_szie, is_train=False)
    trainer = torch.optim.SGD(net.parameters(), lr = 0.01)
    animator = kun.Animator(xlabel='epoch', ylabel='loss',yscale='log',
                        xlim=[1, num_epochs], ylim=[1e-3, 1e2],
                        legend=['train', 'test'])
    for epoch in range(num_epochs):
        kun.train_epoch_ch3(net, train_iter, loss, trainer)
        if epoch == 0 or (epoch + 1) % 20 == 0:
            animator.add(epoch + 1, (evaluate_loss(net, train_iter, loss),
                                    evaluate_loss(net, test_iter, loss)))
    print('weight:', net[0].weight .data.numpy())
    kun.plt.show()

# # 从多项式特征中选取前4个维度，即1, x, x^2/2!, x^3/3!
# # 正常拟合
# train(poly_features[:n_train, :4], poly_features[n_train:, :4], labels[:n_train], labels[n_train:])

# # 从多项式特征中选取前2个维度，即1, x
# # 欠拟合
train(poly_features[:n_train, :2], poly_features[n_train:, :2], labels[:n_train], labels[n_train:])

# # 从多项式特征中选取所有维度
# # 过拟合
# train(poly_features[:n_train, :], poly_features[n_train:, :], labels[:n_train], labels[n_train:])