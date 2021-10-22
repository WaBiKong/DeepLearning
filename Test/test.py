import torchvision
from torchvision import transforms

import kun
import torch
from matplotlib import pyplot as plt


batch_size = 18
train_iter, test_iter = kun.load_data_mnist(batch_size, 28)


def get_mnist_labels(labels):
    """返回Fashion-MNIST数据集的文本标签"""
    text_labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    return [text_labels[int(i)] for i in labels]


# 样本可视化函数
def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5):
    """Plot a list of image."""
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.flatten()
    for i, (ax, img) in enumerate(zip(axes, imgs)):
        if torch.is_tensor(img):
            # 图片张量
            ax.imshow(img.numpy())
        else:
            # PIL图片
            ax.imshow(img)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        if titles:
            ax.set_title(titles[i])
    return axes


X, y = next(iter(train_iter))
print(X.shape)
show_images(X.reshape(18, 28, 28), 2, 9, titles=get_mnist_labels(y))
plt.show()

trans = [transforms.ToTensor()]

mnist_train = torchvision.datasets.MNIST(
    root="../data", train=True, transform=trans, download=True
)
