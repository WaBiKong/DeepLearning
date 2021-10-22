import os
import torch
import torchvision
from matplotlib import pyplot as plt
from torch import nn
from d2l import torch as d2l

# 下载解压热狗数据集
d2l.DATA_HUB['hotdog'] = (d2l.DATA_URL + 'hotdog.zip',
                          'fba480ffa8aa7e0febbb511d181409f899b9baa5')
data_dir = d2l.download_extract('hotdog')

train_imgs = torchvision.datasets.ImageFolder(os.path.join(data_dir, 'train'))
test_imgs = torchvision.datasets.ImageFolder(os.path.join(data_dir, 'test'))

hotdogs = [train_imgs[i][0] for i in range(8)]
not_hotdogs = [train_imgs[-i - 1][0] for i in range(8)]
d2l.show_images(hotdogs + not_hotdogs, 2, 8, scale=1.4)
plt.show()

# 使用三个RGB通道的均值和标准偏差，以标准化每个通道
# 具体而言，通道的平均值将从该通道的每个值中减去，然后将结果除以该通道的标准差
normalize = torchvision.transforms.Normalize([0.485, 0.456, 0.406],
                                             [0.229, 0.224, 0.225])

train_augs = torchvision.transforms.Compose([
    # 从图像中裁切随机大小和随机长宽比的区域，然后将该区域缩放为224×224输入图像
    torchvision.transforms.RandomResizedCrop(224),
    torchvision.transforms.RandomHorizontalFlip(),  # 左右翻转
    torchvision.transforms.ToTensor(),
    normalize
])
test_augs = torchvision.transforms.Compose([
    # 将图像的高度和宽度都缩放到256像素
    torchvision.transforms.Resize(256),
    # 裁剪中央224×224区域作为输入
    torchvision.transforms.CenterCrop(224),
    torchvision.transforms.ToTensor(),
    normalize
])


def train_fine_tuning(net, learning_rate, batch_size=128, num_epochs=5, param_group=True):
    train_iter = torch.utils.data.DataLoader(
        torchvision.datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=train_augs),
        batch_size=batch_size, shuffle=True
    )
    test_iter = torch.utils.data.DataLoader(
        torchvision.datasets.ImageFolder(os.path.join(data_dir, 'test'), transform=test_augs),
        batch_size=batch_size, shuffle=False
    )
    devices = d2l.try_all_gpus()
    loss = nn.CrossEntropyLoss(reduction="none")
    if param_group:  # param_group=True则对随机初始化后的最后一层进行梯度更新
        # 把不是最后一层的所有层的参数拿出来
        params_lx = [
            param for name, param in net.named_parameters()
            # 如果name不是最后一层
            if name not in ["fc.weight", "fc.bias"]
        ]
        trainer = torch.optim.SGD(
            [
                {'params': params_lx},  # 使用默认学习率
                {
                    # 最后一层学习率为10倍的默认学习率
                    'params': net.fc.parameters(),
                    'lr': learning_rate * 10
                }
            ],
            lr=learning_rate, weight_decay=0.001
        )
    else:
        trainer = torch.optim.SGD(net.parameters(), lr=learning_rate, weight_decay=0.001)
    d2l.train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs, devices)


finetune_net = torchvision.models.resnet18(pretrained=True)
# net.fc:表示此网络的最后一层
finetune_net.fc = nn.Linear(finetune_net.fc.in_features, 2)
nn.init.xavier_uniform_(finetune_net.fc.weight)  # 只对最后一层的参数随机初始化
train_fine_tuning(finetune_net, 5e-5)
