import torch
from torch import nn

from ComputerVision.anchorBox import multibox_prior


# 预测每个锚框类别函数,输出通道为每个像素的情况：
# 每个像素锚框数 * 类别 （+1是指背景也算一类）
def cls_predictor(num_inputs, num_anchors, num_classes):
    return nn.Conv2d(num_inputs, num_anchors * (num_classes + 1),
                     kernel_size=(3, 3), padding=1)


# 边界框预测函数，即预测每个锚框偏移量
# 输出通道为：每个像素的多个锚框的四个偏移量
def bbox_predictor(num_inputs, num_anchors):
    return nn.Conv2d(num_inputs, num_anchors * 4, kernel_size=(3, 3), padding=1)


# 将通道移到最后一维
# permute函数将原来的维度：0, 1, 2, 3变为：0, 2, 3, 1，即把第一维放到最后
def flatten_pred(pred):
    return torch.flatten(pred.permute(0, 2, 3, 1), start_dim=1)


def concat_preds(preds):
    return torch.cat([flatten_pred(p) for p in preds], dim=1)


# 高和宽减半块
# 应⽤了在subsec_vgg-blocks中的VGG 模块设计
def down_sample_blk(in_channels, out_channels):
    blk = []
    # 两个填充为1的3 x 3的卷积层，不改变特征图形状
    for _ in range(2):
        blk.append(nn.Conv2d(in_channels, out_channels,
                             kernel_size=(3, 3), padding=1))
        blk.append(nn.BatchNorm2d(out_channels))
        blk.append(nn.ReLU())
        in_channels = out_channels
    # 步幅为2的最大池化层，特征图高宽减半
    blk.append(nn.MaxPool2d(2))
    return nn.Sequential(*blk)


# 基本网络快：⽤于从输⼊图像中抽取特征
def base_net():
    blk = []
    num_filters = [3, 16, 32, 64]
    # 将3个高宽减半块叠加在一起
    # 3次的通道数变化分别为：3->16, 16->32, 32->64
    for i in range(len(num_filters) - 1):
        blk.append(down_sample_blk(num_filters[i], num_filters[i + 1]))
    return nn.Sequential(*blk)


def get_blk(i):
    if i == 0:  # 基本快
        blk = base_net()
    elif i == 1:  # 高宽减半块
        blk = down_sample_blk(64, 128)
    elif i == 4:  # 使⽤全局最⼤池将⾼度和宽度都降到1
        blk = nn.AdaptiveMaxPool2d((1, 1))
    else:  # 高宽减半块，i == 2 || 3
        blk = down_sample_blk(128, 128)
    return blk


# 为每个块定义前向计算
# 此处的输出包括：
# (i) CNN 特征图Y
# (ii) 在当前尺度下根据Y ⽣成的锚框
# (iii) 预测的这些锚框的类别和偏移量（基于Y ）
def blk_forward(X, blk, size, ratio, cls_predictor, bbox_predictor):
    Y = blk(X)
    # 生成以每个像素为中心具有不同形状的锚框
    anchors = multibox_prior(Y, sizes=size, ratios=ratio)
    # 生成预测的类型
    cls_preds = cls_predictor(Y)
    # 生成预测的边界框
    bbox_preds = bbox_predictor(Y)
    return Y, anchors, cls_preds, bbox_preds


sizes = [[0.2, 0.272], [0.37, 0.447], [0.54, 0.619], [0.71, 0.79], [0.88, 0.961]]
ratios = [[1, 2, 0.5]] * 5
# 每个像素点生成锚框数
num_anchors = len(sizes[0]) + len(ratios[0]) - 1


# 完整的模型 TinySSD
class TinySSD(nn.Module):
    def __init__(self, num_classes, **kwargs):
        super(TinySSD, self).__init__(**kwargs)
        self.num_classes = num_classes
        idx_to_in_channels = [64, 128, 128, 128, 128]
        for i in range(5):
            # 即赋值语句'self.blk_i = get_blk(i)'
            setattr(self, f'blk_{i}', get_blk(i))
            setattr(self, f'cls_{i}', cls_predictor(idx_to_in_channels[i],
                                                    num_anchors, num_classes))
            setattr(self, f'bbox_{i}', bbox_predictor(idx_to_in_channels[i],
                                                      num_anchors))

    def forward(self, X):
        anchors, cls_preds, bbox_preds = [None] * 5, [None] * 5, [None] * 5
        for i in range(5):
            # 'getattr(self, 'blk_%d' % i)'即访问'self.blk_i'
            # \ 表示换行
            X, anchors[i], cls_preds[i], bbox_preds[i] = \
                blk_forward(X,
                            getattr(self, f'blk_{i}'), sizes[i], ratios[i],
                            getattr(self, f'cls_{i}'), getattr(self, f'bbox_{i}'))
        anchors = torch.cat(anchors, dim=1)
        cls_preds = concat_preds(cls_preds)
        # 将最后一维reshape为self.num_classes + 1，方便做softmaxtorch.Size([32, 5444, 2])
        cls_preds = cls_preds.reshape(
            cls_preds.shape[0], -1, self.num_classes + 1)
        bbox_preds = concat_preds(bbox_preds)
        return anchors, cls_preds, bbox_preds


# 定义损失函数和评价函数
# 锚框类别的损失使⽤的交叉熵损失函数：分类问题
cls_loss = nn.CrossEntropyLoss(reduction='none')
# 正类锚框偏移量的损失使⽤L1范数(预测值和真实值之差的绝对值)损失：回归问题
bbox_loss = nn.L1Loss(reduction='none')


# 掩码变量bbox_masks 令负类锚框和填充锚框不参与损失的计算
def cls_bbox_loss(cls_preds, cls_labels, bbox_preds, bbox_labels, bbox_masks):
    batch_size, num_classes = cls_preds.shape[0], cls_preds.shape[2]
    cls = cls_loss(cls_preds.reshape(-1, num_classes),
                   cls_labels.reshape(-1)).reshape(batch_size, -1).mean(dim=1)
    bbox = bbox_loss(bbox_preds * bbox_masks,
                     bbox_labels * bbox_masks).mean(dim=1)
    # 将锚框类别和偏移量的损失相加，以获得模型的最终损失函数
    return cls + bbox


def cls_eval(cls_preds, cls_labels):
    # 由于类别预测结果放在最后⼀维，'argmax'需要指定最后⼀维。
    return float((cls_preds.argmax(dim=-1).type(
        cls_labels.dtype) == cls_labels).sum())


def bbox_eval(bbox_preds, bbox_labels, bbox_masks):
    return float((torch.abs((bbox_labels - bbox_preds) * bbox_masks)).sum())
