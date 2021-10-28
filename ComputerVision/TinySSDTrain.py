import matplotlib.pyplot as plt

from kun import Animator, Accumulator

from ComputerVision.MyData import load_data_bananas
from ComputerVision.anchorBox import multibox_target
from ComputerVision.TinySSD import *
from DeepLearningComputing.GPU import try_gpu

# 读取香蕉数据集
batch_size = 16
train_iter, _ = load_data_bananas(batch_size)
# 初始化其参数并定义优化算法
device, net = try_gpu(), TinySSD(num_classes=1)
# 使用了权重衰退weight_decay=5e-4
trainer = torch.optim.SGD(net.parameters(), lr=0.2, weight_decay=5e-4)
# 训练模型
num_epochs = 20
animator = Animator(xlabel='epoch', xlim=[1, num_epochs],
                    legend=['class error', 'bbox mae'])
net = net.to(device)
for epoch in range(num_epochs):
    # 训练精确度的和，训练精确度的和中的示例数
    # 绝对误差的和，绝对误差的和中的示例数
    metric = Accumulator(4)
    net.train()
    for features, target in train_iter:
        timer.start()
        trainer.zero_grad()
        X, Y = features.to(device), target.to(device)
        # 生成多尺度的锚框，为每个锚框预测类别和偏移量
        anchors, cls_preds, bbox_preds = net(X)
        # 为每个锚框标注类别和偏移量
        bbox_labels, bbox_masks, cls_labels = multibox_target(anchors, Y)
        # 根据类别和偏移量的预测和标注值计算损失函数
        l = cls_bbox_loss(cls_preds, cls_labels, bbox_preds, bbox_labels,
                          bbox_masks)
        l.mean().backward()
        trainer.step()
        metric.add(cls_eval(cls_preds, cls_labels), cls_labels.numel(),
                   bbox_eval(bbox_preds, bbox_labels, bbox_masks),
                   bbox_labels.numel())
    cls_err, bbox_mae = 1 - metric[0] / metric[1], metric[2] / metric[3]
    animator.add(epoch + 1, (cls_err, bbox_mae))
print(f'class err {cls_err:.2e}, bbox mae {bbox_mae:.2e}')
print(f'{len(train_iter.dataset) / timer.stop():.1f} examples/sec on '
      f'{str(device)}')

plt.show()

torch.save(net.state_dict(), './TinySSD.pt')

