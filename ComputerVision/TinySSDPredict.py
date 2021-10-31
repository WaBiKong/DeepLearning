import matplotlib.pyplot as plt
import torch
import torchvision
from torch.nn import functional as F

from ComputerVision.TinySSD import TinySSD
from ComputerVision.anchorBox import multibox_detection, show_bboxes
from DeepLearningComputing.GPU import try_gpu


X = torchvision.io.read_image('../data/90.png').unsqueeze(0).float()
print(X.shape)
img = X.squeeze(0).permute(1, 2, 0).long()
net_state_dict = torch.load('./TinySSD.pt')
device, net = try_gpu(), TinySSD(num_classes=1)
net.load_state_dict(net_state_dict)
net = net.to(device)


# 使⽤multibox_detection函数根据锚框及其预测偏移量得到预测边界框
# 通过⾮极⼤值抑制来移除相似的预测边界框。
def predict(X):
    net.eval()
    anchors, cls_preds, bbox_preds = net(X.to(device))
    cls_probs = F.softmax(cls_preds, dim=2).permute(0, 2, 1)
    output = multibox_detection(cls_probs, bbox_preds, anchors)
    idx = [i for i, row in enumerate(output[0]) if row[0] != -1]
    return output[0, idx]


output = predict(X)


# 筛选所有置信度不低于threshold的边界框，做为最终输出
def display(img, output, threshold):
    plt.figure(figsize=(4, 4))  # 设置图层大小，未设置则使用默认大小
    fig = plt.imshow(img)
    for row in output:
        score = float(row[1])
        if score < threshold:
            continue
        h, w = img.shape[0:2]
        bbox = [row[2:6] * torch.tensor((w, h, w, h), device=row.device)]
        show_bboxes(fig.axes, bbox, '%.2f' % score, 'b')


display(img, output.cpu(), threshold=0.9)
plt.show()
