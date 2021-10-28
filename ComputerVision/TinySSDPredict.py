import matplotlib.pyplot as plt
import torch
import torchvision
from d2l.torch import d2l
from torch.nn import functional as F
from ComputerVision.TinySSD import TinySSD

from ComputerVision.anchorBox import multibox_detection
from DeepLearningComputing.GPU import try_gpu

X = torchvision.io.read_image('../data/banana.jpg').unsqueeze(0).float()
img = X.squeeze(0).permute(1, 2, 0).long()
net_state_dict = torch.load('./TinySSD.pt')
device, net = try_gpu(), TinySSD(num_classes=1)
net.load_state_dict(net_state_dict)
net = net.to(device)


def predict(X):
    net.eval()
    anchors, cls_preds, bbox_preds = net(X.to(device))
    cls_probs = F.softmax(cls_preds, dim=2).permute(0, 2, 1)
    output = multibox_detection(cls_probs, bbox_preds, anchors)
    idx = [i for i, row in enumerate(output[0]) if row[0] != -1]
    return output[0, idx]


output = predict(X)


def display(img, output, threshold):
    d2l.set_figsize((5, 5))
    fig = d2l.plt.imshow(img)
    for row in output:
        score = float(row[1])
        if score < threshold:
            continue
        h, w = img.shape[0:2]
        bbox = [row[2:6] * torch.tensor((w, h, w, h), device=row.device)]
        d2l.show_bboxes(fig.axes, bbox, '%.2f' % score, 'b')


display(img, output.cpu(), threshold=0.9)
plt.show()
