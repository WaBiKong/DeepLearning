import torchvision
from matplotlib import pyplot as plt
from d2l import torch as d2l

d2l.set_figsize()
img = d2l.Image.open('../data/cat1.jpg')
print(type(img))
plt.imshow(img)
plt.show()


# aug: 使用什么方法数据增广
def apply(img, aug, num_rows=2, num_cols=4, scale=1.5):
    Y = [aug(img) for _ in range(num_rows * num_cols)]
    d2l.show_images(Y, num_rows, num_cols, scale=scale)
    plt.show()


# RandomHorizontalFlip()：torchvision.transforms中提供的数据增广方法
# 它有50%的机率时图像左右翻转
# RandomVerticalFlip()：torchvision.transforms中提供的数据增广方法
# 它有50%的机率时图像上下翻转
apply(img, torchvision.transforms.RandomHorizontalFlip())  # 左右翻转
apply(img, torchvision.transforms.RandomVerticalFlip())  # 上下翻转

# 随机裁剪一个面积为原始面积10%到100%的区域: scale=(0.1, 1)
# 该区域的宽高比从0.5到2之间随机取值: ratio=(0.5, 2)
# 然后，区域的宽度和高度都被缩放到200像素: (200, 200)
shape_aug = torchvision.transforms.RandomResizedCrop(
    (200, 200), scale=(0.1, 1), ratio=(0.5, 2)
)
apply(img, shape_aug)

# 改变图像颜色的四个方面：亮度、对比度、饱和度和色调
# 随机更改图像的亮度，随机值为原始图像的50%(1−0.5)到150%(1+0.5)之间
apply(img, torchvision.transforms.ColorJitter(brightness=0.5, contrast=0,
                                              saturation=0, hue=0))
# 随机更改图像的色调
apply(img, torchvision.transforms.ColorJitter(brightness=0, contrast=0,
                                              saturation=0, hue=0.5))
# 创建一个 RandomColorJitter 实例
# 同时随机更改图像的亮度（brightness）、对比度（contrast）、饱和度（saturation）和色调（hue）
color_aug = torchvision.transforms.ColorJitter(brightness=0.5, contrast=0.5,
                                               saturation=0.5, hue=0.5)
apply(img, color_aug)

# 使用一个 Compose 实例来综合上面定义的不同的图像增广方法，并将它们应用到每个图像
augs = torchvision.transforms.Compose([
    torchvision.transforms.RandomHorizontalFlip(), color_aug, shape_aug
])
apply(img, augs)
