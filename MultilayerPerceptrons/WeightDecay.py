# 权重衰退
import torch
import kun

# 初始化参数
n_train, n_test, num_inputs, batch_size = 100, 100, 200, 5
true_w, true_b = torch.ones((num_inputs, 1)) * 0.01, 0.05
# 生成数据集
train_data = kun.synthetic_data(true_w, true_b, n_train)
train_iter = kun.load_array(train_data, batch_size)
test_data = kun.synthetic_data(true_w, true_b, n_test)
test_iter = kun.load_array(test_data, batch_size, is_train=False)

# 初始化模型参数
def init_params():
    w = torch.normal(0, 1, size=(num_inputs, 1), requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    return [w, b]

# 定义L2范数惩罚
def l2_penalty(w):
    return torch.sum(w.pow(2)) / 2

def train(lambd):
    w, b = init_params()
    net, loss = lambda X: kun.linreg(X, w, b), kun.squared_loss
    num_epochs, lr = 100, 0.003
    animator = kun.Animator(xlabel='epochs', ylabel='loss', yscale='log',
                            xlim=[5, num_epochs], legend=['train', 'test'])
    for epoch in range(num_epochs):
        for X, y in train_iter:
            with torch.enable_grad():
                # 增加了L2范数惩罚项，广播机制使l2_penalty(w)成为一个长度为'batch_size'的向量
                l = loss(net(X), y) + lambd * l2_penalty(w)
            l.sum().backward()
            kun.sgd([w, b], lr, batch_size)
        if(epoch + 1) % 5 == 0:
            animator.add(epoch + 1, (kun.evaluate_loss(net, train_iter, loss),
                                     kun.evaluate_loss(net, test_iter, loss)))

    print('w的L2范数是：', torch.norm(w).item())
    kun.plt.show()

# 忽视正则化直接训练
train(lambd=0)

# # 使用权重衰退
# train(lambd=3)
