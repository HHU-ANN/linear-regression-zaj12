# 最终在main函数中传入一个维度为6的numpy数组，输出预测值

import os

try:
    import numpy as np
except ImportError as e:
    os.system("sudo pip3 install numpy")
    import numpy as np


def ridge(data):
    lam = 0.5
    x, y = read_data()
    weight = np.dot(np.linalg.inv(np.dot(x.T, x) + np.eye(x.shape[1]) * lam), np.dot(x.T, y))
    return data @ weight
    
def lasso(data):
    x, y = read_data()
    # 归一化
    x = (x - np.mean(x, axis=0)) / np.std(x, axis=0)
    y = (y - np.mean(y)) / np.std(y)

    # 初始化参数
    def init(dim):
        weight = np.zeros(dim)
        b = 0
        return weight, b

    # 定义lasso损失函数
    def l1_loss(x, y, w, b, lamb):
        samples = x.shape[0]
        y_hat = np.dot(x, w) + b
        loss = np.sum((y_hat - y) ** 2) / samples + lamb * np.sum(abs(w))
        dw = np.dot(x.T, (y_hat - y)) / samples + lamb * np.sign(w)
        db = np.sum((y_hat - y)) / samples
        return y_hat, loss, dw, db

    # 训练过程
    def lasso_train(x, y, learning_rate = 0.1, epochs = 500):
        loss_list = []
        weight, b = init(x.shape[1])
        for i in range(1, epochs):
            y_hat, loss, dw, db = l1_loss(x, y, weight, b, 0.5)
            weight += -learning_rate * dw
            b += -learning_rate * db
            loss_list.append(loss)

        return loss_list, weight, b

    loss_list, weight, b = lasso_train(x, y)
    y_pred = np.dot(data, weight) + b
    return loss_list, y_pred

def read_data(path='./data/exp02/'):
    x = np.load(path + 'X_train.npy')
    y = np.load(path + 'y_train.npy')
    return x, y

'''
features = np.array([
    [2.0133330e+03, 1.6400000e+01, 2.8932480e+02, 5.0000000e+00, 2.4982030e+01, 1.2154348e+02],
    [2.0126670e+03, 2.3000000e+01, 1.3099450e+02, 6.0000000e+00, 2.4956630e+01, 1.2153765e+02],
    [2.0131670e+03, 1.9000000e+00, 3.7213860e+02, 7.0000000e+00, 2.4972930e+01, 1.2154026e+02],
    [2.0130000e+03, 5.2000000e+00, 2.4089930e+03, 0.0000000e+00, 2.4955050e+01, 1.2155964e+02],
    [2.0134170e+03, 1.8500000e+01, 2.1757440e+03, 3.0000000e+00, 2.4963300e+01, 1.2151243e+02],
    [2.0130000e+03, 1.3700000e+01, 4.0820150e+03, 0.0000000e+00, 2.4941550e+01, 1.2150381e+02],
    [2.0126670e+03, 5.6000000e+00, 9.0456060e+01, 9.0000000e+00, 2.4974330e+01, 1.2154310e+02],
    [2.0132500e+03, 1.8800000e+01, 3.9096960e+02, 7.0000000e+00, 2.4979230e+01, 1.2153986e+02],
    [2.0130000e+03, 8.1000000e+00, 1.0481010e+02, 5.0000000e+00, 2.4966740e+01, 1.2154067e+02],
    [2.0135000e+03, 6.5000000e+00, 9.0456060e+01, 9.0000000e+00, 2.4974330e+01, 1.2154310e+02]
    ])
labels = np.array([41.2, 37.2, 40.5, 22.3, 28.1, 15.4, 50. , 40.6, 52.5, 63.9])

loss_list, y_pred = lasso(features)
print(loss_list)
print(y_pred, labels)
'''