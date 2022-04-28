# Created by RitsukiShuto on 2022/04/21.
# 最小二乗法<least-squares method>
#
import matplotlib.pyplot as plt
import numpy as np

# CSVを読み込む
data = np.loadtxt("./housing.csv")
print(data)

x = data[:, 9]
y = data[:, 13]

# x = np.reshape(506, 1)

#print(x)
#print(y)

plt.scatter(x, y)   # 散布図を描画
plt.show()

def reg1dim(x, y):
    n = len(x)
    a = ((np.dot(x, y) - y.sum() * x.sum() / n) /
        ((x ** 2).sum() - x.sum() ** 2 / n))

    b = (y.sum() - a * x.sum()) / n

    return a, b

a, b = reg1dim(x, y)

plt.scatter(x, y, color = "k")
plt.plot([0, x.max()], [b, a * x.max() + b])
plt.show()