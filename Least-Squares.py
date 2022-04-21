# Created by RitsukiShuto on 2022/04/21.
# 最小二乗法<least-squares method>
#
import matplotlib.pyplot as plt
import numpy as np

# データ
x = np.array([1, 2, 4, 6, 7])
y = np.array([1, 3, 3, 5, 4])

# plt.scatter(x, y)   # 散布図を描画
# plt.show()

def reg1dim(x, y):
    a = np.dot(x, y) / (x ** 2).sum()

    return a

a = reg1dim(x, y)

plt.scatter(x, y, color = "k")
plt.plot([0, x.max()], [0, a * x.max()])
plt.show()