# Created by RitsukiShuto on 2022/05/19.
# DNN.py
#
import numpy as np
import matplotlib.pyplot as plt

# 人工データを作成
N1 = 100
N2 = 100
N3 = 100

X1 = np.zeros([N1, 2])
X2 = np.zeros([N2, 2])
X3 = np.zeros([N3, 2])

Y1 = np.ones(N1, dtype='int')
Y2 = np.ones(N2, dtype='int' ) * 2
Y3 = np.ones(N3, dtype='int' ) * 3

X1[:, 0] = np.random.normal(80, 5, N1)
X1[:, 1] = np.random.normal(80, 5, N1)

X2[:, 0] = np.random.normal(80, 2, N2)
X2[:, 1] = np.random.normal(20, 5, N2)

X3[:, 0] = np.random.normal(30, 6, N3)
X3[:, 1] = np.random.normal(90, 4, N3)

X = np.concatenate([X1, X2, X3], 0)
Y = np.concatenate([Y1, Y2, Y3], 0)

print(X.shape)
print(Y.shape)

# 描画
fig = plt.figure(figsize = (8, 8))
plt.scatter(X1[:, 0], X1[:, 1], marker='o', c = 'red', s=20)
plt.scatter(X2[:, 0], X2[:, 1], marker='o', c = 'blue', s=20)
plt.scatter(X3[:, 0], X3[:, 1], marker='o', c = 'green', s=20)

# plt.show()

# 
