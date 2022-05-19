# Created by RitsukiShuto on 2022/05/19.
# DNN.py
#
import numpy as np

N1 = 100
N2 = 100
N3 = 100

X1 = np.zeros([N1, 2])
X2 = np.zeros([N2, 2])
X3 = np.zeros([N3, 2])

Y1 = np.ones(N1, dtype='int')
Y2 = np.ones(N2, dtype='int' ) * 2
Y3 = np.ones(N3, dtype='int' ) * 3