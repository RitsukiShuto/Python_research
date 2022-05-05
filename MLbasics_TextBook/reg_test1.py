# Created by RitsukiShuto on 2022/04/30.
# LinearRegression
# leg_test1.py
#
from matplotlib import projections
import linearreg
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d

n = 100
scale = 10

np.random.seed(0)
X = np.random.random((n, 2)) * scale    # 100*2の行列を生成

w0 = 1
w1 = 2
w2 = 3

y = w0 + w1 * X[:, 0] + w2 * X[:, 1] + np.random.random(n)

model = linearreg.LinearRegression()
model.fit(X, y)

print("係数", model.w_)
print("(1, 1)に対する予測値:", model.predict(np.array([1, 1])))

x_mesh, y_mesh = np.meshgrid(np.linspace(0, scale, 20),
                             np.linspace(0, scale, 20))

z_mesh = (model.w_[0] + model.w_[1] * x_mesh.ravel() +
          model.w_[2] * y_mesh.ravel()).reshape(x_mesh.shape)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d',)
ax.scatter(X[:, 0], X[:, 1], y, color="k")
ax.plot_wireframe(x_mesh, y_mesh, z_mesh, color="r")

plt.show()