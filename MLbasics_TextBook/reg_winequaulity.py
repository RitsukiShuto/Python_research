# Created by RitsukiShuto on 2022/05/06.
# reg_winequaulity.py
#
import linearreg
import numpy as np
import csv

# データの読み込み
Xy = []
with open("./data/wine.csv") as fp:
    for row in csv.reader(fp, delimiter=";"):
        Xy.append(row)

Xy = np.array(Xy[1:], dtype=np.float64)

# 訓練データとテストデータに分割
np.random.seed(0)
np.random.shuffle(Xy)
train_X = Xy[:-1000, :-1]
train_y = Xy[:-1000, -1]
test_X = Xy[-1000:, :-1]
test_y = Xy[-1000:, -1]

# 学習
model = linearreg.LinearRegression()
model.fit(train_X, train_y)

# テストデータにモデルを適用
y = model.predict(test_X)

# 結果を表示
print("最初の５つの正解と予測値:")
for i in range(5):
    print("{:1.0f} {:5.3f}".format(test_y[i], y[i]))

print()
print("RMSE:", np.sqrt(((test_y - y) ** 2).mean()))     # 二乗誤差で予測値の評価