import numpy as np
from scipy.stats import norm
import tensorflow as tf
import sys
import math
from keras.layers import Input, Dense, Lambda, Layer, Concatenate
from keras.models import Model, model_from_json
from keras import backend, metrics, optimizers

# 描画
import matplotlib.pyplot as plt

# 2次元正規分布から1点サンプリングする補助関数です
def sampling(args):
 from keras import backend
 # batch_size = 200
 latent_dim = 2
 epsilon_std = 1.0
 z_mean, z_log_var = args
 batch_size = tf.shape(z_mean)[0]
 epsilon = backend.random_normal(shape=(batch_size, latent_dim),mean=0., stddev=epsilon_std)
 return z_mean + backend.exp(z_log_var / 2) * epsilon

# Keras の Layer クラスを継承してオリジナルの損失関数を付加するレイヤーをつくります
class CustomVariationalLayer(Layer):
 def __init__(self, **kwargs):
    self.is_placeholder = True
    super(CustomVariationalLayer, self).__init__(**kwargs)
    def vae_loss(self, x, x_decoded_mean, z_log_var, z_mean): # オリジナルの損失関数
      # 入力と出力の交差エントロピー
      # 精度
      beta = 1000
      xent_loss = beta * original_dim * metrics.mse(x,x_decoded_mean)
      # xent_loss = original_dim * metrics.binary_crossentropy(x,x_decoded_mean)
      # 事前分布と事後分布のKL情報量
      kl_loss = - 0.5 * backend.sum(1 + z_log_var -backend.square(z_mean) - backend.exp(z_log_var), axis=-1)

      return backend.mean(xent_loss + kl_loss)

 def call(self, inputs):
   x = inputs[0]
   x_decoded_mean = inputs[1]
   z_log_var = inputs[2]
   z_mean = inputs[3]
   loss = self.vae_loss(x, x_decoded_mean, z_log_var, z_mean)
   self.add_loss(loss, inputs=inputs) # オリジナルの損失関数を付加

   return x
 
# この自作レイヤーの出力を一応定義しておきますが、今回この出力は全く使いません
# main文
# データ読み込み
N = 200
D1 = 2
D2 = 2
val1 = 0.2
val2 = 0.2
Z = np.linspace(-1,1,N)
Z_theta = math.pi * Z**2


# 教師あり
# ビュー1
X1 = np.zeros([N, D1])
X1[:,0] = Z
X1[:,1] = np.sin(Z_theta)
X1[:, 0] += np.random.normal(0, val1, N)
X1[:, 1] += np.random.normal(0, val1, N)

# ビュー2
X2 = np.zeros([N, D2])
X2[:,0] = Z
X2[:,1] = np.cos(Z_theta)
X2[:, 0] += np.random.normal(0, val2, N)
X2[:, 1] += np.random.normal(0, val2, N)

# 教師なし
NN = 2000
ZZ = np.linspace(-1,1,NN)
ZZ_theta = math.pi * ZZ**2

# ビュー1
X1_un = np.zeros([NN, D1])
X1_un[:,0] = ZZ
X1_un[:,1] = np.sin(ZZ_theta)
X1_un[:, 0] += np.random.normal(0, val1, NN)
X1_un[:, 1] += np.random.normal(0, val1, NN)

# ビュー2
X2_un = np.zeros([NN, D2])
X2_un[:,0] = ZZ
X2_un[:,1] = np.cos(ZZ_theta)
X2_un[:, 0] += np.random.normal(0, val2, NN)
X2_un[:, 1] += np.random.normal(0, val2, NN)


# 検証用
# ビュー1
X1_set = np.zeros([N, D1])
X1_set[:,0] = Z
X1_set[:,1] = np.sin(Z_theta)
X1_set[:, 0] += np.random.normal(0, val1, N)
X1_set[:, 1] += np.random.normal(0, val1, N)

# ビュー2
X2_set = np.zeros([N, D2])
X2_set[:,0] = Z
X2_set[:,1] = np.cos(Z_theta)
X2_set[:, 0] += np.random.normal(0, val2, N)
X2_set[:, 1] += np.random.normal(0, val2, N)
batch_size = N
original_dim = 2 # data_dim
latent_dim = 2
intermediate_dim = 50
epochs = 500
epsilon_std = 1.0


# 変分自己符号化器を構築します
# エンコーダ
x1 = Input(batch_shape=(batch_size, original_dim))
h11 = Dense(intermediate_dim, activation='softplus')(x1)
h12 = Dense(intermediate_dim, activation='softplus')(h11)
z1_mean = Dense(latent_dim)(h12)
z1_log_var = Dense(latent_dim)(h12)
z1 = Lambda(sampling, output_shape=(latent_dim,))([z1_mean,z1_log_var])
encoder1 = Model(x1, z1_mean) # エンコーダのみ分離
x2 = Input(batch_shape=(batch_size, original_dim))
h21 = Dense(intermediate_dim, activation='softplus')(x2)
h22 = Dense(intermediate_dim, activation='softplus')(h21)
z2_mean = Dense(latent_dim)(h22)
z2_log_var = Dense(latent_dim)(h22)
z2 = Lambda(sampling, output_shape=(latent_dim,))([z2_mean,z2_log_var])
encoder2 = Model(x2, z2_mean) # エンコーダのみ分離

# デコーダ
decoder_input1 = Input(shape=(latent_dim,))
decoder_h11 = Dense(intermediate_dim, activation='softplus')(decoder_input1)
decoder_h12 = Dense(intermediate_dim, activation='softplus')(decoder_h11)
decoder_mean1 = Dense(original_dim, activation=None)(decoder_h12)
decoder1 = Model(decoder_input1, decoder_mean1) # デコーダのみ分離
x_decoded_mean1 = decoder1(z1)
decoder_input2 = Input(shape=(latent_dim,))
decoder_h21 = Dense(intermediate_dim, activation='softplus')(decoder_input2)
decoder_h22 = Dense(intermediate_dim, activation='softplus')(decoder_h21)
decoder_mean2 = Dense(original_dim, activation=None)(decoder_h22)
decoder2 = Model(decoder_input2, decoder_mean2) # デコーダのみ分離
x_decoded_mean2 = decoder2(z2)

# カスタマイズした損失関数を付加する訓練用レイヤー
y1 = CustomVariationalLayer()([x1, x_decoded_mean1, z1_log_var,z1_mean])
y2 = CustomVariationalLayer()([x2, x_decoded_mean2, z2_log_var,z2_mean])
vae = Model([x1, x2], [y1, y2])
vae.compile(optimizer=optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08), loss=[None, None])
# vae.compile(optimizer='rmsprop', loss=None)
vae.summary()


#============================================================
# モデルを訓練します
add_no = 200

# 学習
print(X1.shape)
print(X2.shape)
vae.fit(x=[X1, X2], y=None, shuffle=True, batch_size=batch_size, initial_epoch=0, epochs=500)

# 仮ラベル
encoder1 = Model(x1, z1_mean) # エンコーダのみ分離
x1_train_encoded = encoder1.predict(X1_un, batch_size=batch_size)
encoder2 = Model(x2, z2_mean) # エンコーダのみ分離
x2_train_encoded = encoder2.predict(X2_un, batch_size=batch_size)
x1_decode = decoder1.predict(x2_train_encoded)
x2_decode = decoder1.predict(x1_train_encoded)

# 増やす
squared_error = (X1_un-x1_decode)**2
sum_squared_error = np.sum(squared_error, axis=1)
X1_min_No = np.argsort(sum_squared_error)[::-1]
squared_error = (X2_un-x2_decode)**2
sum_squared_error = np.sum(squared_error, axis=1)
X2_min_No = np.argsort(sum_squared_error)[::-1]
X1 = np.concatenate([X1, X1_un[X1_min_No[0:add_no]]], 0)
X2 = np.concatenate([X2, x2_decode[X1_min_No[0:add_no]]], 0)
X1 = np.concatenate([X1, x1_decode[X2_min_No[0:add_no]]], 0)
X2 = np.concatenate([X2, X2_un[X2_min_No[0:add_no]]], 0)
X1_un = np.delete(X1_un, X1_min_No[0:add_no], 0)
X2_un = np.delete(X2_un, X2_min_No[0:add_no], 0)

# 学習
print(X1.shape)
print(X2.shape)
vae.fit(x=[X1, X2], y=None, shuffle=True, batch_size=batch_size,initial_epoch=500, epochs=1000)

# 仮ラベル
encoder1 = Model(x1, z1_mean) # エンコーダのみ分離
x1_train_encoded = encoder1.predict(X1_un, batch_size=batch_size)
encoder2 = Model(x2, z2_mean) # エンコーダのみ分離
x2_train_encoded = encoder2.predict(X2_un, batch_size=batch_size)
x1_decode = decoder1.predict(x2_train_encoded)
x2_decode = decoder1.predict(x1_train_encoded)

# 増やす
squared_error = (X1_un-x1_decode)**2
sum_squared_error = np.sum(squared_error, axis=1)
X1_min_No = np.argsort(sum_squared_error)[::-1]
squared_error = (X2_un-x2_decode)**2
sum_squared_error = np.sum(squared_error, axis=1)
X2_min_No = np.argsort(sum_squared_error)[::-1]
X1 = np.concatenate([X1, X1_un[X1_min_No[0:add_no]]], 0)
X2 = np.concatenate([X2, x2_decode[X1_min_No[0:add_no]]], 0)
X1 = np.concatenate([X1, x1_decode[X2_min_No[0:add_no]]], 0)
X2 = np.concatenate([X2, X2_un[X2_min_No[0:add_no]]], 0)
X1_un = np.delete(X1_un, X1_min_No[0:add_no], 0)
X2_un = np.delete(X2_un, X2_min_No[0:add_no], 0)

# 学習
print(X1.shape)
print(X2.shape)
vae.fit(x=[X1, X2], y=None, shuffle=True, batch_size=batch_size,initial_epoch=1000, epochs=1500)

# 仮ラベル
encoder1 = Model(x1, z1_mean) # エンコーダのみ分離
x1_train_encoded = encoder1.predict(X1_un, batch_size=batch_size)
encoder2 = Model(x2, z2_mean) # エンコーダのみ分離
x2_train_encoded = encoder2.predict(X2_un, batch_size=batch_size)
x1_decode = decoder1.predict(x2_train_encoded)
x2_decode = decoder1.predict(x1_train_encoded)

# 増やす
squared_error = (X1_un-x1_decode)**2
sum_squared_error = np.sum(squared_error, axis=1)
X1_min_No = np.argsort(sum_squared_error)[::-1]
squared_error = (X2_un-x2_decode)**2
sum_squared_error = np.sum(squared_error, axis=1)
X2_min_No = np.argsort(sum_squared_error)[::-1]
X1 = np.concatenate([X1, X1_un[X1_min_No[0:add_no]]], 0)
X2 = np.concatenate([X2, x2_decode[X1_min_No[0:add_no]]], 0)
X1 = np.concatenate([X1, x1_decode[X2_min_No[0:add_no]]], 0)
X2 = np.concatenate([X2, X2_un[X2_min_No[0:add_no]]], 0)
X1_un = np.delete(X1_un, X1_min_No[0:add_no], 0)
X2_un = np.delete(X2_un, X2_min_No[0:add_no], 0)

# 学習
print(X1.shape)
print(X2.shape)
vae.fit(x=[X1, X2], y=None, shuffle=True, batch_size=batch_size,initial_epoch=1500, epochs=2000)

# 仮ラベル
encoder1 = Model(x1, z1_mean) # エンコーダのみ分離
x1_train_encoded = encoder1.predict(X1_un, batch_size=batch_size)
encoder2 = Model(x2, z2_mean) # エンコーダのみ分離
x2_train_encoded = encoder2.predict(X2_un, batch_size=batch_size)
x1_decode = decoder1.predict(x2_train_encoded)
x2_decode = decoder1.predict(x1_train_encoded)

# 増やす
squared_error = (X1_un-x1_decode)**2
sum_squared_error = np.sum(squared_error, axis=1)
X1_min_No = np.argsort(sum_squared_error)[::-1]
squared_error = (X2_un-x2_decode)**2
sum_squared_error = np.sum(squared_error, axis=1)
X2_min_No = np.argsort(sum_squared_error)[::-1]
X1 = np.concatenate([X1, X1_un[X1_min_No[0:add_no]]], 0)
X2 = np.concatenate([X2, x2_decode[X1_min_No[0:add_no]]], 0)
X1 = np.concatenate([X1, x1_decode[X2_min_No[0:add_no]]], 0)
X2 = np.concatenate([X2, X2_un[X2_min_No[0:add_no]]], 0)
X1_un = np.delete(X1_un, X1_min_No[0:add_no], 0)
X2_un = np.delete(X2_un, X2_min_No[0:add_no], 0)
print(X1.shape)
print(X2.shape)
vae.fit(x=[X1, X2], y=None, shuffle=True, batch_size=batch_size, initial_epoch=2000, epochs=2500)


#============================================================
# 結果を表示します
print(X1.shape)
print(X2.shape)
predict = vae.predict([X1_set, X2_set])
from sklearn.metrics import mean_squared_error
# mse1 = mean_squared_error(predict[0], X1_set)
# mse2 = mean_squared_error(predict[1], X2_set)
# np.set_printoptions(suppress=True)
# print('{:.30f}'.format(mse1))
# print('{:.30f}'.format(mse2))

# 新規データの予測
encoder1 = Model(x1, z1_mean) # エンコーダのみ分離
x1_train_encoded = encoder1.predict(X1_set, batch_size=batch_size)
encoder2 = Model(x2, z2_mean) # エンコーダのみ分離
x2_train_encoded = encoder2.predict(X2_set, batch_size=batch_size)
x1_decode = decoder1.predict(x2_train_encoded)
x2_decode = decoder2.predict(x1_train_encoded)
mse1 = mean_squared_error(x1_decode, X1_set)
mse2 = mean_squared_error(x2_decode, X2_set)
np.set_printoptions(suppress=True)
print('{:.30f}'.format(mse1))
print('{:.30f}'.format(mse2))

# 描画
fig1 = plt.figure(figsize=(6, 6))
ax1 = fig1.add_subplot(2, 2, 1)